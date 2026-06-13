from __future__ import print_function

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import dopt_rsag as hvd
from cifar_models import cifar_resnet18, cifar_vgg16
from compression import compressors


hvd.init()

os.environ["HOROVOD_FUSION_THRESHOLD"] = "0"
os.environ["HOROVOD_CACHE_CAPACITY"] = "0"
os.environ["HOROVOD_CYCLE_TIME"] = "0"

parser = argparse.ArgumentParser(
    description="CIFAR-10/100 benchmark with VGG-16 and ResNet-18",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--model",
    type=str,
    default="cifar10_resnet18",
    choices=["cifar10_resnet18", "cifar10_vgg16", "cifar100_resnet18", "cifar100_vgg16"],
)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--base-lr", type=float, default=0.1)
parser.add_argument("--warmup-epochs", type=int, default=5)
parser.add_argument("--lr-decay-epochs", type=str, default="150,220")
parser.add_argument("--lr-decay-factor", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight-decay", type=float, default=5e-4)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--data-dir", type=str, default=None)
parser.add_argument("--download-dataset", action="store_true", default=False)
parser.add_argument("--print-freq", type=int, default=50)
parser.add_argument("--convergence-log-every", type=int, default=0)
parser.add_argument("--convergence-output", type=str, default="")
parser.add_argument("--comm-stats-output", type=str, default="")
parser.add_argument("--comm-stats-every", type=int, default=1)
parser.add_argument("--fp16", action="store_true", default=False)
parser.add_argument("--no-cuda", action="store_true", default=False)
parser.add_argument("--use-adasum", action="store_true", default=False)
parser.add_argument("--mgwfbp", action="store_true", default=False)
parser.add_argument("--asc", action="store_true", default=False)
parser.add_argument("--nstreams", type=int, default=1)
parser.add_argument("--threshold", type=int, default=536870912)
parser.add_argument("--rdma", action="store_true", default=False)
parser.add_argument("--compressor", type=str, default="none", choices=compressors.keys())
parser.add_argument("--density", type=float, default=1.0)
parser.add_argument("--exclude-parts", type=str, default="")
parser.add_argument("--overlap-profile", action="store_true", default=False)
parser.add_argument("--overlap-summary", action="store_true", default=False)
parser.add_argument("--overlap-timeline", action="store_true", default=False)
parser.add_argument("--overlap-summary-mode", type=str, default="strict", choices=["strict", "light"])
parser.add_argument("--overlap-timeline-mode", type=str, default="light", choices=["light", "strict"])
parser.add_argument("--overlap-log-every", type=int, default=10)
parser.add_argument("--overlap-warmup", type=int, default=0)
parser.add_argument("--overlap-output", type=str, default="")
parser.add_argument("--overlap-timeline-output", type=str, default="")
parser.add_argument("--overlap-console", type=int, default=1)
parser.add_argument("--compress-rank", type=int, default=8)
parser.add_argument("--compress-warmup", type=int, default=1000)
parser.add_argument("--compress-min-numel", type=int, default=16384)
parser.add_argument("--rank-reset-on-change", action="store_true", default=False)
parser.add_argument("--active-prefix-enabled", type=int, default=1, choices=[0, 1])
parser.add_argument("--embedding-policy", type=str, default="word", choices=["off", "word", "broad"])
parser.add_argument("--rank-schedule", type=str, default="fixed", choices=["fixed", "aggressive", "gentle", "update_norm_stable"])
parser.add_argument("--stable-rank-levels", type=str, default="")
parser.add_argument("--update-norm-stable-tol", type=float, default=0.01)
parser.add_argument("--update-norm-critical-tol", type=float, default=0.3)
parser.add_argument("--update-norm-patience", type=int, default=100)
parser.add_argument("--update-norm-smoothing", type=float, default=0.9)
parser.add_argument("--update-norm-debug-every", type=int, default=0)
args = parser.parse_args()
args.lr_decay_epochs = [int(epoch) for epoch in args.lr_decay_epochs.split(",") if epoch]
args.dataset = "cifar100" if args.model.startswith("cifar100_") else "cifar10"
if args.data_dir is None:
    args.data_dir = "./{}_data".format(args.dataset)

args.cuda = not args.no_cuda and torch.cuda.is_available()
local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
os.environ["HOROVOD_NUM_NCCL_STREAMS"] = str(args.nstreams)

overlap_enabled = args.overlap_summary or args.overlap_timeline
overlap_needs_sync = (
    (args.overlap_summary and args.overlap_summary_mode == "strict")
    or (args.overlap_timeline and args.overlap_timeline_mode == "strict")
)
os.environ["DEAR_OVERLAP_PROFILE"] = "1" if overlap_enabled else "0"
os.environ["DEAR_OVERLAP_SUMMARY"] = "1" if args.overlap_summary else "0"
os.environ["DEAR_OVERLAP_TIMELINE"] = "1" if args.overlap_timeline else "0"
os.environ["DEAR_OVERLAP_NEEDS_SYNC"] = "1" if overlap_needs_sync else "0"
os.environ["DEAR_OVERLAP_LOG_EVERY"] = str(args.overlap_log_every)
os.environ["DEAR_OVERLAP_WARMUP"] = str(args.overlap_warmup)
os.environ["DEAR_OVERLAP_OUTPUT"] = args.overlap_output
os.environ["DEAR_OVERLAP_TIMELINE_OUTPUT"] = args.overlap_timeline_output
os.environ["DEAR_OVERLAP_CONSOLE"] = str(args.overlap_console)
os.environ["DEAR_COMM_STATS_OUTPUT"] = args.comm_stats_output
os.environ["DEAR_COMM_STATS_EVERY"] = str(args.comm_stats_every)
if args.convergence_output and hvd.rank() == 0:
    convergence_dir = os.path.dirname(args.convergence_output)
    if convergence_dir:
        os.makedirs(convergence_dir, exist_ok=True)
    with open(args.convergence_output, "w") as f:
        f.write("kind,epoch,step,global_step,elapsed_time_s,loss,top1,lr,samples_seen\n")

if args.cuda:
    torch.cuda.set_device(local_rank)

cudnn.benchmark = True

seed = args.seed + hvd.rank()
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if args.cuda:
    torch.cuda.manual_seed(seed)

RANK_SCHEDULES = {
    "fixed": None,
    "update_norm_stable": None,
    "aggressive": {
        0: args.compress_rank,
        args.compress_warmup + 6000: max(1, args.compress_rank // 2),
    },
    "gentle": {
        0: args.compress_rank,
        args.compress_warmup + 12000: max(1, args.compress_rank // 2),
    },
}

DATASET_CONFIGS = {
    "cifar10": {
        "class": datasets.CIFAR10,
        "num_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    "cifar100": {
        "class": datasets.CIFAR100,
        "num_classes": 100,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
}
dataset_config = DATASET_CONFIGS[args.dataset]


# 动态rank的config打印
print("===== Dynamic_rank Training Config =====")
print(f"compress_rank: {args.compress_rank}")
print(f"compress_warmup: {args.compress_warmup}")
print(f"compress_min_numel: {args.compress_min_numel}")
print(f"rank_schedule: {args.rank_schedule}")
if args.rank_schedule == "update_norm_stable":
    print(f"stable_rank_levels: {args.stable_rank_levels or 'auto'}")
    print(f"update_norm_stable_tol: {args.update_norm_stable_tol}")
    print(f"update_norm_critical_tol: {args.update_norm_critical_tol}")
    print(f"update_norm_patience: {args.update_norm_patience}")
    print(f"update_norm_smoothing: {args.update_norm_smoothing}")
    print(f"update_norm_debug_every: {args.update_norm_debug_every}")
print("========================================")

def hvd_barrier():
    token = torch.tensor([1.0], device="cuda" if args.cuda else "cpu")
    hvd.allreduce(token, name="{}_setup_barrier".format(args.dataset))


def build_model():
    if args.model.endswith("_resnet18"):
        return cifar_resnet18(num_classes=dataset_config["num_classes"])
    return cifar_vgg16(num_classes=dataset_config["num_classes"])


def build_dataloaders():
    normalize = transforms.Normalize(
        mean=dataset_config["mean"],
        std=dataset_config["std"],
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset_class = dataset_config["class"]
    if hvd.rank() == 0:
        train_dataset = dataset_class(root=args.data_dir, train=True, download=True, transform=train_transform)
        test_dataset = dataset_class(root=args.data_dir, train=False, download=True, transform=test_transform)
    else:
        train_dataset = None
        test_dataset = None
    hvd_barrier()
    if hvd.rank() != 0:
        train_dataset = dataset_class(root=args.data_dir, train=True, download=False, transform=train_transform)
        test_dataset = dataset_class(root=args.data_dir, train=False, download=False, transform=test_transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=hvd.size(),
        rank=hvd.rank(),
        shuffle=True,
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=hvd.size(),
        rank=hvd.rank(),
        shuffle=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=args.cuda,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=args.cuda,
    )
    return train_loader, test_loader, train_sampler


def adjust_learning_rate(optimizer, epoch, batch_idx, num_batches):
    lr = args.base_lr

    if args.warmup_epochs > 0 and hvd.size() > 1 and epoch < args.warmup_epochs:
        epoch_progress = epoch + float(batch_idx + 1) / num_batches
        lr *= (epoch_progress * (hvd.size() - 1) / args.warmup_epochs + 1.0) / hvd.size()

    for decay_epoch in args.lr_decay_epochs:
        if epoch >= decay_epoch:
            lr *= args.lr_decay_factor

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def log(message):
    if hvd.rank() == 0:
        print(message, flush=True)


run_start_time = time.perf_counter()
global_train_step = 0


def record_convergence(kind, epoch, step, global_step, loss, top1, lr, samples_seen):
    if hvd.rank() != 0:
        return
    elapsed_time_s = time.perf_counter() - run_start_time
    loss_text = "" if loss is None else "%.9f" % float(loss)
    top1_text = "" if top1 is None else "%.6f" % float(top1)
    print(
        "CONVERGENCE kind=%s epoch=%d step=%d global_step=%d elapsed_time_s=%.6f loss=%s top1=%s lr=%.8g samples_seen=%d"
        % (kind, epoch, step, global_step, elapsed_time_s, loss_text, top1_text, lr, samples_seen),
        flush=True,
    )
    if args.convergence_output:
        with open(args.convergence_output, "a") as f:
            f.write(
                "%s,%d,%d,%d,%.9f,%s,%s,%.12g,%d\n"
                % (kind, epoch, step, global_step, elapsed_time_s, loss_text, top1_text, lr, samples_seen)
            )


def accuracy(output, target):
    pred = output.argmax(dim=1)
    correct = pred.eq(target).float().sum()
    return correct, torch.tensor(float(target.size(0)), device=target.device)


model = build_model()
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
if args.cuda:
    criterion.cuda()

initial_lr = args.base_lr
if args.warmup_epochs > 0 and hvd.size() > 1:
    initial_lr = args.base_lr / hvd.size()
optimizer = optim.SGD(
    model.parameters(),
    lr=initial_lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
)

if hvd.size() > 1:
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        model=model,
        compression=compressors[args.compressor](
            device=torch.device("cuda", local_rank) if args.cuda else torch.device("cpu"),
            rank=args.compress_rank,
            rank_schedule=RANK_SCHEDULES[args.rank_schedule],
            warmup_steps=args.compress_warmup,
            min_compression_numel=args.compress_min_numel,
            update_norm_stable_rank=args.rank_schedule == "update_norm_stable",
            stable_rank_levels=args.stable_rank_levels,
            update_norm_stable_tol=args.update_norm_stable_tol,
            update_norm_critical_tol=args.update_norm_critical_tol,
            update_norm_patience=args.update_norm_patience,
            update_norm_smoothing=args.update_norm_smoothing,
            update_norm_debug_every=args.update_norm_debug_every,
            rank_reset_on_change=args.rank_reset_on_change,
            embedding_policy=args.embedding_policy,
        ),
        is_sparse=args.density < 1,
        density=args.density,
        seq_layernames=None,
        layerwise_times=None,
        norm_clip=None,
        threshold=args.threshold,
        writer=None,
        gradient_path="./",
        fp16=args.fp16,
        mgwfbp=args.mgwfbp,
        rdma=args.rdma,
        exclude_parts=args.exclude_parts,
        active_prefix_enabled=bool(args.active_prefix_enabled),
    )
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

train_loader, test_loader, train_sampler = build_dataloaders()


def train_one_epoch(epoch):
    global global_train_step
    model.train()
    train_sampler.set_epoch(epoch)
    running_loss = 0.0

    for step, (data, target) in enumerate(train_loader, start=1):
        lr = adjust_learning_rate(optimizer, epoch, step - 1, len(train_loader))
        if args.cuda:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().item())
        global_train_step += 1
        samples_seen = int(global_train_step * args.batch_size * hvd.size())
        running_loss += loss_value
        if args.convergence_log_every > 0 and global_train_step % args.convergence_log_every == 0:
            record_convergence(
                "train",
                epoch,
                step,
                global_train_step,
                loss_value,
                None,
                lr,
                samples_seen,
            )
        if step % args.print_freq == 0 or step == len(train_loader):
            log(
                "Epoch {:03d} Step {:04d}/{:04d} lr {:.5f} loss {:.4f}".format(
                    epoch,
                    step,
                    len(train_loader),
                    lr,
                    running_loss / step,
                )
            )


def evaluate(epoch):
    model.eval()
    correct_total = torch.tensor(0.0, device="cuda" if args.cuda else "cpu")
    sample_total = torch.tensor(0.0, device="cuda" if args.cuda else "cpu")
    loss_total = torch.tensor(0.0, device="cuda" if args.cuda else "cpu")

    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            output = model(data)
            loss = criterion(output, target)
            correct, batch_size = accuracy(output, target)
            correct_total += correct
            sample_total += batch_size
            loss_total += loss.detach() * batch_size

    correct_total = hvd.allreduce(correct_total, name="{}_eval_correct".format(args.dataset))
    sample_total = hvd.allreduce(sample_total, name="{}_eval_samples".format(args.dataset))
    loss_total = hvd.allreduce(loss_total, name="{}_eval_loss".format(args.dataset))
    avg_loss = (loss_total / sample_total).item()
    top1 = (correct_total / sample_total * 100.0).item()
    log("Epoch {:03d} validation loss {:.4f} top1 {:.2f}%".format(epoch, avg_loss, top1))
    if args.convergence_output:
        record_convergence(
            "validation",
            epoch,
            0,
            global_train_step,
            avg_loss,
            top1,
            optimizer.param_groups[0].get("lr", 0.0),
            int(global_train_step * args.batch_size * hvd.size()),
        )


log(
    "CIFAR benchmark start dataset={} model={} data_dir={} compressor={} batch_size={} workers={} world_size={} base_lr={} warmup_epochs={} decay_epochs={} decay_factor={}".format(
        args.dataset,
        args.model,
        args.data_dir,
        args.compressor,
        args.batch_size,
        args.workers,
        hvd.size(),
        args.base_lr,
        args.warmup_epochs,
        args.lr_decay_epochs,
        args.lr_decay_factor,
    )
)
for epoch in range(args.epochs):
    train_one_epoch(epoch)
    evaluate(epoch)
