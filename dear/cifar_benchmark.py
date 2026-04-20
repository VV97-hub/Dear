from __future__ import print_function

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as tv_models
import torchvision.transforms as transforms

import dopt_rsag as hvd
from compression import compressors


hvd.init()

os.environ["HOROVOD_FUSION_THRESHOLD"] = "0"
os.environ["HOROVOD_CACHE_CAPACITY"] = "0"
os.environ["HOROVOD_CYCLE_TIME"] = "0"

parser = argparse.ArgumentParser(
    description="CIFAR-10 benchmark with VGG-16 and ResNet-18",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--model", type=str, default="cifar10_resnet18", choices=["cifar10_resnet18", "cifar10_vgg16"])
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--base-lr", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight-decay", type=float, default=5e-4)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--data-dir", type=str, default="./cifar10_data")
parser.add_argument("--print-freq", type=int, default=50)
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
parser.add_argument("--rank-schedule", type=str, default=None, choices=[None, "aggressive", "gentle"])
args = parser.parse_args()

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
    None: None,
    "aggressive": {0: args.compress_rank, 6000: max(1, args.compress_rank // 2)},
    "gentle": {0: args.compress_rank, 12000: max(1, args.compress_rank // 2)},
}


# 动态rank的config打印
print("===== Dynamic_rank Training Config =====")
print(f"compress_rank: {args.compress_rank}")
print(f"compress_warmup: {args.compress_warmup}")
print(f"compress_min_numel: {args.compress_min_numel}")
print(f"rank_schedule: {args.rank_schedule}")
print("========================================")

def hvd_barrier():
    token = torch.tensor([1.0], device="cuda" if args.cuda else "cpu")
    hvd.allreduce(token, name="cifar10_setup_barrier")


def build_model():
    if args.model == "cifar10_resnet18":
        return tv_models.resnet18(num_classes=10)
    return tv_models.vgg16(num_classes=10)


def build_dataloaders():
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
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

    if hvd.rank() == 0:
        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transform)
    else:
        train_dataset = None
        test_dataset = None
    hvd_barrier()
    if hvd.rank() != 0:
        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=False, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=False, transform=test_transform)

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


def adjust_learning_rate(optimizer, epoch):
    lr = args.base_lr * hvd.size()
    if epoch >= 100:
        lr *= 0.1
    if epoch >= 150:
        lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def log(message):
    if hvd.rank() == 0:
        print(message, flush=True)


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

lr_scaler = hvd.size() if not args.use_adasum else 1
optimizer = optim.SGD(
    model.parameters(),
    lr=args.base_lr * lr_scaler,
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
    )
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

train_loader, test_loader, train_sampler = build_dataloaders()


def train_one_epoch(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    lr = adjust_learning_rate(optimizer, epoch)
    running_loss = 0.0

    for step, (data, target) in enumerate(train_loader, start=1):
        if args.cuda:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
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

    correct_total = hvd.allreduce(correct_total, name="cifar10_eval_correct")
    sample_total = hvd.allreduce(sample_total, name="cifar10_eval_samples")
    loss_total = hvd.allreduce(loss_total, name="cifar10_eval_loss")
    avg_loss = (loss_total / sample_total).item()
    top1 = (correct_total / sample_total * 100.0).item()
    log("Epoch {:03d} validation loss {:.4f} top1 {:.2f}%".format(epoch, avg_loss, top1))


log(
    "CIFAR benchmark start model={} compressor={} batch_size={} workers={} world_size={}".format(
        args.model,
        args.compressor,
        args.batch_size,
        args.workers,
        hvd.size(),
    )
)
for epoch in range(args.epochs):
    train_one_epoch(epoch)
    evaluate(epoch)
