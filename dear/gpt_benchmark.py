from __future__ import print_function

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, BertTokenizer, GPT2Config, GPT2LMHeadModel
from transformers.utils import logging

import dopt_rsag as hvd
from compression import compressors


logging.set_verbosity_error()
hvd.init()

os.environ["HOROVOD_FUSION_THRESHOLD"] = "0"
os.environ["HOROVOD_CACHE_CAPACITY"] = "0"
os.environ["HOROVOD_CYCLE_TIME"] = "0"


MODEL_CONFIGS = {
    "gpt_125m": dict(n_layer=12, n_embd=768, n_head=12),
    "gpt_160m": dict(n_layer=16, n_embd=768, n_head=12),
    "gpt_230m": dict(n_layer=16, n_embd=1024, n_head=16),
}


parser = argparse.ArgumentParser(
    description="GPT-style causal language-model benchmark for DeAR/ACPR",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--model", type=str, default="gpt_230m", choices=MODEL_CONFIGS.keys())
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--seq-len", type=int, default=128)
parser.add_argument("--data-file", type=str, default="./wikitext-local/train-00000-of-00001.parquet")
parser.add_argument("--tokenizer-dir", type=str, default="./bert-base-uncased-local")
parser.add_argument("--max-train-tokens", type=int, default=2000000)
parser.add_argument("--num-warmup-batches", type=int, default=10)
parser.add_argument("--num-batches-per-iter", type=int, default=10)
parser.add_argument("--num-iters", type=int, default=100)
parser.add_argument("--learning-rate", type=float, default=3e-4)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--no-cuda", action="store_true", default=False)
parser.add_argument("--fp16", action="store_true", default=False)
parser.add_argument("--mgwfbp", action="store_true", default=False)
parser.add_argument("--asc", action="store_true", default=False)
parser.add_argument("--nstreams", type=int, default=1)
parser.add_argument("--threshold", type=int, default=536870912)
parser.add_argument("--rdma", action="store_true", default=False)
parser.add_argument("--compressor", type=str, default="none", choices=compressors.keys())
parser.add_argument("--density", type=float, default=1.0)
parser.add_argument("--exclude-parts", type=str, default="")
parser.add_argument("--loss-log-every", type=int, default=0)
parser.add_argument("--convergence-output", type=str, default="")
parser.add_argument("--comm-stats-output", type=str, default="")
parser.add_argument("--comm-stats-every", type=int, default=1)
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
parser.add_argument("--compress-rank", type=int, default=16)
parser.add_argument("--compress-warmup", type=int, default=500)
parser.add_argument("--compress-refresh-k", type=int, default=0)
parser.add_argument("--compress-min-numel", type=int, default=16384)
parser.add_argument("--rank-reset-on-change", action="store_true", default=False)
parser.add_argument("--active-prefix-enabled", type=int, default=1, choices=[0, 1])
parser.add_argument("--embedding-policy", type=str, default="word", choices=["off", "word", "broad"])
parser.add_argument("--rank-schedule", type=str, default="fixed", choices=["fixed", "aggressive", "gentle", "update_norm_stable"])
parser.add_argument("--stable-rank-levels", type=str, default="")
parser.add_argument("--update-norm-stable-tol", type=float, default=0.05)
parser.add_argument("--update-norm-critical-tol", type=float, default=0.8)
parser.add_argument("--update-norm-patience", type=int, default=20)
parser.add_argument("--update-norm-smoothing", type=float, default=0.8)
parser.add_argument("--update-norm-debug-every", type=int, default=0)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

seed = args.seed + hvd.rank()
np.random.seed(seed)
torch.manual_seed(seed)
if args.cuda:
    torch.cuda.manual_seed(seed)

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
        f.write("step,elapsed_time_s,loss,lr,samples_seen,tokens_seen\n")

RANK_SCHEDULES = {
    "fixed": None,
    "update_norm_stable": None,
    "aggressive": {
        0: args.compress_rank,
        args.compress_warmup + 250: max(1, round(args.compress_rank * 0.75)),
        args.compress_warmup + 450: max(1, args.compress_rank // 2),
    },
    "gentle": {
        0: args.compress_rank,
        args.compress_warmup + 1000: max(1, round(args.compress_rank * 0.75)),
        args.compress_warmup + 3000: max(1, args.compress_rank // 2),
    },
}


def log(message):
    if hvd.rank() == 0:
        print(message, flush=True)


class TokenBlockDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        usable = (len(token_ids) - 1) // seq_len * seq_len
        if usable <= 0:
            raise ValueError("not enough tokens to build GPT training blocks")
        self.tokens = torch.tensor(token_ids[: usable + 1], dtype=torch.long)
        self.seq_len = int(seq_len)
        self.length = usable // self.seq_len

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        start = index * self.seq_len
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return {"input_ids": x, "labels": y}


def build_dataset():
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_dir)
    raw = load_dataset("parquet", data_files=args.data_file, split="train")
    texts = [row["text"] for row in raw if row.get("text") and row["text"].strip()]
    joined = "\n\n".join(texts)
    token_ids = tokenizer(
        joined,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    if args.max_train_tokens > 0:
        token_ids = token_ids[: args.max_train_tokens]
    return tokenizer, TokenBlockDataset(token_ids, args.seq_len)


tokenizer, train_dataset = build_dataset()
sampler = DistributedSampler(
    train_dataset,
    num_replicas=hvd.size(),
    rank=hvd.rank(),
    shuffle=True,
    seed=args.seed,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=sampler,
    num_workers=0,
    pin_memory=args.cuda,
    drop_last=True,
)

config_kwargs = MODEL_CONFIGS[args.model]
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=args.seq_len,
    n_ctx=args.seq_len,
    use_cache=False,
    **config_kwargs
)
model = GPT2LMHeadModel(config)
model.config.use_cache = False
if args.cuda:
    model.cuda()
cudnn.benchmark = True

num_params = sum(p.numel() for p in model.parameters())
log(
    "GPT benchmark start model={} params={} seq_len={} batch_size={} world_size={} dataset_blocks={} compressor={} rank_schedule={} refresh_k={}".format(
        args.model,
        num_params,
        args.seq_len,
        args.batch_size,
        hvd.size(),
        len(train_dataset),
        args.compressor,
        args.rank_schedule,
        args.compress_refresh_k,
    )
)

optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
if hvd.size() > 1:
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        model=model,
        compression=compressors[args.compressor](
            device=torch.device("cuda") if args.cuda else torch.device("cpu"),
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
        refresh_k=args.compress_refresh_k,
        active_prefix_enabled=bool(args.active_prefix_enabled),
    )
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

run_start_time = time.perf_counter()
global_step = 0
data_iter = iter(train_loader)


def current_lr():
    return float(optimizer.param_groups[0].get("lr", 0.0))


def record_convergence(step, loss_value):
    if hvd.rank() != 0:
        return
    elapsed_time_s = time.perf_counter() - run_start_time
    samples_seen = int((step + 1) * args.batch_size * hvd.size())
    tokens_seen = samples_seen * args.seq_len
    print(
        "CONVERGENCE step={} elapsed_time_s={:.6f} loss={:.6f} lr={:.8g} samples_seen={} tokens_seen={}".format(
            step,
            elapsed_time_s,
            loss_value,
            current_lr(),
            samples_seen,
            tokens_seen,
        ),
        flush=True,
    )
    if args.convergence_output:
        with open(args.convergence_output, "a") as f:
            f.write(
                "{},{:.9f},{:.9f},{:.12g},{},{}\n".format(
                    step,
                    elapsed_time_s,
                    loss_value,
                    current_lr(),
                    samples_seen,
                    tokens_seen,
                )
            )


def next_batch():
    global data_iter
    try:
        return next(data_iter)
    except StopIteration:
        sampler.set_epoch(global_step)
        data_iter = iter(train_loader)
        return next(data_iter)


def train_step(log_loss=False, capture_loss=False):
    global global_step
    batch = next_batch()
    if args.cuda:
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}

    if overlap_enabled and hasattr(optimizer, "profile_step_begin"):
        optimizer.profile_step_begin()
    optimizer.zero_grad()

    if overlap_needs_sync and args.cuda:
        torch.cuda.synchronize()
    if overlap_enabled and hasattr(optimizer, "profile_forward_start"):
        optimizer.profile_forward_start()
    outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
    loss = outputs.loss
    if overlap_needs_sync and args.cuda:
        torch.cuda.synchronize()
    if overlap_enabled and hasattr(optimizer, "profile_forward_done"):
        optimizer.profile_forward_done()

    loss_value = None
    if log_loss or capture_loss:
        loss_value = float(loss.detach().item())
    if log_loss:
        record_convergence(global_step, loss_value)

    if overlap_enabled and hasattr(optimizer, "profile_backward_start"):
        optimizer.profile_backward_start()
    if overlap_needs_sync and args.cuda:
        torch.cuda.synchronize()
    loss.backward()
    if overlap_needs_sync and args.cuda:
        torch.cuda.synchronize()
    if overlap_enabled and hasattr(optimizer, "profile_backward_done"):
        optimizer.profile_backward_done()

    optimizer.step()
    global_step += 1
    return loss_value


for _ in range(args.num_warmup_batches):
    train_step(log_loss=False)

if args.cuda:
    torch.cuda.synchronize()

for iter_idx in range(args.num_iters):
    iter_start = time.perf_counter()
    last_loss = None
    for _ in range(args.num_batches_per_iter):
        should_log = args.loss_log_every > 0 and global_step % args.loss_log_every == 0
        capture_loss = _ == args.num_batches_per_iter - 1
        loss_value = train_step(log_loss=should_log, capture_loss=capture_loss)
        if loss_value is not None:
            last_loss = loss_value
    if args.cuda:
        torch.cuda.synchronize()
    iter_time = time.perf_counter() - iter_start
    samples = args.num_batches_per_iter * args.batch_size * hvd.size()
    tokens = samples * args.seq_len
    if hvd.rank() == 0:
        loss_text = "n/a" if last_loss is None else "{:.6f}".format(last_loss)
        print(
            "Iter #{}: {:.2f} samples/s, {:.2f} tokens/s, {:.6f} s/iter, loss {}".format(
                iter_idx,
                samples / iter_time,
                tokens / iter_time,
                iter_time,
                loss_text,
            ),
            flush=True,
        )
