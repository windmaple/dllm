"""Train a GPT-2 (124M) masked-diffusion LM on OpenWebText (default), or a miniGPT on TinyStories.

    python train.py --max_iters 200 --eval_interval 100   # quick smoke test (defaults = the OWT run)
    python train.py --dataset tinystories --data_dir data --n_layer 6 --n_head 6 --n_embd 384 \
        --block_size 256 --batch_size 64 --grad_accum 1 ...   # miniGPT on TinyStories, see README

Distributed data parallel (one process per device, --batch_size is per process):

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 torchrun --standalone --nproc_per_node=4 train.py --device cpu  # gloo
    eval $(python -m torch_tpu._internal.distributed.distributed | sed 's/^/export /')
    torchrun --standalone --nproc_per_node=$WORLD_SIZE train.py --device tpu  # TPU chips / tpu_dist
"""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import owt
from data import EOS_ID, MASK_ID, UNK_ID, CompactTokenizer
from diffusion import diffusion_loss, generate
from model import MiniGPT, MiniGPTConfig


class StoryDataset:
    """Each sample = one story from its beginning, truncated / right-padded with <eos> to block_size."""

    def __init__(self, data_dir, split, block_size):
        self.tokens = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
        self.offsets = np.load(os.path.join(data_dir, f"{split}_offsets.npy"))
        self.lens = np.diff(np.append(self.offsets, len(self.tokens)))
        self.block_size = block_size

    def batch(self, batch_size, rng):
        idx = rng.integers(0, len(self.offsets), batch_size)
        out = np.full((batch_size, self.block_size), EOS_ID, dtype=np.int64)
        for b, i in enumerate(idx):
            n = min(self.lens[i], self.block_size)
            out[b, :n] = self.tokens[self.offsets[i]: self.offsets[i] + n]
        return torch.from_numpy(out)


def get_lr(it, args):
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters
    ratio = (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * min(ratio, 1.0))) * (args.lr - args.min_lr)


def setup_device(name):
    """Resolve --device to a torch.device. Under DDP each TPU process only sees its own chip."""
    if name == "auto":
        # torch_tpu, when installed, is auto-loaded by `import torch` and registers the "tpu" accelerator
        acc = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else None
        name = "tpu" if acc is not None and acc.type == "tpu" else "cpu"
    return torch.device(name)


DIST_BACKEND = {"tpu": "tpu_dist", "cpu": "gloo"}


@torch.no_grad()
def evaluate(model, ds, args, n_batches, device, ctx, static_shapes, mask_id, rank=0, world_size=1):
    """Each rank evaluates its own fixed set of n_batches batches; results are averaged across ranks."""
    model.eval()
    rng = np.random.default_rng(1234 + rank)  # fixed eval batches -> comparable numbers
    torch.manual_seed(1234 + rank)
    totals = torch.zeros(2, device=device)
    for _ in range(n_batches):
        with ctx:
            elbo, ce = diffusion_loss(model, ds.batch(args.batch_size, rng).to(device), static_shapes=static_shapes,
                                      mask_id=mask_id)
        totals += torch.stack([elbo.detach().float(), ce.detach().float()])
    if world_size > 1:
        dist.all_reduce(totals)
    elbo, ce = (totals / (n_batches * world_size)).tolist()
    model.train()
    return elbo, ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="owt", choices=["owt", "tinystories"])
    ap.add_argument("--data_dir", default=None, help="default: /dev/shm/owt (owt) or data (tinystories)")
    ap.add_argument("--out_dir", default="out")
    # model (GPT-2 small)
    ap.add_argument("--n_layer", type=int, default=12)
    ap.add_argument("--n_head", type=int, default=12)
    ap.add_argument("--n_embd", type=int, default=768)
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.0)
    # optimization
    ap.add_argument("--batch_size", type=int, default=10, help="per process, per micro-step")
    ap.add_argument("--grad_accum", type=int, default=6, help="micro-steps per optimizer step")
    ap.add_argument("--max_iters", type=int, default=50000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min_lr", type=float, default=6e-5)
    ap.add_argument("--warmup_iters", type=int, default=2000)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # logging
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--eval_interval", type=int, default=1000)
    ap.add_argument("--eval_batches", type=int, default=10)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--device", default="auto", choices=["auto", "tpu", "cpu"])
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"],
                    help="bfloat16 = autocast mixed precision (recommended on TPU)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    # Weights & Biases (rank 0 only; needs `pip install wandb` and WANDB_API_KEY / `wandb login`)
    ap.add_argument("--wandb", action="store_true", help="log metrics and samples to Weights & Biases")
    ap.add_argument("--wandb_project", default="dllm")
    ap.add_argument("--wandb_run_name", default=None)
    args = ap.parse_args()
    if args.data_dir is None:
        args.data_dir = "/dev/shm/owt" if args.dataset == "owt" else "data"

    # torchrun sets RANK / LOCAL_RANK / WORLD_SIZE
    ddp = "LOCAL_RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1)) if ddp else 1
    master = rank == 0
    log = print if master else (lambda *a, **k: None)

    device = setup_device(args.device)
    if ddp:
        dist.init_process_group(backend=DIST_BACKEND[device.type])
    if master:
        os.makedirs(args.out_dir, exist_ok=True)
    if args.threads:
        torch.set_num_threads(args.threads)
    static_shapes = device.type == "tpu"  # avoid data-dependent shapes -> XLA recompiles
    ctx = (torch.autocast(device_type=device.type, dtype=torch.bfloat16) if args.dtype == "bfloat16"
           else nullcontext())
    torch.manual_seed(args.seed)  # same init on every rank (DDP also broadcasts rank 0's weights)

    if args.dataset == "tinystories":
        tok = CompactTokenizer(os.path.join(args.data_dir, "vocab.json"))
        Dataset, mask_id, banned_ids, prompt = StoryDataset, MASK_ID, (MASK_ID, UNK_ID), "Once upon a time"
    else:
        tok = owt.GPT2Tokenizer()
        Dataset, mask_id, prompt = owt.PackedDataset, owt.MASK_ID, "In a shocking finding, scientists"
        banned_ids = range(owt.GPT2_VOCAB, owt.VOCAB_SIZE)  # <mask> + vocab padding
    train_ds = Dataset(args.data_dir, "train", args.block_size)
    val_ds = Dataset(args.data_dir, "val", args.block_size)

    cfg = MiniGPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                        n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = MiniGPT(cfg).to(device)
    # different data / masking noise on every rank
    torch.manual_seed(args.seed + rank)
    rng = np.random.default_rng(args.seed + rank)
    log(f"device={device} world_size={world_size} dtype={args.dtype} threads={torch.get_num_threads()} | "
        f"params: {model.num_params() / 1e6:.2f}M | global batch {args.batch_size * args.grad_accum * world_size}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))

    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    start_it = 0
    wandb_id = None
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_it = ck["iter"] + 1
        wandb_id = ck.get("wandb_id")
        log(f"resumed from iter {ck['iter']}")

    train_model = model
    if args.compile:
        train_model = torch.compile(model, backend="tpu") if device.type == "tpu" else torch.compile(model)
    if ddp:
        train_model = DDP(train_model)  # on TPU each process sees a single device, so no device_ids
    log_f = open(os.path.join(args.out_dir, "log.jsonl"), "a") if master else None
    run = None
    if args.wandb and master:
        import wandb
        # on --resume, keep logging to the run stored in the checkpoint
        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, id=wandb_id, resume="allow",
                         config={**vars(args), "world_size": world_size, "global_batch": args.batch_size * args.grad_accum * world_size,
                                 "device": device.type, "params": model.num_params()})
        samples = wandb.Table(columns=["iter", "sample"], log_mode="INCREMENTAL")
    tokens_per_iter = args.batch_size * args.grad_accum * world_size * args.block_size

    def to_cpu(obj):
        if torch.is_tensor(obj):
            return obj.cpu()
        if isinstance(obj, dict):
            return {k: to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(to_cpu(v) for v in obj)
        return obj

    def save(it):
        # every rank materializes the state (keeps TPU ranks in lock-step), only rank 0 writes it
        state = to_cpu({"model": model.state_dict(), "optimizer": opt.state_dict()})
        if master:
            torch.save({**state, "config": asdict(cfg), "iter": it, "args": vars(args),
                        "wandb_id": run.id if run else wandb_id}, ckpt_path)

    model.train()
    t0 = time.time()
    for it in range(start_it, args.max_iters):
        lr = get_lr(it, args)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        elbo = ce = 0.0
        for micro in range(args.grad_accum):
            x0 = train_ds.batch(args.batch_size, rng).to(device)
            # only all-reduce gradients on the last micro-step
            sync = nullcontext() if not ddp or micro == args.grad_accum - 1 else train_model.no_sync()
            with sync:
                with ctx:
                    m_elbo, m_ce = diffusion_loss(train_model, x0, static_shapes=static_shapes, mask_id=mask_id)
                (m_elbo / args.grad_accum).backward()
            elbo = elbo + m_elbo.detach() / args.grad_accum
            ce = ce + m_ce.detach() / args.grad_accum
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        # Logging, eval and sampling run on every rank (only rank 0 prints / writes) so that
        # TPU ranks hit the same host syncs (.item(), .cpu()) and stay in lock-step.
        if it % args.log_interval == 0:
            t_elbo, t_ce, t_gnorm = elbo.item(), ce.item(), gnorm.item()  # syncs the device
            dt = (time.time() - t0) / (args.log_interval if it > start_it else 1)
            t0 = time.time()
            hbm = torch.accelerator.max_memory_allocated() / 2**30 if device.type == "tpu" else None
            log(f"iter {it:5d} | elbo {t_elbo:.3f} | masked-ce {t_ce:.3f} | "
                f"gnorm {t_gnorm:.2f} | lr {lr:.2e} | {dt * 1000:.0f} ms/it"
                + (f" | peak hbm {hbm:.1f}GiB" if hbm is not None else ""), flush=True)
            if master:
                log_f.write(json.dumps({"iter": it, "train_elbo": t_elbo, "train_ce": t_ce, "lr": lr}) + "\n")
                log_f.flush()
            if run:
                metrics = {"train/elbo": t_elbo, "train/masked_ce": t_ce, "train/grad_norm": t_gnorm, "lr": lr,
                           "perf/ms_per_iter": dt * 1000, "perf/tokens_per_s": tokens_per_iter / dt,
                           "tokens": (it + 1) * tokens_per_iter}
                if hbm is not None:
                    metrics["perf/peak_hbm_gib"] = hbm
                run.log(metrics, step=it)

        if (it > 0 and it % args.eval_interval == 0) or it == args.max_iters - 1:
            v_elbo, v_ce = evaluate(model, val_ds, args, args.eval_batches, device, ctx, static_shapes, mask_id,
                                    rank, world_size)
            log(f"==> iter {it} | val elbo {v_elbo:.3f} (ppl bound {math.exp(v_elbo):.1f}) | val masked-ce {v_ce:.3f}")
            if master:
                log_f.write(json.dumps({"iter": it, "val_elbo": v_elbo, "val_ce": v_ce}) + "\n")
                log_f.flush()
            if run:
                run.log({"val/elbo": v_elbo, "val/masked_ce": v_ce, "val/ppl_bound": math.exp(v_elbo)}, step=it)
            save(it)
            with ctx:
                # random-order unmasking: confidence-based sampling over the whole sequence collapses
                ids, _ = generate(model, tok.encode(prompt), length=128, steps=128, temperature=0.8,
                                  strategy="random", record=False, device=device, mask_id=mask_id,
                                  banned_ids=banned_ids)
            log("sample:", repr(tok.decode(ids)[:400]), flush=True)
            if run:
                samples.add_data(it, tok.decode(ids))
                run.log({"samples": samples}, step=it)
            model.train()
            t0 = time.time()

    save(args.max_iters - 1)
    log("done")
    if run:
        run.finish()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
