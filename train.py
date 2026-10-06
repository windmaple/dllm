"""Train a miniGPT masked-diffusion LM on TinyStories.

    python train.py                       # default config (CPU friendly)
    python train.py --max_iters 200 --eval_interval 100   # quick smoke test
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from data import EOS_ID, CompactTokenizer
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


@torch.no_grad()
def evaluate(model, ds, args, n_batches):
    model.eval()
    rng = np.random.default_rng(1234)  # fixed eval batches -> comparable numbers
    torch.manual_seed(1234)
    elbos, ces = [], []
    for _ in range(n_batches):
        elbo, ce = diffusion_loss(model, ds.batch(args.batch_size, rng))
        elbos.append(elbo.item())
        ces.append(ce.item())
    model.train()
    return float(np.mean(elbos)), float(np.mean(ces))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="out")
    # model (shrunk-down GPT-2)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--n_embd", type=int, default=384)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)
    # optimization
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_iters", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr", type=float, default=1e-4)
    ap.add_argument("--warmup_iters", type=int, default=200)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # logging
    ap.add_argument("--log_interval", type=int, default=20)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=10)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tok = CompactTokenizer(os.path.join(args.data_dir, "vocab.json"))
    train_ds = StoryDataset(args.data_dir, "train", args.block_size)
    val_ds = StoryDataset(args.data_dir, "val", args.block_size)

    cfg = MiniGPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                        n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = MiniGPT(cfg).to(device)
    print(f"device={device} threads={torch.get_num_threads()} | params: {model.num_params() / 1e6:.2f}M")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))

    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    start_it = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_it = ck["iter"] + 1
        print(f"resumed from iter {ck['iter']}")

    train_model = torch.compile(model) if args.compile else model
    log_f = open(os.path.join(args.out_dir, "log.jsonl"), "a")

    def save(it):
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "config": asdict(cfg),
                    "iter": it, "args": vars(args)}, ckpt_path)

    model.train()
    t0 = time.time()
    for it in range(start_it, args.max_iters):
        lr = get_lr(it, args)
        for g in opt.param_groups:
            g["lr"] = lr

        x0 = train_ds.batch(args.batch_size, rng).to(device)
        elbo, ce = diffusion_loss(train_model, x0)
        opt.zero_grad(set_to_none=True)
        elbo.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if it % args.log_interval == 0:
            dt = (time.time() - t0) / (args.log_interval if it > start_it else 1)
            t0 = time.time()
            print(f"iter {it:5d} | elbo {elbo.item():.3f} | masked-ce {ce.item():.3f} | "
                  f"gnorm {gnorm:.2f} | lr {lr:.2e} | {dt * 1000:.0f} ms/it", flush=True)
            log_f.write(json.dumps({"iter": it, "train_elbo": elbo.item(), "train_ce": ce.item(), "lr": lr}) + "\n")
            log_f.flush()

        if (it > 0 and it % args.eval_interval == 0) or it == args.max_iters - 1:
            v_elbo, v_ce = evaluate(model, val_ds, args, args.eval_batches)
            print(f"==> iter {it} | val elbo {v_elbo:.3f} (ppl bound {math.exp(v_elbo):.1f}) | val masked-ce {v_ce:.3f}")
            log_f.write(json.dumps({"iter": it, "val_elbo": v_elbo, "val_ce": v_ce}) + "\n")
            log_f.flush()
            save(it)
            ids, _ = generate(model, tok.encode("Once upon a time"), length=128, steps=64,
                              temperature=0.8, record=False, device=device)
            print("sample:", repr(tok.decode(ids)[:400]), flush=True)
            model.train()
            t0 = time.time()

    save(args.max_iters - 1)
    print("done")


if __name__ == "__main__":
    main()
