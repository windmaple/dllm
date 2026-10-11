"""Generate many samples in parallel: each device denoises a batch of sequences, all devices at once.

Denoising one sequence is inherently sequential (step after step), so the parallelism is across
samples: --num_samples sequences per device are denoised as one batch, and under torchrun every
TPU chip (one process each, different seeds) does so simultaneously; rank 0 gathers the results.

    python sample.py --prompt "In a shocking finding, scientists" --num_samples 16      # one device
    eval $(python -m torch_tpu._internal.distributed.distributed | sed 's/^/export /')
    torchrun --standalone --nproc_per_node=$WORLD_SIZE sample.py --num_samples 64 --out samples.jsonl
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

import owt
from data import EOS_ID, MASK_ID, UNK_ID, CompactTokenizer
from diffusion import generate
from model import MiniGPT, MiniGPTConfig
from train import DIST_BACKEND, setup_device


def load_checkpoint(path, data_dir, device):
    """Model + tokenizer + special ids for an OWT (GPT-2) or TinyStories checkpoint."""
    ck = torch.load(path, map_location="cpu")
    dataset = ck["args"].get("dataset", "tinystories")
    if dataset == "owt":
        tok, mask_id, eos_id = owt.GPT2Tokenizer(), owt.MASK_ID, owt.EOT_ID
        banned_ids = range(owt.GPT2_VOCAB, owt.VOCAB_SIZE)
    else:
        tok, mask_id, eos_id = CompactTokenizer(os.path.join(data_dir, "vocab.json")), MASK_ID, EOS_ID
        banned_ids = (MASK_ID, UNK_ID)
    model = MiniGPT(MiniGPTConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, tok, dict(mask_id=mask_id, banned_ids=banned_ids), eos_id, dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/ckpt.pt")
    ap.add_argument("--data_dir", default="data", help="TinyStories vocab.json location")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--num_samples", type=int, default=16, help="sequences per device, denoised as one batch")
    ap.add_argument("--length", type=int, default=256)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--strategy", default="random", choices=["confidence", "random"])
    ap.add_argument("--block_len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "tpu", "cpu"])
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--out", default=None, help="write all samples to this .jsonl (default: print a few)")
    args = ap.parse_args()

    ddp = "LOCAL_RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1)) if ddp else 1
    device = setup_device(args.device)
    if ddp:
        dist.init_process_group(backend=DIST_BACKEND[device.type])
    torch.manual_seed(args.seed + rank)  # different samples on every device

    model, tok, ids_kw, eos_id, _ = load_checkpoint(args.ckpt, args.data_dir, device)
    prompt_ids = tok.encode(args.prompt) if args.prompt else None

    t0 = time.time()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.dtype == "bfloat16"):
        x, _ = generate(model, prompt_ids, length=args.length, steps=args.steps, temperature=args.temperature,
                        top_k=args.top_k, strategy=args.strategy, block_len=args.block_len, record=False,
                        device=device, num_samples=args.num_samples, **ids_kw)
    x = x.view(args.num_samples, args.length)
    dt = time.time() - t0

    if ddp:  # gather every device's batch (and the slowest device's time) on all ranks
        xs = [torch.empty_like(x, device=device) for _ in range(world_size)]
        dist.all_gather(xs, x.to(device))
        x = torch.cat([t.cpu() for t in xs])
        t = torch.tensor([dt], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        dt = t.item()

    if rank == 0:
        n = x.shape[0]
        print(f"{n} samples ({world_size} device(s) x {args.num_samples}) of {args.length} tokens in "
              f"{args.steps} steps: {dt:.1f}s -> {n / dt:.1f} samples/s, {n * args.length / dt:,.0f} tokens/s")
        texts = [tok.decode(row.tolist(), stop_at_eos=False) for row in x]
        if args.out:
            with open(args.out, "w") as f:
                for i, text in enumerate(texts):
                    f.write(json.dumps({"device": i // args.num_samples, "seed": args.seed + i // args.num_samples,
                                        "index": i % args.num_samples, "text": text}) + "\n")
            print(f"saved {args.out}")
        for text in texts[:3]:
            print("---\n" + text[:400])
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
