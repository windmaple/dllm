"""Masked (absorbing-state) discrete diffusion, a la MDLM / LLaDA.

Forward process: at noise level t in [0, 1], every token is independently
replaced by <mask> with probability t. At t=1 the sequence is all masks.

Training objective (continuous-time ELBO of masked diffusion):
    L = E_{t, x_t} [ (1/t) * sum_{i masked} -log p_theta(x_0^i | x_t) ] / L
i.e. a re-weighted masked-LM loss with a random masking ratio.

Reverse process (generation): start from all masks and, over `steps` steps,
progressively commit the model's predictions for some of the masked
positions, keeping committed tokens fixed.
"""

import math

import torch
import torch.nn.functional as F

from data import EOS_ID, MASK_ID, UNK_ID


def diffusion_loss(model, x0, eps=1e-3):
    """x0: (B, L) clean tokens. Returns (elbo_loss, masked_token_ce)."""
    B, L = x0.shape
    # Stratified sampling of t over the batch -> lower-variance loss estimate.
    u = (torch.rand(1, device=x0.device) + torch.arange(B, device=x0.device) / B) % 1.0
    t = eps + (1 - eps) * u  # (B,)
    is_masked = torch.rand(B, L, device=x0.device) < t[:, None]
    xt = torch.where(is_masked, MASK_ID, x0)

    logits = model(xt, only=is_masked)  # (N_masked, V)
    ce = F.cross_entropy(logits.float(), x0[is_masked], reduction="none")
    w = (1.0 / t)[:, None].expand(B, L)[is_masked]
    elbo = (ce * w).sum() / (B * L)
    return elbo, ce.mean()


def _num_to_unmask(n_masked, steps):
    """Linear schedule: how many tokens to reveal at each of `steps` steps."""
    base = n_masked // steps
    counts = torch.full((steps,), base, dtype=torch.long)
    counts[: n_masked % steps] += 1
    return counts


@torch.no_grad()
def generate(model, prompt_ids=None, length=256, steps=128, temperature=1.0, top_k=None,
             strategy="confidence", block_len=None, record=True, device="cpu", generator=None):
    """Sample one sequence by iterative unmasking.

    strategy:
      "confidence": at each step reveal the masked positions whose sampled token
                    has the highest probability (LLaDA's low-confidence remasking).
      "random":     reveal a uniformly random subset of masked positions
                    (the vanilla MDLM ancestral sampler).
    block_len: if set, use semi-autoregressive decoding (LLaDA): the sequence is split
      into blocks of `block_len` tokens that are denoised left to right, in parallel
      within each block (the model still attends to the whole sequence). This stops
      confidence-based sampling from committing far-away, easy tokens (e.g. trailing
      <eos> padding) too early.
    Returns (final_tokens, history) where history is a list of (tokens, newly_revealed_mask)
    tensors, one per step (step 0 = initial state).
    """
    model.eval()
    x = torch.full((1, length), MASK_ID, dtype=torch.long, device=device)
    if prompt_ids:
        p = torch.tensor(prompt_ids[:length], device=device)
        x[0, : len(p)] = p
    history = [(x[0].clone().cpu(), torch.zeros(length, dtype=torch.bool))] if record else []

    # Partition the masked positions into blocks and give each block a share of the steps.
    masked_pos = (x[0] == MASK_ID).nonzero()[:, 0]
    if block_len is None:
        blocks = [masked_pos]
    else:
        blocks = [b for b in torch.split(masked_pos, block_len) if len(b)]
    steps_per_block = max(1, steps // len(blocks))

    for blk in blocks:
        in_block = torch.zeros(1, length, dtype=torch.bool, device=device)
        in_block[0, blk] = True
        n_steps = min(steps_per_block, len(blk))
        for k in _num_to_unmask(len(blk), n_steps).tolist():
            logits = model(x).float()  # (1, L, V)
            logits[..., MASK_ID] = -math.inf  # never predict the special tokens
            logits[..., UNK_ID] = -math.inf
            logits = logits / max(temperature, 1e-5)
            if top_k is not None:
                kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
                logits = logits.masked_fill(logits < kth, -math.inf)
            probs = logits.softmax(-1)
            if temperature > 0:
                x0 = torch.multinomial(probs[0], 1, generator=generator).T  # (1, L)
            else:
                x0 = probs.argmax(-1)
            conf = probs.gather(-1, x0[..., None])[..., 0]  # p(chosen token)

            if strategy == "confidence":
                score = conf
            elif strategy == "random":
                score = torch.rand(conf.shape, device=device, generator=generator)
            else:
                raise ValueError(strategy)
            eligible = (x == MASK_ID) & in_block
            score = score.masked_fill(~eligible, -math.inf)
            idx = torch.topk(score[0], k).indices
            reveal = torch.zeros_like(eligible)
            reveal[0, idx] = True
            x = torch.where(reveal, x0, x)
            if record:
                history.append((x[0].clone().cpu(), reveal[0].clone().cpu()))
    return x[0].cpu(), history
