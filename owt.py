"""OpenWebText data + the GPT-2 tokenizer used for the GPT-2-size model.

The data is the pre-tokenized Kaggle dataset windmaple/openwebtext-gpt2 (produced with nanoGPT's
openwebtext/prepare.py): every document is tokenized with GPT-2's BPE, terminated by <|endoftext|>, and
concatenated into one flat uint16 stream (train.bin: ~9.04B tokens, val.bin: ~4.4M tokens). Unlike
TinyStories (one short story per sample), training samples are random block_size windows of that stream.

Vocabulary: the 50,257 GPT-2 tokens, plus <mask> = 50257 (the absorbing state of the diffusion
process), padded to 50,304 (a multiple of 128) for the TPU's matrix units. Ids >= 50258 never occur.

Usage:
    python owt.py --out_dir /dev/shm/owt     # downloads train.bin / val.bin (needs `pip install kaggle`
                                             # and KAGGLE_USERNAME / KAGGLE_KEY or ~/.kaggle/kaggle.json)
"""

import argparse
import os

import numpy as np

GPT2_VOCAB = 50257
EOT_ID = 50256  # <|endoftext|>: document separator
MASK_ID = 50257
VOCAB_SIZE = 50304  # padded

KAGGLE_DATASET = "windmaple/openwebtext-gpt2"


class GPT2Tokenizer:
    """tiktoken GPT-2 BPE with the same interface as data.CompactTokenizer."""

    def __init__(self):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = VOCAB_SIZE

    def encode(self, text):
        return self.enc.encode_ordinary(text)

    def token_bytes(self, i):
        """Raw bytes of a GPT-2 token id (< 50257)."""
        return self.enc.decode_single_token_bytes(int(i))

    def token_str(self, i):
        if i == EOT_ID:
            return "<eot>"
        if i == MASK_ID:
            return "<mask>"
        if i >= GPT2_VOCAB:
            return "<pad>"
        return self.enc.decode_single_token_bytes(int(i)).decode("utf-8", errors="replace")

    def decode(self, ids, stop_at_eos=False):
        out = []
        for i in ids:
            i = int(i)
            if i == EOT_ID and stop_at_eos:
                break
            if i < GPT2_VOCAB:
                out.append(i)
        return self.enc.decode(out)


class PackedDataset:
    """Random block_size windows of a flat uint16 token stream."""

    def __init__(self, data_dir, split, block_size):
        self.tokens = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
        self.block_size = block_size

    def batch(self, batch_size, rng):
        import torch
        idx = rng.integers(0, len(self.tokens) - self.block_size, batch_size)
        return torch.from_numpy(np.stack([self.tokens[i: i + self.block_size] for i in idx]).astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/dev/shm/owt")
    args = ap.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=args.out_dir, unzip=True, quiet=False)
    for split in ("train", "val"):
        n = os.path.getsize(os.path.join(args.out_dir, f"{split}.bin")) // 2
        print(f"{split}: {n:,} tokens")


if __name__ == "__main__":
    main()
