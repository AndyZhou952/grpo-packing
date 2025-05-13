import argparse
import numpy as np
from tqdm import trange
from mindspore.mindrecord import FileWriter

def make_fake_mindrecord(
    output_path,
    num_samples=4,
    max_prompt_length=2048,
    seq_length=4096,
    vocab_size=32000,
    pad_token_id=151643,
    seed=42,
    pack_num=4, # used to control seq_length, no packing here.
):
    rng = np.random.default_rng(seed)
    pack_num = pack_num or 1
    max_sample_length = seq_length // (pack_num * 2)
    local_max_prompt = min(max_prompt_length, max_sample_length)

    schema = {
        "prompt_ids": {"type": "int64", "shape": [-1]},
        "pretrain_ids": {"type": "int64", "shape": [-1]},
        "loss_mask": {"type": "int64", "shape": [-1]},
    }

    writer = FileWriter(file_name=output_path, shard_num=1, overwrite=True)
    writer.add_schema(schema)

    for _ in trange(num_samples):
        sample_len = rng.integers(2, max_sample_length + 1)
        prompt_len = rng.integers(1, min(local_max_prompt, sample_len - 1) + 1)
        resp_len = sample_len - prompt_len

        prompt_ids = np.full(max_prompt_length, pad_token_id, np.int64)
        prompt_ids[:prompt_len] = rng.integers(1, vocab_size, prompt_len)

        pretrain_ids = np.full(seq_length, pad_token_id, np.int64)
        pretrain_ids[:resp_len] = rng.integers(1, vocab_size, resp_len)

        loss_mask = np.full(seq_length, pad_token_id, np.int64)
        loss_mask[:resp_len] = 1
        loss_mask[:prompt_len] = 0

        writer.write_raw_data([
            {"prompt_ids": prompt_ids,
             "pretrain_ids": pretrain_ids,
             "loss_mask": loss_mask}
        ])

    writer.commit()
    print(f"Wrote {num_samples} samples to {output_path}")

def get_fake_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_path", default='.')
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--max_prompt_length", type=int, default=2048)
    p.add_argument("--seq_length", type=int, default=4096)
    p.add_argument("--vocab_size", type=int, default=32000)
    p.add_argument("--pad_token_id", type=int, default=151643)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pack_num", type=int, default = 4)
    return p.parse_args()

if __name__ == "__main__":
    args = get_fake_args()
    make_fake_mindrecord(
        output_path=args.output_path,
        num_samples=args.num_samples,
        max_prompt_length=args.max_prompt_length,
        seq_length=args.seq_length,
        vocab_size=args.vocab_size,
        pad_token_id=args.pad_token_id,
        seed=args.seed,
        pack_num=args.pack_num,
    )
