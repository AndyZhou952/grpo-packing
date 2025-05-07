import time
import types
import numpy as np
import mindspore as ms
import logging
logger = logging.getLogger(__name__)


def pack_grpo_data(self, prompt_completion_ids, prompts_mask, responses_mask, advantages, pack_num=1):
    """ pack_grpo_data """
    data_dict_list = []
    bs = prompt_completion_ids.shape[0]
    advantages = advantages.reshape(-1)
    logger.info(f"advantages shape in pack: {advantages.shape}")
    for i in range(bs):
        sample_prompt_mask = prompts_mask[i]
        sample_response_mask = responses_mask[i]
        indices = np.nonzero(sample_prompt_mask)[0]
        if len(indices) > 0:
            prompt_start_idx = indices[0]
        else:
            logger.warning(f"prompts_mask is all zero for index {i}!")
            continue
        indices = np.nonzero(sample_response_mask)[0]
        if len(indices) > 0:
            response_end_index = indices[-1]
        else:
            logger.warning(f"responses_mask is all zero for index {i}!")
            continue
        data_dict = {"prompt_completion_ids": prompt_completion_ids[i],
                     "prompt_mask": prompts_mask[i],
                     "response_mask": responses_mask[i],
                     "advantage": advantages[i],
                     "prompt_start_idx": prompt_start_idx,
                     "response_end_index": response_end_index}
        data_dict_list.append(data_dict)
    pack_group = self.create_pack_group(data_dict_list, pack_num)
    result = [self.pack_grouped_data(p, pack_num) for p in pack_group]
    return result

def pack_grpo_data_new(self, prompt_completion_ids, prompts_mask, responses_mask, advantages, pack_num=1):
    bs, seq_len = prompts_mask.shape
    advantages = advantages.reshape(-1)
    logger.info(f"advantages shape in pack: {advantages.shape}")

    # determine if prompts and responses are non-empty
    has_prompt = prompts_mask.any(axis=1)
    has_response = responses_mask.any(axis=1)

    # warnings
    zero_prompts = np.where(~has_prompt)[0]
    zero_responses = np.where(has_prompt & ~has_response)[0]
    if zero_prompts.size> 0:
        logger.warning(
            "prompts_mask is all zero for indices [%s]!",
            ", ".join(map(str, zero_prompts.tolist()))
        )
    if zero_responses.size > 0:
        logger.warning(
            "responses_mask is all zero for indices [%s]!",
            ", ".join(map(str, zero_responses.tolist()))
        )

    # identify prompt_start_idx and response_end_index
    first_prompt = prompts_mask.argmax(axis=1)
    last_from_end = np.flip(responses_mask, axis=1).argmax(axis=1)
    last_response = seq_len - 1 - last_from_end

    # keep only those with both prompt and response
    valid_idx = np.where(has_prompt & has_response)[0]

    data_dict_list = [
        {
            "prompt_completion_ids": prompt_completion_ids[i],
            "prompt_mask": prompts_mask[i],
            "response_mask": responses_mask[i],
            "advantage": advantages[i],
            "prompt_start_idx": int(first_prompt[i]),
            "response_end_index": int(last_response[i]),
        }
        for i in valid_idx
    ]

    pack_group = self.create_pack_group(data_dict_list, pack_num)
    result = [self.pack_grouped_data(p, pack_num) for p in pack_group]
    return result

def generate_fake_batch(unique_prompts, generations_per_prompt, seq_len):
    rng = np.random.default_rng(0)
    bs = unique_prompts * generations_per_prompt

    prompt_completion_ids = rng.integers(5, 32000, (bs, seq_len), dtype=np.int32)
    base_masks = rng.random((unique_prompts, seq_len)) > 0.2
    prompts_mask = np.repeat(base_masks, generations_per_prompt, axis=0).astype(np.int32)

    response_mask = (rng.random((bs, seq_len)) > 0.5).astype(np.int32)

    advantages = rng.standard_normal((bs, seq_len)).astype(np.float32)
    return (prompt_completion_ids, prompts_mask, response_mask, advantages)

def test_equivalence():
    ids, prompts, responses, advantages = generate_fake_batch(4, 8, 1024)
    dummy_self = types.SimpleNamespace(create_pack_group=lambda x, y: [x],
                                       pack_grouped_data=lambda x, y: x) # placeholder really
    out_old = pack_grpo_data(dummy_self, ids, prompts, responses, advantages)
    out_new = pack_grpo_data_new(dummy_self, ids, prompts, responses, advantages)
    assert all(all(all(np.array_equal(d1[k], d2[k]) if isinstance(d1[k], np.ndarray) else d1[k] == d2[k] for k in d1) for d1, d2 in zip(g1, g2)) for g1, g2 in zip(out_old, out_new)), "mismatch"
    print("all good!")

def benchmark(unique_prompts  = 4, generations_per_prompt = 8, seq_len = 8192):
    ids, prompts, responses, advantages = generate_fake_batch(unique_prompts, generations_per_prompt, seq_len)
    dummy_old = types.SimpleNamespace(create_pack_group=lambda x, y: [x],
                                       pack_grouped_data=lambda x, y: x)
    dummy_new = types.SimpleNamespace(create_pack_group=lambda x, y: [x],
                                       pack_grouped_data=lambda x, y: x)

    t0 = time.time()
    for _ in range(100):
        _ = pack_grpo_data(dummy_old, ids, prompts, responses, advantages)
    t_old = (time.time() - t0)/100

    t0 = time.time()
    for _ in range(100):
        _ = pack_grpo_data_new(dummy_new, ids, prompts, responses, advantages)
    t_new = (time.time() - t0)/100

    print(f"unique prompts={unique_prompts} generations_per_prompt={generations_per_prompt} seq_len={seq_len}  "
          f"old={t_old * 1e3:.2f}ms  new={t_new * 1e3:.2f}ms")

if __name__ == "__main__":
    test_equivalence()
    benchmark()