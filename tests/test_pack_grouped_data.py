import numpy as np
import types
import time
import logging
logger = logging.getLogger(__name__)

# helpers
def pad_sequence_to_length(sequence, target_length, pad_value):
    """Pad sequence to target length with specified pad value."""
    current_length = len(sequence)
    if current_length < target_length:
        return np.pad(
            sequence,
            (0, target_length - current_length),
            mode="constant",
            constant_values=pad_value,
        )
    return sequence[:target_length]

def create_pack_group(self, data_dict_list, pack_num):
    sample_num = len(data_dict_list)
    pack_group, each_group = [], []
    current_group_length = 0
    for i in range(sample_num):
        sample_length = data_dict_list[i]["response_end_index"] - data_dict_list[i]["prompt_start_idx"] + 2
        needed_length = current_group_length + sample_length + (pack_num - len(each_group) - 1)
        if len(each_group) >= pack_num or needed_length > self.grpo_config.seq_length:
            pack_group.append(each_group)
            each_group = []
            current_group_length = 0
        each_group.append(data_dict_list[i])
        current_group_length += sample_length
    if each_group:
        pack_group.append(each_group)
    return pack_group

def pack_grpo_data(self, prompt_completion_ids, prompts_mask, responses_mask, advantages, pack_num=1):
    bs, seq_len = prompts_mask.shape
    advantages = advantages.reshape(-1)
    logger.info(f"advantages shape in pack: {advantages.shape}")

    # determine if prompts and responses are non-empty
    has_prompt = prompts_mask.any(axis=1)
    has_response = responses_mask.any(axis=1)

    # warnings
    zero_prompts = np.where(~has_prompt)[0]
    zero_responses = np.where(has_prompt & ~has_response)[0]
    if zero_prompts.size > 0:
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

# functions to be tested old ver
def pack_grouped_data(self, pack_list, pack_num=1):
    real_sample_num = len(pack_list)
    dummy_sample_num = pack_num - real_sample_num
    pad_to_length = self.grpo_config.seq_length - dummy_sample_num
    pad_token_id = self.tokenizer.eos_token_id

    prompt_completion_ids = []
    actual_sequence_length = []
    responses_mask = []
    sample_index = []
    sample_valid_length = []
    advantages = []
    occupied_length = 0
    for i, data_dict in enumerate(pack_list):
        sample_prompt_completion_ids = data_dict["prompt_completion_ids"]
        sample_response_mask = data_dict["response_mask"]
        sample_advantage = data_dict["advantage"]
        prompt_start_idx = data_dict["prompt_start_idx"]
        response_end_index = data_dict["response_end_index"]
        sample_length = response_end_index - prompt_start_idx + 2

        sample_prompt_completion_ids = sample_prompt_completion_ids[prompt_start_idx:response_end_index + 1]
        sample_prompt_completion_ids = np.pad(
            sample_prompt_completion_ids, (0, 1), mode="constant", constant_values=pad_token_id,
        )

        sample_response_mask = sample_response_mask[prompt_start_idx:response_end_index + 1]
        sample_response_mask = np.pad(
            sample_response_mask, (0, 1), mode="constant", constant_values=0,
        )

        sample_actual_sequence_length = occupied_length + sample_length
        this_sample_index = np.array([i] * sample_length)
        sample_advantage = np.array([sample_advantage] * sample_length)

        if i == real_sample_num - 1:
            sample_prompt_completion_ids = pad_sequence_to_length(
                sample_prompt_completion_ids, pad_to_length - occupied_length, pad_token_id
            )
            sample_response_mask = pad_sequence_to_length(
                sample_response_mask, pad_to_length - occupied_length, 0
            )
            sample_advantage = pad_sequence_to_length(
                sample_advantage, pad_to_length - occupied_length, 0
            )
            this_sample_index = pad_sequence_to_length(
                this_sample_index, pad_to_length - occupied_length, i
            )
            sample_actual_sequence_length = pad_to_length

        prompt_completion_ids.append(sample_prompt_completion_ids)
        responses_mask.append(sample_response_mask)
        advantages.append(sample_advantage)
        actual_sequence_length.append(sample_actual_sequence_length)
        sample_index.append(this_sample_index)
        sample_valid_length.append(np.sum(sample_response_mask))

        occupied_length += sample_length

    for i in range(dummy_sample_num):
        prompt_completion_ids.append(np.array([pad_token_id]))
        responses_mask.append(np.array([0]))
        advantages.append(np.array([0]))
        actual_sequence_length.append(actual_sequence_length[-1]+1)
        sample_index.append(np.array([real_sample_num + i]))
        sample_valid_length.append(1)

    result = {
        "prompt_completion_ids": np.concatenate(prompt_completion_ids, axis=0),
        "responses_mask": np.concatenate(responses_mask, axis=0),
        "advantages": np.concatenate(advantages, axis=0),
        "actual_sequence_length": np.array(actual_sequence_length),
        "sample_index": np.concatenate(sample_index, axis=0),
        "sample_valid_length": np.array(sample_valid_length)
    }

    return result

# functions to be tested - new ver
def pack_grouped_data_new(self, pack_list, pack_num=1):
    real_sample_num = len(pack_list)
    dummy_sample_num = pack_num - real_sample_num
    pad_to_length = self.grpo_config.seq_length - dummy_sample_num
    pad_token_id = self.tokenizer.eos_token_id

    total_sequence_slots = self.grpo_config.seq_length
    total_samples = real_sample_num + dummy_sample_num

    # preallocate
    prompt_completion_ids = np.full(total_sequence_slots, pad_token_id, dtype=int)
    responses_mask = np.zeros(total_sequence_slots, dtype=int)
    advantages = np.zeros(total_sequence_slots, dtype=float)
    sample_index = np.zeros(total_sequence_slots, dtype=int)
    actual_sequence_length = np.zeros(total_samples, dtype=int)
    sample_valid_length = np.zeros(total_samples, dtype=int)

    occupied_length = 0

    for i, data_dict in enumerate(pack_list):
        sample_prompt_completion_ids = data_dict["prompt_completion_ids"]
        sample_response_mask = data_dict["response_mask"]
        sample_advantage_value = data_dict["advantage"]
        prompt_start_idx = data_dict["prompt_start_idx"]
        response_end_index = data_dict["response_end_index"]

        original_length = response_end_index - prompt_start_idx + 2

        segment = sample_prompt_completion_ids[prompt_start_idx:response_end_index + 1]
        tmp_prompt_ids = pad_sequence_to_length(segment, original_length, pad_token_id)
        mask_segment = sample_response_mask[prompt_start_idx:response_end_index + 1]
        tmp_responses_mask = pad_sequence_to_length(mask_segment, original_length, 0)

        tmp_sample_index = np.full(original_length, i, dtype=int)
        tmp_advantages = np.full(original_length, sample_advantage_value, dtype=float)

        if i == real_sample_num - 1:
            tmp_prompt_ids = pad_sequence_to_length(tmp_prompt_ids, pad_to_length - occupied_length, pad_token_id)
            tmp_responses_mask = pad_sequence_to_length(tmp_responses_mask, pad_to_length - occupied_length, 0)
            tmp_advantages = pad_sequence_to_length(tmp_advantages, pad_to_length - occupied_length, 0)
            tmp_sample_index = pad_sequence_to_length(tmp_sample_index, pad_to_length - occupied_length, i)
            write_length = pad_to_length - occupied_length
            actual_sequence_length[i] = pad_to_length
        else:
            write_length = original_length
            actual_sequence_length[i] = occupied_length + original_length

        prompt_completion_ids[occupied_length:occupied_length + write_length] = tmp_prompt_ids
        responses_mask[occupied_length:occupied_length + write_length] = tmp_responses_mask
        advantages[occupied_length:occupied_length + write_length] = tmp_advantages
        sample_index[occupied_length:occupied_length + write_length] = tmp_sample_index

        sample_valid_length[i] = int(tmp_responses_mask.sum())
        occupied_length += write_length

    # fill dummy, prompt completion ids already pad_token_id, responses_mask and advantages already zero
    start = occupied_length
    end = start + dummy_sample_num
    sample_index[start:end] = np.arange(real_sample_num, real_sample_num + dummy_sample_num, dtype=int)
    base_length = actual_sequence_length[real_sample_num - 1]
    actual_sequence_length[real_sample_num:real_sample_num + dummy_sample_num] = \
        base_length + np.arange(1, dummy_sample_num + 1, dtype=int)
    sample_valid_length[real_sample_num:real_sample_num + dummy_sample_num] = 1

    result = {
        "prompt_completion_ids":  prompt_completion_ids,
        "responses_mask":         responses_mask,
        "advantages":             advantages,
        "actual_sequence_length": actual_sequence_length,
        "sample_index":           sample_index,
        "sample_valid_length":    sample_valid_length,
    }
    return result

def generate_synthetic_batch(num_unique_prompts,
                             num_generations_per_prompt,
                             sequence_length,
                             vocabulary_size=30000):
    rng = np.random.default_rng(42)
    batch_size = num_unique_prompts * num_generations_per_prompt

    token_id_matrix = rng.integers(
        low=1,
        high=vocabulary_size,
        size=(batch_size, sequence_length),
        dtype=np.int32
    )

    prompt_mask_matrix = np.zeros((batch_size, sequence_length), dtype=np.int32)
    response_mask_matrix = np.zeros((batch_size, sequence_length), dtype=np.int32)
    advantage_values = rng.standard_normal(batch_size).astype(np.float32)

    for sample_idx in range(batch_size):
        max_prompt_length = sequence_length // 4
        prompt_length = rng.integers(1, max_prompt_length + 1)

        max_response_length = sequence_length - prompt_length - 1
        response_length = rng.integers(
            1,
            min(max_response_length, sequence_length // 4) + 1
        )

        prompt_start_index = rng.integers(
            0,
            sequence_length - prompt_length - response_length
        )

        response_start_index = prompt_start_index + prompt_length

        prompt_mask_matrix[
            sample_idx,
            prompt_start_index:response_start_index
        ] = 1
        response_mask_matrix[
            sample_idx,
            response_start_index:response_start_index + response_length
        ] = 1

    return token_id_matrix, prompt_mask_matrix, response_mask_matrix, advantage_values

def test_equivalence(num_unique_prompts=4,
                     num_generations_per_prompt=8,
                     sequence_length=1024):
    batch_ids, prompt_mask, response_mask, advantage_values = generate_synthetic_batch(
        num_unique_prompts,
        num_generations_per_prompt,
        sequence_length
    )

    dummy_old = types.SimpleNamespace(
        grpo_config=types.SimpleNamespace(seq_length=sequence_length),
        tokenizer=types.SimpleNamespace(eos_token_id=5)
    )
    dummy_old.create_pack_group = types.MethodType(create_pack_group, dummy_old)
    dummy_old.pack_grouped_data = types.MethodType(pack_grouped_data, dummy_old)

    dummy_new = types.SimpleNamespace(
        grpo_config=types.SimpleNamespace(seq_length=sequence_length),
        tokenizer=types.SimpleNamespace(eos_token_id=5)
    )
    dummy_new.create_pack_group = types.MethodType(create_pack_group, dummy_new)
    dummy_new.pack_grouped_data = types.MethodType(pack_grouped_data_new, dummy_new)

    output_old = pack_grpo_data(
        dummy_old,
        batch_ids,
        prompt_mask,
        response_mask,
        advantage_values,
        pack_num=num_generations_per_prompt
    )
    output_new = pack_grpo_data(
        dummy_new,
        batch_ids,
        prompt_mask,
        response_mask,
        advantage_values,
        pack_num=num_generations_per_prompt
    )

    assert len(output_old) == len(output_new), "Number of groups differs"
    for group_idx, (group_old, group_new) in enumerate(zip(output_old, output_new)):
        for key in group_old:
            assert np.array_equal(group_old[key], group_new[key]), (
                f"Mismatch in group {group_idx}, key '{key}'"
            )
    print("all good!")

def benchmark(num_unique_prompts=4,
              num_generations_per_prompt=8,
              sequence_length=8192):
    batch_ids, prompt_mask, response_mask, advantage_values = generate_synthetic_batch(
        num_unique_prompts,
        num_generations_per_prompt,
        sequence_length
    )

    dummy_old = types.SimpleNamespace(
        grpo_config=types.SimpleNamespace(seq_length=sequence_length),
        tokenizer=types.SimpleNamespace(eos_token_id=5)
    )
    dummy_old.create_pack_group = types.MethodType(create_pack_group, dummy_old)
    dummy_old.pack_grouped_data = types.MethodType(pack_grouped_data, dummy_old)

    dummy_new = types.SimpleNamespace(
        grpo_config=types.SimpleNamespace(seq_length=sequence_length),
        tokenizer=types.SimpleNamespace(eos_token_id=5)
    )
    dummy_new.create_pack_group = types.MethodType(create_pack_group, dummy_new)
    dummy_new.pack_grouped_data = types.MethodType(pack_grouped_data_new, dummy_new)

    start_time = time.time()
    for _ in range(100):
        _ = pack_grpo_data(
            dummy_old,
            batch_ids,
            prompt_mask,
            response_mask,
            advantage_values,
            pack_num=num_generations_per_prompt
        )
    elapsed_old = (time.time() - start_time) / 100

    start_time = time.time()
    for _ in range(100):
        _ = pack_grpo_data(
            dummy_new,
            batch_ids,
            prompt_mask,
            response_mask,
            advantage_values,
            pack_num=num_generations_per_prompt
        )
    elapsed_new = (time.time() - start_time) / 100

    speedup = elapsed_old / elapsed_new
    print(
        f"unique_prompts={num_unique_prompts}  "
        f"generations_per_prompt={num_generations_per_prompt}  "
        f"sequence_length={sequence_length}  "
        f"old={elapsed_old*1e3:.2f}ms  "
        f"new={elapsed_new*1e3:.2f}ms  "
        f"speed-up x{speedup:.2f}"
    )

if __name__ == "__main__":
    test_equivalence() # all good!
    benchmark() # old ~ 16.62ms  new ~ 7.88ms (total time running pack_grpo_data)