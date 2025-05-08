## Optimization 1
`grpo_models.py` function `batch_unsorted_segment_sum` ~#261

test script: `tests/test_batch_unsorted_segment_sum.py`

For batch size = 128, sequence length = 8,192, segment number = 32,
**original time ~126.76 ms  new ~ 6.85 ms, speed-up x18.5**.

GRPO training on the fly testing (ALL following default config in the branch 0.4.0): 
**original time ~ 6ms, new ~ 2.5 ms, speed-up x2.4**

**original version:**

```python
def batch_unsorted_segment_sum(input_ids, segments_ids, num_segments):
    slice = P.StridedSlice()
    bs, seq_len = input_ids.shape
    output = ops.zeros((bs, num_segments), input_ids.dtype)
    for b in range(bs):
        current_input = slice(input_ids, (b, 0), (b + 1, seq_len), (1, 1))
        current_segments = slice(segments_ids, (b, 0), (b + 1, seq_len), (1, 1))
        seg_sum = ops.unsorted_segment_sum(current_input, current_segments, num_segments)
        output[b] = seg_sum
    return output
```

**updated version:**

```python
# also remove self.slice definition
def batch_unsorted_segment_sum_new(input_ids, segments_ids, num_segments):
    bs, _ = input_ids.shape
    offsets = ops.arange(0, bs * num_segments, num_segments)
    # ensure segment_id uniqueness after reshape
    seg_off = segments_ids + offsets.view(bs, 1)
    flat_sum = ops.unsorted_segment_sum(
        input_ids.view(-1),
        seg_off.view(-1),
        bs * num_segments
    )
    return flat_sum.view(bs, num_segments)
```

## Optimization 2

`grpo_trainer.py` function `pack_grpo_data` ~#417 Vectorization

test script: `tests/test_pack_grpo_data.py`

For unique prompts = 4, generations per prompt = 8, sequence length = 8,192,
**original time ~3.87 ms  new ~ 0.89 ms, speed-up x4.35**.

GRPO training on the fly testing (ALL following default config in the branch 0.4.0, except that we set **pack_num=3**): 
**original time ~ 24.09ms, new ~ 19.41 ms, speed-up x1.24**

Remark: the logger performs slightly different from the original version for efficiency, but essentially report the same information.

**original version:**

```python
def pack_grpo_data(self, prompt_completion_ids, prompts_mask, responses_mask, advantages, pack_num=1):
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
    for i, pack_list in enumerate(pack_group):
        packed = self.pack_grouped_data(pack_list, pack_num)
        result.append(packed)
    return result
```

**updated version:**

```python
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
```

## Optimization 3
`grpo_models.py` function `pack_grouped_data`

Changes: 
- [x] pre-allocated array; 
- [x] vectorized dummy fills

test script: `tests/test_pack_grouped_data.py`

For unique prompts = 4, generations per prompt = 8, sequence length = 8,192,
**original time ~16.62 ms  new ~ 7.88 ms, speed-up x2.11**.

GRPO training on the fly testing (ALL following default config in the branch 0.4.0, except that we set **pack_num=3**): 
**original time ~ 21.24 ms, new ~ 17.57 ms, speed-up x1.21**

**original version:**

```python
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
```

**updated version:**

```python
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
```