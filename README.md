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