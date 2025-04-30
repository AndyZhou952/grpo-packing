## Optimization 1
`grpo_models.py` function `batch_unsorted_segment_sum` ~#261

test script: `tests/test_batch_unsorted_segment_sum.py`

For batch size = 128, sequence length = 8,192, segment number = 32,
**original time ~126.76 ms  new ~ 6.85 ms**.

GRPO training on the fly testing: **original time ~ 6ms, new ~ 2.5 ms**

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


