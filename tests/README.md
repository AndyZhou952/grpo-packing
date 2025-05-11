## Comparing the master ver. and the dev ver. (w/ packing)

### grpo_models.py
#### robustness around handling dummy input in packing
```
# we have this line b/c all dummy sample has length 1
# may break if real sample has length 1
real_sample_num = ops.sum(sample_valid_len != 1, dtype=mstype.int32)
```

Quick fixes: inside `pack_grouped_data`
```
# in the original implementation
sample_valid_length.append(0) # set to 0
# in the new implementation
sample_valid_length[real_sample_num:real_sample_num + dummy_sample_num] = 0 # set to 0
```

#### early mask out in the padding version
```
# packing version
per_token_logps = self.policy_model(input_ids, None, None, None, False, False,
                                            samples, actual_seq_length, False, False)
per_token_logps = per_token_logps * responses_mask
per_token_kl = self.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1  # [bs, seq_len]
per_token_loss = self.exp(per_token_logps - ops.stop_gradient(per_token_logps)) * advantages # [bs, seq_len]
per_token_loss = - (per_token_loss - self.beta * per_token_kl)  # [bs, seq_len]
masked_per_token_loss = per_token_loss * responses_mask
```
```
# master version
per_token_logps = self.policy_model(input_ids, samples=samples)  # [bs, seq_len]
per_token_kl = self.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1  # [bs, seq_len]
per_token_loss = self.exp(per_token_logps - ops.stop_gradient(per_token_logps)) * advantages  # [bs, seq_len]
per_token_loss = -(per_token_loss - self.beta * per_token_kl)  # [bs, seq_len]
responses_mask = responses_mask[:, 1:]
masked_per_token_loss = per_token_loss * responses_mask
```
Essentially equivalent, packing version might be slightly better to avoid computation waste. However, if the goal is to
align with the master ver., may remove `per_token_logps = per_token_logps * responses_mask`

#### undetermined: offset_actual_seq_length & policy model (likely ok)
