import numpy as np


def create_fake_data_case1(
    batch_size=64,
    seq_length=8192,
    rollout=8,
    first_response_idx=6144,
    last_prompt_idx=6143,
    seed=0
):
    """
    Case 1: max_pack_num=1, real_pack_num=1.
    """
    np.random.seed(seed)
    pad_token_id = 151643
    all_prompt_completion_ids = np.full((batch_size, seq_length), pad_token_id, dtype=int)
    all_prompts_mask = np.zeros((batch_size, seq_length), dtype=int)
    all_responses_mask = np.zeros((batch_size, seq_length), dtype=int)
    advantages = np.random.randn(batch_size, 1).astype(float)

    # rollout groups
    num_groups = int(np.ceil(batch_size / rollout))
    group_prompt_starts = np.random.randint(0, last_prompt_idx + 1, size=num_groups)
    group_prompt_ids = [
        np.random.randint(1, 10000, size=(last_prompt_idx - start + 1,))
        for start in group_prompt_starts
    ]
    # random response ends
    response_end_idxs = np.random.randint(first_response_idx, seq_length, size=batch_size)

    for i in range(batch_size):
        grp = i // rollout
        prompt_start = group_prompt_starts[grp]
        response_end = response_end_idxs[i]
        # prompt mask and ids
        all_prompts_mask[i, prompt_start:last_prompt_idx + 1] = 1
        all_prompt_completion_ids[i, prompt_start:last_prompt_idx + 1] = group_prompt_ids[grp]
        # response mask and ids
        all_responses_mask[i, first_response_idx:response_end + 1] = 1
        all_prompt_completion_ids[i, first_response_idx:response_end + 1] = \
            np.random.randint(1, 10000, size=(response_end - first_response_idx + 1,))

    return all_prompt_completion_ids, all_prompts_mask, all_responses_mask, advantages


def create_fake_data_case2(
    batch_size=64,
    seq_length=8192,
    rollout=8,
    first_response_idx=6144,
    last_prompt_idx=6143,
    seed=0
):
    """
    Case 2: max_pack_num=4, real_pack_num=4.
    """
    np.random.seed(seed)
    pad_token_id = 151643
    all_prompt_completion_ids = np.full((batch_size, seq_length), pad_token_id, dtype=int)
    all_prompts_mask = np.zeros((batch_size, seq_length), dtype=int)
    all_responses_mask = np.zeros((batch_size, seq_length), dtype=int)
    advantages = np.random.randn(batch_size, 1).astype(float)

    # fixed prompt start for all
    prompt_start = last_prompt_idx
    sample_length = seq_length // 4  # each sample length
    # compute response_end so that sample_length = response_end - prompt_start + 2
    response_end = prompt_start + sample_length - 2  # <= seq_length-1

    for i in range(batch_size):
        # prompt mask and ids
        all_prompts_mask[i, prompt_start:last_prompt_idx + 1] = 1
        all_prompt_completion_ids[i, prompt_start:last_prompt_idx + 1] = \
            np.random.randint(1, 10000)
        # response mask and ids
        all_responses_mask[i, first_response_idx:response_end + 1] = 1
        all_prompt_completion_ids[i, first_response_idx:response_end + 1] = \
            np.random.randint(1, 10000, size=(response_end - first_response_idx + 1,))

    return all_prompt_completion_ids, all_prompts_mask, all_responses_mask, advantages

ids1, prompts_mask1, responses_mask1, advantages1 = create_fake_data_case1()
ids2, prompts_mask2, responses_mask2, advantages2 = create_fake_data_case2()

np.savez(
    "case1.npz",
    prompt_completion_ids=ids1,
    prompts_mask=prompts_mask1,
    responses_mask=responses_mask1,
    advantages=advantages1,
)

np.savez(
    "case2.npz",
    prompt_completion_ids=ids2,
    prompts_mask=prompts_mask2,
    responses_mask=responses_mask2,
    advantages=advantages2,
)
print("Saved case1.npz and case2.npz")
