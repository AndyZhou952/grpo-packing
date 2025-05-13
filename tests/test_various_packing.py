# Implementations following the original dev ver (not the optimized ver.). Test cases are generated with the help of AI. Verified for the most parts 

import numpy as np

def create_pack_group(data_list, pack_num, seq_length):
    """Group samples for packing without exceeding seq_length."""
    pack_groups = []
    current_group = []
    current_length = 0
    for item in data_list:
        sample_length = item["response_end_index"] - item["prompt_start_idx"] + 2  # +2 for inclusive span + pad
        needed_length = current_length + sample_length + (pack_num - len(current_group) - 1)
        if len(current_group) >= pack_num or needed_length > seq_length:
            pack_groups.append(current_group)
            current_group = []
            current_length = 0
        current_group.append(item)
        current_length += sample_length
    if current_group:
        pack_groups.append(current_group)
    return pack_groups

def pack_grouped_data(pack_list, pack_num, seq_length, pad_id):
    """
    Pack up to `pack_num` samples into one length-`seq_length` sequence,
    adding 1-token dummy sequences if real samples < pack_num.
    """
    real_num   = len(pack_list)
    dummy_num  = pack_num - real_num
    pad_to_len = seq_length - dummy_num

    occupied_length = 0
    concat_ids, concat_rmask, concat_adv, concat_idx = [], [], [], []
    actual_seq_lengths, sample_valid_lengths = [], []

    for idx, item in enumerate(pack_list):
        start = item["prompt_start_idx"]
        end   = item["response_end_index"]

        seq_ids   = item["prompt_completion_ids"][start : end + 1]
        seq_rmask = item["response_mask"][start : end + 1]

        need_inter_pad = not (idx == real_num - 1 and dummy_num == 0) # change for robustness, in fact does not matter
        if need_inter_pad:                                          
            seq_ids   = np.pad(seq_ids,   (0, 1), constant_values=pad_id)
            seq_rmask = np.pad(seq_rmask, (0, 1), constant_values=0)

        seq_adv = np.full(seq_ids.shape, item["advantage"], dtype=np.float32)
        seq_idx = np.full(seq_ids.shape, idx,               dtype=np.int32)

        if idx == real_num - 1:
            target_len = pad_to_len - occupied_length
            seq_ids   = np.pad(seq_ids,   (0, max(0, target_len - seq_ids.size)),   constant_values=pad_id)[:target_len]
            seq_rmask = np.pad(seq_rmask, (0, max(0, target_len - seq_rmask.size)), constant_values=0)     [:target_len]
            seq_adv   = np.pad(seq_adv,   (0, max(0, target_len - seq_adv.size)),   constant_values=0)     [:target_len]
            seq_idx   = np.pad(seq_idx,   (0, max(0, target_len - seq_idx.size)),   constant_values=idx)   [:target_len]
            actual_seq_lengths.append(pad_to_len)
        else:
            actual_seq_lengths.append(occupied_length + seq_ids.size)

        occupied_length += seq_ids.size

        concat_ids.append(seq_ids)
        concat_rmask.append(seq_rmask)
        concat_adv.append(seq_adv)
        concat_idx.append(seq_idx)
        sample_valid_lengths.append(int(seq_rmask.sum()))

    for d in range(dummy_num):
        concat_ids.append(np.array([pad_id],       np.int32))
        concat_rmask.append(np.array([0],          np.int32))
        concat_adv.append(np.array([0],            np.float32))
        concat_idx.append(np.array([real_num + d], np.int32))
        last_len = actual_seq_lengths[-1] if actual_seq_lengths else 0
        actual_seq_lengths.append(last_len + 1)
        sample_valid_lengths.append(1)

    return {
        "prompt_completion_ids":  np.concatenate(concat_ids),
        "responses_mask":         np.concatenate(concat_rmask),
        "advantages":             np.concatenate(concat_adv),
        "sample_index":           np.concatenate(concat_idx),
        "actual_sequence_length": np.array(actual_seq_lengths, dtype=np.int32),
        "sample_valid_length":    np.array(sample_valid_lengths, dtype=np.int32),
    }

def pack_grpo_data(prompt_ids, prompt_mask, resp_mask, advantages, pack_num, seq_length):
    data_entries = []
    bs = prompt_ids.shape[0]
    if advantages.ndim == 2:
        adv_values = advantages[:, 0]
    else:
        adv_values = advantages

    for i in range(bs):
        if resp_mask[i].sum() == 0:
            continue
        prompt_start_idx = int(np.argmax(prompt_mask[i] == 1)) if np.any(prompt_mask[i]) else 0
        resp_indices     = np.nonzero(resp_mask[i])[0]
        response_end_index = int(resp_indices[-1]) if resp_indices.size > 0 else -1
        data_entries.append({
            "prompt_completion_ids": prompt_ids[i],
            "prompt_mask":           prompt_mask[i],
            "response_mask":         resp_mask[i],
            "advantage":             float(adv_values[i]),
            "prompt_start_idx":      prompt_start_idx,
            "response_end_index":    response_end_index
        })

    grouped = create_pack_group(data_entries, pack_num, seq_length)
    return [
        pack_grouped_data(group, pack_num, seq_length, pad_id=0)
        for group in grouped
    ]

def _construct_inputs_packing(all_packed, ref_batch_size, idx):
    ids_batch, seq_len_batch = [], []
    for i in range(ref_batch_size):
        entry = all_packed[i + idx * ref_batch_size]
        ids_batch.append(entry["prompt_completion_ids"])
        seq_len_batch.append(entry["actual_sequence_length"])
    return np.array(ids_batch), np.array(seq_len_batch)


import os
import unittest
import numpy as np


class TestGRPOPaddingAndLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import mindspore as ms
        except ImportError:
            pass

    def setUp(self):
        self.batch_size = 4
        self.seq_length = 16
        self.pad_id     = 0
        np.random.seed(42)

        self.base_ids   = np.full((self.batch_size, self.seq_length), self.pad_id, dtype=np.int32)
        self.base_pmask = np.zeros((self.batch_size, self.seq_length), dtype=np.int32)
        self.base_rmask = np.zeros((self.batch_size, self.seq_length), dtype=np.int32)

        self.adv_values = np.random.uniform(-1.0, 1.0, size=self.batch_size).astype(np.float32)
        self.base_adv   = np.zeros((self.batch_size, self.seq_length), dtype=np.float32)

        for i in range(self.batch_size):
            prompt_len    = 1
            resp_len      = 2
            prompt_token  = np.random.randint(1, 100, size=prompt_len, dtype=np.int32)
            response_tok  = np.random.randint(1, 100, size=resp_len, dtype=np.int32)

            self.base_ids[i, 0:prompt_len]                         = prompt_token
            self.base_ids[i, prompt_len : prompt_len + resp_len]   = response_tok

            self.base_pmask[i, 0:prompt_len]                        = 1
            self.base_rmask[i, prompt_len : prompt_len + resp_len]  = 1

            # repeat advantage over prompt+response
            self.base_adv[i, 0 : prompt_len + resp_len] = self.adv_values[i]

    def test_create_pack_group(self):
        samples = [
            {"prompt_start_idx": 0, "response_end_index": 5},
            {"prompt_start_idx": 0, "response_end_index": 8},
            {"prompt_start_idx": 0, "response_end_index": 6},
            {"prompt_start_idx": 0, "response_end_index": 7},
        ]
        groups = create_pack_group(samples, pack_num=4, seq_length=10)

        # all samples accounted for
        self.assertEqual(sum(len(g) for g in groups), len(samples))
        # Never more than pack_num in any group
        for g in groups:
            self.assertLessEqual(len(g), 4)                    

    def test_pack_grouped_data_padding(self):
        sampleA = {
            "prompt_completion_ids": np.array([5,6,7,8], dtype=np.int32),
            "response_mask":        np.array([0,1,1,1], dtype=np.int32),
            "advantage":            1.0,
            "prompt_start_idx":     0,
            "response_end_index":   3
        }
        sampleB = {
            "prompt_completion_ids": np.array([9,10,11], dtype=np.int32),
            "response_mask":        np.array([0,1,1], dtype=np.int32),
            "advantage":            1.0,
            "prompt_start_idx":     0,
            "response_end_index":   2
        }
        packed = pack_grouped_data([sampleA, sampleB], pack_num=4, seq_length=11, pad_id=self.pad_id)

        # shape sanity
        self.assertEqual(packed["prompt_completion_ids"].size, 11)
        self.assertEqual(packed["responses_mask"].size,        11)
        self.assertEqual(packed["advantages"].size,           11)
        self.assertEqual(packed["sample_index"].size,         11)
        # dummy indices at end
        self.assertTrue(np.array_equal(packed["sample_index"][-2:], np.array([2,3], dtype=np.int32)))
        # dummy masks & adv are zero
        self.assertEqual(packed["responses_mask"][-2:].sum(), 0)
        self.assertEqual(packed["advantages"][-2:].sum(),     0.0)
        # last actual_sequence_length = full length
        self.assertEqual(packed["actual_sequence_length"][-1], 11)
        # dummy valid length = 1
        self.assertTrue(np.all(packed["sample_valid_length"][-2:] == 1))

    def test_pack_grpo_data_cases(self):
        # No-packing scenario
        pack_results = pack_grpo_data(self.base_ids, self.base_pmask, self.base_rmask,
                                      self.base_adv, pack_num=1, seq_length=self.seq_length)
        self.assertEqual(len(pack_results), self.batch_size)
        for i, packed in enumerate(pack_results):
            np.testing.assert_array_equal(packed["prompt_completion_ids"], self.base_ids[i])
            np.testing.assert_array_equal(packed["responses_mask"],       self.base_rmask[i])
            self.assertTrue(np.all(packed["sample_index"] == 0))
            self.assertEqual(packed["actual_sequence_length"][-1], self.seq_length)
            self.assertEqual(packed["sample_valid_length"][0], int(self.base_rmask[i].sum()))

        # Full-packing scenario
        packed_all_list = pack_grpo_data(self.base_ids, self.base_pmask, self.base_rmask,
                                         self.base_adv, pack_num=4, seq_length=self.seq_length)
        self.assertEqual(len(packed_all_list), 1)
        packed_all = packed_all_list[0]

        # sequence length
        self.assertEqual(packed_all["prompt_completion_ids"].size, self.seq_length)
        # indices cover 0..3
        unique_idx = np.unique(packed_all["sample_index"])
        self.assertTrue(np.array_equal(np.sort(unique_idx), np.arange(self.batch_size)))

        # each sample contributes (prompt+response+1 PAD)
        for idx in range(self.batch_size):
            segment_len = np.sum(packed_all["sample_index"] == idx)
            orig_content = int(self.base_pmask[idx].sum() + self.base_rmask[idx].sum())
            expected_len = orig_content + 1                        # ← CHANGED
            self.assertEqual(segment_len, expected_len)

        # valid lengths = number of response tokens
        self.assertTrue(np.array_equal(
            packed_all["sample_valid_length"],
            self.base_rmask.sum(axis=1).astype(np.int32)
        ))
        # final actual_sequence_length = full length
        self.assertEqual(packed_all["actual_sequence_length"][-1], self.seq_length)

    def test_construct_inputs_packing(self):
        single_packs = pack_grpo_data(self.base_ids, self.base_pmask, self.base_rmask,
                                      self.base_adv, pack_num=1, seq_length=self.seq_length)
        ref_batch_size = 2

        part0_ids, part0_seq_lens = _construct_inputs_packing(single_packs, ref_batch_size, idx=0)
        part1_ids, part1_seq_lens = _construct_inputs_packing(single_packs, ref_batch_size, idx=1)

        self.assertEqual(part0_ids.shape[0], ref_batch_size)
        self.assertEqual(part1_ids.shape[0], ref_batch_size)

        np.testing.assert_array_equal(part0_ids[0], self.base_ids[0])
        np.testing.assert_array_equal(part0_ids[1], self.base_ids[1])
        np.testing.assert_array_equal(part1_ids[0], self.base_ids[2])
        np.testing.assert_array_equal(part1_ids[1], self.base_ids[3])

        np.testing.assert_array_equal(part0_seq_lens[0], single_packs[0]["actual_sequence_length"])
        np.testing.assert_array_equal(part0_seq_lens[1], single_packs[1]["actual_sequence_length"])
        np.testing.assert_array_equal(part1_seq_lens[0], single_packs[2]["actual_sequence_length"])
        np.testing.assert_array_equal(part1_seq_lens[1], single_packs[3]["actual_sequence_length"])

    def test_loss_equivalence_packed_vs_unpacked(self):
        if not os.path.exists("caseA.npz"):
            # generate synthetic data
            np.savez("caseA.npz",
                     prompt_completion_ids=self.base_ids,
                     prompts_mask=self.base_pmask,
                     responses_mask=self.base_rmask,
                     advantages=self.base_adv)
            packed_all = pack_grpo_data(self.base_ids, self.base_pmask, self.base_rmask,
                                        self.base_adv, pack_num=4, seq_length=self.seq_length)[0]
            np.savez("caseC.npz",
                     prompt_completion_ids=packed_all["prompt_completion_ids"].reshape(1, -1),
                     prompts_mask=self.base_pmask.reshape(1, -1),
                     responses_mask=packed_all["responses_mask"].reshape(1, -1),
                     advantages=packed_all["advantages"].reshape(1, -1))

        dataA = np.load("caseA.npz")
        dataC = np.load("caseC.npz")
        A_ids, A_rmask, A_adv = dataA["prompt_completion_ids"], dataA["responses_mask"], dataA["advantages"]
        C_ids, C_rmask, C_adv = dataC["prompt_completion_ids"], dataC["responses_mask"], dataC["advantages"]

        policy_logps_A = - (A_ids % 10) / 10.0
        policy_logps_C = - (C_ids % 10) / 10.0
        ref_logps_A    = policy_logps_A - 0.1
        ref_logps_C    = policy_logps_C - 0.1
        beta = 0.01

        kl_A = np.exp(ref_logps_A - policy_logps_A) - (ref_logps_A - policy_logps_A) - 1
        per_token_loss_A = - (np.exp(policy_logps_A - policy_logps_A) * A_adv - beta * kl_A)
        masked_loss_A    = per_token_loss_A * A_rmask
        avg_loss_per_seq = masked_loss_A.sum(axis=1) / A_rmask.sum(axis=1)
        total_loss_unpacked = np.mean(avg_loss_per_seq)

        kl_C = np.exp(ref_logps_C - policy_logps_C) - (ref_logps_C - policy_logps_C) - 1
        per_token_loss_C = - (np.exp(policy_logps_C - policy_logps_C) * C_adv - beta * kl_C)
        masked_loss_C    = per_token_loss_C * C_rmask

        original_resp_lengths = [int(A_rmask[i].sum()) for i in range(self.batch_size)]
        segment_avgs = []
        idx = 0
        for j, resp_len in enumerate(original_resp_lengths):
            seg_length = resp_len + 2                             # ← CHANGED
            segment_loss = masked_loss_C[0, idx : idx + seg_length].sum() / resp_len
            segment_avgs.append(segment_loss)
            idx += seg_length

        avg_loss_per_segment = np.array(segment_avgs)
        total_loss_packed    = np.mean(avg_loss_per_segment)

        for k in range(self.batch_size):
            self.assertAlmostEqual(avg_loss_per_seq[k], avg_loss_per_segment[k], places=5)
        self.assertAlmostEqual(total_loss_unpacked, total_loss_packed, places=5)

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
