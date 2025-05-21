# non pack
prompt_completion_ids = np.array(
    [100, 101, 102, 103, 104, 105, 106, 107, 151643, 151643, 151643, 151643], dtype=int
)
prompts_mask = np.array(
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int
)
responses_mask = np.array(
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=int
)
ref_per_token_logps = np.array(
    [-2.8476415, -3.519992 , -2.6247745, -2.5297177,
     -3.9755175, -3.6510897, -2.9360797, -3.1581213,
     -3.0084007, -3.426522 , -2.560301 , -2.611104 ], dtype=np.float32
)
advantages = np.float32(-0.74015701)

# packed
prompt_completion_ids_packed = np.array(
    [100, 101, 102, 103, 104, 105, 106, 107, 151643, 151643, 151643, 151643], dtype=int
)
responses_mask_packed = np.array(
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=int
)
ref_per_token_logps_packed = np.array(
    [-2.8476415, -3.519992 , -2.6247745, -2.5297177,
     -3.9755175, -3.6510897, -2.9360797, -3.1581213,
      0.       ,  0.       ,  0.       ,  0.       ], dtype=np.float32
)
advantages_packed = np.array(
    [-0.74015701, -0.74015701, -0.74015701, -0.74015701,
     -0.74015701, -0.74015701, -0.74015701, -0.74015701,
     -0.74015701,  0.        ,  0.        ,  0.        ], dtype=np.float32
)
actual_sequence_length = np.array([12], dtype=int)
sample_index = np.array([0]*12, dtype=int)
sample_valid_len = np.array([4], dtype=int)
