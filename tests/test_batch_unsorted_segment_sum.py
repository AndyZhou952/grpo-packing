import time
import numpy as np
import mindspore as ms
from mindspore import Tensor, ops
from mindspore.ops import operations as P

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

def batch_unsorted_segment_sum_new(input_ids, segments_ids, num_segments):
    bs, _ = input_ids.shape
    offsets = ops.arange(0, bs * num_segments, num_segments)
    seg_off = segments_ids + offsets.view(bs, 1)
    flat_sum = ops.unsorted_segment_sum(
        input_ids.view(-1),
        seg_off.view(-1),
        bs * num_segments
    )
    return flat_sum.view(bs, num_segments)

def test_equivalence(bs, seq_len, num_segments):
    np.random.seed(0)
    a = Tensor(np.random.randint(0, 10, (bs, seq_len)), ms.float32)
    b = Tensor(np.random.randint(0,num_segments,(bs,seq_len)), ms.int32)

    out1 = batch_unsorted_segment_sum(a, b, num_segments)
    out2 = batch_unsorted_segment_sum_new(a, b, num_segments)
    assert np.allclose(out1.asnumpy(), out2.asnumpy()), "mismatch"
    print("all good!")

def benchmark(bs, seq_len, num_segments):
    ms.set_context(mode = ms.GRAPH_MODE)
    np.random.seed(1)
    a = Tensor(np.random.rand(bs,seq_len).astype(np.float32))
    b = Tensor(np.random.randint(0,num_segments,(bs,seq_len)), ms.int32)

    t0 = time.time()
    for _ in range(100):
        _ = batch_unsorted_segment_sum(a, b, num_segments)
    t_old = (time.time() - t0)/100

    t0 = time.time()
    for _ in range(100):
        _ = batch_unsorted_segment_sum_new(a, b, num_segments)
    t_new = (time.time() - t0)/100

    print(f"bs={bs} seq={seq_len} segs={num_segments}  "
          f"old={t_old * 1e3:.2f}ms  new={t_new * 1e3:.2f}ms  speed-up x{t_old/t_new:.2f}")

if __name__ == "__main__":
    test_equivalence(bs = 128, seq_len = 8192, num_segments = 32) # all good!
    benchmark(bs = 128, seq_len = 8192, num_segments = 32) # old ~ 126.76ms  new ~ 6.85ms