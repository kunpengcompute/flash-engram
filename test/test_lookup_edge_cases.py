# test_lookup_edge_cases.py
import numpy as np
import pytest

try:
    from flash_engram import core_engram
except ImportError:
    core_engram = None


def _make_valid_args(dtype=np.float32):
    vocab, dim = 100, 32
    bs, slen, heads = 2, 5, 4
    rng = np.random.default_rng(1)
    weight = rng.normal(size=(vocab, dim)).astype(dtype)
    indices = rng.integers(0, vocab, size=(bs, slen, heads)).astype(np.int64)
    offsets = np.zeros(heads, dtype=np.int64)
    output = np.zeros((bs * slen * heads, dim), dtype=dtype)
    return weight, indices, offsets, output, vocab, dim, bs, slen, heads


def _call_lookup(func, weight, indices, offsets, output, vocab, dim, bs, slen, heads, offsets_need=0):
    func(vocab, dim,
         weight.ctypes.data, indices.ctypes.data,
         bs, slen, heads,
         output.ctypes.data, offsets.ctypes.data,
         offsets_need)


# -------------------- F32 边界测试 --------------------
@pytest.mark.skipif(core_engram is None, reason="core_engram 未编译")
class TestLookupF32EdgeCases:

    # 零 embedding_dim 是安全的：memcpy 0 字节
    def test_zero_embedding_dim(self):
        weight = np.zeros((100, 0), dtype=np.float32)
        indices = np.random.randint(0, 100, size=(2, 5, 4)).astype(np.int64)
        offsets = np.zeros(4, dtype=np.int64)
        output = np.zeros((40, 0), dtype=np.float32)
        _call_lookup(core_engram._lookup_cpp_f32, weight, indices, offsets, output, 100, 0, 2, 5, 4)
        assert output.shape == (40, 0)


# -------------------- F16 边界测试 --------------------
@pytest.mark.skipif(core_engram is None, reason="core_engram 未编译")
class TestLookupF16EdgeCases:

    def test_zero_embedding_dim(self):
        weight = np.zeros((100, 0), dtype=np.float16)
        indices = np.random.randint(0, 100, size=(2, 5, 4)).astype(np.int64)
        offsets = np.zeros(4, dtype=np.int64)
        output = np.zeros((40, 0), dtype=np.float16)
        _call_lookup(core_engram._lookup_cpp_f16, weight, indices, offsets, output, 100, 0, 2, 5, 4)
        assert output.shape == (40, 0)