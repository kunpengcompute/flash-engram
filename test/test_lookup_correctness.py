# test_lookup_correctness.py
import numpy as np
import pytest

try:
    from flash_engram import core_engram
except ImportError:
    core_engram = None

# ---------- Python 参考实现 ----------
def _python_lookup(weight, indices, offsets, offsets_need):
    if offsets_need:
        indices = indices + offsets
    return weight[indices].reshape(-1, weight.shape[-1])


def _generate_lookup_data(dtype=np.float32, batch=3, seq=7, heads=5, vocab=200, dim=32, seed=42):
    rng = np.random.default_rng(seed)
    weight = rng.normal(size=(vocab, dim)).astype(dtype)
    indices = rng.integers(0, vocab, size=(batch, seq, heads)).astype(np.int64)
    offsets = rng.integers(0, vocab // 10, size=heads).astype(np.int64)
    return weight, indices, offsets


def _generate_safe_offset_data(dtype=np.float32, batch=2, seq=4, heads=3, vocab=100, dim=32, seed=99):
    """生成数据，确保 indices + offsets 不越界"""
    rng = np.random.default_rng(seed)
    weight = rng.normal(size=(vocab, dim)).astype(dtype)
    offsets = rng.integers(0, vocab // 10, size=heads).astype(np.int64)
    max_offset = offsets.max()
    indices = rng.integers(0, vocab - max_offset - 1, size=(batch, seq, heads)).astype(np.int64)
    return weight, indices, offsets


# ---------- 测试类 ----------
@pytest.mark.skipif(core_engram is None, reason="core_engram 未编译")
class TestLookupF32:
    def test_basic_no_offset(self):
        weight, indices, offsets = _generate_lookup_data(np.float32, seed=1)
        bs, slen, heads = indices.shape
        vocab, dim = weight.shape
        indices_in = indices.copy()
        output = np.zeros((bs * slen * heads, dim), dtype=np.float32)
        core_engram._lookup_cpp_f32(vocab, dim, weight.ctypes.data, indices_in.ctypes.data,
                                    bs, slen, heads, output.ctypes.data, offsets.ctypes.data, 0)
        expected = _python_lookup(weight, indices, offsets, False)
        np.testing.assert_allclose(output, expected, rtol=1e-7)

    def test_zero_offsets_with_need(self):
        weight, indices, _ = _generate_lookup_data(np.float32, seed=4)
        bs, slen, heads = indices.shape
        vocab, dim = weight.shape
        offsets = np.zeros(heads, dtype=np.int64)

        output0 = np.zeros((bs * slen * heads, dim), dtype=np.float32)
        indices0 = indices.copy()
        core_engram._lookup_cpp_f32(vocab, dim, weight.ctypes.data, indices0.ctypes.data,
                                    bs, slen, heads, output0.ctypes.data, offsets.ctypes.data, 0)

        output1 = np.zeros_like(output0)
        indices1 = indices.copy()
        core_engram._lookup_cpp_f32(vocab, dim, weight.ctypes.data, indices1.ctypes.data,
                                    bs, slen, heads, output1.ctypes.data, offsets.ctypes.data, 1)

        np.testing.assert_allclose(output0, output1, rtol=1e-7)
        np.testing.assert_array_equal(indices0, indices1)

    def test_lookup_with_offsets_manual(self):
        """手动计算偏移后的索引，然后以 offsets_need=0 且 offsets 全零调用，验证查表"""
        weight, indices, offsets = _generate_safe_offset_data(np.float32, seed=42)
        bs, slen, heads = indices.shape
        vocab, dim = weight.shape

        # 手动计算偏移后的索引
        shifted_indices = indices + offsets   # shape (bs, slen, heads)
        expected_output = weight[shifted_indices].reshape(-1, dim)

        # 调用 C++，传入已偏移的索引，且 offsets 全零、不启用内部偏移
        zero_offsets = np.zeros(heads, dtype=np.int64)
        output = np.zeros((bs * slen * heads, dim), dtype=np.float32)
        core_engram._lookup_cpp_f32(vocab, dim, weight.ctypes.data,
                                    shifted_indices.ctypes.data,
                                    bs, slen, heads,
                                    output.ctypes.data, zero_offsets.ctypes.data, 0)

        np.testing.assert_allclose(output, expected_output, rtol=1e-7)


@pytest.mark.skipif(core_engram is None, reason="core_engram 未编译")
class TestLookupF16:
    def test_basic_no_offset(self):
        weight, indices, offsets = _generate_lookup_data(np.float16, seed=5)
        bs, slen, heads = indices.shape
        vocab, dim = weight.shape
        indices_in = indices.copy()
        output = np.zeros((bs * slen * heads, dim), dtype=np.float16)
        core_engram._lookup_cpp_f16(vocab, dim, weight.ctypes.data, indices_in.ctypes.data,
                                    bs, slen, heads, output.ctypes.data, offsets.ctypes.data, 0)
        expected = _python_lookup(weight, indices, offsets, False)
        np.testing.assert_allclose(output, expected, rtol=1e-3, atol=1e-3)

    def test_lookup_with_offsets_manual(self):
        """手动计算偏移后的索引，然后以 offsets_need=0 且 offsets 全零调用，验证查表"""
        weight, indices, offsets = _generate_safe_offset_data(np.float16, seed=99)
        bs, slen, heads = indices.shape
        vocab, dim = weight.shape

        shifted_indices = indices + offsets
        expected_output = weight[shifted_indices].reshape(-1, dim)

        zero_offsets = np.zeros(heads, dtype=np.int64)
        output = np.zeros((bs * slen * heads, dim), dtype=np.float16)
        core_engram._lookup_cpp_f16(vocab, dim, weight.ctypes.data,
                                    shifted_indices.ctypes.data,
                                    bs, slen, heads,
                                    output.ctypes.data, zero_offsets.ctypes.data, 0)

        np.testing.assert_allclose(output, expected_output, rtol=1e-3, atol=1e-3)