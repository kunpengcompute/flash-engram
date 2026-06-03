# test_ngram_hashes_correctness.py
import numpy as np
import pytest

try:
    from flash_engram import core_engram
except ImportError:
    core_engram = None

# ---------- 纯 Python 参考实现 ----------
def _python_ngram_hashes(input_ids, multipliers, head_vocab_sizes, n_head_per_ngram, pad_id):
    batch_size, seq_len = input_ids.shape
    out = np.zeros((batch_size, seq_len, 2 * n_head_per_ngram), dtype=np.int64)
    for b in range(batch_size):
        tokens = input_ids[b]
        for j in range(seq_len):
            x0 = tokens[j] * multipliers[0]
            x1 = (pad_id if j == 0 else tokens[j - 1]) * multipliers[1]
            x2 = (pad_id if j <= 1 else tokens[j - 2]) * multipliers[2]
            mix2 = x0 ^ x1
            mix3 = mix2 ^ x2
            for h in range(n_head_per_ngram):
                out[b, j, h] = mix2 % head_vocab_sizes[h]
                out[b, j, n_head_per_ngram + h] = mix3 % head_vocab_sizes[n_head_per_ngram + h]
    return out


def _generate_random_params(seed=42):
    rng = np.random.default_rng(seed)
    batch_size = int(rng.integers(1, 4))
    seq_len = int(rng.integers(1, 20))
    n_head = 8
    max_ngram = 3
    pad_id = 0
    input_ids = rng.integers(0, 100, size=(batch_size, seq_len)).astype(np.int64)
    multipliers = rng.integers(1, 1000, size=3).astype(np.int64)
    head_vocab_sizes = rng.integers(2, 100, size=2 * n_head).astype(np.int64)
    # 创建 output 数组
    output = np.zeros((batch_size, seq_len, max_ngram * n_head), dtype=np.int64)
    return input_ids, output, batch_size, seq_len, multipliers, head_vocab_sizes, max_ngram, n_head, pad_id

# ---------- 测试类 ----------
@pytest.mark.skipif(core_engram is None, reason="core_engram 模块未编译，跳过测试")
class TestGetNgramHashesCpp:
    """测试 _get_ngram_hashes_cpp 的正确性"""

    def test_basic(self):
        input_ids, output, bs, slen, mult, hv, max_ng, nh, pid = _generate_random_params(seed=1)
        core_engram._get_ngram_hashes_cpp(input_ids, output, bs, slen, mult, hv, max_ng, nh, pid)
        expected = _python_ngram_hashes(input_ids, mult, hv, nh, pid)
        np.testing.assert_array_equal(output[..., :2 * nh], expected)

    def test_seq_len_one(self):
        input_ids = np.array([[5]], dtype=np.int64)
        bs, slen = 1, 1
        nh = 3
        max_ng = 3
        mult = np.array([11, 13, 17], dtype=np.int64)
        hv = np.array([2, 3, 4, 5, 6, 7], dtype=np.int64)
        pid = -1
        output = np.zeros((bs, slen, max_ng * nh), dtype=np.int64)
        core_engram._get_ngram_hashes_cpp(input_ids, output, bs, slen, mult, hv, max_ng, nh, pid)
        expected = _python_ngram_hashes(input_ids, mult, hv, nh, pid)
        np.testing.assert_array_equal(output[..., :2 * nh], expected)

    def test_vocab_size_one(self):
        input_ids = np.array([[7, 8, 9]], dtype=np.int64)
        bs, slen = 1, 3
        nh = 2
        max_ng = 3
        mult = np.array([1, 2, 3], dtype=np.int64)
        hv = np.array([1, 5, 3, 10], dtype=np.int64)
        pid = 0
        output = np.zeros((bs, slen, max_ng * nh), dtype=np.int64)
        core_engram._get_ngram_hashes_cpp(input_ids, output, bs, slen, mult, hv, max_ng, nh, pid)
        expected = _python_ngram_hashes(input_ids, mult, hv, nh, pid)
        np.testing.assert_array_equal(output[..., :2 * nh], expected)

    def test_large_odd_seq_len(self):
        rng = np.random.default_rng(2)
        bs, slen = 2, 17
        nh = 4
        max_ng = 3
        input_ids = rng.integers(0, 200, size=(bs, slen)).astype(np.int64)
        mult = rng.integers(1, 100, size=3).astype(np.int64)
        hv = rng.integers(2, 50, size=2 * nh).astype(np.int64)
        pid = -2
        output = np.zeros((bs, slen, max_ng * nh), dtype=np.int64)
        core_engram._get_ngram_hashes_cpp(input_ids, output, bs, slen, mult, hv, max_ng, nh, pid)
        expected = _python_ngram_hashes(input_ids, mult, hv, nh, pid)
        np.testing.assert_array_equal(output[..., :2 * nh], expected)