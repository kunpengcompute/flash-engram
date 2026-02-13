'''
    Copyright (c) 2026 Huawei Technologies Co., Ltd.
    flash-engram is licensed under Mulan PSL v2.
    You can use this software according to the terms and conditions of the Mulan PSL v2.
    You may obtain a copy of Mulan PSL v2 at:
            http://license.coscl.org.cn/MulanPSL2
    THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
    EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
    MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
    See the Mulan PSL v2 for more details.
'''

import numpy as np
import torch
from typing import Union, List

try:
    from .core_engram import _get_ngram_hashes_cpp, _lookup_cpp_f32, _lookup_cpp_f16
except ImportError as e:
    raise ImportError(
        "C++ extension not found."
    ) from e

def _get_ngram_hashes(
        input_ids: np.ndarray,
        batch_size: int,
        seq_len: int,
        multipliers: np.ndarray,
        head_vocab_sizes: np.ndarray,
        max_ngram_size: int,
        n_head_per_ngram: int,
        pad_id: int
    ) -> np.ndarray:
    # input_ids: [B, T]
    # output_ids: [B, T, max_ngram_size * n_head_per_ngram]
    output_ids = np.empty((batch_size, seq_len, max_ngram_size * n_head_per_ngram), dtype=input_ids.dtype)
    _get_ngram_hashes_cpp(input_ids, output_ids, batch_size, seq_len, multipliers, head_vocab_sizes, max_ngram_size, n_head_per_ngram, pad_id)
    return output_ids

def _lookup_f32(
        num_embeddings: int,
        embedding_dim: int,
        weight: torch.Tensor,
        indices: torch.Tensor,
        batch_size: int,
        seq_len: int,
        heads: int,
        offsets: torch.Tensor,
        offsets_need: int = 0
    ) -> torch.Tensor:
    output = torch.empty(batch_size, seq_len, heads, embedding_dim, dtype=torch.float32)
    _lookup_cpp_f32(num_embeddings, embedding_dim, weight.data_ptr(), indices.data_ptr(), batch_size, seq_len, heads,
        output.data_ptr(), offsets.data_ptr(), offsets_need)
    return output

def _lookup_f16(
        num_embeddings: int,
        embedding_dim: int,
        weight: torch.Tensor,
        indices: torch.Tensor,
        batch_size: int,
        seq_len: int,
        heads: int,
        offsets: torch.Tensor,
        offsets_need: int = 0
    ) -> torch.Tensor:
    output = torch.empty(batch_size, seq_len, heads, embedding_dim, dtype=torch.float16)
    _lookup_cpp_f16(num_embeddings, embedding_dim, weight.data_ptr(), indices.data_ptr(), batch_size, seq_len, heads,
        output.data_ptr(), offsets.data_ptr(), offsets_need)
    return output
