/*
    Copyright (c) 2026 Huawei Technologies Co., Ltd.
    flash-engram is licensed under Mulan PSL v2.
    You can use this software according to the terms and conditions of the Mulan PSL v2.
    You may obtain a copy of Mulan PSL v2 at:
            http://license.coscl.org.cn/MulanPSL2
    THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
    EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
    MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
    See the Mulan PSL v2 for more details.
*/

#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <vector>
#include <iomanip>
#include <chrono>
#include <random>
#include <iostream>
#include <numeric>

#include <arm_sve.h>
#include <stdint.h>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void memcpy_sve(void* dst, const void* src, size_t n) {
    if (n == 0) return;

    unsigned char* d = (unsigned char*)dst;
    const unsigned char* s = (const unsigned char*)src;

    const size_t vl = svcntb(); // vector length in bytes (32)
    
    const size_t step = vl * 4;
    const size_t np= (n & ~(step - 1));
    for (size_t i = 0; i < np; i += step) {
        svuint8_t v0 = svld1_u8(svptrue_b8(), s);
        svuint8_t v1 = svld1_u8(svptrue_b8(), s + vl);
        svuint8_t v2 = svld1_u8(svptrue_b8(), s + 2 * vl);
        svuint8_t v3 = svld1_u8(svptrue_b8(), s + 3 * vl);

        svst1_u8(svptrue_b8(), d, v0);
        svst1_u8(svptrue_b8(), d + vl, v1);
        svst1_u8(svptrue_b8(), d + 2 * vl, v2);
        svst1_u8(svptrue_b8(), d + 3 * vl, v3);
        s += 4 * vl;
        d += 4 * vl;
    }
    const size_t np2 = (n & ~(vl - 1));
    for (size_t i = np; i < np2; i += vl) {
        svuint8_t v0 = svld1_u8(svptrue_b8(), s);
        svst1_u8(svptrue_b8(), d, v0);
        s += vl;
        d += vl;
    }
    for (size_t i = np2; i < n; ++i) {
        *d++ = *s++;
    }
}


void _get_ngram_hashes_cpp(py::array_t<int64_t> input_ids, py::array_t<int64_t> output_ids, int batch_size, int seq_len,
                            py::array_t<int64_t> multipliers, py::array_t<int64_t> head_vocab_sizes,
                            int max_ngram_size, int n_head_per_ngram, int pad_id) {
    /*
    input_ids: [batch_size, seq_len]
    output_ids: [batch_size, seq_len, ngram * head]
    */
    auto input_buf = input_ids.request();
    int64_t* input_ptr = static_cast<int64_t*>(input_buf.ptr);
    auto output_buf = output_ids.request();
    int64_t* output_ptr = static_cast<int64_t*>(output_buf.ptr);
    auto multipliers_buf = multipliers.request();
    int64_t* multipliers_ptr = static_cast<int64_t*>(multipliers_buf.ptr);
    auto head_vocab_sizes_buf = head_vocab_sizes.request();
    int64_t* head_vocab_sizes_ptr = static_cast<int64_t*>(head_vocab_sizes_buf.ptr);

    int desired = seq_len / 1000;
    int threads = std::min(desired < 1 ? 1 : desired, omp_get_max_threads());
    for (int i = 0; i < batch_size; ++i) {
        const int64_t* shift0 = input_ptr + i * seq_len;
        int64_t* output = output_ptr + i * seq_len * max_ngram_size * n_head_per_ngram;
        if (seq_len == 1) {
            int64_t x0 = shift0[0] * multipliers_ptr[0];
            int64_t x1 = pad_id * multipliers_ptr[1];
            int64_t x2 = pad_id * multipliers_ptr[2];
            int64_t mix2 =  x0 ^ x1;
            int64_t mix3 =  mix2 ^ x2;
            int64_t* output_head = output;
            for (size_t h = 0; h < n_head_per_ngram; ++h) {
                output_head[h] = mix2 % head_vocab_sizes_ptr[h];
                output_head[n_head_per_ngram + h] = mix3 % head_vocab_sizes_ptr[n_head_per_ngram + h];
            }
        }
        else {
            std::vector<int64_t> shift1(seq_len);
            std::vector<int64_t> shift2(seq_len);
            shift1[0] = pad_id;
            std::copy(shift0, shift0 + seq_len - 1, shift1.begin() + 1);
            shift2[0] = pad_id;
            shift2[1] = pad_id;
            std::copy(shift0, shift0 + seq_len - 2, shift2.begin() + 2);

            #pragma omp parallel for num_threads(threads) schedule(static)
            for (size_t j = 0; j < seq_len; ++j) {
                int64_t x0 = shift0[j] * multipliers_ptr[0];
                int64_t x1 = shift1[j] * multipliers_ptr[1];
                int64_t x2 = shift2[j] * multipliers_ptr[2];
                int64_t mix2 =  x0 ^ x1;
                int64_t mix3 =  mix2 ^ x2;
                int64_t* output_head = output + j * max_ngram_size * n_head_per_ngram;
                for (size_t h = 0; h < n_head_per_ngram; ++h) {
                    output_head[h] = mix2 % head_vocab_sizes_ptr[h];
                    output_head[n_head_per_ngram + h] = mix3 % head_vocab_sizes_ptr[n_head_per_ngram + h];
                }
            }
        }
    }
}

void make_offsets(int64_t* input, int64_t* offsets, size_t batch_size, size_t seq_len, size_t heads) {
    int desired = batch_size * seq_len / 2500;
    int threads = std::min(desired < 1 ? 1 : desired, omp_get_max_threads());
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t i = 0; i < batch_size * seq_len; ++i) {
        size_t base = i * heads;
        for (int h = 0; h < heads; ++h) {
            input[base + h] += offsets[h];
        }
    }
}

void _lookup_cpp_f32(size_t num_embeddings, size_t embedding_dim, int64_t weight,
        int64_t indices, size_t batch_size, size_t seq_len, size_t heads,
        int64_t output, int64_t offsets, int offsets_need = 0) {
    float* weight_ptr = reinterpret_cast<float*>(weight);
    float* output_ptr = reinterpret_cast<float*>(output);
    int64_t* indices_ptr = reinterpret_cast<int64_t*>(indices);
    int64_t* offsets_ptr = reinterpret_cast<int64_t*>(offsets);

    size_t vector_size_bytes = embedding_dim * sizeof(float);
    if (offsets_need == 1) {
        make_offsets(indices_ptr, offsets_ptr, batch_size, seq_len, heads);
    }
    
    int total_num = batch_size * seq_len * heads;
    int desired = total_num / 160;
    int threads = std::min(desired < 1 ? 1 : desired, omp_get_max_threads());

    size_t prefetch_distance = 16;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t i = 0; i < total_num; ++i) {
        int64_t idx = indices_ptr[i];
        if (i + prefetch_distance < total_num) {
            int64_t future_idx = indices_ptr[i + prefetch_distance];
            const float* future_src = weight_ptr + future_idx * embedding_dim;
            __builtin_prefetch(future_src, 0, 0);
        }
        const float* src = weight_ptr + idx * embedding_dim;
        float* dst = output_ptr + i * embedding_dim;
        memcpy_sve(dst, src, vector_size_bytes);
    }
}

void _lookup_cpp_f16(size_t num_embeddings, size_t embedding_dim, int64_t weight,
        int64_t indices, size_t batch_size, size_t seq_len, size_t heads,
        int64_t output, int64_t offsets, int offsets_need = 0) {
    __fp16* weight_ptr = reinterpret_cast<__fp16*>(weight);
    __fp16* output_ptr = reinterpret_cast<__fp16*>(output);
    int64_t* indices_ptr = reinterpret_cast<int64_t*>(indices);
    int64_t* offsets_ptr = reinterpret_cast<int64_t*>(offsets);

    size_t vector_size_bytes = embedding_dim * sizeof(__fp16);
    if (offsets_need == 1) {
        make_offsets(indices_ptr, offsets_ptr, batch_size, seq_len, heads);
    }
    
    int total_num = batch_size * seq_len * heads;
    int desired = total_num / 160;
    int threads = std::min(desired < 1 ? 1 : desired, omp_get_max_threads());

    size_t prefetch_distance = 16;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t i = 0; i < total_num; ++i) {
        int64_t idx = indices_ptr[i];
        if (i + prefetch_distance < total_num) {
            int64_t future_idx = indices_ptr[i + prefetch_distance];
            const __fp16* future_src = weight_ptr + future_idx * embedding_dim;
            __builtin_prefetch(future_src, 0, 0);
        }
        const __fp16* src = weight_ptr + idx * embedding_dim;
        __fp16* dst = output_ptr + i * embedding_dim;
        memcpy_sve(dst, src, vector_size_bytes);
    }
}


PYBIND11_MODULE(core_engram, m) {
    m.doc() = "pybind11 core_engram plugin";
    m.def("_get_ngram_hashes_cpp", &_get_ngram_hashes_cpp, "A function for hash");
    m.def("_lookup_cpp_f32", &_lookup_cpp_f32, "A function for embedding f32 lookup");
    m.def("_lookup_cpp_f16", &_lookup_cpp_f16, "A function for embedding f16 lookup");
    m.attr("__version__") = "0.0.1";
}