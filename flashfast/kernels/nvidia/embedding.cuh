#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>

#include "utils.cuh"

/*
    Embedding kernel.
*/
template <typename T_TOKEN = int64_t, typename T_DATA = half>
__global__ __forceinline__ void embeddingKernel(T_TOKEN* tokens, T_DATA* weight, T_DATA* embedding, 
                                                int batch_size, int seq_length, int embedding_dim, int tokens_stride_bs, int tokens_stride_seq) {
    int tid = threadIdx.x;
    int batch_id = blockIdx.y;
    int seq_id = blockIdx.x;

    embedding_dim = embedding_dim >> 3;

    if (tid < embedding_dim) {
        int index = (batch_id * seq_length + seq_id) * embedding_dim + tid;
        T_TOKEN token_id = tokens[batch_id * tokens_stride_bs + seq_id * tokens_stride_seq];

        reinterpret_cast<float4*>(embedding)[index] = reinterpret_cast<float4*>(weight)[token_id * embedding_dim + tid];
    }
}