#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>

#include "utils.cuh"

#define SHARED_MEM_MAX_ROWS 64

/*
    GEMV kernel using FP16 to accumulate.
    Modified from fast_gemv_acc_fp16_residual_kernel
    Support MP = 1~8
*/
template <typename T = half, typename T2 = half2>
__global__ __forceinline__ void fast_gemv_acc_fp16_residual_kernel(T* mat, T* vec, T* r, T* res, unsigned int m, 
                                              unsigned int k, unsigned int n, unsigned int num_per_thread) {
  
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = (blockIdx.x * blockDim.y + threadIdx.y) << 1;
  unsigned int bid = blockIdx.z * gridDim.y + blockIdx.y;
  if (bid >= m) return;
  T2 vec_val[4];
  T2 mat_val[8];

  // T2 temp_sum = {convert_fp32_to_fp16<T>(0.0f), convert_fp32_to_fp16<T>(0.0f)};
  T2 sum[2];
  sum[0] = {convert_fp32_to_fp16<T>(0.0f), convert_fp32_to_fp16<T>(0.0f)};
  sum[1] = {convert_fp32_to_fp16<T>(0.0f), convert_fp32_to_fp16<T>(0.0f)};
  T2 gsum;

#pragma unroll
  for (int iter = 0; iter < DIV_UP(num_per_thread, 8); iter++) {
    unsigned int j = (tid + iter * blockDim.x) << 3;
    if (j >= k) {break;}
    *(float4*)(&vec_val[0]) = *(float4*)(&vec[bid * k + j]);
    *(float4*)(&mat_val[0]) = *(float4*)(&mat[row * k + j]);
    *(float4*)(&mat_val[4]) = *(float4*)(&mat[(row + 1) * k + j]);

    sum[0] = __hadd2(sum[0], __hmul2(vec_val[0], mat_val[0]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[1], mat_val[1]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[2], mat_val[2]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[3], mat_val[3]));

    sum[1] = __hadd2(sum[1], __hmul2(vec_val[0], mat_val[4]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[1], mat_val[5]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[2], mat_val[6]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[3], mat_val[7]));
  }

  gsum.x = __hadd(sum[0].x, sum[0].y);
  gsum.y = __hadd(sum[1].x, sum[1].y);

  static __shared__ T warpLevelSums[WARP_SIZE];

  gsum.x = blockReduceSum(gsum.x, warpLevelSums);
  gsum.y = blockReduceSum(gsum.y, warpLevelSums);

  T2 to_add = *(T2*)(&r[row]);

  if (tid == 0) {
    *(T2*)(&res[bid * n + row]) = __hadd2(gsum, to_add);
  }
}

template <typename T = half, typename T2 = half2>
__global__ __forceinline__ void fast_gemv_acc_fp16_bias_relu_kernel(
                                T* x, T* w1, T* b1, 
                                int bs, int dim, int h_dim, 
                                T* res) {
  
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.x * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  T2 x_val[4];
  T2 w1_val[4];

  // T2 temp_sum = {convert_fp32_to_fp16<T>(0.0f), convert_fp32_to_fp16<T>(0.0f)};
  T2 temp_sum_1 = {convert_fp32_to_fp16<T>(0.0f), convert_fp32_to_fp16<T>(0.0f)};

#pragma unroll
  for (int iter = 0; iter < DIV_UP((dim >> 3), blockDim.x); iter++) {
    unsigned int j = (start_idx + iter * blockDim.x) << 3;
    if (j >= dim) {break;}
      
      // float4 vec_val = vec4[j];
      // float4 mat_val = mat4[row * (n >> 3) + j];
      *(float4*)(&x_val[0]) = *(float4*)(&x[j]);
      *(float4*)(&w1_val[0]) = *(float4*)(&w1[row * dim + j]);

      temp_sum_1 = __hadd2(temp_sum_1, __hmul2(x_val[0],  w1_val[0]));  
      temp_sum_1 = __hadd2(temp_sum_1, __hmul2(x_val[1],  w1_val[1]));
      temp_sum_1 = __hadd2(temp_sum_1, __hmul2(x_val[2],  w1_val[2]));
      temp_sum_1 = __hadd2(temp_sum_1, __hmul2(x_val[3],  w1_val[3])); 
  }

  float sum_1 = convert_fp16_to_fp32(__hadd(temp_sum_1.x, temp_sum_1.y));

  static __shared__ float warpLevelSums[WARP_SIZE];

  sum_1 = blockReduceSum(sum_1, warpLevelSums);

  sum_1 += convert_fp16_to_fp32(b1[row]);

  if (tid == 0) {
    if (sum_1 < 1e-5f) sum_1 = 1e-5f;
    res[row] = convert_fp32_to_fp16<T>(sum_1);
  }
}


/*
    QKV projects + rope kernel for KVcache with any shape. Only need the strides of bs and seqlen dim. 
    Modified from fast_gemv_acc_fp16_with_llama2_rope_kernel
    Support only bs = 1, seqlen = 1.
    Support MP = 1~8
*/
template <ROPE_TYPE ROPE, FREQ_ALIGNED ALIGN = FREQ_ALIGNED::YES, typename T = half, typename T2 = half2>
__global__ __forceinline__ void fast_gemv_acc_fp16_qkv_rope_kernel(
                          T* WQ, T* WK, T* WV, T* BQ, T* BK, T* BV, 
                          T* X, float* f, int* f_offsets, T* Q, T* K, T *V, 
                          int m, int k, int n, int n_kv, int kv_stride_bs, int kv_stride_seq,
                          int len, int hn, int hn_kv, int hs, unsigned int num_per_thread) {
  
  // each thread load num_per_thread elements from global
  size_t tid = threadIdx.x;
  size_t row = (blockIdx.x * blockDim.y + threadIdx.y) << 1;
  size_t bid = blockIdx.z * gridDim.y + blockIdx.y;
  if (bid >= m) return;
  T* weight_ptr = (row < n) ? &WQ[0] : 
                    (row < n + n_kv) ? &WK[0] : &WV[0];
  T* bias_ptr = (row < n) ? ((BQ == nullptr) ? nullptr : &BQ[0]) : 
                    (row < n + n_kv) ? ((BK == nullptr) ? nullptr : &BK[0]) : 
                    ((BV == nullptr) ? nullptr : &BV[0]);
  
  float* freq;
  if (ALIGN) {
    if (ROPE == ROPE_TYPE::FULL_ROPE)
      freq = &f[len * hs];
    else if (ROPE == ROPE_TYPE::HALF_ROPE)
      freq = &f[len * hs / 2];
  }
  else {
    if (ROPE == ROPE_TYPE::FULL_ROPE)
      freq = &f[f_offsets[bid] * hs];
    else if (ROPE == ROPE_TYPE::HALF_ROPE)
      freq = &f[f_offsets[bid] * hs / 2];
  }

  int qkv_row = (row < n) ? row : row % n_kv;

  size_t kv_offset = (len + bid % m) * kv_stride_seq;
  size_t kv_index = kv_offset + qkv_row;
  T2 vec_val[4];
  T2 mat_val[8];

  // T2 temp_sum = {convert_fp32_to_fp16<T>(0.0f), convert_fp32_to_fp16<T>(0.0f)};
  T2 sum[2];
  sum[0] = {convert_fp32_to_fp16<T>(0.0f), convert_fp32_to_fp16<T>(0.0f)};
  sum[1] = {convert_fp32_to_fp16<T>(0.0f), convert_fp32_to_fp16<T>(0.0f)};
  float2 gsum;

#pragma unroll
  for (int iter = 0; iter < DIV_UP(num_per_thread, 8); iter++) {
    unsigned int j = (tid + iter * blockDim.x) << 3;
    if (j >= k) {break;}
    *(float4*)(&vec_val[0]) = *(float4*)(&X[bid * k + j]);
    *(float4*)(&mat_val[0]) = *(float4*)(weight_ptr + (qkv_row) * k + j);
    *(float4*)(&mat_val[4]) = *(float4*)(weight_ptr + (qkv_row + 1) * k + j);

    sum[0] = __hadd2(sum[0], __hmul2(vec_val[0], mat_val[0]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[1], mat_val[1]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[2], mat_val[2]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[3], mat_val[3]));

    sum[1] = __hadd2(sum[1], __hmul2(vec_val[0], mat_val[4]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[1], mat_val[5]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[2], mat_val[6]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[3], mat_val[7]));
  }

  gsum.x = convert_fp16_to_fp32(sum[0].x) + convert_fp16_to_fp32(sum[0].y);
  gsum.y = convert_fp16_to_fp32(sum[1].x) + convert_fp16_to_fp32(sum[1].y);

  static __shared__ float warpLevelSums[WARP_SIZE];

  gsum.x = blockReduceSum(gsum.x, warpLevelSums);
  gsum.y = blockReduceSum(gsum.y, warpLevelSums);

  if (tid > 0) {return;}
  // add bias
  if (bias_ptr != nullptr) {
    T2 bias = *(T2*)(bias_ptr + qkv_row);
    gsum.x += convert_fp16_to_fp32(bias.x);
    gsum.y += convert_fp16_to_fp32(bias.y);
  }

  if (ROPE == ROPE_TYPE::FULL_ROPE) {
    // Full RoPE
    if (row < n) {
      int idx = row % hs;
      float2 to_rotate = *(float2*)(&freq[idx]);
      float2 gres;
      gres.x = to_rotate.x * gsum.x - to_rotate.y * gsum.y;
      gres.y = to_rotate.x * gsum.y + to_rotate.y * gsum.x;
      *(T2*)(&Q[bid * n + row]) = convert_2fp32_to_2fp16_rn<T2>(gres);
    }
    else if (row < n + n_kv){
      int idx = row % hs;
      float2 to_rotate = *(float2*)(&freq[idx]);
      float2 gres;
      gres.x = to_rotate.x * gsum.x - to_rotate.y * gsum.y;
      gres.y = to_rotate.x * gsum.y + to_rotate.y * gsum.x;
      *(T2*)(&K[kv_index]) = convert_2fp32_to_2fp16_rn<T2>(gres);
    }
    else if (row < n + n_kv * 2){
      *(T2*)(&V[kv_index]) = convert_2fp32_to_2fp16_rn<T2>(gsum);
    }
  }
  else if (ROPE == ROPE_TYPE::HALF_ROPE) {
    // Half RoPE
    if (row < n) {
      int idx = row % hs;
      if (idx < hs / 2){
          float2 update_sum;
          float2 to_rotate = *(float2*)(&freq[idx]);
          update_sum.x = to_rotate.x * gsum.x - to_rotate.y * gsum.y;
          update_sum.y = to_rotate.x * gsum.y + to_rotate.y * gsum.x;
          *(T2*)(&Q[bid * n + row]) = convert_2fp32_to_2fp16_rn<T2>(update_sum);
      }
      else{
        *(T2*)(&Q[bid * n + row]) = convert_2fp32_to_2fp16_rn<T2>(gsum);
      }
    }
    else if (row < n + n_kv){
      int idx = row % hs;
      if (idx < hs / 2){
          float2 update_sum;
          float2 to_rotate = *(float2*)(&freq[idx]);
          update_sum.x = to_rotate.x * gsum.x - to_rotate.y * gsum.y;
          update_sum.y = to_rotate.x * gsum.y + to_rotate.y * gsum.x;
          *(T2*)(&K[kv_index]) = convert_2fp32_to_2fp16_rn<T2>(update_sum);
      }
      else{
        *(T2*)(&K[kv_index]) = convert_2fp32_to_2fp16_rn<T2>(gsum);
      }
    }
    else if (row < n + n_kv * 2){
      *(T2*)(&V[kv_index]) = convert_2fp32_to_2fp16_rn<T2>(gsum);
    }
  }
  else {
    // No RoPE
    if (row < n) {
      *(T2*)(&Q[bid * n + row]) = convert_2fp32_to_2fp16_rn<T2>(gsum);
    }
    else if (row < n + n_kv){
      *(T2*)(&K[kv_index]) = convert_2fp32_to_2fp16_rn<T2>(gsum);
    }
    else if (row < n + n_kv * 2){
      *(T2*)(&V[kv_index]) = convert_2fp32_to_2fp16_rn<T2>(gsum);
    }
  }
}
