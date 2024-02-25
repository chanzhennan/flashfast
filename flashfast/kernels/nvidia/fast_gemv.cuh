#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>

#include "utils.cuh"

#define SHARED_MEM_MAX_ROWS 64

/*
    GEMV kernel using FP32 to accumulate. This implementation comes directly from
    https://github.com/wangsiping97/FastGEMV.
*/
template <typename T = half>
__global__ __forceinline__ void fast_gemv_acc_fp32_kernel(
                          T* mat, T* vec, T* res, unsigned int n,
                          unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  float4* mat4 = reinterpret_cast<float4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

#pragma unroll
  for (int iter = 0; iter < DIV_UP(num_per_thread, 8); iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < n >> 3) {
      float4 vec_val = vec4[j];
      float4 mat_val = mat4[row * (n >> 3) + j];
      half2* vec_h1 = (half2*)&vec_val.x;
      half2* vec_h2 = (half2*)&vec_val.y;
      half2* vec_h3 = (half2*)&vec_val.z;
      half2* vec_h4 = (half2*)&vec_val.w;
      half2* mat_h1 = (half2*)&mat_val.x;
      half2* mat_h2 = (half2*)&mat_val.y;
      half2* mat_h3 = (half2*)&mat_val.z;
      half2* mat_h4 = (half2*)&mat_val.w;
      sum += convert_fp16_to_fp32(vec_h1->x) * convert_fp16_to_fp32(mat_h1->x);
      sum += convert_fp16_to_fp32(vec_h1->y) * convert_fp16_to_fp32(mat_h1->y);
      sum += convert_fp16_to_fp32(vec_h2->x) * convert_fp16_to_fp32(mat_h2->x);
      sum += convert_fp16_to_fp32(vec_h2->y) * convert_fp16_to_fp32(mat_h2->y);
      sum += convert_fp16_to_fp32(vec_h3->x) * convert_fp16_to_fp32(mat_h3->x);
      sum += convert_fp16_to_fp32(vec_h3->y) * convert_fp16_to_fp32(mat_h3->y);
      sum += convert_fp16_to_fp32(vec_h4->x) * convert_fp16_to_fp32(mat_h4->x);
      sum += convert_fp16_to_fp32(vec_h4->y) * convert_fp16_to_fp32(mat_h4->y);
    }
  }

  sum = warpReduceSum(sum, blockDim.x);

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = convert_fp32_to_fp16<T>(sum);
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums[threadIdx.y][laneId]
            : 0.0;
  // Final reduce using first warp
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) {
    res[row] = convert_fp32_to_fp16<T>(sum);
  }
}


/*
    GEMV kernel using FP16 to accumulate.
    Modified from fast_gemv_acc_fp16_kernel
    Support MP = 1~8
*/
template <typename T = half>
__global__ __forceinline__ void fast_gemv_acc_fp16_kernel(T* mat, T* bias, T* vec, T* res, unsigned int m, 
                                              unsigned int k, unsigned int n, unsigned int num_per_thread) {
  
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = (blockIdx.x * blockDim.y + threadIdx.y) << 1;
  unsigned int bid = blockIdx.z * gridDim.y + blockIdx.y;
  if (bid >= m) return;
  half2 vec_val[4];
  half2 mat_val[8];

  // half2 temp_sum = {convert_fp32_to_fp16<T>(0.0f), convert_fp32_to_fp16<T>(0.0f)};
  half2 sum[2];
  sum[0] = {convert_fp32_to_fp16<T>(0.0f), convert_fp32_to_fp16<T>(0.0f)};
  sum[1] = {convert_fp32_to_fp16<T>(0.0f), convert_fp32_to_fp16<T>(0.0f)};
  half2 gsum;

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

  if (tid == 0) {
    if (bias != nullptr)
      *(half2*)(&res[bid * n + row]) = __hadd2(gsum, *(half2*)(&bias[row]));
    else
      *(half2*)(&res[bid * n + row]) = gsum;
  }
}

