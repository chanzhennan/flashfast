/*
    Utility functions. 
*/

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../utils.h"

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

__device__ __forceinline__ float warpReduceSum(float sum_val,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    sum_val += __shfl_down_sync(0xffffffff, sum_val, 16);  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    sum_val += __shfl_down_sync(0xffffffff, sum_val, 8);  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    sum_val += __shfl_down_sync(0xffffffff, sum_val, 4);  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    sum_val += __shfl_down_sync(0xffffffff, sum_val, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    sum_val += __shfl_down_sync(0xffffffff, sum_val, 1);  // 0-1, 2-3, 4-5, etc.
  return sum_val;
}

__device__ __forceinline__ half warpReduceSum(half result,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result, 16));  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result, 8));  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result, 4));  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result, 2));  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result, 1));  // 0-1, 2-3, 4-5, etc.
  return result;
}

__device__ __forceinline__ half2 warpReduceSum(half2 result,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result, 16));  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result, 8));  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result, 4));  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result, 2));  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result, 1));  // 0-1, 2-3, 4-5, etc.
  return result;
}

__device__ __forceinline__ nv_bfloat16 warpReduceSum(nv_bfloat16 result,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result, 16));  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result, 8));  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result, 4));  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result, 2));  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result, 1));  // 0-1, 2-3, 4-5, etc.
  return result;
}

__device__ __forceinline__ nv_bfloat162 warpReduceSum(nv_bfloat162 result,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result, 16));  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result, 8));  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result, 4));  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result, 2));  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result, 1));  // 0-1, 2-3, 4-5, etc.
  return result;
}

__device__ __forceinline__ float warpReduceMax(float max_val,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, 16));  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, 8));  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, 4));  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, 2));  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, 1));  // 0-1, 2-3, 4-5, etc.
  return max_val;
}

__device__ __forceinline__ float blockReduceSum(float reducing, float *shared_mem)
{
    // Helper function for reduce softmax exp sum.
    const int32_t WPT = blockDim.x / 32;
    int32_t WPTB = WPT == 20 ? 32 : WPT;
    const int32_t lane_id = threadIdx.x % 32;
    const int32_t warp_id = threadIdx.x / 32;

# pragma unroll
    for (int32_t mask = 16; mask >= 1; mask /= 2) {
        reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
    }

    if (lane_id == 0) shared_mem[warp_id] = reducing;
    __syncthreads();

    if (lane_id < WPTB) reducing = lane_id < WPT ? shared_mem[lane_id] : 0.0f;

# pragma unroll
    for (int32_t mask = WPTB / 2; mask >= 1; mask /= 2) {
        reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
    }
    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}

__device__ __forceinline__ half blockReduceSum(half reducing, half *shared_mem)
{
    // Helper function for reduce softmax exp sum.
    const int32_t WPT = blockDim.x / 32;
    const int32_t lane_id = threadIdx.x % 32;
    const int32_t warp_id = threadIdx.x / 32;

# pragma unroll
    for (int32_t mask = 16; mask >= 1; mask /= 2) {
        reducing = __hadd(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }

    if (lane_id == 0) shared_mem[warp_id] = reducing;
    __syncthreads();

    if (lane_id < WPT) reducing = shared_mem[lane_id];

# pragma unroll
    for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
        reducing = __hadd(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }
    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}

__device__ __forceinline__ nv_bfloat16 blockReduceSum(nv_bfloat16 reducing, nv_bfloat16 *shared_mem)
{
    // Helper function for reduce softmax exp sum.
    const int32_t WPT = blockDim.x / 32;
    const int32_t lane_id = threadIdx.x % 32;
    const int32_t warp_id = threadIdx.x / 32;

# pragma unroll
    for (int32_t mask = 16; mask >= 1; mask /= 2) {
        reducing = __hadd(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }

    if (lane_id == 0) shared_mem[warp_id] = reducing;
    __syncthreads();

    if (lane_id < WPT) reducing = shared_mem[lane_id];

# pragma unroll
    for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
        reducing = __hadd(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }
    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}

__device__ __forceinline__ float convert_fp16_to_fp32(half data)        {return __half2float(data);}
__device__ __forceinline__ float convert_fp16_to_fp32(nv_bfloat16 data) {return __bfloat162float(data);}

__device__ __forceinline__ float2 convert_2fp16_to_2fp32(half2 data)        {return __half22float2(data);}
__device__ __forceinline__ float2 convert_2fp16_to_2fp32(nv_bfloat162 data) {return __bfloat1622float2(data);}

template <typename T> __device__ __forceinline__ T convert_fp32_to_fp16(float data)                        {return;}
template <>           __device__ __forceinline__ half convert_fp32_to_fp16<half>(float data)               {return __float2half(data);}
template <>           __device__ __forceinline__ nv_bfloat16 convert_fp32_to_fp16<nv_bfloat16>(float data) {return __float2bfloat16(data);}

template <typename T> __device__ __forceinline__ T convert_fp32_to_fp16_rn(float data)                        {return;}
template <>           __device__ __forceinline__ half convert_fp32_to_fp16_rn<half>(float data)               {return __float2half_rn(data);}
template <>           __device__ __forceinline__ nv_bfloat16 convert_fp32_to_fp16_rn<nv_bfloat16>(float data) {return __float2bfloat16_rn(data);}

template <typename T> __device__ __forceinline__ T convert_2fp32_to_2fp16_rn(float2 data)                          {return;}
template <>           __device__ __forceinline__ half2 convert_2fp32_to_2fp16_rn<half2>(float2 data)               {return __float22half2_rn(data);}
template <>           __device__ __forceinline__ nv_bfloat162 convert_2fp32_to_2fp16_rn<nv_bfloat162>(float2 data) {return __float22bfloat162_rn(data);}