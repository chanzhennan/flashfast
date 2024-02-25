#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>

#include "utils.cuh"

/*
    RMSNorm kernel.
*/
template <typename T = half>
__global__ __forceinline__ void rmsnorm_kernel(
                    T* x, T* rw, T* o, int bs, int dim){
  
  int bid = blockIdx.x;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int j = tid << 3;

  if (j >= dim) {return;}

  half2 x_val[4];
  half2 w_val[4];
  float pow_sum = 0.0f;

  *(float4*)(&x_val[0]) = *(float4*)(&x[bid * dim + j]);
  *(float4*)(&w_val[0]) = *(float4*)(&rw[j]);

  // RMSNorm (float)
#pragma unroll
  for (int i = 0; i < 4; i++){
    pow_sum += convert_fp16_to_fp32(x_val[i].x) * convert_fp16_to_fp32(x_val[i].x);
    pow_sum += convert_fp16_to_fp32(x_val[i].y) * convert_fp16_to_fp32(x_val[i].y);
  }

  // block reduce to get mean
  static __shared__ float warpLevelSums[WARP_SIZE];
  
  pow_sum = blockReduceSum(pow_sum, warpLevelSums);
  if (tid == 0){
    warpLevelSums[0] = rsqrtf(__fdividef(pow_sum, (float)dim) + 1e-5f);
  }
  __syncthreads();

  // normalization
  float scaling = warpLevelSums[0];
#pragma unroll
  for (int i = 0; i < 4; i++){
    x_val[i].x = convert_fp32_to_fp16<T>(convert_fp16_to_fp32(x_val[i].x) * scaling);
    x_val[i].y = convert_fp32_to_fp16<T>(convert_fp16_to_fp32(x_val[i].y) * scaling);
  }
  x_val[0] = __hmul2(x_val[0], w_val[0]);
  x_val[1] = __hmul2(x_val[1], w_val[1]);
  x_val[2] = __hmul2(x_val[2], w_val[2]);
  x_val[3] = __hmul2(x_val[3], w_val[3]);

  // store intermediate value
  *(float4*)(&o[bid * dim + j]) = *(float4*)(&x_val[0]);
}


/*
    RMSNorm kernel with input eps
*/
template <typename T = half>
__global__ __forceinline__ void rmsnorm_kernel(
                    T* x, T* rw, T* o, int bs, int dim, float eps){
  
  int bid = blockIdx.x;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int j = tid << 3;

  if (j >= dim) {return;}

  half2 x_val[4];
  half2 w_val[4];
  float pow_sum = 0.0f;

  *(float4*)(&x_val[0]) = *(float4*)(&x[bid * dim + j]);
  *(float4*)(&w_val[0]) = *(float4*)(&rw[j]);

  // RMSNorm (float)
#pragma unroll
  for (int i = 0; i < 4; i++){
    pow_sum += convert_fp16_to_fp32(x_val[i].x) * convert_fp16_to_fp32(x_val[i].x);
    pow_sum += convert_fp16_to_fp32(x_val[i].y) * convert_fp16_to_fp32(x_val[i].y);
  }

  // block reduce to get mean
  static __shared__ float warpLevelSums[WARP_SIZE];
  
  pow_sum = blockReduceSum(pow_sum, warpLevelSums);
  if (tid == 0){
    warpLevelSums[0] = rsqrtf(__fdividef(pow_sum, (float)dim) + eps);
  }
  __syncthreads();

  // normalization
  float scaling = warpLevelSums[0];
#pragma unroll
  for (int i = 0; i < 4; i++){
    x_val[i].x = convert_fp32_to_fp16<T>(convert_fp16_to_fp32(x_val[i].x) * scaling);
    x_val[i].y = convert_fp32_to_fp16<T>(convert_fp16_to_fp32(x_val[i].y) * scaling);
  }
  x_val[0] = __hmul2(x_val[0], w_val[0]);
  x_val[1] = __hmul2(x_val[1], w_val[1]);
  x_val[2] = __hmul2(x_val[2], w_val[2]);
  x_val[3] = __hmul2(x_val[3], w_val[3]);

  // store intermediate value
  *(float4*)(&o[bid * dim + j]) = *(float4*)(&x_val[0]);
}


/*
    LayerNorm kernel.
*/
template <typename T = half>
__global__ __forceinline__ void layernorm_kernel(
                    T* x, T* rw, T* rb, T* o, int bs, int dim){
  
  int bid = blockIdx.x;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int j = tid << 3;

  if (j >= dim) {return;}

  half2 x_val[4];
  half2 w_val[4];
  half2 b_val[4];
  float mean_sum = 0.0f;
  float pow_sum = 0.0f;

  *(float4*)(&x_val[0]) = *(float4*)(&x[bid * dim + j]);
  *(float4*)(&w_val[0]) = *(float4*)(&rw[j]);
  *(float4*)(&b_val[0]) = *(float4*)(&rb[j]);

  // Mean (float)
#pragma unroll
  for (int i = 0; i < 4; i++){
    mean_sum += convert_fp16_to_fp32(x_val[i].x);
    mean_sum += convert_fp16_to_fp32(x_val[i].y);
  }

  // block reduce to get mean
  static __shared__ float warpLevelSums[WARP_SIZE];
  mean_sum = blockReduceSum(mean_sum, warpLevelSums);
  if (tid == 0){
    warpLevelSums[0] = __fdividef(mean_sum, (float)dim);
  }
  __syncthreads();

  // Norm (float)
  float mean = warpLevelSums[0];
#pragma unroll
  for (int i = 0; i < 4; i++){
    pow_sum += (convert_fp16_to_fp32(x_val[i].x) - mean) * (convert_fp16_to_fp32(x_val[i].x) - mean);
    pow_sum += (convert_fp16_to_fp32(x_val[i].y) - mean) * (convert_fp16_to_fp32(x_val[i].y) - mean);
  }

  pow_sum = blockReduceSum(pow_sum, warpLevelSums);
  if (tid == 0){
    warpLevelSums[0] = rsqrtf(__fdividef(pow_sum, (float)dim) + 1e-5f);
  }
  __syncthreads();

  // normalization
  float scaling = warpLevelSums[0];
#pragma unroll
  for (int i = 0; i < 4; i++){
    x_val[i].x = convert_fp32_to_fp16<T>((convert_fp16_to_fp32(x_val[i].x) - mean) * scaling);
    x_val[i].y = convert_fp32_to_fp16<T>((convert_fp16_to_fp32(x_val[i].y) - mean) * scaling);
  }
  x_val[0] = __hadd2(__hmul2(x_val[0], w_val[0]), b_val[0]);
  x_val[1] = __hadd2(__hmul2(x_val[1], w_val[1]), b_val[1]);
  x_val[2] = __hadd2(__hmul2(x_val[2], w_val[2]), b_val[2]);
  x_val[3] = __hadd2(__hmul2(x_val[3], w_val[3]), b_val[3]);

  // store intermediate value
  *(float4*)(&o[bid * dim + j]) = *(float4*)(&x_val[0]);
}


/*
    LayerNorm kernel with input eps
*/
template <typename T = half>
__global__ __forceinline__ void layernorm_kernel(
                    T* x, T* rw, T* rb, T* o, int bs, int dim, float eps){
  
  int bid = blockIdx.x;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int j = tid << 3;

  if (j >= dim) {return;}

  half2 x_val[4];
  half2 w_val[4];
  half2 b_val[4];
  float mean_sum = 0.0f;
  float pow_sum = 0.0f;

  *(float4*)(&x_val[0]) = *(float4*)(&x[bid * dim + j]);
  *(float4*)(&w_val[0]) = *(float4*)(&rw[j]);
  *(float4*)(&b_val[0]) = *(float4*)(&rb[j]);

  // Mean (float)
#pragma unroll
  for (int i = 0; i < 4; i++){
    mean_sum += convert_fp16_to_fp32(x_val[i].x);
    mean_sum += convert_fp16_to_fp32(x_val[i].y);
  }

  // block reduce to get mean
  static __shared__ float warpLevelSums[WARP_SIZE];
  mean_sum = blockReduceSum(mean_sum, warpLevelSums);
  if (tid == 0){
    warpLevelSums[0] = __fdividef(mean_sum, (float)dim);
  }
  __syncthreads();

  // Norm (float)
  float mean = warpLevelSums[0];
#pragma unroll
  for (int i = 0; i < 4; i++){
    pow_sum += (convert_fp16_to_fp32(x_val[i].x) - mean) * (convert_fp16_to_fp32(x_val[i].x) - mean);
    pow_sum += (convert_fp16_to_fp32(x_val[i].y) - mean) * (convert_fp16_to_fp32(x_val[i].y) - mean);
  }

  pow_sum = blockReduceSum(pow_sum, warpLevelSums);
  if (tid == 0){
    warpLevelSums[0] = rsqrtf(__fdividef(pow_sum, (float)dim) + eps);
  }
  __syncthreads();

  // normalization
  float scaling = warpLevelSums[0];
#pragma unroll
  for (int i = 0; i < 4; i++){
    x_val[i].x = convert_fp32_to_fp16<T>((convert_fp16_to_fp32(x_val[i].x) - mean) * scaling);
    x_val[i].y = convert_fp32_to_fp16<T>((convert_fp16_to_fp32(x_val[i].y) - mean) * scaling);
  }
  x_val[0] = __hadd2(__hmul2(x_val[0], w_val[0]), b_val[0]);
  x_val[1] = __hadd2(__hmul2(x_val[1], w_val[1]), b_val[1]);
  x_val[2] = __hadd2(__hmul2(x_val[2], w_val[2]), b_val[2]);
  x_val[3] = __hadd2(__hmul2(x_val[3], w_val[3]), b_val[3]);

  // store intermediate value
  *(float4*)(&o[bid * dim + j]) = *(float4*)(&x_val[0]);
}


/*
    add residual + RMSNorm fused kernel.
*/
template <typename T = half>
__global__ __forceinline__ void residual_rmsnorm_kernel(
                                T* x, T* r, T* rw, 
                                int bs, int dim, 
                                T* o, T* ro){
  
  int bid = blockIdx.x;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int j = tid << 3;

  if (j >= dim) {return;}

  half2 x_val[4];
  half2 i_val[4];
  half2 r_val[4];
  half2 w_val[4];
  float pow_sum = 0.0f;

  *(float4*)(&x_val[0]) = *(float4*)(&x[bid * dim + j]);
  *(float4*)(&r_val[0]) = *(float4*)(&r[bid * dim + j]);
  *(float4*)(&w_val[0]) = *(float4*)(&rw[j]);

  // residual
  i_val[0] = __hadd2(x_val[0], r_val[0]);
  i_val[1] = __hadd2(x_val[1], r_val[1]);
  i_val[2] = __hadd2(x_val[2], r_val[2]);
  i_val[3] = __hadd2(x_val[3], r_val[3]);

  // store intermediate value
  *(float4*)(&ro[bid * dim + j]) = *(float4*)(&i_val[0]);

  // RMSNorm (float)
#pragma unroll
  for (int i = 0; i < 4; i++){
    pow_sum += convert_fp16_to_fp32(i_val[i].x) * convert_fp16_to_fp32(i_val[i].x);
    pow_sum += convert_fp16_to_fp32(i_val[i].y) * convert_fp16_to_fp32(i_val[i].y);
  }

  // block reduce to get mean
  static __shared__ float warpLevelSums[WARP_SIZE];
  
  pow_sum = blockReduceSum(pow_sum, warpLevelSums);
  if (tid == 0){
    warpLevelSums[0] = rsqrtf(__fdividef(pow_sum, (float)dim) + 1e-5f);
  }
  __syncthreads();

  // normalization
  float scaling = warpLevelSums[0];
#pragma unroll
  for (int i = 0; i < 4; i++){
    x_val[i].x = convert_fp32_to_fp16<T>(convert_fp16_to_fp32(i_val[i].x) * scaling);
    x_val[i].y = convert_fp32_to_fp16<T>(convert_fp16_to_fp32(i_val[i].y) * scaling);
  }
  x_val[0] = __hmul2(x_val[0], w_val[0]);
  x_val[1] = __hmul2(x_val[1], w_val[1]);
  x_val[2] = __hmul2(x_val[2], w_val[2]);
  x_val[3] = __hmul2(x_val[3], w_val[3]);

  // store intermediate value
  *(float4*)(&o[bid * dim + j]) = *(float4*)(&x_val[0]);
}


/*
    add residual + RMSNorm fused kernel with input eps
*/
template <typename T = half>
__global__ __forceinline__ void residual_rmsnorm_kernel(
                                T* x, T* r, T* rw, 
                                int bs, int dim, float eps,
                                T* o, T* ro){
  
  int bid = blockIdx.x;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int j = tid << 3;

  if (j >= dim) {return;}

  half2 x_val[4];
  half2 i_val[4];
  half2 r_val[4];
  half2 w_val[4];
  float pow_sum = 0.0f;

  *(float4*)(&x_val[0]) = *(float4*)(&x[bid * dim + j]);
  *(float4*)(&r_val[0]) = *(float4*)(&r[bid * dim + j]);
  *(float4*)(&w_val[0]) = *(float4*)(&rw[j]);

  // residual
  i_val[0] = __hadd2(x_val[0], r_val[0]);
  i_val[1] = __hadd2(x_val[1], r_val[1]);
  i_val[2] = __hadd2(x_val[2], r_val[2]);
  i_val[3] = __hadd2(x_val[3], r_val[3]);

  // store intermediate value
  *(float4*)(&ro[bid * dim + j]) = *(float4*)(&i_val[0]);

  // RMSNorm (float)
#pragma unroll
  for (int i = 0; i < 4; i++){
    pow_sum += convert_fp16_to_fp32(i_val[i].x) * convert_fp16_to_fp32(i_val[i].x);
    pow_sum += convert_fp16_to_fp32(i_val[i].y) * convert_fp16_to_fp32(i_val[i].y);
  }

  // block reduce to get mean
  static __shared__ float warpLevelSums[WARP_SIZE];
  
  pow_sum = blockReduceSum(pow_sum, warpLevelSums);
  if (tid == 0){
    warpLevelSums[0] = rsqrtf(__fdividef(pow_sum, (float)dim) + eps);
  }
  __syncthreads();

  // normalization
  float scaling = warpLevelSums[0];
#pragma unroll
  for (int i = 0; i < 4; i++){
    x_val[i].x = convert_fp32_to_fp16<T>(convert_fp16_to_fp32(i_val[i].x) * scaling);
    x_val[i].y = convert_fp32_to_fp16<T>(convert_fp16_to_fp32(i_val[i].y) * scaling);
  }
  x_val[0] = __hmul2(x_val[0], w_val[0]);
  x_val[1] = __hmul2(x_val[1], w_val[1]);
  x_val[2] = __hmul2(x_val[2], w_val[2]);
  x_val[3] = __hmul2(x_val[3], w_val[3]);

  // store intermediate value
  *(float4*)(&o[bid * dim + j]) = *(float4*)(&x_val[0]);
}



/*
    add residual + LayerNorm fused kernel.
*/
template <typename T = half>
__global__ __forceinline__ void residual_layernorm_kernel(
                                T* x, T* r, T* rw, T* rb, 
                                int bs, int dim, 
                                T* o, T* ro){
  
  int bid = blockIdx.x;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int j = tid << 3;

  if (j >= dim) {return;}

  half2 x_val[4];
  half2 i_val[4];
  half2 r_val[4];
  half2 w_val[4];
  half2 b_val[4];
  float mean_sum = 0.0f;
  float pow_sum = 0.0f;

  *(float4*)(&x_val[0]) = *(float4*)(&x[bid * dim + j]);
  *(float4*)(&r_val[0]) = *(float4*)(&r[bid * dim + j]);
  *(float4*)(&w_val[0]) = *(float4*)(&rw[j]);
  *(float4*)(&b_val[0]) = *(float4*)(&rb[j]);

  // residual
  i_val[0] = __hadd2(x_val[0], r_val[0]);
  i_val[1] = __hadd2(x_val[1], r_val[1]);
  i_val[2] = __hadd2(x_val[2], r_val[2]);
  i_val[3] = __hadd2(x_val[3], r_val[3]);

  // store intermediate value
  *(float4*)(&ro[bid * dim + j]) = *(float4*)(&i_val[0]);

  // Mean (float)
#pragma unroll
  for (int i = 0; i < 4; i++){
    mean_sum += convert_fp16_to_fp32(i_val[i].x);
    mean_sum += convert_fp16_to_fp32(i_val[i].y);
  }

  // block reduce to get mean
  static __shared__ float warpLevelSums[WARP_SIZE];
  mean_sum = blockReduceSum(mean_sum, warpLevelSums);
  if (tid == 0){
    warpLevelSums[0] = __fdividef(mean_sum, (float)dim);
  }
  __syncthreads();

  // Norm (float)
  float mean = warpLevelSums[0];
#pragma unroll
  for (int i = 0; i < 4; i++){
    pow_sum += (convert_fp16_to_fp32(i_val[i].x) - mean) * (convert_fp16_to_fp32(i_val[i].x) - mean);
    pow_sum += (convert_fp16_to_fp32(i_val[i].y) - mean) * (convert_fp16_to_fp32(i_val[i].y) - mean);
  }
  
  pow_sum = blockReduceSum(pow_sum, warpLevelSums);
  if (tid == 0){
    warpLevelSums[0] = rsqrtf(__fdividef(pow_sum, (float)dim) + 1e-5f);
  }
  __syncthreads();

  // normalization
  float scaling = warpLevelSums[0];
#pragma unroll
  for (int i = 0; i < 4; i++){
    x_val[i].x = convert_fp32_to_fp16<T>((convert_fp16_to_fp32(i_val[i].x) - mean) * scaling);
    x_val[i].y = convert_fp32_to_fp16<T>((convert_fp16_to_fp32(i_val[i].y) - mean) * scaling);
  }
  x_val[0] = __hadd2(__hmul2(x_val[0], w_val[0]), b_val[0]);
  x_val[1] = __hadd2(__hmul2(x_val[1], w_val[1]), b_val[1]);
  x_val[2] = __hadd2(__hmul2(x_val[2], w_val[2]), b_val[2]);
  x_val[3] = __hadd2(__hmul2(x_val[3], w_val[3]), b_val[3]);

  // store intermediate value
  *(float4*)(&o[bid * dim + j]) = *(float4*)(&x_val[0]);
}

/*
    add residual + LayerNorm fused kernel with input eps
*/
template <typename T = half>
__global__ __forceinline__ void residual_layernorm_kernel(
                                T* x, T* r, T* rw, T* rb, 
                                int bs, int dim, float eps,
                                T* o, T* ro){
  
  int bid = blockIdx.x;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int j = tid << 3;

  if (j >= dim) {return;}

  half2 x_val[4];
  half2 i_val[4];
  half2 r_val[4];
  half2 w_val[4];
  half2 b_val[4];
  float mean_sum = 0.0f;
  float pow_sum = 0.0f;

  *(float4*)(&x_val[0]) = *(float4*)(&x[bid * dim + j]);
  *(float4*)(&r_val[0]) = *(float4*)(&r[bid * dim + j]);
  *(float4*)(&w_val[0]) = *(float4*)(&rw[j]);
  *(float4*)(&b_val[0]) = *(float4*)(&rb[j]);

  // residual
  i_val[0] = __hadd2(x_val[0], r_val[0]);
  i_val[1] = __hadd2(x_val[1], r_val[1]);
  i_val[2] = __hadd2(x_val[2], r_val[2]);
  i_val[3] = __hadd2(x_val[3], r_val[3]);

  // store intermediate value
  *(float4*)(&ro[bid * dim + j]) = *(float4*)(&i_val[0]);

  // Mean (float)
#pragma unroll
  for (int i = 0; i < 4; i++){
    mean_sum += convert_fp16_to_fp32(i_val[i].x);
    mean_sum += convert_fp16_to_fp32(i_val[i].y);
  }

  // block reduce to get mean
  static __shared__ float warpLevelSums[WARP_SIZE];
  mean_sum = blockReduceSum(mean_sum, warpLevelSums);
  if (tid == 0){
    warpLevelSums[0] = __fdividef(mean_sum, (float)dim);
  }
  __syncthreads();

  // Norm (float)
  float mean = warpLevelSums[0];
#pragma unroll
  for (int i = 0; i < 4; i++){
    pow_sum += (convert_fp16_to_fp32(i_val[i].x) - mean) * (convert_fp16_to_fp32(i_val[i].x) - mean);
    pow_sum += (convert_fp16_to_fp32(i_val[i].y) - mean) * (convert_fp16_to_fp32(i_val[i].y) - mean);
  }
  
  pow_sum = blockReduceSum(pow_sum, warpLevelSums);
  if (tid == 0){
    warpLevelSums[0] = rsqrtf(__fdividef(pow_sum, (float)dim) + eps);
  }
  __syncthreads();

  // normalization
  float scaling = warpLevelSums[0];
#pragma unroll
  for (int i = 0; i < 4; i++){
    x_val[i].x = convert_fp32_to_fp16<T>((convert_fp16_to_fp32(i_val[i].x) - mean) * scaling);
    x_val[i].y = convert_fp32_to_fp16<T>((convert_fp16_to_fp32(i_val[i].y) - mean) * scaling);
  }
  x_val[0] = __hadd2(__hmul2(x_val[0], w_val[0]), b_val[0]);
  x_val[1] = __hadd2(__hmul2(x_val[1], w_val[1]), b_val[1]);
  x_val[2] = __hadd2(__hmul2(x_val[2], w_val[2]), b_val[2]);
  x_val[3] = __hadd2(__hmul2(x_val[3], w_val[3]), b_val[3]);

  // store intermediate value
  *(float4*)(&o[bid * dim + j]) = *(float4*)(&x_val[0]);
}