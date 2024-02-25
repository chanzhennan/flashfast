#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>
#include <cub/cub.cuh>

#include "utils.cuh"

#define FLOAT_BANK_SIZE 32
#define HALF_BANK_SIZE 64
#define MAX_HEAD_SIZE 128
#define MAX_LEN_GROUP 64
#define MAX_LOOP_SPACE 2


///////////////////////////////  splitKV scaling kernel  ////////////////////////////////////////

template <int SPLIT, typename T = half>
__global__ void decode_splitKV_scaling_kernel(float* H_F, int dim, int hs, int hn, float* S, T* H){

  // HF: [bs, 1, hn, SPLIT, hs]
  // S: [bs, hn, SPLIT]

  unsigned int h_offset = blockIdx.x * dim + blockIdx.y * hs;
  unsigned int tid = threadIdx.x;

  float real_sum = 0.0f;
  float partial_sum[SPLIT] = {0.0f}; 
#pragma unroll
  for (int s = 0; s < SPLIT; s++){
    partial_sum[s] = S[(blockIdx.x * hn + blockIdx.y) * SPLIT + s];
    real_sum += partial_sum[s];
  }

  float real_value = 0.0f;
#pragma unroll
  for (int s = 0; s < SPLIT; s++){
    float res = H_F[h_offset * SPLIT + s * hs + tid];
    float scaling = __fdividef(partial_sum[s], real_sum);
    real_value += res * scaling;
  }

  H[h_offset + tid] = convert_fp32_to_fp16<T>(real_value);
}


///////////////////////////////  fall back kernel  ////////////////////////////////////////

/*
    Decode MHA kernel w/ or w/o alibi masked for KVcache with any shape. Only need the strides of bs and seqlen dim. 
    Naive decode MQA/GQA kernel w/ or w/o alibi masked for KVcache with any shape. Only need the strides of bs and seqlen dim. 
    Efficient for MHA, when hn == hn_kv
    Not efficient at all for MQA/GQA
*/
template <int BLOCK_SIZE, int FLOOP, MASK_TYPE MASK = MASK_TYPE::NO_MASK, typename T = half, typename T2 = half2>
__global__ __forceinline__ void decode_fall_back_kernel(
                T* Q, T* K, T* V, float* slopes, const float p_scale, 
                int hn, int hn_kv, int hs, int len, int kv_stride_bs, int kv_stride_seq, int loop, T* H) {
  int dim = hn * hs;
  unsigned int q_head_id = blockIdx.y;
  unsigned int kv_head_id = blockIdx.y / (hn / hn_kv);
  unsigned int q_offset = blockIdx.x * dim + q_head_id * hs;
  unsigned int kv_offset = blockIdx.x * kv_stride_bs + kv_head_id * hs;
  
  unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
  
  int hs_shift = hs >> 3;
  int len_group = DIV_UP(BLOCK_SIZE, hs_shift); // make sure len_group is a power-of-two number
  int pad_len = DIV_UP(len, hs_shift) * hs_shift;

  unsigned int k_row = tid / hs_shift;
  unsigned int k_col = tid % hs_shift;
  unsigned int j = k_col << 3;

  T2 q_val[4], k_val[4];
  float s_temp;
  T temp_sum;
  T PD[8] = {convert_fp32_to_fp16<T>(0.0f)};
  
  float slope; // for alibi mask
  if (MASK == MASK_TYPE::ALIBI_MASK) 
    slope = slopes[kv_head_id];

  // 4KB
  __shared__ float s_shmem[FLOOP * (FLOAT_BANK_SIZE + 8)];
  // 512B
  __shared__ float temp_max_val[MAX_LEN_GROUP];
  // 35KB
  __shared__ T av_shmem[MAX_LEN_GROUP][MAX_HEAD_SIZE + 8]; 
  // 1KB
  __shared__ T half_space[MAX_LOOP_SPACE][MAX_HEAD_SIZE + 8];
  __shared__ float float_space[MAX_LOOP_SPACE];
  __shared__ float block_max, block_sum;

  // load Q
  *(float4*)(&q_val[0]) = *(float4*)(&Q[q_offset + j]);

  // to execute reduction in the loop
  float qxk_result = 0.0f;
  float sum_val = 0.0f, temp_sum_val = 0.0f;
  float max_val = -1e20f, temp_max_val_thread = -1e20f;

  // Q x K
#pragma unroll
  for (int row = k_row; row < pad_len; row += len_group){
    // if (row >= len) {s_shmem[row] = 0; break;}
    
    *(float4*)(&k_val[0]) = row < len ? 
      *(float4*)(&K[(size_t)row * kv_stride_seq + kv_offset + j]) : 
      *(float4*)(&PD[0]);

    s_temp = convert_fp16_to_fp32(q_val[0].x) * convert_fp16_to_fp32(k_val[0].x);
    s_temp += convert_fp16_to_fp32(q_val[0].y) * convert_fp16_to_fp32(k_val[0].y);
    s_temp += convert_fp16_to_fp32(q_val[1].x) * convert_fp16_to_fp32(k_val[1].x);
    s_temp += convert_fp16_to_fp32(q_val[1].y) * convert_fp16_to_fp32(k_val[1].y);
    s_temp += convert_fp16_to_fp32(q_val[2].x) * convert_fp16_to_fp32(k_val[2].x);
    s_temp += convert_fp16_to_fp32(q_val[2].y) * convert_fp16_to_fp32(k_val[2].y);
    s_temp += convert_fp16_to_fp32(q_val[3].x) * convert_fp16_to_fp32(k_val[3].x);
    s_temp += convert_fp16_to_fp32(q_val[3].y) * convert_fp16_to_fp32(k_val[3].y);
    s_temp = warpReduceSum(s_temp, (hs >> 3));

    // now for all k_col == 0, a temp_s is stored in the register
    // push into share memory
    unsigned int s_row = row / FLOAT_BANK_SIZE;
    unsigned int s_col = row % FLOAT_BANK_SIZE;

    if (k_col == 0){
      qxk_result = s_temp * p_scale;
      if (MASK == MASK_TYPE::ALIBI_MASK) 
        qxk_result += row < len ? slope * row : 0.0f; // for alibi mask
      temp_max_val_thread = max(temp_max_val_thread, qxk_result);
      s_shmem[s_row * (FLOAT_BANK_SIZE + 8) + s_col] = qxk_result;
    }
  }

  if (k_col == 0){
    temp_max_val[k_row] = temp_max_val_thread;
  }

  // make sure QxK and temporary max and sum are stored into shmem
  __syncthreads();
  
  max_val = tid < len_group ? temp_max_val[tid] : -1e20f;
  max_val = warpReduceMax(max_val, min(len_group, 32));
  
  if (tid % 32 == 0){
    float_space[(tid / 32)] = max_val;
  }
  __syncthreads();

  if (tid == 0){
  #pragma unroll
    for (int r = 1; r < (len_group >> 5); r++){
      float_space[0] = max(float_space[0], float_space[r]);
    } 
  }
  __syncthreads();

  __shared__ typename cub::BlockReduce<float, BLOCK_SIZE>::TempStorage temp_storage;
  
#pragma unroll
  for (int l = 0; l < loop; l += 1){
    unsigned int lid = l * BLOCK_SIZE + tid;
    unsigned int l_row = lid / FLOAT_BANK_SIZE;
    unsigned int l_col = lid % FLOAT_BANK_SIZE;
    s_shmem[l_row * (FLOAT_BANK_SIZE + 8) + l_col] = lid < len ? 
      __expf(s_shmem[l_row * (FLOAT_BANK_SIZE + 8) + l_col] - float_space[0]) : 0.0f;
    temp_sum_val += s_shmem[l_row * (FLOAT_BANK_SIZE + 8) + l_col];
  }
  temp_sum_val = cub::BlockReduce<float, BLOCK_SIZE>(temp_storage).Reduce(temp_sum_val, cub::Sum());
  if (tid == 0){block_sum = temp_sum_val;}
  __syncthreads();
  
  float a;
  T result;
  float2 pv[4];

#pragma unroll
  for (int i = 0; i < 4; i++){
    pv[i].x = 0.0f;
    pv[i].y = 0.0f;
  }

  // V partition
  unsigned int v_row = tid / (hs >> 3);
  unsigned int v_col = tid % (hs >> 3);

#pragma unroll
  for (int row = v_row; row < pad_len; row += len_group){

    *(float4*)(&k_val[0]) = row < len ? 
      *(float4*)(&V[(size_t)row * kv_stride_seq + kv_offset + (v_col << 3)]) : 
      *(float4*)(&PD[0]);
    
    unsigned int a_row = row / FLOAT_BANK_SIZE;
    unsigned int a_col = row % FLOAT_BANK_SIZE;

    a = s_shmem[a_row * (FLOAT_BANK_SIZE + 8) + a_col] * 
      __fdividef(1.0f, block_sum + 1e-6f);

    pv[0].x = pv[0].x + a * convert_fp16_to_fp32(k_val[0].x);
    pv[0].y = pv[0].y + a * convert_fp16_to_fp32(k_val[0].y);
    pv[1].x = pv[1].x + a * convert_fp16_to_fp32(k_val[1].x);
    pv[1].y = pv[1].y + a * convert_fp16_to_fp32(k_val[1].y);
    pv[2].x = pv[2].x + a * convert_fp16_to_fp32(k_val[2].x);
    pv[2].y = pv[2].y + a * convert_fp16_to_fp32(k_val[2].y);
    pv[3].x = pv[3].x + a * convert_fp16_to_fp32(k_val[3].x);
    pv[3].y = pv[3].y + a * convert_fp16_to_fp32(k_val[3].y);
  }

  T2 pv_half[4];

  pv_half[0] = convert_2fp32_to_2fp16_rn<T2>(pv[0]);
  pv_half[1] = convert_2fp32_to_2fp16_rn<T2>(pv[1]);
  pv_half[2] = convert_2fp32_to_2fp16_rn<T2>(pv[2]);
  pv_half[3] = convert_2fp32_to_2fp16_rn<T2>(pv[3]);

  // transpoisition allows efficient mem access of V matrix
  *(float4*)(&av_shmem[v_row][(v_col << 3)]) = *(float4*)(&pv_half[0]);
  __syncthreads();

  unsigned int o_row = tid % len_group;
  unsigned int o_col = tid / len_group;

  *(float4*)(&pv_half[0]) = *(float4*)(&av_shmem[o_row][(o_col << 3)]);

  unsigned int real_len_group = min(len_group, 32);

  pv_half[0] = warpReduceSum(pv_half[0], real_len_group);
  pv_half[1] = warpReduceSum(pv_half[1], real_len_group);
  pv_half[2] = warpReduceSum(pv_half[2], real_len_group);
  pv_half[3] = warpReduceSum(pv_half[3], real_len_group);

  if (o_row % 32 == 0){
    *(float4*)(&av_shmem[(o_row / 32)][(o_col << 3)]) = *(float4*)(&pv_half[0]);
  }
  __syncthreads();

  if (o_row == 0){
  #pragma unroll
    for (int r = 1; r < (len_group >> 5); r++){
      pv_half[0] = __hadd2(pv_half[0], *(T2*)(&av_shmem[r][(o_col << 3) + 0]));
      pv_half[1] = __hadd2(pv_half[1], *(T2*)(&av_shmem[r][(o_col << 3) + 2]));
      pv_half[2] = __hadd2(pv_half[2], *(T2*)(&av_shmem[r][(o_col << 3) + 4]));
      pv_half[3] = __hadd2(pv_half[3], *(T2*)(&av_shmem[r][(o_col << 3) + 6]));
    } 
    *(float4*)(&H[q_offset + (o_col << 3)]) = *(float4*)(&pv_half[0]);
  }
}



///////////////////////////////  async softmax kernel  ////////////////////////////////////////

/*
    Decode MHA kernel w/ or w/o alibi masked with FD++ for KVcache with any shape. Only need the strides of bs and seqlen dim. 
    Naive decode MQA/GQA kernel w/ or w/o alibi masked with FD++ for KVcache with any shape. Only need the strides of bs and seqlen dim. 
    Efficient for MHA, when hn == hn_kv
    Not efficient at all for MQA/GQA
*/
template <int BLOCK_SIZE, int FLOOP, MASK_TYPE MASK = MASK_TYPE::NO_MASK, typename T = half, typename T2 = half2>
__global__ __forceinline__ void decode_async_softmax_kernel(
                T* Q, T* K, T* V, float* slopes, const float p_scale, const float max_val, 
                int hn, int hn_kv, int hs, int len, int kv_stride_bs, int kv_stride_seq, int loop, T* H) {
  int dim = hn * hs;
  unsigned int q_head_id = blockIdx.y;
  unsigned int kv_head_id = blockIdx.y / (hn / hn_kv);
  unsigned int q_offset = blockIdx.x * dim + q_head_id * hs;
  unsigned int kv_offset = blockIdx.x * kv_stride_bs + kv_head_id * hs;
  
  unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
  
  int hs_shift = hs >> 3;
  int len_group = DIV_UP(BLOCK_SIZE, hs_shift); // make sure len_group is a power-of-two number
  int pad_len = DIV_UP(len, len_group) * len_group; // a little speedup if use hs_shift here, but computation error will be introduced

  unsigned int kv_row = tid / hs_shift;
  unsigned int kv_col = tid % hs_shift;
  unsigned int j = kv_col << 3;

  T2 q_val[4], k_val[4], v_val[4], s_temp;
  T temp_sum;
  T PD[8] = {convert_fp32_to_fp16<T>(0.0f)};

  float slope, max_val_alibi; // for alibi mask
  if (MASK == MASK_TYPE::ALIBI_MASK) 
    slope = slopes[kv_head_id];
    max_val_alibi = max_val + slope * (len - 1);

  // 4KB
  __shared__ float s_shmem[FLOOP * (FLOAT_BANK_SIZE + 8)];
  // 35KB
  __shared__ T av_shmem[MAX_LEN_GROUP][MAX_HEAD_SIZE + 8]; 
  __shared__ float block_sum;

  // load Q
  *(float4*)(&q_val[0]) = *(float4*)(&Q[q_offset + j]);

  float temp_sum_val = 0.0f;
  float a;
  float4 pv[2];

#pragma unroll
  for (int i = 0; i < 2; i++){
    pv[i].x = 0.0f;
    pv[i].y = 0.0f;
    pv[i].z = 0.0f;
    pv[i].w = 0.0f;
  }

#pragma unroll
  for (int row = kv_row; row < pad_len; row += len_group){

    // Q x K
    *(float4*)(&k_val[0]) = row < len ? 
      *(float4*)(&K[(size_t)row * kv_stride_seq + kv_offset + j]) : 
      *(float4*)(&PD[0]);
    *(float4*)(&v_val[0]) = row < len ? 
      *(float4*)(&V[(size_t)row * kv_stride_seq + kv_offset + j]) : 
      *(float4*)(&PD[0]);

    s_temp = __hmul2(q_val[0], k_val[0]);
    s_temp = __hadd2(s_temp, __hmul2(q_val[1], k_val[1]));
    s_temp = __hadd2(s_temp, __hmul2(q_val[2], k_val[2]));
    s_temp = __hadd2(s_temp, __hmul2(q_val[3], k_val[3]));

    temp_sum = __hadd(s_temp.x, s_temp.y);
    temp_sum = warpReduceSum(temp_sum, (hs >> 3));

    // now for all k_col == 0, a temp_s is stored in the register
    // push into share memory
    unsigned int s_row = kv_row / FLOAT_BANK_SIZE;
    unsigned int s_col = kv_row % FLOAT_BANK_SIZE;

    __syncthreads();

    // softmax without scaling
    if (kv_col == 0){
      if (MASK == MASK_TYPE::ALIBI_MASK) 
        s_shmem[s_row * (FLOAT_BANK_SIZE + 8) + s_col] = row < len ? 
          __expf(convert_fp16_to_fp32(temp_sum) * p_scale 
                + slope * row - max_val_alibi) : 0.0f;
      else 
        s_shmem[s_row * (FLOAT_BANK_SIZE + 8) + s_col] = row < len ? 
          __expf(convert_fp16_to_fp32(temp_sum) * p_scale - max_val) : 0.0f;
    }

    __syncthreads();
    // calculate the intermediate sum
    unsigned int t_row = tid / FLOAT_BANK_SIZE;
    unsigned int t_col = tid % FLOAT_BANK_SIZE;
    temp_sum_val += (tid < len_group) ? 
        s_shmem[t_row * (FLOAT_BANK_SIZE + 8) + t_col] : 0.0f;

    // AxV
    a = s_shmem[s_row * (FLOAT_BANK_SIZE + 8) + s_col];

    pv[0].x = pv[0].x + a * convert_fp16_to_fp32(v_val[0].x);
    pv[0].y = pv[0].y + a * convert_fp16_to_fp32(v_val[0].y);
    pv[0].z = pv[0].z + a * convert_fp16_to_fp32(v_val[1].x);
    pv[0].w = pv[0].w + a * convert_fp16_to_fp32(v_val[1].y);
    pv[1].x = pv[1].x + a * convert_fp16_to_fp32(v_val[2].x);
    pv[1].y = pv[1].y + a * convert_fp16_to_fp32(v_val[2].y);
    pv[1].z = pv[1].z + a * convert_fp16_to_fp32(v_val[3].x);
    pv[1].w = pv[1].w + a * convert_fp16_to_fp32(v_val[3].y);
  }

  unsigned int real_len_group = min(len_group, 32);

  unsigned int o_row = tid % len_group;
  unsigned int o_col = tid / len_group;

  __shared__ typename cub::BlockReduce<float, BLOCK_SIZE>::TempStorage temp_storage;
  temp_sum_val = cub::BlockReduce<float, BLOCK_SIZE>(temp_storage).Reduce(temp_sum_val, cub::Sum());
  if (tid == 0){block_sum = temp_sum_val;}
  __syncthreads();

  float scaling = __fdividef(1.0f, block_sum + 1e-6f);

  T2 pv_half[4];

  pv_half[0] = {convert_fp32_to_fp16<T>(pv[0].x * scaling), convert_fp32_to_fp16<T>(pv[0].y * scaling)};
  pv_half[1] = {convert_fp32_to_fp16<T>(pv[0].z * scaling), convert_fp32_to_fp16<T>(pv[0].w * scaling)};
  pv_half[2] = {convert_fp32_to_fp16<T>(pv[1].x * scaling), convert_fp32_to_fp16<T>(pv[1].y * scaling)};
  pv_half[3] = {convert_fp32_to_fp16<T>(pv[1].z * scaling), convert_fp32_to_fp16<T>(pv[1].w * scaling)};

  // transpoisition allows efficient mem access of V matrix
  *(float4*)(&av_shmem[kv_row][(kv_col << 3)]) = *(float4*)(&pv_half[0]);
  __syncthreads();

  *(float4*)(&pv_half[0]) = *(float4*)(&av_shmem[o_row][(o_col << 3)]);

  pv_half[0] = warpReduceSum(pv_half[0], real_len_group);
  pv_half[1] = warpReduceSum(pv_half[1], real_len_group);
  pv_half[2] = warpReduceSum(pv_half[2], real_len_group);
  pv_half[3] = warpReduceSum(pv_half[3], real_len_group);

  if (o_row % 32 == 0){
    *(float4*)(&av_shmem[(o_row >> 5)][(o_col << 3)]) = *(float4*)(&pv_half[0]);
  }
  __syncthreads();

  if (o_row == 0){
  #pragma unroll
    for (int r = 1; r < (len_group >> 5); r++){
      pv_half[0] = __hadd2(pv_half[0], *(T2*)(&av_shmem[r][(o_col << 3) + 0]));
      pv_half[1] = __hadd2(pv_half[1], *(T2*)(&av_shmem[r][(o_col << 3) + 2]));
      pv_half[2] = __hadd2(pv_half[2], *(T2*)(&av_shmem[r][(o_col << 3) + 4]));
      pv_half[3] = __hadd2(pv_half[3], *(T2*)(&av_shmem[r][(o_col << 3) + 6]));
    } 
    *(float4*)(&H[q_offset + (o_col << 3)]) = *(float4*)(&pv_half[0]);
  }
}


///////////////////////////////  async softmax & splitKV kernel  ////////////////////////////////////////

/*
    Decode splitKV MHA kernel w/ or w/o alibi masked with FD++ for KVcache with any shape. Only need the strides of bs and seqlen dim. 
    Naive decode splitKV MQA/GQA kernel w/ or w/o alibi masked with FD++ for KVcache with any shape. Only need the strides of bs and seqlen dim. 
    Efficient for MHA, when hn == hn_kv
    Not efficient at all for MQA/GQA
*/
template <int BLOCK_SIZE, int FLOOP, int SPLIT, MASK_TYPE MASK = MASK_TYPE::NO_MASK, typename T = half, typename T2 = half2>
__global__ void decode_splitKV_kernel(
                T* Q, T* K, T* V, float* slopes, const float p_scale, const float max_val, 
                int hn, int hn_kv, int hs, int len, int kv_stride_bs, int kv_stride_seq, int loop, float* S, float* H) {
  int dim = hn * hs;
  unsigned int q_head_id = blockIdx.y;
  unsigned int kv_head_id = blockIdx.y / (hn / hn_kv);
  unsigned int q_offset = blockIdx.x * dim + q_head_id * hs;
  unsigned int kv_offset = blockIdx.x * kv_stride_bs + kv_head_id * hs;

  int cur_len = DIV_UP(len, SPLIT);
  unsigned int len_offset = blockIdx.z * cur_len;
  
  unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
  
  int len_group = DIV_UP(BLOCK_SIZE, (hs >> 3)); // make sure len_group is a power-of-two number
  //int len_span = DIV_UP(BLOCK_SIZE, hs); // must not exceed 4
  int pad_len = DIV_UP(cur_len, len_group) * len_group;

  unsigned int kv_row = tid / (hs >> 3);
  unsigned int kv_col = tid % (hs >> 3);
  unsigned int j = kv_col << 3;
  // int BK = blockDim.x * blockDim.y / (hs >> 3);

  // T2 q_val[4], k_val[8], v_val[8], s_temp;
  T2 q_val[4], kv_val[8], s_temp;
  T temp_sum = convert_fp32_to_fp16<T>(0.0f);
  T PD[8] = {convert_fp32_to_fp16<T>(0.0f)};

  float slope, max_val_alibi; // for alibi mask
  if (MASK == MASK_TYPE::ALIBI_MASK)
    slope = slopes[q_head_id];
    max_val_alibi = max_val + slope * (len - 1);

  // 4KB
  __shared__ float s_shmem[MAX_LEN_GROUP];
  // 512B
  // __shared__ float temp_max_val[MAX_LEN_GROUP];
  // 35KB
  __shared__ float av_shmem[MAX_LEN_GROUP][(MAX_HEAD_SIZE >> 1) + 8]; 
  __shared__ float re_shmem[(MAX_LEN_GROUP >> 5)][MAX_HEAD_SIZE + 8]; 
  // 1KB
  __shared__ float block_sum[2];

  // load Q
  *(float4*)(&q_val[0]) = *(float4*)(&Q[q_offset + j]);

  // to execute reduction in the loop
  // float qxk_result = 0.0f;
  float temp_sum_val = 0.0f;

  float a;
  float4 pv[2];

#pragma unroll
  for (int i = 0; i < 2; i++){
    pv[i].x = 0.0f;
    pv[i].y = 0.0f;
    pv[i].z = 0.0f;
    pv[i].w = 0.0f;
  }

  unsigned int k_start = 0;
  unsigned int v_start = 4;

#pragma unroll
  for (int row = kv_row; row < pad_len; row += len_group){

    // Q x K
    *(float4*)(&kv_val[k_start]) = ((row < cur_len) && (len_offset + row < len)) ? 
      *(float4*)(&K[(size_t)(row + len_offset) * kv_stride_seq + kv_offset + j]) : 
      *(float4*)(&PD[0]);
    *(float4*)(&kv_val[v_start]) = ((row < cur_len) && (len_offset + row < len)) ? 
      *(float4*)(&V[(size_t)(row + len_offset) * kv_stride_seq + kv_offset + j]) : 
      *(float4*)(&PD[0]);

    s_temp = __hmul2(q_val[0], kv_val[k_start]);
    s_temp = __hadd2(s_temp, __hmul2(q_val[1], kv_val[k_start + 1]));
    s_temp = __hadd2(s_temp, __hmul2(q_val[2], kv_val[k_start + 2]));
    s_temp = __hadd2(s_temp, __hmul2(q_val[3], kv_val[k_start + 3]));

    temp_sum = __hadd(s_temp.x, s_temp.y);
    temp_sum = warpReduceSum(temp_sum, (hs >> 3));

    // now for all k_col == 0, a temp_s is stored in the register
    // push into share memory

    // softmax without scaling
    if (kv_col == 0){
      if (MASK == MASK_TYPE::ALIBI_MASK) 
        s_shmem[kv_row] = ((row < cur_len) && (len_offset + row < len)) ? 
          __expf(convert_fp16_to_fp32(temp_sum) * p_scale 
                + slope * (len_offset + row) - max_val_alibi) : 0.0f;
      else 
        s_shmem[kv_row] = ((row < cur_len) && (len_offset + row < len)) ? 
          __expf(convert_fp16_to_fp32(temp_sum) * p_scale - max_val) : 0.0f;
    }

    __syncthreads();
    // calculate the intermediate sum
    temp_sum_val += (tid < len_group) ? s_shmem[tid] : 0.0f;

    // AxV
    // a.x = convert_fp32_to_fp16<T>(s_shmem[s_row * (FLOAT_BANK_SIZE + 8) + s_col]);
    // a.y = a.x;
    a = s_shmem[kv_row];

    pv[0].x = pv[0].x + a * convert_fp16_to_fp32(kv_val[v_start].x);
    pv[0].y = pv[0].y + a * convert_fp16_to_fp32(kv_val[v_start].y);
    pv[0].z = pv[0].z + a * convert_fp16_to_fp32(kv_val[v_start + 1].x);
    pv[0].w = pv[0].w + a * convert_fp16_to_fp32(kv_val[v_start + 1].y);
    pv[1].x = pv[1].x + a * convert_fp16_to_fp32(kv_val[v_start + 2].x);
    pv[1].y = pv[1].y + a * convert_fp16_to_fp32(kv_val[v_start + 2].y);
    pv[1].z = pv[1].z + a * convert_fp16_to_fp32(kv_val[v_start + 3].x);
    pv[1].w = pv[1].w + a * convert_fp16_to_fp32(kv_val[v_start + 3].y);

    k_start = 4 - k_start;
    v_start = 4 - v_start;
  }

  unsigned int real_len_group = min(len_group, 32);

  unsigned int o_row = tid % len_group;  
  unsigned int o_col = tid / len_group;

  // calculate the global sum
  temp_sum_val = warpReduceSum(temp_sum_val, real_len_group);

  if (tid % real_len_group == 0 && tid < len_group){
    block_sum[(tid >> 5)] = temp_sum_val; 
  }
  __syncthreads();

  if (tid == 0){
  #pragma unroll
    for (int r = 1; r < (len_group >> 5); r++){
      block_sum[0] += block_sum[r];
    }
    S[(blockIdx.x * hn + blockIdx.y) * SPLIT + blockIdx.z] = block_sum[0];
    // atomicAdd(&S[blockIdx.x * hn + blockIdx.y], block_sum[0]);
  }
  __syncthreads();

  float partial_scaling = __fdividef(1.0f, block_sum[0] + 1e-6f);

  pv[0].x = pv[0].x * partial_scaling;
  pv[0].y = pv[0].y * partial_scaling;
  pv[0].z = pv[0].z * partial_scaling;
  pv[0].w = pv[0].w * partial_scaling;
  pv[1].x = pv[1].x * partial_scaling;
  pv[1].y = pv[1].y * partial_scaling;
  pv[1].z = pv[1].z * partial_scaling;
  pv[1].w = pv[1].w * partial_scaling;

  // transpoisition allows efficient mem access of V matrix
  // first 64
  *(float4*)(&av_shmem[kv_row][(kv_col << 2)]) = *(float4*)(&pv[0]);
  __syncthreads();

  *(float4*)(&pv[0]) = *(float4*)(&av_shmem[o_row][(o_col << 2)]);
  // __syncwarp();

  pv[0].x = warpReduceSum(pv[0].x, real_len_group);
  pv[0].y = warpReduceSum(pv[0].y, real_len_group);
  pv[0].z = warpReduceSum(pv[0].z, real_len_group);
  pv[0].w = warpReduceSum(pv[0].w, real_len_group);

  if (o_row % 32 == 0){
    *(float4*)(&re_shmem[(o_row >> 5)][(o_col << 3)]) = *(float4*)(&pv[0]);
  }

  __syncthreads();

  // another 64
  *(float4*)(&av_shmem[kv_row][(kv_col << 2)]) = *(float4*)(&pv[1]);
  __syncthreads();

  *(float4*)(&pv[1]) = *(float4*)(&av_shmem[o_row][(o_col << 2)]);
  // __syncwarp();

  pv[1].x = warpReduceSum(pv[1].x, real_len_group);
  pv[1].y = warpReduceSum(pv[1].y, real_len_group);
  pv[1].z = warpReduceSum(pv[1].z, real_len_group);
  pv[1].w = warpReduceSum(pv[1].w, real_len_group);

  if (o_row % 32 == 0){
    *(float4*)(&re_shmem[(o_row >> 5)][(o_col << 3) + 4]) = *(float4*)(&pv[1]);
  }

  __syncthreads();

  if (kv_row == 0){
      pv[0].x = re_shmem[0][(kv_col << 3) + 0];
      pv[0].y = re_shmem[0][(kv_col << 3) + 1];
      pv[0].z = re_shmem[0][(kv_col << 3) + 2];
      pv[0].w = re_shmem[0][(kv_col << 3) + 3];
      pv[1].x = re_shmem[0][(kv_col << 3) + 4];
      pv[1].y = re_shmem[0][(kv_col << 3) + 5];
      pv[1].z = re_shmem[0][(kv_col << 3) + 6];
      pv[1].w = re_shmem[0][(kv_col << 3) + 7];
  #pragma unroll
    for (int r = 1; r < (len_group >> 5); r++){
      pv[0].x = pv[0].x + re_shmem[r][(kv_col << 3) + 0];
      pv[0].y = pv[0].y + re_shmem[r][(kv_col << 3) + 1];
      pv[0].z = pv[0].z + re_shmem[r][(kv_col << 3) + 2];
      pv[0].w = pv[0].w + re_shmem[r][(kv_col << 3) + 3];
      pv[1].x = pv[1].x + re_shmem[r][(kv_col << 3) + 4];
      pv[1].y = pv[1].y + re_shmem[r][(kv_col << 3) + 5];
      pv[1].z = pv[1].z + re_shmem[r][(kv_col << 3) + 6];
      pv[1].w = pv[1].w + re_shmem[r][(kv_col << 3) + 7];
    } 
    *(float4*)(&H[q_offset * SPLIT + blockIdx.z * hs + (kv_col << 3)]) = *(float4*)(&pv[0]);
    *(float4*)(&H[q_offset * SPLIT + blockIdx.z * hs + (kv_col << 3) + 4]) = *(float4*)(&pv[1]);
  }
}
