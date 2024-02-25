#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "utils.cuh"



/*
    For llama, rope is used for all of every head. 
    For chatglm, rope is only used for half of every head. 
*/
template <ROPE_TYPE ROPE, FREQ_ALIGNED ALIGN = FREQ_ALIGNED::YES, typename T = half>
__global__ __forceinline__ void update_kv_cache_rope(T *q_src, int src_q_stride_bs, int src_q_stride_seq,
                                     T *k_src, int src_k_stride_bs, int src_k_stride_seq,
                                     T *v_src, int src_v_stride_bs, int src_v_stride_seq,
                                     float *f, int* f_offsets,
                                     T *q_dst, int q_stride_bs, int q_stride_seq, 
                                     T *k_dst, int k_stride_bs, int k_stride_seq, 
                                     T *v_dst, int v_stride_bs, int v_stride_seq, 
                                     int start_len, int hn, int hn_kv, int hs) {

        int batch_id = blockIdx.y;
        int seq_id = blockIdx.x;
        int head_id = blockIdx.z;
        int hs_id = threadIdx.x << 1;

        int dim_id = head_id < hn ? head_id * hs + hs_id :
                    head_id < hn + hn_kv ? (head_id - hn) * hs + hs_id :
                    (head_id - hn - hn_kv) * hs + hs_id;

        if (ROPE == ROPE_TYPE::HALF_ROPE) {
            hs = hs >> 1;
        }

        float* freq;
        if (ALIGN)
            freq = &f[(start_len + seq_id) * hs];
        else
            freq = &f[f_offsets[batch_id + seq_id] * hs]; // If not aligned, one of batch_id and seq_id is 0

        if (head_id < hn){
            // Q
            size_t q_src_index = batch_id * src_q_stride_bs + seq_id * src_q_stride_seq + dim_id;
            size_t q_dst_index = batch_id * q_stride_bs + seq_id * q_stride_seq + dim_id;

            if ((hs_id >= hs && ROPE == ROPE_TYPE::HALF_ROPE) || ROPE == ROPE_TYPE::NO_ROPE){
                *(float *)(&q_dst[q_dst_index]) = *(float *)(&q_src[q_src_index]);
            }
            else{
                float q_val[2], q_res[2], f_val[2];
                q_val[0] = convert_fp16_to_fp32(q_src[q_src_index]);
                q_val[1] = convert_fp16_to_fp32(q_src[q_src_index + 1]);
                f_val[0] = (freq[hs_id]);
                f_val[1] = (freq[hs_id + 1]);
                q_res[0] = q_val[0] * f_val[0] - q_val[1] * f_val[1];
                q_res[1] = q_val[0] * f_val[1] + q_val[1] * f_val[0];
                q_dst[q_dst_index] = convert_fp32_to_fp16_rn<T>(q_res[0]);
                q_dst[q_dst_index + 1] = convert_fp32_to_fp16_rn<T>(q_res[1]);
            }
        }
        else if (head_id < hn + hn_kv){
            // K
            size_t k_src_index = batch_id * src_k_stride_bs + seq_id * src_k_stride_seq + dim_id;
            size_t k_dst_index = batch_id * k_stride_bs + ((size_t)start_len + seq_id) * k_stride_seq + dim_id;

            if ((hs_id >= hs && ROPE == ROPE_TYPE::HALF_ROPE) || ROPE == ROPE_TYPE::NO_ROPE){
                *(float *)(&k_dst[k_dst_index]) = *(float *)(&k_src[k_src_index]);
            }
            else{
                float k_val[2], k_res[2], f_val[2];
                k_val[0] = convert_fp16_to_fp32(k_src[k_src_index]);
                k_val[1] = convert_fp16_to_fp32(k_src[k_src_index + 1]);
                f_val[0] = (freq[hs_id]);
                f_val[1] = (freq[hs_id + 1]);
                k_res[0] = k_val[0] * f_val[0] - k_val[1] * f_val[1];
                k_res[1] = k_val[0] * f_val[1] + k_val[1] * f_val[0];
                k_dst[k_dst_index] = convert_fp32_to_fp16_rn<T>(k_res[0]);
                k_dst[k_dst_index + 1] = convert_fp32_to_fp16_rn<T>(k_res[1]);
            }
        }
        else if (head_id < hn + hn_kv * 2){
            // V
            size_t v_src_index = batch_id * src_v_stride_bs + seq_id * src_v_stride_seq + dim_id;
            size_t v_dst_index = batch_id * v_stride_bs + ((size_t)start_len + seq_id) * v_stride_seq + dim_id;

            *(float *)(&v_dst[v_dst_index]) = *(float *)(&v_src[v_src_index]);
        }
}





/*
    Move k/v to KVcache with any shape. 
*/
template <typename T = half>
__global__ __forceinline__ void update_cache(T* src, T* dst, int src_stride_bs, int src_stride_seq, int dst_stride_bs, int dst_stride_seq){
    int dim_id = blockIdx.y * blockDim.x + threadIdx.x;
    int bs_id = blockIdx.z;
    size_t seq_id = blockIdx.x;

    float4* src_f4 = (float4*)src;
    float4* dst_f4 = (float4*)dst;
    dst_f4[bs_id * dst_stride_bs + seq_id * dst_stride_seq + dim_id] = src_f4[bs_id * src_stride_bs + seq_id * src_stride_seq + dim_id];
}

/*
    Set bias to O.
    Here, m & n are the shape of O for fused QKV, not the shape of Q/K/V
*/
template <typename T = half>
__global__ __forceinline__ void set_bias(T* O, T* bias, int m, int n){
    int rid = blockIdx.x;
    int cid = blockIdx.y * blockDim.x + threadIdx.x;

    if (cid >= n) return;

    float4* O_f4 = (float4*)O;
    float4* bias_f4 = (float4*)bias;
    O_f4[rid * n + cid] = bias_f4[cid];
}