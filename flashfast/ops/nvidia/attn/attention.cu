#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "../../../kernels/nvidia/decode_attn.cuh"
// #include "../../../kernels/nvidia/flash_attn/flash_attn.h"



#define CASE_SPLIT_KV(NUM_SPLIT_KV)                                                                                                                     \
    decode_splitKV_kernel<1024, 4, NUM_SPLIT_KV, MASK><<<dim3(bs, hn, NUM_SPLIT_KV), dim3(1024)>>>(                                                     \
        Q, K, V, alibi_slopes, scale, attn_max, hn, hn_kv, hs, len, kv_stride_bs, kv_stride_seq, DIV_UP(DIV_UP(len, NUM_SPLIT_KV), 1024), S, T);        \
    decode_splitKV_scaling_kernel<NUM_SPLIT_KV><<<dim3(bs, hn), dim3(hs)>>>(T, dim, hs, hn, S, H);                                                      \
    break;

template <MASK_TYPE MASK>
void decode_attention(half* Q, half* K, half* V, float* workspace, float* alibi_slopes, const float scale, const float attn_max, 
            const int bs, const int seqlen, const int len, const int hn, const int hn_kv, const int hs, const int kv_stride_bs, const int kv_stride_seq, const int strategy, 
            half* H){
    int dim = hn * hs;
    if (strategy == 0){
        std::cout << "called decode_fall_back_kernel here " << std::endl;
        decode_fall_back_kernel<1024, 192, MASK><<<dim3(bs, hn, 1), dim3(1024)>>>(
            Q, K, V, 
            alibi_slopes, scale, hn, hn_kv, hs, len, kv_stride_bs, kv_stride_seq, DIV_UP(len, 1024), 
            H
        );
    } else if (strategy == 1){
        decode_async_softmax_kernel<1024, 4, MASK><<<dim3(bs, hn, 1), dim3(1024)>>>(
            Q, K, V, 
            alibi_slopes, scale, attn_max, hn, hn_kv, hs, len, kv_stride_bs, kv_stride_seq, DIV_UP(len, 1024), 
            H
        );
    } else if (strategy > 1){
        float* S = workspace; // size: bs * hn * strategy
        float* T = workspace + bs * hn * strategy; // size: bs * hn * strategy * hs
        switch(strategy) {
            case 1: CASE_SPLIT_KV(1)
            case 2: CASE_SPLIT_KV(2)
            case 3: CASE_SPLIT_KV(3)
            case 4: CASE_SPLIT_KV(4)
            case 5: CASE_SPLIT_KV(5)
            case 6: CASE_SPLIT_KV(6)
            case 7: CASE_SPLIT_KV(7)
            case 8: CASE_SPLIT_KV(8)
            case 9: CASE_SPLIT_KV(9)
            case 10: CASE_SPLIT_KV(10)
            case 11: CASE_SPLIT_KV(11)
            case 12: CASE_SPLIT_KV(12)
            case 13: CASE_SPLIT_KV(13)
            case 14: CASE_SPLIT_KV(14)
            case 15: CASE_SPLIT_KV(15)
            case 16: CASE_SPLIT_KV(16)
            default: throw std::invalid_argument("The given num_split is not supported!");
        }
    }            
}

#undef CASE_SPLIT_KV

// Instantiating template functions explicitly <decode_attention>
#define DECODE_ATTENTION(MASK)                                                                                                                                                          \
    template void decode_attention<MASK>(half* Q, half* K, half* V, float* workspace, float* alibi_slopes, const float scale, const float attn_max,                                     \
            const int bs, const int seqlen, const int len, const int hn, const int hn_kv, const int hs, const int kv_stride_bs, const int kv_stride_seq, const int strategy, half* H)
DECODE_ATTENTION(MASK_TYPE::NO_MASK);
// DECODE_ATTENTION(MASK_TYPE::ALIBI_MASK);
#undef DECODE_ATTENTION


template <MASK_TYPE MASK, SEQ_DIM_TYPE SEQ_DIM>
void Attention(at::Tensor Q, at::Tensor K, at::Tensor V, c10::optional<at::Tensor> Workspace, c10::optional<at::Tensor> Alibi_slopes, 
                const float scale, const float attn_max, const int strategy, at::Tensor H) {
    // Q: [bs, seqlen, hn, hs], must be contiguous
    // K & V: [bs, len, hn_kv, hs]

    int bs = SEQ_DIM == SEQ_DIM_TYPE::SECOND ? Q.size(0) : Q.size(1);
    int seqlen = SEQ_DIM == SEQ_DIM_TYPE::SECOND ? Q.size(1) : Q.size(0);
    int len = SEQ_DIM == SEQ_DIM_TYPE::SECOND ? K.size(1) : K.size(0);
    int hn = Q.size(2);
    int hn_kv = K.size(2);
    int hs = Q.size(3);
    int kv_stride_bs = SEQ_DIM == SEQ_DIM_TYPE::SECOND ? K.stride(0) : K.stride(1);
    int kv_stride_seq = SEQ_DIM == SEQ_DIM_TYPE::SECOND ? K.stride(1) : K.stride(0);

    // Transpose to [bs, seq, hn, hs] for flash attention
    // Do not effect the decode attention for indexing kvcache with kv_stride and stride_seq == stride_bs for Q and H 
    if (SEQ_DIM == SEQ_DIM_TYPE::FIRST) {
        Q = Q.transpose(0, 1);
        K = K.transpose(0, 1);
        V = V.transpose(0, 1);
        H = H.transpose(0, 1);
    }

    int run_strategy;
    if (seqlen > 1) run_strategy = -1;
    else run_strategy = strategy;

    float* workspace = Workspace.has_value() ? reinterpret_cast<float *>(Workspace.value().data_ptr<float>()) : nullptr;
    float* alibi_slopes;
    if (MASK == MASK_TYPE::ALIBI_MASK) {
        alibi_slopes = Alibi_slopes.value().data_ptr<float>();
    }
    else {
        alibi_slopes = nullptr;
    }
        
    if (run_strategy >= 0)
        decode_attention<MASK>(
            reinterpret_cast<half *>(Q.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(K.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(V.data_ptr<at::Half>()), 
            workspace, 
            alibi_slopes,
            scale, attn_max, bs, seqlen, len, hn, hn_kv, hs, kv_stride_bs, kv_stride_seq, strategy, 
            reinterpret_cast<half *>(H.data_ptr<at::Half>())
        );
    else {
        throw std::runtime_error("unspport flash_attention");
        // flash_attention<MASK>(Q, K, V, Alibi_slopes, c10::nullopt, scale, H);
    }
        
}



// Instantiating template functions explicitly <attention>
#define ATTENTION(MASK, SEQ_DIM)                                                                                                                                    \
    template void Attention<MASK, SEQ_DIM>(at::Tensor Q, at::Tensor K, at::Tensor V, c10::optional<at::Tensor> Workspace, c10::optional<at::Tensor> Alibi_slopes,   \
                const float scale, const float attn_max, const int strategy, at::Tensor H)
// ATTENTION(MASK_TYPE::NO_MASK, SEQ_DIM_TYPE::FIRST);
// ATTENTION(MASK_TYPE::ALIBI_MASK, SEQ_DIM_TYPE::FIRST);
ATTENTION(MASK_TYPE::NO_MASK, SEQ_DIM_TYPE::SECOND);
// ATTENTION(MASK_TYPE::ALIBI_MASK, SEQ_DIM_TYPE::SECOND);
#undef ATTENTION




// template <MASK_TYPE MASK>
// void Attention_serving(at::Tensor Q, at::Tensor K, at::Tensor V, c10::optional<at::Tensor> Alibi_slopes, 
//                     c10::optional<at::Tensor> Q_context_length, c10::optional<at::Tensor> K_context_length, c10::optional<at::Tensor> Block_table, 
//                     const float scale,  at::Tensor H) {

//     throw std::runtime_error("unspport flash_attention_serving");
//     // flash_attention_serving<MASK>(Q, K, V, Alibi_slopes, Q_context_length, K_context_length, Block_table, scale, H);
// }

// // Instantiating template functions explicitly <attention_serving>
// #define ATTENTION_SERVING(MASK)                                                                                                                             \
//     template void Attention_serving<MASK>(at::Tensor Q, at::Tensor K, at::Tensor V, c10::optional<at::Tensor> Alibi_slopes,                                 \
//                             c10::optional<at::Tensor> Q_context_length, c10::optional<at::Tensor> K_context_length, c10::optional<at::Tensor> Block_table,  \
//                             const float scale, at::Tensor H)
// ATTENTION_SERVING(MASK_TYPE::NO_MASK);
// ATTENTION_SERVING(MASK_TYPE::ALIBI_MASK);
// #undef ATTENTION_SERVING