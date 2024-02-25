#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "../op.h"


at::Tensor decode_mha_ut(at::Tensor Q, at::Tensor K, at::Tensor V, 
                        const float scale, const float attn_max, const int len, const int strategy) {
    // Q: [bs, seqlen, hn, hs]
    // K: [..., hn, hs]
    // V: [..., hn, hs]

    int bs = Q.size(0);
    int seqlen = Q.size(1);
    int hn = K.size(-2);
    int hs = K.size(-1);
    int dim = hn * hs;

    int kv_stride_bs = K.stride(0);
    int kv_stride_seq = K.stride(1);

    at::Tensor H = torch::empty({bs, seqlen, hn, hs}, 
        at::device(Q.device()).dtype(Q.dtype()));
    
    at::Tensor Workspace = torch::empty({bs, hn, strategy > 1 ? strategy : 0, hs + 1}, 
        at::device(Q.device()).dtype(at::ScalarType::Float));

    c10::optional<at::Tensor> none = c10::nullopt;
    Attention<MASK_TYPE::NO_MASK, SEQ_DIM_TYPE::SECOND>(Q, K, V, strategy > 1 ? Workspace : none, none, scale, attn_max, strategy, H);

    return H;
}

// at::Tensor decode_mha_alibi_masked_ut(at::Tensor Q, at::Tensor K, at::Tensor V, at::Tensor alibi_slopes,
//                         const float scale, const float attn_max, const int len, const int strategy) {
//     // Q: [bs, seqlen, hn, hs]
//     // K: [..., hn, hs]
//     // V: [..., hn, hs]
//     // alibi_slopes: [hn]

//     int bs = Q.size(0);
//     int seqlen = Q.size(1);
//     int hn = K.size(-2);
//     int hs = K.size(-1);
//     int dim = hn * hs;

//     int kv_stride_bs = K.stride(0);
//     int kv_stride_seq = K.stride(1);

//     at::Tensor H = torch::empty({bs, seqlen, hn, hs}, 
//         at::device(Q.device()).dtype(Q.dtype()));
    
//     at::Tensor Workspace = torch::empty({bs, hn, strategy > 1 ? strategy : 0, hs + 1}, 
//         at::device(Q.device()).dtype(at::ScalarType::Float));

//     c10::optional<at::Tensor> none = c10::nullopt;
//     Attention<MASK_TYPE::ALIBI_MASK, SEQ_DIM_TYPE::SECOND>(Q, K, V, strategy > 1 ? Workspace : none, alibi_slopes, scale, attn_max, strategy, H);

//     return H;
// }

// at::Tensor decode_mqa_ut(at::Tensor Q, at::Tensor K, at::Tensor V, 
//                         const float scale, const float attn_max, const int len, const int strategy) {
//     // Q: [bs, seqlen, hn, hs]
//     // K: [..., hn_kv, hs]
//     // V: [..., hn_kv, hs]

//     int bs = Q.size(0);
//     int seqlen = Q.size(1);
//     int hn = Q.size(-2);
//     int hn_kv = K.size(-2);
//     int hs = Q.size(-1);
//     int ngroups = hn / hn_kv;

//     int kv_stride_bs = K.stride(0);
//     int kv_stride_seq = K.stride(1);

//     at::Tensor H = torch::empty({bs, seqlen, hn, hs}, 
//         at::device(Q.device()).dtype(Q.dtype()));
    
//     at::Tensor Workspace = torch::empty({bs, hn, strategy > 1 ? strategy : 0, hs + 1}, 
//         at::device(Q.device()).dtype(at::ScalarType::Float));

//     c10::optional<at::Tensor> none = c10::nullopt;
//     Attention<MASK_TYPE::NO_MASK, SEQ_DIM_TYPE::SECOND>(Q, K, V, strategy > 1 ? Workspace : none, none, scale, attn_max, strategy, H);

//     return H;
// }

// at::Tensor decode_mqa_t_ut(at::Tensor Q, at::Tensor K, at::Tensor V, 
//                         const float scale, const float attn_max, const int len, const int strategy) {
//     // Q: [seqlen, bs, hn, hs]
//     // K: [..., hn_kv, hs]
//     // V: [..., hn_kv, hs]

//     int bs = Q.size(1);
//     int seqlen = Q.size(0);
//     int hn = Q.size(-2);
//     int hn_kv = K.size(-2);
//     int hs = Q.size(-1);
//     int ngroups = hn / hn_kv;

//     int kv_stride_bs = K.stride(1);
//     int kv_stride_seq = K.stride(0);

//     at::Tensor H = torch::empty({seqlen, bs, hn, hs}, 
//         at::device(Q.device()).dtype(Q.dtype()));
    
//     at::Tensor Workspace = torch::empty({bs, hn, strategy > 1 ? strategy : 0, hs + 1}, 
//         at::device(Q.device()).dtype(at::ScalarType::Float));

//     c10::optional<at::Tensor> none = c10::nullopt;
//     Attention<MASK_TYPE::NO_MASK, SEQ_DIM_TYPE::FIRST>(Q, K, V, strategy > 1 ? Workspace : none, none, scale, attn_max, strategy, H);

//     return H;
// }

// at::Tensor decode_serving_mha_ut(at::Tensor Q, at::Tensor K, at::Tensor V, at::Tensor Q_context_length, at::Tensor K_context_length, 
//                         at::Tensor Block_table, const float scale) {

//     bool prefill = Q.dim() == 3 ? true : false;

//     if (prefill) {
//         int tokens = Q.size(0);
//         int hn = Q.size(1);
//         int hs = Q.size(2);
//         at::Tensor H = torch::empty({tokens, hn, hs}, 
//             at::device(Q.device()).dtype(Q.dtype()));

//         c10::optional<at::Tensor> none = c10::nullopt;
//         Attention_serving<MASK_TYPE::NO_MASK>(Q, K, V, none, Q_context_length, K_context_length, none, scale, H);

//         return H;
//     } else {
//         // error here
//         int bs = Q.size(0);
//         int seqlen = Q.size(1);
//         int hn = Q.size(2);
//         int hs = Q.size(3);
//         at::Tensor H = torch::empty({bs, seqlen, hn, hs}, 
//             at::device(Q.device()).dtype(Q.dtype()));

//         c10::optional<at::Tensor> none = c10::nullopt;
//         Attention_serving<MASK_TYPE::NO_MASK>(Q, K, V, none, none, none, Block_table, scale, H);

//         return H;
//     }
// }
