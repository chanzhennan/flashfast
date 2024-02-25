#pragma once

#include "../../kernels/utils.h"



////////////////////////////// op for PyTorch //////////////////////////////
template <MASK_TYPE MASK, SEQ_DIM_TYPE SEQ_DIM>
void Attention(at::Tensor Q, 
            at::Tensor K, 
            at::Tensor V, 
            c10::optional<at::Tensor> Workspace, 
            c10::optional<at::Tensor> Alibi_slopes, 
            const float scale, 
            const float attn_max, 
            const int strategy, 
            at::Tensor H);

// template <MASK_TYPE MASK>
// void Attention_serving(at::Tensor Q, 
//                     at::Tensor K, 
//                     at::Tensor V, 
//                     c10::optional<at::Tensor> Alibi_slopes, 
//                     c10::optional<at::Tensor> Q_context_length,
//                     c10::optional<at::Tensor> K_context_length,
//                     c10::optional<at::Tensor> Block_table, 
//                     const float scale, 
//                     at::Tensor H);