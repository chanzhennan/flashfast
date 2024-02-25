#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "flash_attn.h"

template <MASK_TYPE MASK = MASK_TYPE::NO_MASK>
void flash_attention(at::Tensor Q, at::Tensor K, at::Tensor V, c10::optional<at::Tensor> Alibi_slopes, c10::optional<at::Tensor> Block_table, const float scale, at::Tensor H) {

    const bool seqlenq_ngroups_swapped = MASK == MASK_TYPE::NO_MASK && Q.size(1) == 1 && Q.size(2) > K.size(2);
    if (seqlenq_ngroups_swapped) H = H.reshape({Q.size(0), K.size(2), Q.size(2) / K.size(2), Q.size(3)}).transpose(1, 2);

    c10::optional<at::Tensor> none = c10::nullopt;
    c10::optional<const at::Tensor> cnone = c10::nullopt;
    c10::optional<at::Tensor> Alibi_slopes_optional = Alibi_slopes;
    c10::optional<at::Tensor> Block_table_optional = Block_table;
    c10::optional<at::Tensor> H_optional = H;

    mha_fwd_kvcache(Q, K, V, cnone, cnone, cnone, cnone, cnone, cnone, Block_table_optional, Alibi_slopes_optional, H_optional, scale, true, -1, -1, true, -10086);
}

// Instantiating template functions explicitly <flash_attention>
#define FLASH_ATTENTION(MASK)                                                                                               \
    template void flash_attention<MASK>(at::Tensor Q, at::Tensor K, at::Tensor V,                                           \
                                        c10::optional<at::Tensor> Alibi_slopes, c10::optional<at::Tensor> Block_table, const float scale, at::Tensor H)
FLASH_ATTENTION(MASK_TYPE::NO_MASK);
FLASH_ATTENTION(MASK_TYPE::ALIBI_MASK);
#undef FLASH_ATTENTION


template <MASK_TYPE MASK = MASK_TYPE::NO_MASK>
void flash_attention_serving(at::Tensor Q, at::Tensor K, at::Tensor V, c10::optional<at::Tensor> Alibi_slopes, 
                            c10::optional<at::Tensor> Q_context_length, c10::optional<at::Tensor> K_context_length, c10::optional<at::Tensor> Block_table, 
                            const float scale, at::Tensor H) {
    
    c10::optional<at::Tensor> none = c10::nullopt;
    c10::optional<const at::Tensor> cnone = c10::nullopt;
    c10::optional<at::Tensor> H_optional = H;
    
    if (Q_context_length.has_value() && K_context_length.has_value() && !Block_table.has_value()) {
        // Prefill. 
        // Q: [num_token_q, num_heads, head_size]
        // K/V: [num_token_k, num_heads, head_size]
        // Alibi_slopes: [num_heads] or [bs x num_heads]
        // Q_context_length/K_context_length: bs+1
        int max_seqlen_q = Q_context_length.value().max().item<int>();
        int max_seqlen_k = K_context_length.value().max().item<int>();
        mha_varlen_fwd(Q, K, V, H_optional, Q_context_length.value(), K_context_length.value(), none, Alibi_slopes, max_seqlen_q, max_seqlen_k, 0.0, scale, false, true, -1, -1, false, c10::nullopt);
    } else if (!Q_context_length.has_value() && !K_context_length.has_value() && Block_table.has_value()) {
        // Decode.
        // Q: [num_blocks, page_block_size, num_heads, head_size]
        // K/V: [num_blocks, page_block_size, num_heads, head_size]
        const bool seqlenq_ngroups_swapped = MASK == MASK_TYPE::NO_MASK && Q.size(1) == 1 && Q.size(2) > K.size(2);
        if (seqlenq_ngroups_swapped) H = H.reshape({Q.size(0), K.size(2), Q.size(2) / K.size(2), Q.size(3)}).transpose(1, 2);
        mha_fwd_kvcache(Q, K, V, cnone, cnone, cnone, cnone, cnone, cnone, Block_table, Alibi_slopes, H_optional, scale, true, -1, -1, true, -10086);
    } else {
        throw std::runtime_error("It is not allowed to provide both Block_table and (Q_context_length, K_context_length) at the same time!");
    }
}

// Instantiating template functions explicitly <flash_attention_serving>
#define FLASH_ATTENTION_SERVING(MASK)                                                                                                                       \
    template void flash_attention_serving<MASK>(at::Tensor Q, at::Tensor K, at::Tensor V, c10::optional<at::Tensor> Alibi_slopes,                           \
                            c10::optional<at::Tensor> Q_context_length, c10::optional<at::Tensor> K_context_length, c10::optional<at::Tensor> Block_table,  \
                            const float scale, at::Tensor H)
FLASH_ATTENTION_SERVING(MASK_TYPE::NO_MASK);
FLASH_ATTENTION_SERVING(MASK_TYPE::ALIBI_MASK);
#undef FLASH_ATTENTION_SERVING