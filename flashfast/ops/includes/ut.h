#pragma once

#include "../../kernels/utils.h"

at::Tensor decode_mha_ut(at::Tensor Q, at::Tensor K, at::Tensor V, 
                        const float scale, const float attn_max, const int len, const int strategy);
// at::Tensor decode_mha_alibi_masked_ut(at::Tensor Q, at::Tensor K, at::Tensor V, at::Tensor alibi_slopes,
//                         const float scale, const float attn_max, const int len, const int strategy);
// at::Tensor decode_mqa_ut(at::Tensor Q, at::Tensor K, at::Tensor V, 
//                         const float scale, const float attn_max, const int len, const int strategy);
// at::Tensor decode_mqa_t_ut(at::Tensor Q, at::Tensor K, at::Tensor V, 
//                         const float scale, const float attn_max, const int len, const int strategy);
// at::Tensor decode_serving_mha_ut(at::Tensor Q, at::Tensor K, at::Tensor V, at::Tensor Q_context_length, at::Tensor K_context_length, 
//                         at::Tensor Block_table, const float scale);

