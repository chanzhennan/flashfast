import math
import torch
import torch.nn as nn
from flashfast import decode_serving_mha_ut as decode_serving_mha
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--warmup', type=int, default=0)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--benchmark', default=False, action='store_true')
parser.add_argument('--is_prefill', default=False, action='store_true')
args = parser.parse_args()

MAX_BS = 64
MAX_SEQ_LEN = 4096

### test settings
PREFILL = args.is_prefill
Headnums = [32]
Headdims = [128]
Seqlens = [67, 128, 153, 195, 252, 456, 642, 859, 1023, 1453, 1764, 1999]
Batchsizes = [1, 2, 4, 8, 16, 24, 32, 48, 64] if not PREFILL else [1, 2, 4, 8]
Configs = [[bs, len, hn, hd] for hd in Headdims for hn in Headnums for bs in Batchsizes for len in Seqlens]
if not PREFILL:
    exit()

### benchmark settings
WARM_UP = args.warmup
REP = args.rep
BENCHMARK_FLAG = args.benchmark

### PyTorch implemetation as baseline
def pytorch_attn(q, k, v, scale, mask=None):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    p = torch.matmul(q, k.transpose(2, 3)) * scale
    if mask is not None:
        p = p + mask
    p = torch.softmax(p.float(), dim=-1).half()
    out = torch.matmul(p, v)
    return out.transpose(1, 2)

print("prefill  batch  cache_len  head_num  head_dim  all_close  max_bias  ref_dur(ms)  infini_dur(ms)  speedup")

for BS,LEN,HN,HS in Configs:

    if PREFILL:
        seq_len_list = torch.randint(2, LEN, (BS,), dtype=torch.int32, device="cuda")
        seq_len_sum = torch.sum(seq_len_list, dim=0).item()
        q_context_length = torch.cumsum(seq_len_list, dim=0).to(torch.int32)
        q_context_length = torch.cat((torch.tensor([0], dtype=torch.int32, device="cuda"), q_context_length)).contiguous()
        k_context_length = q_context_length
        block_table = torch.zeros((BS, LEN), dtype=torch.int32, device="cuda")
        q = torch.empty((seq_len_sum, HN, HS), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
        k = torch.empty((seq_len_sum, HN, HS), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
        v = torch.empty((seq_len_sum, HN, HS), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
        o = torch.empty((seq_len_sum, HN, HS), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)

        scale = 1.0 / math.sqrt(HS)
        mask = torch.full((1, 1, LEN, LEN), float("-inf"), device=q.device)
        mask = torch.triu(mask, diagonal=1).type_as(q)

        def pytorch_attn_varlen():
            for i in range(BS):
                o[q_context_length[i]:q_context_length[i+1]] = pytorch_attn(q[q_context_length[i]:q_context_length[i+1]].unsqueeze(0), 
                                                                            k[q_context_length[i]:q_context_length[i+1]].unsqueeze(0), 
                                                                            v[q_context_length[i]:q_context_length[i+1]].unsqueeze(0), 
                                                                            scale, mask[:, :, :seq_len_list[i], :seq_len_list[i]]).squeeze(0)
            return o
        ref_out = pytorch_attn_varlen()
        infini_out = decode_serving_mha(q, k, v, q_context_length, k_context_length, block_table, scale)

        all_close = torch.allclose(ref_out, infini_out, atol=1e-2, rtol=1e-3)
        max_bias = (abs(ref_out - infini_out)).max()

    else:
        pass

    if not BENCHMARK_FLAG:
        print(str(bool(PREFILL)).ljust(len('prefill')) + "  " +
            str(BS).ljust(len('batch')) + "  " +
            str(LEN).ljust(len('cache_len')) + "  " +
            str(HN).ljust(len('head_num')) + "  " +
            str(HS).ljust(len('head_dim')) + "  " +
            str(bool(all_close)).ljust(len('all_close')) + "  " +
            "{:.4f}".format(max_bias).ljust(len('max_bias'))) 
    else:
        if PREFILL:
            ### benchmarking
            for _ in range(WARM_UP):
                _ = pytorch_attn_varlen()

            start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
            end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
            for i in range(REP):
                start_event[i].record()
                _ = pytorch_attn_varlen()
                end_event[i].record()
            torch.cuda.synchronize()
            ref_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)


            for _ in range(WARM_UP):
                _ = decode_serving_mha(q, k, v, q_context_length, k_context_length, block_table, scale)

            start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
            end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
            for i in range(REP):
                start_event[i].record()
                _ = decode_serving_mha(q, k, v, q_context_length, k_context_length, block_table, scale)
                end_event[i].record()
            torch.cuda.synchronize()
            infini_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

        else:
            pass

        print(str(bool(PREFILL)).ljust(len('prefill')) + "  " +
            str(BS).ljust(len('batch')) + "  " +
            str(LEN).ljust(len('cache_len')) + "  " +
            str(HN).ljust(len('head_num')) + "  " +
            str(HS).ljust(len('head_dim')) + "  " +
            str(bool(all_close)).ljust(len('all_close')) + "  " +
            "{:.4f}".format(max_bias).ljust(len('max_bias')) + "  " +
            "{:.4f}".format(torch.mean(ref_dur).item()).ljust(len('ref_dur(ms)')) + "  " +
            "{:.4f}".format(torch.mean(infini_dur).item()).ljust(len('infini_dur(ms)')) +  "  " +
            "{:.4f}".format(torch.mean(ref_dur).item() / torch.mean(infini_dur).item()).ljust(len('speedup'))) 
        