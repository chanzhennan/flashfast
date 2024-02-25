import math
import torch
import torch.nn as nn
from flashfast import decode_mha_ut as decode_mha
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
Seqlens = [67, 128] #, 153, 195, 252, 456, 642, 859, 1023, 1453, 1764, 1999]
Batchsizes = [1] #, 2, 4, 8, 16, 24, 32, 48, 64] if not PREFILL else [1, 2, 4, 8]
Configs = [[bs, len, hn, hd] for hd in Headdims for hn in Headnums for bs in Batchsizes for len in Seqlens]
Strategies = [0, 1, 2, 3, 4, 5, 6] if not PREFILL else [-1]

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

if not BENCHMARK_FLAG:
    print("prefill  batch  cache_len  head_num  head_dim" + 
          "".join([f"  s{str(strategy)[0]}_all_close  s{str(strategy)[0]}_max_bias" for strategy in Strategies]))
else:
    print("prefill  batch  cache_len  head_num  head_dim  ref_dur(ms)" +
          "".join([f"  s{str(strategy)[0]}_infini_dur(ms)  s{str(strategy)[0]}_speedup" for strategy in Strategies]))

for BS,LEN,HN,HS in Configs:

    QLEN = LEN if PREFILL else 1
    Q = torch.empty((BS, QLEN, HN, HS), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    k = torch.empty((MAX_BS, MAX_SEQ_LEN, HN, HS), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    v = torch.empty((MAX_BS, MAX_SEQ_LEN, HN, HS), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    scale = 1.0 / math.sqrt(HS)
    if QLEN > 1:
        mask = torch.full((1, 1, QLEN, QLEN), float("-inf"), device=Q.device)
        mask = torch.triu(mask, diagonal=1).type_as(Q)
    else:
        mask = None

    ref_out = pytorch_attn(Q.reshape((BS, QLEN, HN, HS)), k[:BS, :LEN], v[:BS, :LEN], scale, mask)
    all_close_list = []
    max_bias_list = []
    
    import pdb
    pdb.set_trace()


    for strategy in Strategies:
        infini_out = decode_mha(Q, k[:BS,:LEN], v[:BS,:LEN], scale, 8.0, LEN, strategy)
        all_close = torch.allclose(ref_out, infini_out.reshape((BS, QLEN, HN, HS)), atol=1e-2, rtol=1e-3)
        max_bias = (abs(ref_out - infini_out.reshape((BS, QLEN, HN, HS)))).max()
        all_close_list.append(all_close)
        max_bias_list.append(max_bias)

    if not BENCHMARK_FLAG:
        print(str(bool(PREFILL)).ljust(len('prefill')) + "  " +
            str(BS).ljust(len('batch')) + "  " +
            str(LEN).ljust(len('cache_len')) + "  " +
            str(HN).ljust(len('head_num')) + "  " +
            str(HS).ljust(len('head_dim')) + 
            "".join(["  " + str(bool(all_close_list[i])).ljust(len(f"s{str(strategy)[0]}_all_close")) + 
                     "  " + "{:.4f}".format(max_bias_list[i]).ljust(len(f"s{str(strategy)[0]}_max_bias")) 
                     for i, strategy in enumerate(Strategies)])
        ) 
    else:
        ### benchmarking
        def benchmark(func, args):
            for _ in range(WARM_UP):
                func(*args)
            start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
            end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
            for i in range(REP):
                start_event[i].record()
                func(*args)
                end_event[i].record()
            torch.cuda.synchronize()
            dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
            return dur
        
        ref_dur = benchmark(pytorch_attn, (Q.reshape((BS, QLEN, HN, HS)), k[:BS, :LEN], v[:BS, :LEN], scale))
        infini_dur_list = []
        for strategy in Strategies:
            infini_dur = benchmark(decode_mha, (Q, k[:BS,:LEN], v[:BS,:LEN], scale, 8.0, LEN, strategy)) 
            infini_dur_list.append(infini_dur)

        print(str(bool(PREFILL)).ljust(len('prefill')) + "  " +
            str(BS).ljust(len('batch')) + "  " +
            str(LEN).ljust(len('cache_len')) + "  " +
            str(HN).ljust(len('head_num')) + "  " +
            str(HS).ljust(len('head_dim')) + "  " +
            "{:.4f}".format(torch.mean(ref_dur).item()).ljust(len('ref_dur(ms)')) +
            "".join(["  " + "{:.4f}".format(torch.mean(infini_dur_list[i]).item()).ljust(len(f"s{str(strategy)[0]}_infini_dur(ms)")) +  
                     "  " + "{:.4f}".format(torch.mean(ref_dur).item() / torch.mean(infini_dur_list[i]).item()).ljust(len(f"s{str(strategy)[0]}_speedup"))
                     for i, strategy in enumerate(Strategies)])
        ) 
        