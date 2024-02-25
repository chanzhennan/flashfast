#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>
#include <mma.h>

#include "utils.cuh"

using namespace nvcuda;

template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m8n32k256x8_bz1_kernel(
    const half * __restrict__ a, const half * __restrict__ b, const half * __restrict__ bias, half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ < 800
    return;
#endif
    int bx = blockIdx.x;
    int bz = blockIdx.z;
    int k_start = K / gridDim.z * bz;

    int tid = threadIdx.x;
    int wid = tid >> 5;         // WARP id

    // bx for N, if bx is out of range, return
    if (bx >= N / BN)
        return;

    __shared__ half smem[(BM + BN) * LDK];
    half *s_a = smem;
    half *s_b = smem + BM * LDK;

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 5);      // 0 ~ 7    | 0 1  2 ...  7  
    int load_a_smem_k = (tid & 31) << 3; // 0 ~ 248  | 0 8 16 ... 248(32个数) 
    int load_b_smem_n = (tid >> 5) << 2; // 0 ~ 28   | 0 4  8 ... 28   
    int load_b_smem_k = load_a_smem_k;

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half);
    int load_b_smem_addrs[4];
    #pragma unroll
    for(int i=0; i<4; i++)
        load_b_smem_addrs[i] = s_b_base_addr + OFFSET(load_b_smem_n, load_b_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);

    int load_a_gmem_m = load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = k_start + load_a_smem_k;
    int load_b_gmem_k = k_start + load_b_smem_k;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k, K);

    for (int bk = 0; bk < (K / gridDim.z) / BK; bk++ ) {
        if (load_a_gmem_m < M) {
            asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :
                : "r"(load_a_smem_addr_0),
                    "l"(&a[load_a_gmem_addr]));
        }
        #pragma unroll
        for(int i=0; i<4; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * K]));
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for(int i=0; i<2; i++){
            wmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 128 * i], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[wid * 16 + 128 * i], LDK);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
    }

    wmma::store_matrix_sync(&smem[wid * 8 * LDN], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_m = tid >> 5;   // 0, 1, 2, 3, 4, 5, 6, 7
    int shmem_c_n = tid & 31;   // 0, 1, 2, 3, 4, ..., 31
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(shmem_c_m, bx * BN + shmem_c_n, N);

    
#pragma unroll
    for (int s = 1; s < 8; s++){
        smem[shmem_c_addr] = __hadd(smem[shmem_c_addr], smem[shmem_c_addr + s * 8 * LDN]);
    }

    if (shmem_c_m < M) {
        if (bias != nullptr)
            c[gmem_c_addr] = __hadd(smem[shmem_c_addr], bias[gmem_c_addr % N]);
        else
            c[gmem_c_addr] = smem[shmem_c_addr];
    }
}

/**
 * BM = 16 BN = 16 BK = 256
 * LDK = 256+8 LDN = 16+8
*/
template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m16n16k256x8_bz1_kernel(
    const half * __restrict__ a, const half * __restrict__ b, const half * __restrict__ bias, half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ < 800
    return;
#endif
    int bx = blockIdx.x;        // TILE N
    int by = blockIdx.y;        // TILE M
    int bz = blockIdx.z;        // SPLIT K
    int k_start = 0;

    int tid = threadIdx.x;
    int wid = tid >> 5;         // WARP id

    // bx for N, if bx is out of range, return
    if (bx >= N / BN)
        return;

    __shared__ half smem[(BM + BN) * LDK];
    half *s_a = smem;
    half *s_b = smem + BM * LDK;

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 5) << 1; // 0 ~ 14   | 0 2  4 ... 14  
    int load_a_smem_k = (tid & 31) << 3; // 0 ~ 248  | 0 8 16 ... 248(32个数) 
    int load_b_smem_n = (tid >> 5) << 1; // 0 ~ 14   | 0 2  4 ... 14  
    int load_b_smem_k = load_a_smem_k;

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addrs[2];
    #pragma unroll
    for(int i=0; i<2; i++)
        load_a_smem_addrs[i] = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);
    // int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half);
    int load_b_smem_addrs[2];
    #pragma unroll
    for(int i=0; i<2; i++)
        load_b_smem_addrs[i] = s_b_base_addr + OFFSET(load_b_smem_n, load_b_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = k_start + load_a_smem_k;
    int load_b_gmem_k = k_start + load_b_smem_k;

    int load_a_gmem_addr;
    if(load_a_gmem_m < M)
        load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    else
        load_a_gmem_addr = OFFSET(load_a_gmem_m % M, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k, K);

    // for (int bk = 0; bk < (K / gridDim.z) / BK; bk++) {
    for (int bk = 0; bk < K / BK; bk++) {
        #pragma unroll
        for(int i=0; i<2; i++){
            // if (load_a_gmem_m < M) {
                asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                    : "r"(load_a_smem_addrs[i]), "l"(&a[load_a_gmem_addr + i * K]));
            // }
        }
        #pragma unroll
        for(int i=0; i<2; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * K]));
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for(int i=0; i<2; i++){
            wmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 128 * i], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[wid * 16 + 128 * i], LDK);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
    }

    wmma::store_matrix_sync(&smem[wid * BM * LDN], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    // 对应16*16的shape
    int shmem_c_m = tid >> 4;   // 0, 1, 2, 3, 4, ..., 15  (每个有16个)
    int shmem_c_n = tid & 15;   // 0, 1, 2, 3, 4, ..., 15 （循环16组)
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(by * BM + shmem_c_m, bx * BN + shmem_c_n, N);

    
    #pragma unroll      // 8个warp做的是沿着k维度方向的计算，归约结果
    for (int s = 1; s < 8; s++){
        smem[shmem_c_addr] = __hadd(smem[shmem_c_addr], smem[shmem_c_addr + s * BM * LDN]);
    }

    if ((by * BM + shmem_c_m) < M) {
        if (bias != nullptr)
            c[gmem_c_addr] = __hadd(smem[shmem_c_addr], bias[gmem_c_addr % N]);
        else
            c[gmem_c_addr] = smem[shmem_c_addr];
    }
}



template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m16n32k256x8_bz1_kernel(
    const half * __restrict__ a, const half * __restrict__ b, const half * __restrict__ bias, half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ < 800
    return;
#endif
    int bx = blockIdx.x;
    int bz = blockIdx.z;
    int k_start = K / gridDim.z * bz;

    int tid = threadIdx.x;
    int wid = tid >> 5;         // WARP id

    // bx for N, if bx is out of range, return
    if (bx >= N / BN)
        return;

    __shared__ half smem[(BM + BN) * LDK];
    half *s_a = smem;
    half *s_b = smem + BM * LDK;

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 5) << 1; // 0 ~ 14   | 0 2  4 ... 14  
    int load_a_smem_k = (tid & 31) << 3; // 0 ~ 248  | 0 8 16 ... 248 
    int load_b_smem_n = (tid >> 5) << 2; // 0 ~ 28   | 0 4  8 ... 28   
    int load_b_smem_k = load_a_smem_k;

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addrs[2];
    #pragma unroll
    for (int i=0; i<2; i++){
        load_a_smem_addrs[i] = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);
    }
    int load_b_smem_addrs[4];
    #pragma unroll
    for (int i=0; i<4; i++){
        load_b_smem_addrs[i] = s_b_base_addr + OFFSET(load_b_smem_n, load_b_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);
    }

    int load_a_gmem_m = load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = k_start + load_a_smem_k;
    int load_b_gmem_k = k_start + load_b_smem_k;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k, K);

    for (int bk = 0; bk < (K / gridDim.z) / BK; bk++ ) {
        
        #pragma unroll
        for(int i=0; i<2; i++)
        {   
            if ((load_a_gmem_m + i) < M) {
                asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :
                    : "r"(load_a_smem_addrs[i]), "l"(&a[load_a_gmem_addr + i * K]));
            }
        }

        #pragma unroll
        for(int i=0; i<4; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * K]));
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for(int i=0; i<4; i++){
            wmma::load_matrix_sync(frag_a, &s_a[(wid & 3) * 16 + 64 * i], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[(wid >> 2) * 16 * LDK + (wid & 3) * 16 + 64 * i], LDK);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
    }

    wmma::store_matrix_sync(&smem[(wid & 3) * 16 * LDN + (wid >> 2) * 16], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_m = tid >> 4;           // 0, 1, 2, 3, 4, ..., 15
    int shmem_c_n = (tid & 15) << 1;    // 0, 2, 4, 6, 8, ..., 30
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(shmem_c_m, bx * BN + shmem_c_n, N);

    
#pragma unroll
    for (int s = 1; s < 4; s++){
        *(half2*)(&smem[shmem_c_addr]) = __hadd2(*(half2*)(&smem[shmem_c_addr]), 
            *(half2*)(&smem[shmem_c_addr + s * 16 * LDN]));
    }

    if (shmem_c_m < M) {
        if (bias != nullptr)
            *(half2*)(&c[gmem_c_addr]) = __hadd2(*(half2*)(&smem[shmem_c_addr]), *(half2*)(&bias[gmem_c_addr % N]));
        else
            *(half2*)(&c[gmem_c_addr]) = *(half2*)(&smem[shmem_c_addr]);
    }
}


template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m16n32k256x8_db_kernel(
    const half * __restrict__ a, const half * __restrict__ b, const half * __restrict__ bias, half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ < 800
    return;
#endif
    int bx = blockIdx.x;
    int bz = blockIdx.z;
    int k_start = K / gridDim.z * bz;

    int tid = threadIdx.x;
    int wid = tid >> 5;         // WARP id

    // bx for N, if bx is out of range, return
    if (bx >= N / BN)
        return;

    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + 2 * BM * LDK;
    int s_a_offset = BM * LDK;
    int s_b_offset = BN * LDK;

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 5) << 1; // 0 ~ 14   | 0 2  4 ... 14  
    int load_a_smem_k = (tid & 31) << 3; // 0 ~ 248  | 0 8 16 ... 248 
    int load_b_smem_n = (tid >> 5) << 2; // 0 ~ 28   | 0 4  8 ... 28   
    int load_b_smem_k = load_a_smem_k;

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addrs[2];
    #pragma unroll
    for (int i=0; i<2; i++){
        load_a_smem_addrs[i] = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);
    }
    int load_b_smem_addrs[4];
    #pragma unroll
    for (int i=0; i<4; i++){
        load_b_smem_addrs[i] = s_b_base_addr + OFFSET(load_b_smem_n, load_b_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);
    }

    int load_a_gmem_m = load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = k_start + load_a_smem_k;
    int load_b_gmem_k = k_start + load_b_smem_k;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k, K);

    {
        #pragma unroll
        for(int i=0; i<2; i++)
        {   
            if ((load_a_gmem_m + i) < M) {
                asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :
                    : "r"(load_a_smem_addrs[i]), "l"(&a[load_a_gmem_addr + i * K]));
            }
        }

        #pragma unroll
        for(int i=0; i<4; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * K]));
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }

    #pragma unroll
    for (int bk = 1; bk < (K / gridDim.z) / BK; bk++ ) {

        int smem_sel = (bk & 1) ^ 1;
        int smem_sel_next = ((bk - 1) & 1) ^ 1;

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
        
        #pragma unroll
        for(int i=0; i<2; i++)
        {   
            if ((load_a_gmem_m + i) < M) {
                asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :
                    : "r"(load_a_smem_addrs[i] + smem_sel_next * s_a_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr + i * K]));
            }
        }

        #pragma unroll
        for(int i=0; i<4; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs[i] + smem_sel_next * s_b_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + i * K]));
        }

        for(int i=0; i<4; i++){
            wmma::load_matrix_sync(frag_a, &s_a[smem_sel * s_a_offset + (wid & 3) * 16 + 64 * i], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[smem_sel * s_b_offset + (wid >> 2) * 16 * LDK + (wid & 3) * 16 + 64 * i], LDK);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }

    int smem_sel = (((K / gridDim.z) / BK) & 1) ^ 1;

    for(int i=0; i<4; i++){
        wmma::load_matrix_sync(frag_a, &s_a[smem_sel * s_a_offset + (wid & 3) * 16 + 64 * i], LDK);
        wmma::load_matrix_sync(frag_b, &s_b[smem_sel * s_b_offset + (wid >> 2) * 16 * LDK + (wid & 3) * 16 + 64 * i], LDK);
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    
    __syncthreads();

    wmma::store_matrix_sync(&smem[(wid & 3) * 16 * LDN + (wid >> 2) * 16], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_m = tid >> 4;           // 0, 1, 2, 3, 4, ..., 15
    int shmem_c_n = (tid & 15) << 1;    // 0, 2, 4, 6, 8, ..., 30
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(shmem_c_m, bx * BN + shmem_c_n, N);

    
#pragma unroll
    for (int s = 1; s < 4; s++){
        *(half2*)(&smem[shmem_c_addr]) = __hadd2(*(half2*)(&smem[shmem_c_addr]), 
            *(half2*)(&smem[shmem_c_addr + s * 16 * LDN]));
    }

    if (shmem_c_m < M) {
        if (bias != nullptr)
            *(half2*)(&c[gmem_c_addr]) = __hadd2(*(half2*)(&smem[shmem_c_addr]), *(half2*)(&bias[gmem_c_addr % N]));
        else
            *(half2*)(&c[gmem_c_addr]) = *(half2*)(&smem[shmem_c_addr]);
    }
}


