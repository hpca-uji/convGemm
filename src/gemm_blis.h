/**
 * This file is part of convGemmNHWC
 *
 * Copyright (C) 2021-22 Universitat Politècnica de València and
 *                       Universitat Jaume I
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include <time.h>
#include <stdbool.h>
#include <blis/blis.h>

#define min(a, b) (((a)<(b))?(a):(b))
#define max(a, b) (((a)>(b))?(a):(b))

#define Acol(a1, a2)  A[ (a2)*(ldA)+(a1) ]
#define Bcol(a1, a2)  B[ (a2)*(ldB)+(a1) ]
#define Ccol(a1, a2)  C[ (a2)*(ldC)+(a1) ]
#define Mcol(a1, a2)  M[ (a2)*(ldM)+(a1) ]

#define Arow(a1, a2)  A[ (a1)*(ldA)+(a2) ]
#define Brow(a1, a2)  B[ (a1)*(ldB)+(a2) ]
#define Crow(a1, a2)  C[ (a1)*(ldC)+(a2) ]
#define Mrow(a1, a2)  M[ (a1)*(ldM)+(a2) ]

#ifdef BENCHMARK
extern double t_pack, t_kernel, t_generic;
#define BEGIN_TIMER { double t1 = get_time();
#define END_TIMER(t) double t2 = get_time(); t += t2 - t1; }
#define END_BEGIN_TIMER(t) { double t3 = get_time(); t += t3 - t1; t1 = t3; }
#else
#define BEGIN_TIMER
#define END_TIMER(t)
#define END_BEGIN_TIMER(t)
#endif

extern cntx_t *blis_cntx;
extern sgemm_ukr_ft blis_gemm_kernel;
extern int blis_abi_version;

typedef struct {
    int batches, height, width, channels, kn, kheight, kwidth;
    int vstride, hstride, vpadding, hpadding, vdilation, hdilation, oheight, owidth;
    const float *bias_vector;
    const float *running_mean;
    const float *inv_std;
    const float *gamma;
    const float *beta;
    bool relu;
} conv_p;

void gemm_blis_init();

static inline void gemm_kernel(dim_t m, dim_t n, dim_t k, float *restrict alpha, float *restrict a, float *restrict b,
                        float *restrict beta, float *restrict c, inc_t rs_c0, inc_t cs_c0,
                        auxinfo_t *restrict data, cntx_t *restrict cntx) {
#if BLIS_ABI_VERSION == 3
    blis_gemm_kernel(k, alpha, a, b, beta, c, rs_c0, cs_c0, data, cntx);
#elif BLIS_ABI_VERSION == 4
    blis_gemm_kernel(m, n, k, alpha, a, b, beta, c, rs_c0, cs_c0, data, cntx);
#else
    printf("BLIS_ABI_VERSION %d is not supported yet!\n", BLIS_ABI_VERSION);
    exit(2);
#endif
}

typedef void (*pack_func)(char orderM, char transM, int mc, int nc, const float *M, int ldM, float *Mc, int RR,
                          const conv_p *conv_p, int start_row, int start_col);

typedef void (*post_func)(int mr, int nr, const float *Cc, int ldCc, float beta, float *C, int ldC,
                          const conv_p *conv_p, int start_row, int start_col, bool last);

static inline double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

void gemm_blis_B3A2C0(char orderA, char orderB, char orderC, char transA, char transB, int m, int n, int k, float alpha,
                      const float *A, int ldA, const float *B, int ldB, float beta,
                      float *C, int ldC, float *Ac, pack_func pack_RB, float *Bc, pack_func pack_CB,
                      post_func postprocess, cntx_t *cntx, const conv_p *conv_params);

void gemm_blis_A3B2C0(char orderA, char orderB, char orderC, char transA, char transB, int m, int n, int k, float alpha,
                      const float *A, int ldA, const float *B, int ldB, float beta,
                      float *C, int ldC, float *Ac, pack_func pack_RB, float *Bc, pack_func pack_CB,
                      post_func postprocess, cntx_t *cntx, const conv_p *conv_params);

void
gemm_blis_B3A2C0_orig(char orderA, char orderB, char orderC, char transA, char transB, int m, int n, int k, float alpha,
                      const float *A, int ldA, const float *B, int ldB,
                      float beta, float *C, int ldC, float *Ac, pack_func pack_RB, float *Bc, pack_func pack_CB,
                      post_func postprocess, cntx_t *cntx,
                      const conv_p *conv_params);

void gemm_base_Cresident(char, int, int, int, float, const float *, int, const float *, int, float, float *, int);

void pack_RB(char, char, int, int, const float *, int, float *, int, const conv_p *, int, int);

void
pack_CB(char orderM, char transM, int mc, int nc, const float *M, int ldM, float *Mc, int RR, const conv_p *conv_params,
        int start_row, int start_col);

void unpack_RB(char, char, int, int, float *, int, const float *, int);

void unpack_CB(char, char, int, int, float *, int, const float *, int);

void sxpbyM(int, int, const float *, int, float, float *, int);

/**
 * Estimates the dimension of the panels that will fit into a given level of the cache hierarchy following the
 * principles in the paper "Analytical modeling is enough for high performance BLIS" by T. M. Low et al, 2016.
 *
 * Rule of thumb: subtract 1 line from WL (associativity), which is dedicated to the operand which does not reside in
 * the cache, and distribute the rest between the two other operands proportionally to the ratio n/m  For example,
 * with the conventional algorithm B3A2B1C0 and the L1 cache,  1 line is dedicated to Cr (non-resident in cache)
 * while the remaining lines are distributed  between Ar (mr x kc) and Br (kc x nr) proportionally to the ratio nr/mr
 * to estimate kc.
 *
 * @param NL Number of sets
 * @param CL Bytes per line
 * @param WL Associativity degree
 * @param Sdata Bytes per element (e.g., 8 for FP64)
 * @param m First dimension of block in a higher level of cache
 * @param n Second dimension of block in a higher level of cache
 * @return k, such that a block of size k x n will stays in this level of the cache
 */
static inline int model_level(double NL, double CL, double WL, double Sdata, double m, double n) {
    double CAr = floor((WL - 1) / (1 + n / m)); // Lines of each set for Ar
    if (CAr == 0) { // Special case
        CAr = 1;
        // CBr = WL-2;
    } else {
        // CBr = ceil( ( n / m ) * CAr ); // Lines of each set for Br
    }
    return (int) floor(CAr * NL * CL / (m * Sdata));
}

static inline void gemm_blis_workspace(cntx_t *cntx, int m, int n, int k, int *MC_bs, int *NC_bs, int *KC_bs) {
#if 1
    *MC_bs = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MC, cntx);
    *NC_bs = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NC, cntx);
    *KC_bs = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_KC, cntx);
#else
    // *NC_bs = 3072;
    // *KC_bs = 368; //640
    // *MC_bs = 560; //120
    int MR = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MR, cntx);
    int NR = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NR, cntx);
    int Sdata=4;
    int SL1=64*1024, WL1=4, NL1 = 256, CL1 = SL1 / (WL1 * NL1);
    int SL2=1*1024*1024, WL2=16, NL2 = 2048, CL2 = SL2 / (WL2 * NL2);
    int SL3=4*1024*1024, WL3=16, NL3 = 4096, CL3 = SL3 / (WL3 * NL3);

    *KC_bs = model_level(NL1, CL1, WL1, Sdata, MR, NR);
    if (k > 0 && *KC_bs > k) *KC_bs = k;

    *MC_bs = model_level(NL2, CL2, WL2, Sdata, *KC_bs, NR);
    if (m > 0 && *MC_bs > m) *MC_bs = m;
    *MC_bs = *MC_bs - *MC_bs % MR;
    if (*MC_bs < MR) *MC_bs = MR;

    *NC_bs = model_level(NL3, CL3, WL3, Sdata, *KC_bs, *MC_bs);
    // if (n > 0 && *NC_bs > n) *NC_bs = n;
    // *MC_bs = 448; *NC_bs = 1020; *KC_bs = 512;
    // printf("m=%d n=%d k=%d MC_bs=%d NC_bs=%d KC_bs=%d\n", m, n, k, *MC_bs, *NC_bs, *KC_bs);
#endif
}

static inline void gemm_blis_blocks_sizes(int m, int n, int k, int *MR_bs, int *NR_bs, int *MC_bs, int *NC_bs, int *KC_bs) {
    gemm_blis_init();
    *MR_bs = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MR, blis_cntx);
    *NR_bs = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NR, blis_cntx);
    gemm_blis_workspace(blis_cntx, m, n, k, MC_bs, NC_bs, KC_bs);
    *MC_bs -= *MC_bs % *MR_bs;
    *NC_bs -= *NC_bs % *NR_bs;
}
