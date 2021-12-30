#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#include <omp.h>
#define BLIS_DISABLE_BLAS_DEFS
#include <blis.h>

#include "convGemm.h"
#include "gemm_blis.h"

void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, const float *a, int *lda, const float *b, int *ldb, float *beta, float *c, int *ldc);

static inline void sgemm(char transa, char transb, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb, float beta, float *c, int ldc) {
    sgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void fill_rand(float *a, int n)
{
    for (int i = 0; i < n; i++)
        a[i] = rand() * 1.0 / RAND_MAX;
}

float diff(int n, const float *Cref, const float *C)
{
    float maxdiff = 0.0, cref_diff = 0.0, c_diff = 0.0;
    for (int i = 0; i < n; i++) {
        float d = fabs(Cref[i] - C[i]);
        if (d > maxdiff) {
            maxdiff = d;
            cref_diff = Cref[i];
            c_diff = C[i];
        }
    }
    // printf("=== %e %e %e ===\n", cref_diff, c_diff, maxdiff);
    if (cref_diff == 0.0) return maxdiff;
    else return maxdiff / cref_diff;
}

int main(int argc, char *argv[])
{
    int m, n, k, rep;
    if (strcmp(argv[1], "nchw") == 0) {
        rep = atoi(argv[2]);
        int b   = atoi(argv[3]);
        int h   = atoi(argv[4]);
        int w   = atoi(argv[5]);
        int c   = atoi(argv[6]);
        int kn  = atoi(argv[7]);
        int kh  = atoi(argv[8]);
        int kw  = atoi(argv[9]);
        int vpadding  = argc > 10 ? atoi(argv[10]) : 1;
        int hpadding  = argc > 11 ? atoi(argv[11]) : 1;
        int vstride   = argc > 12 ? atoi(argv[12]) : 1;
        int hstride   = argc > 13 ? atoi(argv[13]) : 1;
        int vdilation = argc > 14 ? atoi(argv[14]) : 1;
        int hdilation = argc > 15 ? atoi(argv[15]) : 1;
        int ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) / vstride + 1;
        int wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) / hstride + 1;
        m = ho * wo * b, n = kn, k = kh * kw * c;
    } else if (argc == 5) {
        m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]), rep = atoi(argv[4]);
    } else {
        fprintf(stderr, "Missing arguments: M N K rep\n");
        return 1;
    }

    bli_init();
    cntx_t* cntx = bli_gks_query_cntx();
    sgemm_ukr_ft gemm_kernel = bli_cntx_get_l3_nat_ukr_dt      (BLIS_FLOAT, BLIS_GEMM, cntx);
    int MR                   = bli_cntx_get_blksz_def_dt       (BLIS_FLOAT, BLIS_MR,   cntx);
    int NR                   = bli_cntx_get_blksz_def_dt       (BLIS_FLOAT, BLIS_NR,   cntx);
    // int PACKMR               = bli_cntx_get_blksz_max_dt       (BLIS_FLOAT, BLIS_MR,   cntx);
    // int PACKNR               = bli_cntx_get_blksz_max_dt       (BLIS_FLOAT, BLIS_NR,   cntx);
    // bool row_pref            = bli_cntx_get_l3_nat_ukr_prefs_dt(BLIS_FLOAT, BLIS_GEMM, cntx);
    int MC, NC, KC;
    gemm_blis_workspace(cntx, m, n, k, &MC, &NC, &KC);
    // BLIS_POOL_ADDR_ALIGN_SIZE, KR
    printf("# MR = %d NR = %d MC = %d NC = %d KC = %d\n", MR, NR, MC, NC, KC);

    int NC2 = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NC, cntx);
    int MC2 = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MC, cntx);
    int KC2 = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_KC, cntx);
    printf("# MR = %d NR = %d BLIS_MC = %d BLIS_NC = %d BLIS_KC = %d\n", MR, NR, MC2, NC2, KC2);
    NC = NC > NC2 ? NC : NC2;
    MC = MC > MC2 ? MC : MC2;
    KC = KC > KC2 ? KC : KC2;

    bli_thread_set_num_threads(omp_get_max_threads());

    /* float *a1 = malloc(PACKMR * k * sizeof(float));
    float *b1 = malloc(k * PACKNR * sizeof(float));
    fill_rand(a1, PACKMR * k);
    fill_rand(b1, k * PACKNR);
    float *c11 = malloc(MR * NR * sizeof(float));
    int rsc = 1; // colunm major
    int csc = MR;
    kernel(k, &alpha, a1, b1, &beta, c11, rsc, csc, NULL, cntx);

    float *c22 = malloc(MR * NR * sizeof(float));
    gemm_base_Cresident('C', MR, NR, k, alpha, a1, PACKMR, b1, PACKNR, beta, c22, MR);

    float maxdiff = 0.0;
    for (int i = 0; i < MR * NR; i++) {
        float d = fabs(c11[i] - c22[i]);
        if (d > maxdiff) maxdiff = d;
    } */

    float *A = malloc(m * k * sizeof(float));
    float *B = malloc(k * n * sizeof(float));
    float *C1 = malloc(m * n * sizeof(float));
    float *C2 = malloc(m * n * sizeof(float));
    float *C3 = malloc(m * n * sizeof(float));
    float *Cref = malloc(m * n * sizeof(float));
    float *Ac = aligned_alloc(4096, omp_get_max_threads() * MC * KC * sizeof(float));
    float *Bc = aligned_alloc(4096, KC * NC * sizeof(float));
    float *Cc = aligned_alloc(4096, MC * NC * sizeof(float));

    float alpha = 1.0;
    float beta = 0.0;

    for (int i = 0; i < rep; i++) {
#ifdef BENCHMARK
        t_pack = 0.0, t_kernel = 0.0, t_generic = 0.0;
#endif
        fill_rand(A, m * k);
        fill_rand(B, k * n);
        double t1 = get_time();
        sgemm('N', 'N', m, n, k, 1.0, A, m, B, k, 0.0, Cref, m);
        double t2 = get_time();
        bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, m, n, k, &alpha, A, 1, m, B, 1, k, &beta, C1, 1, m);
        double t3 = get_time();
        gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'N', m, n, k, 1.0, A, m, B, k, 0.0, C2, m, Ac, pack_RB, Bc, pack_CB, Cc, NULL, cntx, NULL, NULL, NULL, NULL, NULL, NULL, false);
        double t4 = get_time();
        gemm_blis_A3B2C0('C', 'C', 'C', 'N', 'N', m, n, k, 1.0, A, m, B, k, 0.0, C3, m, Ac, pack_RB, Bc, pack_CB, Cc, NULL, cntx, NULL, NULL);
        double t5 = get_time();

        printf("%d %d %d ", m, n, k);
        printf("%e ",  diff(m * n, Cref, C2));
        printf("%e\t", diff(m * n, Cref, C3));
        printf("%e %e %e %e\t", t2 - t1, t3 - t2, t4 - t3, t5 - t4);
        double gflop = 2.0 * m * n * k * 1e-9;
        printf("%e %e %e %e", gflop / (t2 - t1), gflop / (t3 - t2), gflop / (t4 - t3), gflop / (t5 - t4));
#ifdef BENCHMARK
        printf("t_pack = %e t_kernel = %e t_generic %e", t_pack, t_kernel, t_generic);
#endif
        printf("\n");
    }

    return 0;
}
