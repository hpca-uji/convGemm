#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#include <blis.h>

#include "gemm_blis.h"

// void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, const float *a, int *lda, const float *b, int *ldb, float *beta, float *c, int *ldc);

static inline void sgemm(char transa, char transb, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb, float beta, float *c, int ldc) {
    sgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void fill_rand(float *a, int n)
{
    for (int i = 0; i < n; i++)
        a[i] = rand() * 1.0 / RAND_MAX;
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
        fprintf(stderr, "Missing arguments: M N K\n");
        return 1;
    }

    bli_init();
    cntx_t* cntx = bli_gks_query_cntx();
    sgemm_ukr_ft gemm_kernel = bli_cntx_get_l3_nat_ukr_dt      (BLIS_FLOAT, BLIS_GEMM, cntx);
    int MR                   = bli_cntx_get_blksz_def_dt       (BLIS_FLOAT, BLIS_MR,   cntx);
    int NR                   = bli_cntx_get_blksz_def_dt       (BLIS_FLOAT, BLIS_NR,   cntx);
    int PACKMR               = bli_cntx_get_blksz_max_dt       (BLIS_FLOAT, BLIS_MR,   cntx);
    int PACKNR               = bli_cntx_get_blksz_max_dt       (BLIS_FLOAT, BLIS_NR,   cntx);
    bool row_pref            = bli_cntx_get_l3_nat_ukr_prefs_dt(BLIS_FLOAT, BLIS_GEMM, cntx);
    int NC                   = bli_cntx_get_blksz_def_dt       (BLIS_FLOAT, BLIS_NC,   cntx);
    int MC                   = bli_cntx_get_blksz_def_dt       (BLIS_FLOAT, BLIS_MC,   cntx);
    int KC                   = bli_cntx_get_blksz_def_dt       (BLIS_FLOAT, BLIS_KC,   cntx);
    // BLIS_POOL_ADDR_ALIGN_SIZE, KR
    printf("kernel = %p MR = %d NR = %d packmr = %d packnr = %d row_pref = %d NC = %d MC = %d KC = %d\n", gemm_kernel, MR, NR, PACKMR, PACKNR, row_pref, NC, MC, KC);

    /* float *a1 = malloc(PACKMR * k * sizeof(float));
    float *b1 = malloc(k * PACKNR * sizeof(float));
    fill_rand(a1, PACKMR * k);
    fill_rand(b1, k * PACKNR);
    float *c11 = malloc(MR * NR * sizeof(float));
    float alpha = 1.0;
    float beta = 0.0;
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

    int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]);
    float *A = malloc(m * k * sizeof(float));
    float *B = malloc(k * n * sizeof(float));
    float *C = malloc(m * n * sizeof(float));
    float *Cref = malloc(m * n * sizeof(float));
    float *Ac = aligned_alloc(4096, MC * KC * sizeof(float));
    float *Bc = aligned_alloc(4096, KC * NC * sizeof(float));
    float *Cc = aligned_alloc(4096, MC * NC * sizeof(float));

    for (int i = 0; i < 3; i++) {
        fill_rand(A, m * k);
        fill_rand(B, k * n);
        double t1 = get_time();
        sgemm('N', 'N', m, n, k, 1.0, A, m, B, k, 0.0, Cref, m);
#ifdef BENCHMARK
        t_pack = 0.0, t_kernel = 0.0, t_generic = 0.0;
#endif
        double t2 = get_time();
        gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'N', m, n, k, 1.0, A, m, B, k, 0.0, C, m, Ac, Bc, Cc, cntx);
        double t3 = get_time();

        float maxdiff = 0.0, cref_diff = 0.0, c_diff = 0.0;
        for (int i = 0; i < m * n; i++) {
            float d = fabs(Cref[i] - C[i]);
            if (d > maxdiff) {
                maxdiff = d;
                cref_diff = Cref[i];
                c_diff = C[i];
            }
        }
        printf("maxdiff = %e ref = %e res = %e\n", maxdiff, cref_diff, c_diff);
        printf("t_blas = %e t_blis = %e ", t2 - t1, t3 - t2);
#ifdef BENCHMARK
        printf("t_pack = %e t_kernel = %e t_generic %e", t_pack, t_kernel, t_generic);
#endif
        printf("\n");
    }

    for (int i = 0; i < 3; i++) {
        fill_rand(A, m * k);
        fill_rand(B, k * n);
        double t1 = get_time();
        sgemm('N', 'T', m, n, k, 1.0, A, m, B, n, 0.0, Cref, m);
#ifdef BENCHMARK
        t_pack = 0.0, t_kernel = 0.0, t_generic = 0.0;
#endif
        double t2 = get_time();
        gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'T', m, n, k, 1.0, A, m, B, n, 0.0, C, m, Ac, Bc, Cc, cntx);
        double t3 = get_time();

        float maxdiff = 0.0, cref_diff = 0.0, c_diff = 0.0;
        for (int i = 0; i < m * n; i++) {
            float d = fabs(Cref[i] - C[i]);
            if (d > maxdiff) {
                maxdiff = d;
                cref_diff = Cref[i];
                c_diff = C[i];
            }
        }
        printf("maxdiff = %e ref = %e res = %e\n", maxdiff, cref_diff, c_diff);
        printf("t_blas = %e t_blis = %e ", t2 - t1, t3 - t2);
#ifdef BENCHMARK
        printf("t_pack = %e t_kernel = %e t_generic %e", t_pack, t_kernel, t_generic);
#endif
        printf("\n");
    }

    return 0;
}
