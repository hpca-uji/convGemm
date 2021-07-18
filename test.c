#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#include "convGemm.h"
#include "blis.h"
#include "gemm_blis.h"
#include "gemm_nhwc.h"
#include "im2row_nhwc.h"

void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, const float *a, int *lda, const float *b, int *ldb, float *beta, float *c, int *ldc);

inline void sgemm(char transa, char transb, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb, float beta, float *c, int ldc) {
    sgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

float *random_alloc(int n)
{
    float *a = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
        a[i] = (float)rand() / RAND_MAX;
    return a;
}

bool check(int n, float *a, float *b)
{
    for (int i = 0; i < n; i++) {
        float d = fabsf((a[i] - b[i]) / a[i]);
        if (d > 1e-5) {
            printf(": %d %e %e", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[])
{
    if (argc < 7 || argc > 14) {
        printf("program params: b h w c kn kh kw [ vpadding hpadding vstride hstride vdilation hdilation]\n");
        return 1;
    }
    int b  = atoi(argv[1]);
    int h  = atoi(argv[2]);
    int w  = atoi(argv[3]);
    int c  = atoi(argv[4]);
    int kn = atoi(argv[5]);
    int kh = atoi(argv[6]);
    int kw = atoi(argv[7]);
    int vpadding  = argc >  8 ? atoi(argv[ 8]) : 1;
    int hpadding  = argc >  9 ? atoi(argv[ 9]) : 1;
    int vstride   = argc > 10 ? atoi(argv[10]) : 1;
    int hstride   = argc > 11 ? atoi(argv[11]) : 1;
    int vdilation = argc > 12 ? atoi(argv[12]) : 1;
    int hdilation = argc > 13 ? atoi(argv[13]) : 1;
    float alpha = 1.0;
    float beta = 0.0;

    float* ac_pack, *bc_pack;
    alloc_pack_buffs(&ac_pack, &bc_pack);

    float *image = random_alloc(b * h * w * c);
    float *kernel = random_alloc(kn * kh * kw * c);

    int ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) / vstride + 1;
    int wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) / hstride + 1;
    float *aux = calloc(c * kh * kw * ho * wo * b, sizeof(float));
    im2row_nhwc(aux, c * kh * kw, image, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation, 0, ho * wo * b, 0, c * kh * kw);

    float *out_gemm = random_alloc(kn * ho * wo * b);
    float *out_blis = random_alloc(kn * ho * wo * b);
    float *out_nhwc = random_alloc(kn * ho * wo * b);

    sgemm(                          'N', 'N', kn, ho * wo * b, kh * kw * c, alpha, kernel, kn, aux, kh * kw * c, beta, out_gemm, kn);
    gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'N', kn, ho * wo * b, kh * kw * c, alpha, kernel, kn, aux, kh * kw * c, beta, out_blis, kn, ac_pack, bc_pack);
    gemm_nhwc_B3A2C0('C', 'C', 'C', 'N', 'N', kn, ho * wo * b, kh * kw * c, alpha, kernel, kn, aux, kh * kw * c, beta, out_nhwc, kn, ac_pack, bc_pack, image, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation);

    if (!check(kn * ho * wo * b, out_gemm, out_blis)) {
        printf(" error in gemm_blis 'N'\n");
        return 1;
    }
    if (!check(kn * ho * wo * b, out_gemm, out_nhwc)) {
        printf(" error in gemm_nhwc 'N'\n");
        return 2;
    }

    sgemm(                          'N', 'T', kn, ho * wo * b, kh * kw * c, alpha, kernel, kn, aux, ho * wo * b, beta, out_gemm, kn);
    gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'T', kn, ho * wo * b, kh * kw * c, alpha, kernel, kn, aux, ho * wo * b, beta, out_blis, kn, ac_pack, bc_pack);
    gemm_nhwc_B3A2C0('C', 'C', 'C', 'N', 'T', kn, ho * wo * b, kh * kw * c, alpha, kernel, kn, aux, ho * wo * b, beta, out_nhwc, kn, ac_pack, bc_pack, image, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation);

    float *trans_gemm = random_alloc(c * kh * kw * kn);
    float *trans_blis = random_alloc(c * kh * kw * kn);
    float *trans_nhwc = random_alloc(c * kh * kw * kn);

    sgemm(                          'N', 'T', kn, kh * kw * c, ho * wo * b, alpha, out_gemm, kn, aux, kh * kw * c, beta, trans_gemm, kn);
    gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'T', kn, kh * kw * c, ho * wo * b, alpha, out_gemm, kn, aux, kh * kw * c, beta, trans_blis, kn, ac_pack, bc_pack);
    gemm_nhwc_B3A2C0('C', 'C', 'C', 'N', 'T', kn, kh * kw * c, ho * wo * b, alpha, out_gemm, kn, NULL, kh * kw * c, beta, trans_nhwc, kn, ac_pack, bc_pack, image, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation);

    if (!check(c * kh * kw * kn, trans_gemm, trans_blis)) {
        printf(" error in gemm_blis 'T'\n");
        return 3;
    }
    if (!check(c * kh * kw * kn, trans_gemm, trans_nhwc)) {
        printf(" error in gemm_nhwc 'T'\n");
        return 4;
    }

    printf(": Ok\n");

    return 0;
}
