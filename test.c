#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#include <blis.h>

#include "convGemm.h"
#include "gemm_blis.h"
#include "gemm_nhwc.h"
#include "im2row_nhwc.h"
#include "gemm_nchw.h"
#include "im2col_nchw.h"

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

    bli_init();
    cntx_t *cntx = bli_gks_query_cntx();
    int NC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NC, cntx);
    int MC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MC, cntx);
    int KC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_KC, cntx);
    float *ac_pack = aligned_alloc(4096, MC * KC * sizeof(float));
    float *bc_pack = aligned_alloc(4096, KC * NC * sizeof(float));
    float *cc_pack = aligned_alloc(4096, MC * NC * sizeof(float));

    float *image = random_alloc(b * h * w * c);
    float *kernel = random_alloc(kn * kh * kw * c);
    float *bias_vector = random_alloc(kn);

    int ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) / vstride + 1;
    int wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) / hstride + 1;
    float *aux = calloc(c * kh * kw * ho * wo * b, sizeof(float));
    double t1 = get_time();
    im2row_nhwc(aux, c * kh * kw, image, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation, 0, ho * wo * b, 0, c * kh * kw);
    double t2 = get_time();
    double t_im2row = t2 - t1;
    printf("\t%e", t_im2row);

    float *out_gemm = random_alloc(kn * ho * wo * b);
    float *out_blis = random_alloc(kn * ho * wo * b);
    float *out      = random_alloc(kn * ho * wo * b);

    t1 = get_time();
    sgemm(                          'N', 'N', kn, ho * wo * b, kh * kw * c, alpha, kernel, kn, aux, kh * kw * c, beta, out_gemm, kn);
    t2 = get_time();
    gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'N', kn, ho * wo * b, kh * kw * c, alpha, kernel, kn, aux, kh * kw * c, beta, out_blis, kn, ac_pack, bc_pack, cc_pack, cntx);
    double t3 = get_time();
    gemm_nhwc_B3A2C0('C', 'C', 'C', 'N', 'N', kn, ho * wo * b, kh * kw * c, alpha, kernel, kn, NULL, kh * kw * c, beta, out, kn, ac_pack, bc_pack, cc_pack, cntx, image, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation, bias_vector);
    double t4 = get_time();
    double t_gemm = t2 - t1;
    double t_blis = t3 - t2;
    double t_nhwc = t4 - t3;
    printf(" %e %e %e", t_gemm, t_blis, t_nhwc);

    if (!check(kn * ho * wo * b, out_gemm, out_blis)) {
        printf(" error in gemm_blis 'N' NHWC\n");
        return 1;
    }
    for(int j = 0; j < ho * wo * b; j++) // add bias
        for(int i = 0; i < kn; i++)
            out_gemm[i + j * kn] += bias_vector[i];
    if (!check(kn * ho * wo * b, out_gemm, out)) {
        printf(" error in gemm_nhwc 'N' NHWC\n");
        return 2;
    }

    float *trans_gemm = random_alloc(c * kh * kw * kn);
    float *trans_blis = random_alloc(c * kh * kw * kn);
    float *trans_nhwc = random_alloc(c * kh * kw * kn);

    t1 = get_time();
    sgemm(                          'N', 'T', kn, kh * kw * c, ho * wo * b, alpha, out_gemm, kn, aux, kh * kw * c, beta, trans_gemm, kn);
    t2 = get_time();
    gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'T', kn, kh * kw * c, ho * wo * b, alpha, out_gemm, kn, aux, kh * kw * c, beta, trans_blis, kn, ac_pack, bc_pack, cc_pack, cntx);
    t3 = get_time();
    gemm_nhwc_B3A2C0('C', 'C', 'C', 'N', 'T', kn, kh * kw * c, ho * wo * b, alpha, out_gemm, kn, NULL, kh * kw * c, beta, trans_nhwc, kn, ac_pack, bc_pack, cc_pack, cntx, image, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation, NULL);
    t4 = get_time();
    t_gemm = t2 - t1;
    t_blis = t3 - t2;
    t_nhwc = t4 - t3;
    printf("\t%e %e %e", t_gemm, t_blis, t_nhwc);

    if (!check(c * kh * kw * kn, trans_gemm, trans_blis)) {
        printf(" error in gemm_blis 'T' NHWC\n");
        return 3;
    }
    if (!check(c * kh * kw * kn, trans_gemm, trans_nhwc)) {
        printf(" error in gemm_nhwc 'T' NHWC\n");
        return 4;
    }

    memset(aux, 0, c * kh * kw * ho * wo * b * sizeof(float));
    t1 = get_time();
    im2col_nchw(aux, b * ho * wo, image, b, c, h, w, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation, 0, c * kh * kw, 0, b * ho * wo);
    t2 = get_time();
    double t_im2col = t2 - t1;
    printf("\t%e", t_im2col);

    t1 = get_time();
    sgemm(                          'N', 'N', ho * wo * b, kn, kh * kw * c, alpha, aux, ho * wo * b, kernel, kh * kw * c, beta, out_gemm, ho * wo * b);
    t2 = get_time();
    gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'N', ho * wo * b, kn, kh * kw * c, alpha, aux, ho * wo * b, kernel, kh * kw * c, beta, out_blis, ho * wo * b, ac_pack, bc_pack, cc_pack, cntx);
    t3 = get_time();
    gemm_nchw_B3A2C0('C', 'C', 'C', 'N', 'N', ho * wo * b, kn, kh * kw * c, alpha, NULL, ho * wo * b, kernel, kh * kw * c, beta, out, ho * wo * b, ac_pack, bc_pack, cc_pack, cntx, image, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation, bias_vector);
    t4 = get_time();
    t_gemm = t2 - t1;
    t_blis = t3 - t2;
    double t_nchw = t4 - t3;
    printf(" %e %e %e", t_gemm, t_blis, t_nchw);

    if (!check(kn * ho * wo * b, out_gemm, out_blis)) {
        printf(" error in gemm_blis 'N' NCHW\n");
        return 1;
    }
    for(int j = 0; j < kn; j++) // add bias
        for(int i = 0; i < ho * wo * b; i++)
            out_gemm[j * ho * wo * b + i] += bias_vector[j];
    // transpose first and second dimension
    /* for (int i = 0; i < b; i++)
        for (int j = 0; j < kn; j++)
            for (int x = 0; x < ho; x++)
                for (int y = 0; y < wo; y++)
                    out_gemm[i * kn * ho * wo +
                        j      * ho * wo +
                        x           * wo +
                        y] = out_blis[j * b * ho * wo +
                                  i     * ho * wo +
                                  x          * wo +
                                  y]; */
    if (!check(kn * ho * wo * b, out_gemm, out)) {
        printf(" error in gemm_nchw 'N' NCHW\n");
        return 2;
    }
    printf("\n");

    return 0;
}
