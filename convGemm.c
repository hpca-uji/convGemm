#include <stdio.h>
#include <stdlib.h>

#include <blis.h>

#include "convGemm.h"
#include "gemm_blis.h"
#include "gemm_nhwc.h"
#include "im2row_nhwc.h"
#include "gemm_nchw.h"
#include "im2col_nchw.h"

int alloc_pack_buffs(float** Ac_pack, float** Bc_pack, float** Cc_pack)
{
    bli_init();
    cntx_t *cntx = bli_gks_query_cntx();
    int NC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NC, cntx);
    int MC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MC, cntx);
    int KC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_KC, cntx);

    *Ac_pack = aligned_alloc(4096, MC * KC * sizeof(float));
    *Bc_pack = aligned_alloc(4096, KC * NC * sizeof(float));
    *Cc_pack = aligned_alloc(4096, MC * NC * sizeof(float));

    if(*Ac_pack == NULL || *Bc_pack == NULL || *Cc_pack == NULL) return 1;
    return 0;
}

void sconvGemmNHWC(char trans,
                    unsigned c, unsigned kh, unsigned kw, unsigned kn,
                    float alpha, float *in,
                    unsigned b, unsigned h, unsigned w,
                    unsigned vpadding, unsigned hpadding,
                    unsigned vstride, unsigned hstride,
                    unsigned vdilation, unsigned hdilation,
                    float *x, float beta,
                    float *out, float *bias_vector,
                    float *ac_pack, float *bc_pack, float *cc_pack)
{
    /*
     * computes
     *    out = alpha * in * im2row(x) + beta * out + bias_vector
     * or
     *    out = alpha * in * transpose(im2row(x)) + beta * out
     */

    int ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) / vstride + 1;
    int wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) / hstride + 1;
    // printf("im2row_nhwc h=%d ho=%d vp=%d vs=%d vd=%d\n", h, ho, vpadding, vstride, vdilation);
    // printf("im2row_nhwc w=%d wo=%d hp=%d hs=%d hd=%d\n", w, wo, hpadding, hstride, hdilation);
    // printf("sconvGemmNHWC trans=%c  bias_vector=", trans);
    // if (bias_vector) for (int i = 0; i < kn; i++) printf(" %g", bias_vector[i]);
    // printf("\n");
#if 0
    float *aux = (float *) calloc(c * kh * kw * ho * wo * b, sizeof(float));
    im2row_nhwc(aux, c * kh * kw, x, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation, 0, ho * wo * b, 0, c * kh * kw);
    if (trans == 'N') {
        sgemm('N', 'N', kn, ho * wo * b, kh * kw * c, alpha, in, kn, aux, kh * kw * c, beta, out, kn);
        // gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'N', kn, ho * wo * b, kh * kw * c, alpha, in, kn, aux, kh * kw * c, beta, out, kn, ac_pack, bc_pack);
        if (bias_vector) {
            // #pragma omp parallel for
            for(int j = 0; j < ho * wo * b; j++)
                for(int i = 0; i < kn; i++)
                    out[i + j * kn] += bias_vector[i];
        }
    } else {
        sgemm('N', 'T', kn, kh * kw * c, ho * wo * b, alpha, in, kn, aux, kh * kw * c, beta, out, kn);
        // gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'T', kn, kh * kw * c, ho * wo * b, alpha, in, kn, aux, kh * kw * c, beta, out, kn, ac_pack, bc_pack);
    }
    free(aux);
#else
    cntx_t *cntx = bli_gks_query_cntx();
    if (trans == 'N') {
        gemm_nhwc_B3A2C0('C', 'C', 'C', 'N', 'N', kn, ho * wo * b, kh * kw * c, alpha, in, kn, NULL, kh * kw * c, beta, out, kn, ac_pack, bc_pack, cc_pack, cntx, x, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation, bias_vector);
    } else {
        gemm_nhwc_B3A2C0('C', 'C', 'C', 'N', 'T', kn, kh * kw * c, ho * wo * b, alpha, in, kn, NULL, kh * kw * c, beta, out, kn, ac_pack, bc_pack, cc_pack, cntx, x, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation, NULL);
    }
#endif
}

void sconvGemmNHWC_back(unsigned kn, unsigned kh, unsigned kw, unsigned c,
                        float alpha, float *weights,
                        unsigned b, unsigned h, unsigned w,
                        unsigned hstride, unsigned vstride,
                        unsigned hpadding, unsigned vpadding,
                        unsigned vdilation, unsigned hdilation,
                        float *dy, float *dx,
                        float *ac_pack, float *bc_pack, float *cc_pack)
{
    /*
     * Computes: dx = row2im(dy * transpose(weights))
     */
    int ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) / vstride + 1;
    int wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) / hstride + 1;
    float *aux = (float *) calloc(c * kh * kw * ho * wo * b, sizeof(float));
    sgemm('T', 'N', c * kh * kw, ho * wo * b, kn, 1.0, weights, kn, dy, kn, 0.0, aux, c * kh * kw);
    row2im_nhwc(aux, dx, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation);

    free(aux);
}

void sconvGemmNCHW(float *ref, char trans,
                    unsigned kn, unsigned c, unsigned kh, unsigned kw,
                    float alpha, float *in,
                    unsigned h, unsigned w, unsigned b,
                    unsigned vpadding, unsigned hpadding,
                    unsigned hstride, unsigned vstride,
                    unsigned vdilation, unsigned hdilation,
                    float *x, float beta,
                    float *out, float *bias_vector,
                    float *ac_pack, float *bc_pack, float *cc_pack)
{
    cntx_t *cntx = bli_gks_query_cntx();
    int ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) / vstride + 1;
    int wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) / hstride + 1;
    float *aux2 = (float *) calloc(kn * ho * wo * b, sizeof(float));
#if 0
    float *aux  = (float *) calloc(c * kh * kw * ho * wo * b, sizeof(float));
    im2col_nchw(aux, b * ho * wo, x, b, c, h, w, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation, 0, c * kh * kw, 0, b * ho * wo);
    if (trans == 'N') {
        sgemm('N', 'N', ho * wo * b, kn, kh * kw * c, alpha, aux, ho * wo * b, in, kh * kw * c, beta, aux2, ho * wo * b);
        // gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'N', ho * wo * b, kn, kh * kw * c, alpha, aux, ho * wo * b, in, kh * kw * c, beta, aux2, ho * wo * b, ac_pack, bc_pack, cc_pack, cntx);
        if (bias_vector) { // add bias
            // #pragma omp parallel for
            for(int i = 0; i < kn; i++)
                for(int j = 0; j < ho * wo * b; j++)
                    aux2[i * ho * wo * b + j] += bias_vector[i];
        }
        // transpose first and second dimension
        for (int i = 0; i < b; i++)
            for (int j = 0; j < kn; j++)
                for (int x = 0; x < ho; x++)
                    for (int y = 0; y < wo; y++)
                        out[i * kn * ho * wo +
                            j      * ho * wo +
                            x           * wo +
                            y] = aux2[j * b * ho * wo +
                                      i     * ho * wo +
                                      x          * wo +
                                      y];
        for (int i = 0; i < kn * ho * wo * b; i++)
            if (fabsf(out[i] - ref[i]) > 1e-4) {
                printf("%d %e %e\n", i, out[i], ref[i]);
                abort();
            }
    } else {
        /* TODO wrong dimensions
        sgemm('T', 'N', kh * kw * c, kn, ho * wo * b, alpha, aux, ho * wo * b, in, kh * kw * c, beta, aux2, kh * kw * c);
        for (int i = 0; i < kh * kw * c * kn; i++)
            if (fabsf(out[i] - ref[i]) > 1e-4) {
                printf("%d %e %e\n", i, out[i], ref[i]);
                abort();
            } */
    }
    free(aux);
#else
    if (trans == 'N') {
        gemm_nchw_B3A2C0('C', 'C', 'C', 'N', 'N', ho * wo * b, kn, kh * kw * c, alpha, NULL, ho * wo * b, in, kh * kw * c, beta, out, ho * wo * b, ac_pack, bc_pack, cc_pack, cntx, x, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation, bias_vector);
        for (int i = 0; i < kn * ho * wo * b; i++)
            if (fabsf(out[i] - ref[i]) > 1e-4) {
                printf("%d %e %e\n", i, out[i], ref[i]);
                abort();
            }
    } else {
        abort(); // TODO not implemented
    }
#endif
    free(aux2);
}
