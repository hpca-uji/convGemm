#include <stdio.h>
#include <stdlib.h>

#include <blis.h>

#include "convGemm.h"
#include "gemm_blis.h"
#include "gemm_nhwc.h"
#include "gemm_back_nhwc.h"
#include "im2row_nhwc.h"
#include "gemm_nchw.h"
#include "gemm_back_nchw.h"
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
                    unsigned b, unsigned h, unsigned w, unsigned c,
                    unsigned kn, unsigned kh, unsigned kw,
                    unsigned vpadding, unsigned hpadding,
                    unsigned vstride, unsigned hstride,
                    unsigned vdilation, unsigned hdilation,
                    float alpha, const float *in,
                    const float *x, float beta,
                    float *out, const float *bias_vector,
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
#if 0
    float *aux = (float *) calloc(c * kh * kw * ho * wo * b, sizeof(float));
    im2row_nhwc(aux, c * kh * kw, x, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation);
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

void sconvGemmNHWC_back(unsigned b, unsigned h, unsigned w, unsigned c,
                        unsigned kn, unsigned kh, unsigned kw,
                        unsigned vstride, unsigned hstride,
                        unsigned vpadding, unsigned hpadding,
                        unsigned vdilation, unsigned hdilation,
                        float alpha, const float *weights,
                        const float *dy, float *dx,
                        float *ac_pack, float *bc_pack, float *cc_pack)
{
    /*
     * Computes: dx = dx + row2im(transpose(weights) * dy)
     */
    int ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) / vstride + 1;
    int wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) / hstride + 1;
#if 0
    float *aux = (float *) calloc(c * kh * kw * ho * wo * b, sizeof(float));
    sgemm('T', 'N', c * kh * kw, ho * wo * b, kn, alpha, weights, kn, dy, kn, 0.0, aux, c * kh * kw);
    row2im_nhwc(ho * wo * b, c * kh * kw, aux, c * kh * kw, dx, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation, 0, 0);
    free(aux);
#else
    cntx_t *cntx = bli_gks_query_cntx();
    gemm_back_nhwc_B3A2C0('C', 'C', 'C', 'T', 'N', c * kh * kw, ho * wo * b, kn, alpha, weights, kn, dy, kn, 1.0, NULL, c * kh * kw, ac_pack, bc_pack, cc_pack, cntx, dx, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation);
#endif
}

void sconvGemmNCHW(char trans,
                    unsigned b, unsigned c, unsigned h, unsigned w,
                    unsigned kn, unsigned kh, unsigned kw,
                    unsigned vpadding, unsigned hpadding,
                    unsigned vstride, unsigned hstride,
                    unsigned vdilation, unsigned hdilation,
                    float alpha, const float *in,
                    const float *x, float beta,
                    float *out, const float *bias_vector,
                    float *ac_pack, float *bc_pack, float *cc_pack)
{
    cntx_t *cntx = bli_gks_query_cntx();
    int ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) / vstride + 1;
    int wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) / hstride + 1;
#if 0
    float *aux  = (float *) calloc(c * kh * kw * ho * wo * b, sizeof(float));
    float *aux2 = (float *) calloc(kn * ho * wo * b, sizeof(float));
    im2col_nchw(aux, b * ho * wo, x, b, c, h, w, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation);
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
                        out[((i * kn + j) * ho + x) * wo + y] = aux2[((j * b + i) * ho + x) * wo + y];
    } else {
        // transpose first and second dimension
        for (int i = 0; i < kn; i++)
            for (int j = 0; j < b; j++)
                for (int x = 0; x < ho; x++)
                    for (int y = 0; y < wo; y++)
                        aux2[((i * b + j) * ho + x) * wo + y] = in[((j * kn + i) * ho + x) * wo + y];
        sgemm('T', 'N', kh * kw * c, kn, ho * wo * b, alpha, aux, ho * wo * b, aux2, ho * wo * b, beta, out, kh * kw * c);
    }
    free(aux);
    free(aux2);
#else
    if (trans == 'N') {
        gemm_nchw_B3A2C0('C', 'C', 'C', 'N', 'N', ho * wo * b, kn, kh * kw * c, alpha, NULL, ho * wo * b, in, kh * kw * c, beta, out, ho * wo * b, ac_pack, bc_pack, cc_pack, cntx, x, b, c, h, w, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation, bias_vector);
    } else {
        gemm_nchw_B3A2C0('C', 'C', 'C', 'T', 'N', kh * kw * c, kn, ho * wo * b, alpha, NULL, ho * wo * b, in, ho * wo * b, beta, out, kh * kw * c, ac_pack, bc_pack, cc_pack, cntx, x, b, c, h, w, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation, bias_vector);
    }
#endif
}

void sconvGemmNCHW_back(unsigned b, unsigned c, unsigned h, unsigned w,
                        unsigned kn, unsigned kh, unsigned kw,
                        unsigned vstride, unsigned hstride,
                        unsigned vpadding, unsigned hpadding,
                        unsigned vdilation, unsigned hdilation,
                        float alpha, const float *weights,
                        const float *dy, float *dx,
                        float *ac_pack, float *bc_pack, float *cc_pack)
{
    /*
     * Computes: dx = col2im(dy * transpose(weights))
     */
    int ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) / vstride + 1;
    int wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) / hstride + 1;
#if 0
    float *aux = (float *) calloc(c * kh * kw * ho * wo * b, sizeof(float));
    float *aux2 = (float *) calloc(kn * ho * wo * b, sizeof(float));
    // transpose first and second dimension
    transpose_nchw(kn * ho * wo, b, dy, kn * ho * wo, 0.0, aux2, b, ho, wo, 0, 0);
    sgemm('N', 'T', b * ho * wo, c * kh * kw, kn, alpha, aux2, b * ho * wo, weights, c * kh * kw, 0.0, aux, b * ho * wo);
    col2im_nchw(c * kh * kw, b * ho * wo, aux, b * ho * wo, dx, b, c, h, w, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation, 0, 0);
    free(aux);
    free(aux2);
#else
    cntx_t *cntx = bli_gks_query_cntx();
    gemm_back_nchw_B3A2C0('C', 'C', 'C', 'N', 'T', b * ho * wo, c * kh * kw, kn, alpha, dy, b * ho * wo, weights, c * kh * kw, 1.0, NULL, b * ho * wo, ac_pack, bc_pack, cc_pack, cntx, dx, b, c, h, w, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation);
#endif
}
