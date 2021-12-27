#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <blis.h>

#include "convGemm.h"
#include "gemm_blis.h"
#include "im2row_nhwc.h"
#include "im2col_nchw.h"

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

    cntx_t *cntx = bli_gks_query_cntx();
    int ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) / vstride + 1;
    int wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) / hstride + 1;
    convol_dim dim = { b, h, w, c, kn, kh, kw, vstride, hstride, vpadding, hpadding, vdilation, hdilation, ho, wo };
    if (trans == 'N') {
        gemm_blis_A3B2C0('C', 'C', 'C', 'N', 'N', kn, ho * wo * b, kh * kw * c, alpha, in, kn, x, kh * kw * c, beta, out, kn, ac_pack, pack_RB, bc_pack, pack_CB_nhwc, cc_pack, bias_vector == NULL ? NULL : add_bias_nhwc, cntx, &dim, bias_vector);
    } else {
        gemm_blis_A3B2C0('C', 'C', 'C', 'N', 'T', kn, kh * kw * c, ho * wo * b, alpha, in, kn, x, kh * kw * c, beta, out, kn, ac_pack, pack_RB, bc_pack, pack_CB_nhwc, cc_pack, NULL, cntx, &dim, NULL);
    }
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

    cntx_t *cntx = bli_gks_query_cntx();
    int ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) / vstride + 1;
    int wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) / hstride + 1;
    convol_dim dim = { b, h, w, c, kn, kh, kw, vstride, hstride, vpadding, hpadding, vdilation, hdilation, ho, wo };
    gemm_blis_B3A2C0('C', 'C', 'C', 'T', 'N', c * kh * kw, ho * wo * b, kn, alpha, weights, kn, dy, kn, 1.0, dx, c * kh * kw, ac_pack, pack_RB, bc_pack, pack_CB, cc_pack, post_row2im_nhwc, cntx, &dim, NULL);
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
    convol_dim dim = { b, h, w, c, kn, kh, kw, vstride, hstride, vpadding, hpadding, vdilation, hdilation, ho, wo };
    if (trans == 'N') {
        gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'N', ho * wo * b, kn, kh * kw * c, alpha, x, ho * wo * b, in, kh * kw * c, beta, out, ho * wo * b, ac_pack, pack_RB_nchw, bc_pack, pack_CB, cc_pack, add_bias_transpose_nchw, cntx, &dim, bias_vector);
    } else {
        gemm_blis_B3A2C0('C', 'C', 'C', 'T', 'N', kh * kw * c, kn, ho * wo * b, alpha, x, ho * wo * b, in, ho * wo * b, beta, out, kh * kw * c, ac_pack, pack_RB_nchw, bc_pack, pack_CB_nchw_trans, cc_pack, NULL, cntx, &dim, NULL);
    }
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

    cntx_t *cntx = bli_gks_query_cntx();
    int ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) / vstride + 1;
    int wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) / hstride + 1;
    convol_dim dim = { b, h, w, c, kn, kh, kw, vstride, hstride, vpadding, hpadding, vdilation, hdilation, ho, wo };
    gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'T', b * ho * wo, c * kh * kw, kn, alpha, dy, b * ho * wo, weights, c * kh * kw, 1.0, dx, b * ho * wo, ac_pack, pack_RB_nchw_trans, bc_pack, pack_CB, cc_pack, post_col2im_nchw, cntx, &dim, NULL);
}
