#include <stdio.h>
#include <stdlib.h>

#include <blis.h>

#include "convGemm.h"
#include "gemm_blis.h"
#include "gemm_nhwc.h"
#include "im2row_nhwc.h"

static cntx_t* cntx;
static int MR, NR, MC, NC, KC, PACKMR, PACKNR;
static sgemm_ukr_ft gemm_kernel;

int alloc_pack_buffs(float** Ac_pack, float** Bc_pack)
{
    bli_init();
    cntx_t *cntx = bli_gks_query_cntx();
    int NC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NC, cntx);
    int MC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MC, cntx);
    int KC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_KC, cntx);

    *Ac_pack = aligned_alloc(4096, MC * KC * sizeof(float));
    *Bc_pack = aligned_alloc(4096, KC * NC * sizeof(float));

    if(*Ac_pack == NULL || *Bc_pack == NULL) return 1;
    return 0;
}

void sconvGemmNHWC(char trans,
                    unsigned kn, unsigned kh, unsigned kw, unsigned c,
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
    if (trans == 'N') {
        gemm_nhwc_B3A2C0('C', 'C', 'C', 'N', 'N', kn, ho * wo * b, kh * kw * c, alpha, in, kn, NULL, kh * kw * c, beta, out, kn, ac_pack, bc_pack, cc_pack, cntx, x, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation);
        if (bias_vector) {
            // #pragma omp parallel for
            for(int j = 0; j < ho * wo * b; j++)
                for(int i = 0; i < kn; i++)
                    out[i + j * kn] += bias_vector[i];
        }
    } else {
        gemm_nhwc_B3A2C0('C', 'C', 'C', 'N', 'T', kn, kh * kw * c, ho * wo * b, alpha, in, kn, NULL, kh * kw * c, beta, out, kn, ac_pack, bc_pack, cc_pack, cntx, x, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation);
    }
#endif
}

void row2im_nhwc(const float *rows, float *x, int n, int h, int w, int c, int hh, int ww, int kh, int kw, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation)
{
    // #pragma omp parallel for
    for (int nn = 0; nn < n; nn++)
        for (int xx = 0; xx < hh; xx++)
            for (int yy = 0; yy < ww; yy++) {
                int row = nn * hh * ww + xx * ww + yy;
                for (int cc = 0; cc < c; cc++)
                    for (int ii = 0; ii < kh; ii++) {
                        int x_x = vstride * xx + vdilation * ii - vpadding;
                        if (0 <= x_x && x_x < h)
                            for (int jj = 0; jj < kw; jj++) {
                                int x_y = hstride * yy + hdilation * jj - hpadding;
                                if (0 <= x_y && x_y < w) {
                                    int col = cc * kh * kw + ii * kw + jj;
                                    // x[nn, x_x, x_y, cc] += rows[row, col]
                                    x[nn  * h * w * c +
                                      x_x     * w * c +
                                      x_y         * c +
                                      cc] += rows[row * c * kh * kw + col];
                                }
                            }
                    }
            }
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
