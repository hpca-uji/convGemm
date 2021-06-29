#include <stdio.h>
#include <stdlib.h>

void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, const float *a, int *lda, const float *b, int *ldb, float *beta, float *c, int *ldc);

inline void sgemm(char transa, char transb, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb, float beta, float *c, int ldc) {
    sgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}


int alloc_pack_buffs(float** Ac_pack, float** Bc_pack)
{
    /*
    // *Ac_pack =  (float *) aligned_alloc(ALIGN, BLOCK_MC * BLOCK_KC * sizeof(float));
    // *Bc_pack =  (float *) aligned_alloc(ALIGN, BLOCK_KC * BLOCK_NC * sizeof(float));
    *Ac_pack =  (float *) aligned_alloc(4096, MC * KC * sizeof(float));
    *Bc_pack =  (float *) aligned_alloc(4096, KC * NC * sizeof(float));

    if(Ac_pack == NULL || Bc_pack == NULL)
        return 1;
    */
    return 0;
}

void im2row_nhwc(float *rows, const float *x, int n, int h, int w, int c, int hh, int ww, int kh, int kw, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation)
{
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
                                    // rows[row, col] = x[nn, x_x, x_y, cc]
                                    rows[row * c * kh * kw + col] = x[
                                        nn  * h * w * c +
                                        x_x     * w * c +
                                        x_y         * c +
                                        cc];
                                }
                            }
                    }
            }
}

void sconvGemmNHWC(char trans,
                         unsigned kn, unsigned kh, unsigned kw, unsigned c,
                         float alpha, float *weights,
                         unsigned b, unsigned h, unsigned w,
                         unsigned vpadding, unsigned hpadding,
                         unsigned vstride, unsigned hstride,
                         unsigned vdilation, unsigned hdilation,
                         float *x, float beta,
                         float *out,
                         float *ac_pack, float *bc_pack)
{
    int ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) / vstride + 1;
    int wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) / hstride + 1;
    float *Aux = (float *) calloc(c * kh * kw * ho * wo * b, sizeof(float));
    // printf("im2row_nhwc h=%d ho=%d vp=%d vs=%d vd=%d\n", h, ho, vpadding, vstride, vdilation);
    // printf("im2row_nhwc w=%d wo=%d hp=%d hs=%d hd=%d\n", w, wo, hpadding, hstride, hdilation);
    im2row_nhwc(Aux, x, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation);
    // printf("sgemm\n");
    sgemm('N', 'N', kn, ho * wo * b, kh * kw * c, 1.0, weights, kn, Aux, kh * kw * c, 0.0, out, kn);
    free(Aux);
}
