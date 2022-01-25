#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <blis.h>

#include "gemm_blis.h"
#include "im2row_nhwc.h"

void pack_CB_nhwc(char orderM, char transM, int mc, int nc, const float *restrict M, int ldM, float *restrict Mc, int RR, const convol_dim *d, int start_i, int start_j)
{
/*
  BLIS pack for M-->Mc using implicit im2row
*/
    if (((transM == 'N') && (orderM == 'C')) || ((transM == 'T') && (orderM == 'R')))
    {
        // initial kernel positions
        int start_ky =  start_i % d->kwidth;
        int start_kx = (start_i / d->kwidth) % d->kheight;
        int start_c  = (start_i / d->kwidth) / d->kheight;
        #pragma omp parallel for
        for (int j = 0; j < nc; j += RR) {
            int k = j * mc;
            int nr = min(nc - j, RR);
            int ky = start_ky;
            int kx = start_kx;
            int c  = start_c;
            // initial pixel positions
            int start_y  =  (start_j + j) % d->owidth;
            int start_x  = ((start_j + j) / d->owidth) % d->oheight;
            int start_b  = ((start_j + j) / d->owidth) / d->oheight;
            for (int i = 0; i < mc; i++) {
                int y = start_y;
                int x = start_x;
                int b = start_b;
                int jj = 0;
                for (; jj < nr; jj++) {
                    // Mc[k] = Mcol(i,j+jj);
                    int ix = d->vstride * x + d->vdilation * kx - d->vpadding;
                    int iy = d->hstride * y + d->hdilation * ky - d->hpadding;
                    if (0 <= ix && ix < d->height && 0 <= iy && iy < d->width) {
                        Mc[k] = M[((b * d->height + ix) * d->width + iy) * d->channel + c];
                    } else Mc[k] = 0.0;
                    k++;
                    // next pixel position
                    y++; if (y >= d->owidth) { y = 0;
                    x++; if (x >= d->oheight) { x = 0;
                    b++; } }
                }
                for (; jj < RR; jj++) {
                    Mc[k] = 0.0;
                    k++;
                }
                // k += (RR - nr);
                // next kernel position
                ky++; if (ky >= d->kwidth) { ky = 0;
                kx++; if (kx >= d->kheight) { kx = 0;
                c++; } }
            }
        }
    } else {
        int start_y  =  (start_i) % d->owidth;
        int start_x  = ((start_i) / d->owidth) % d->oheight;
        int start_b  = ((start_i) / d->owidth) / d->oheight;
        #pragma omp parallel for
        for (int j = 0; j < nc; j += RR) {
            int k = j * mc;
            int nr = min(nc - j, RR);
            int y = start_y;
            int x = start_x;
            int b = start_b;
            int start_ky =  (start_j + j) % d->kwidth;
            int start_kx = ((start_j + j) / d->kwidth) % d->kheight;
            int start_c  = ((start_j + j) / d->kwidth) / d->kheight;
            for (int i = 0; i < mc; i++) {
                int ky = start_ky;
                int kx = start_kx;
                int c  = start_c;
                int jj = 0;
                for (; jj < nr; jj++) {
                    // Mc[k] = Mcol(j+jj,i);
                    int ix = d->vstride * x + d->vdilation * kx - d->vpadding;
                    int iy = d->hstride * y + d->hdilation * ky - d->hpadding;
                    if (0 <= ix && ix < d->height && 0 <= iy && iy < d->width) {
                        Mc[k] = M[((b * d->height + ix) * d->width + iy) * d->channel + c];
                    } else Mc[k] = 0.0;
                    k++;
                    // next kernel position
                    ky++; if (ky >= d->kwidth) { ky = 0;
                    kx++; if (kx >= d->kheight) { kx = 0;
                    c++; } }
                }
                for (; jj < RR; jj++) {
                    Mc[k] = 0.0;
                    k++;
                }
                // k += (RR - nr);
                // next pixel position
                y++; if (y >= d->owidth) { y = 0;
                x++; if (x >= d->oheight) { x = 0;
                b++; } }
            }
        }
    }
}

void im2row_nhwc(float *restrict rows, int ld, const float *restrict in, int batch, int height, int width, int channel, int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation)
{
#if 1
    #pragma omp parallel for
    for (int b = 0; b < batch; b++)
        for (int x = 0; x < oheight; x++)
            for (int y = 0; y < owidth; y++) {
                int row = b * oheight * owidth + x * owidth + y;
                    for (int kx = 0; kx < kheight; kx++) {
                        int ix = vstride * x + vdilation * kx - vpadding;
                        if (0 <= ix && ix < height)
                            for (int ky = 0; ky < kwidth; ky++) {
                                int iy = hstride * y + hdilation * ky - hpadding;
                                if (0 <= iy && iy < width) for (int c = 0; c < channel; c++) {
                                    int col = c * kheight * kwidth + kx * kwidth + ky;
                                    rows[row * channel * kheight * kwidth + col] = in[((b * height + ix) * width + iy) * channel + c];
                                }
                            }
                    }
            }
#else
    assert(start_row < oheight * owidth * batch);
    assert(end_row <= oheight * owidth * batch);
    assert(start_col < channel * kheight * kwidth);
    assert(end_col <= channel * kheight * kwidth);
    // starting values for the first row
    // int row = (b * oheight + x) * owidth + y;
    int y =  start_row % owidth;
    int x = (start_row / owidth) % oheight;
    int b = (start_row / owidth) / oheight;
    // starting values for the first column
    // int col = (c * kheight + kx) * kwidth + ky;
    int start_ky =  start_col % kwidth;
    int start_kx = (start_col / kwidth) % kheight;
    int start_c  = (start_col / kwidth) / kheight;

    // #pragma omp parallel for
    for (int row = 0; row < end_row - start_row; row++) {
        for (int col = 0, c = start_c, kx = start_kx, ky = start_ky; col < end_col - start_col; col++) {
            int ix = vstride * x + vdilation * kx - vpadding;
            int iy = hstride * y + hdilation * ky - hpadding;
            if (0 <= ix && ix < height && 0 <= iy && iy < width) {
                // rows[row, col] = in[b, ix, iy, c]
                rows[row * ld + col] = in[((b * height + ix) * width + iy) * channel + c];
            } else rows[row * ld + col] = 0;
            ky++; if (ky >= kwidth) { ky = 0;
            kx++; if (kx >= kheight) { kx = 0;
            c++; } }
        }
        y++; if (y >= owidth) { y = 0;
        x++; if (x >= oheight) { x = 0;
        b++; } }
    }
#endif
}

void row2im_nhwc(int m, int n, const float *restrict rows, int ld, float *restrict out, int batch, int height, int width, int channel, int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation)
{
    #pragma omp parallel for
    for (int b = 0; b < batch; b++)
        for (int x = 0; x < oheight; x++)
            for (int y = 0; y < owidth; y++) {
                int row = b * oheight * owidth + x * owidth + y;
                    for (int kx = 0; kx < kheight; kx++) {
                        int ix = vstride * x + vdilation * kx - vpadding;
                        if (0 <= ix && ix < height)
                            for (int ky = 0; ky < kwidth; ky++) {
                                int iy = hstride * y + hdilation * ky - hpadding;
                                if (0 <= iy && iy < width) for (int c = 0; c < channel; c++) {
                                    int col = c * kheight * kwidth + kx * kwidth + ky;
                                    // out[b, x_x, x_y, cc] += rows[row, col]
                                    out[((b  * height + ix) * width + iy) * channel + c] += rows[row * channel * kheight * kwidth + col];
                                }
                            }
                    }
            }
}

void post_row2im_nhwc(int n, int m, const float *restrict rows, int ldr, float beta, float *restrict out, int ldout, const convol_dim *d, int start_col, int start_row, bool last)
{
    /* int m = oheight * owidth * batch;
    int n = channel * kheight * kwidth; */

    // starting values for the first column
    // int col = (c * kheight + kx) * kwidth + ky;
    int start_ky =  start_col % d->kwidth;
    int start_kx = (start_col / d->kwidth) % d->kheight;
    int start_c  = (start_col / d->kwidth) / d->kheight;

    for (int row = 0; row < m; row++) {
        // int row = (b * oheight + x) * owidth + y;
        int y =  (start_row + row) % d->owidth;
        int x = ((start_row + row) / d->owidth) % d->oheight;
        int b = ((start_row + row) / d->owidth) / d->oheight;
        for (int col = 0, c = start_c, kx = start_kx, ky = start_ky; col < n; col++) {
            int ix = d->vstride * x + d->vdilation * kx - d->vpadding;
            int iy = d->hstride * y + d->hdilation * ky - d->hpadding;
            if (0 <= ix && ix < d->height && 0 <= iy && iy < d->width) {
                // in[b, ix, iy, c] += rows[row, col]
                #pragma omp atomic
                out[((b * d->height + ix) * d->width + iy) * d->channel + c] += rows[row * ldr + col];
            }
            ky++; if (ky >= d->kwidth) { ky = 0;
            kx++; if (kx >= d->kheight) { kx = 0;
            c++; } }
        }
    }
}

inline void add_bias_bn_relu_nhwc_inline(int mr, int nr, const float *restrict Cc, int ldCc, float beta, float *restrict C, int ldC, const convol_dim *dim, int start_row, int start_col)
{
    for (int j = 0; j < nr; j++) {
        const float *in = Cc + j * ldCc;
        float *out = C + start_row + (start_col + j) * ldC;
        for (int i = 0, ri = start_row; i < mr; i++, ri++) {
            float tmp = in[i];
            if (beta != 0.0) tmp += out[i];
            tmp += dim->bias_vector[ri]; // add bias
            tmp = (tmp - dim->running_mean[ri]) * dim->inv_std[ri]; // batchnorm
            tmp = (tmp * dim->gamma[ri]) + dim->beta[ri];
            if (tmp < 0) tmp = 0; // relu
            out[i] = tmp;
        }
    }
}

inline void add_bias_bn_nhwc_inline(int mr, int nr, const float *restrict Cc, int ldCc, float beta, float *restrict C, int ldC, const convol_dim *dim, int start_row, int start_col)
{
    for (int j = 0; j < nr; j++) {
        const float *in = Cc + j * ldCc;
        float *out = C + start_row + (start_col + j) * ldC;
        for (int i = 0, ri = start_row; i < mr; i++, ri++) {
            float tmp = in[i];
            if (beta != 0.0) tmp += out[i];
            tmp += dim->bias_vector[ri]; // add bias
            tmp = (tmp - dim->running_mean[ri]) * dim->inv_std[ri]; // batchnorm
            tmp = (tmp * dim->gamma[ri]) + dim->beta[ri];
            out[i] = tmp;
        }
    }
}

inline void add_bias_relu_nhwc_inline(int mr, int nr, const float *restrict Cc, int ldCc, float beta, float *restrict C, int ldC, const convol_dim *dim, int start_row, int start_col)
{
    for (int j = 0; j < nr; j++) {
        const float *in = Cc + j * ldCc;
        float *out = C + start_row + (start_col + j) * ldC;
        for (int i = 0, ri = start_row; i < mr; i++, ri++) {
            float tmp = in[i];
            if (beta != 0.0) tmp += out[i];
            tmp += dim->bias_vector[ri]; // add bias
            if (tmp < 0) tmp = 0; // relu
            out[i] = tmp;
        }
    }
}

inline void add_bias_nhwc_inline(int mr, int nr, const float *restrict Cc, int ldCc, float beta, float *restrict C, int ldC, const convol_dim *dim, int start_row, int start_col)
{
    for (int j = 0; j < nr; j++) {
        const float *in = Cc + j * ldCc;
        float *out = C + start_row + (start_col + j) * ldC;
        for (int i = 0, ri = start_row; i < mr; i++, ri++) {
            float tmp = in[i];
            if (beta != 0.0) tmp += out[i];
            tmp += dim->bias_vector[ri]; // add bias
            out[i] = tmp;
        }
    }
}

void add_bias_nhwc(int mr, int nr, const float *restrict Cc, int ldCc, float beta, float *restrict C, int ldC, const convol_dim *dim, int start_row, int start_col, bool last)
{
    if (!last) {

        sxpbyM(mr, nr, Cc, ldCc, beta, C + start_row + start_col * ldC, ldC);

    } else if (dim->bias_vector && dim->running_mean && dim->relu) { // fused convgemm + bn + relu

        if (beta == 0.0)
            add_bias_bn_relu_nhwc_inline(mr, nr, Cc, ldCc, 0.0, C, ldC, dim, start_row, start_col);
        else
            add_bias_bn_relu_nhwc_inline(mr, nr, Cc, ldCc, 1.0, C, ldC, dim, start_row, start_col);

    } else if (dim->bias_vector && dim->running_mean && !dim->relu) { // fused convgemm + bn

        if (beta == 0.0)
            add_bias_bn_nhwc_inline(mr, nr, Cc, ldCc, 0.0, C, ldC, dim, start_row, start_col);
        else
            add_bias_bn_nhwc_inline(mr, nr, Cc, ldCc, 1.0, C, ldC, dim, start_row, start_col);

    } else if (dim->bias_vector && !dim->running_mean && dim->relu) { // fused convgemm + relu

        if (beta == 0.0)
            add_bias_relu_nhwc_inline(mr, nr, Cc, ldCc, 0.0, C, ldC, dim, start_row, start_col);
        else
            add_bias_relu_nhwc_inline(mr, nr, Cc, ldCc, 1.0, C, ldC, dim, start_row, start_col);

    } else if (!dim->bias_vector && !dim->running_mean && !dim->relu) { // fused convgemm + bias

        if (beta == 0.0)
            add_bias_nhwc_inline(mr, nr, Cc, ldCc, 0.0, C, ldC, dim, start_row, start_col);
        else
            add_bias_nhwc_inline(mr, nr, Cc, ldCc, 1.0, C, ldC, dim, start_row, start_col);

    } else { // Unoptimized fallback
        add_bias_nhwc_inline(mr, nr, Cc, ldCc, beta, C, ldC, dim, start_row, start_col);
    }
}
