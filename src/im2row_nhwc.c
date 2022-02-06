/**
 * This file is part of convGemmNHWC
 *
 * Copyright (C) 2021-22 Universitat Politècnica de València and
 *                       Universitat Jaume I
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include <stdbool.h>

#include "gemm_blis.h"
#include "im2row_nhwc.h"

/*
 * BLIS pack for M-->Mc using implicit im2row
*/
void pack_CB_nhwc(char orderM, char transM, int mc, int nc, const float *restrict M, int ldM, float *restrict Mc,
                  int RR, const conv_p *conv_params, int start_row, int start_col) {
    if (((transM == 'N') && (orderM == 'C')) || ((transM == 'T') && (orderM == 'R'))) {
        // initial kernel positions
        int start_ky = start_row % conv_params->kwidth;
        int start_kx = (start_row / conv_params->kwidth) % conv_params->kheight;
        int start_c = (start_row / conv_params->kwidth) / conv_params->kheight;
#pragma omp parallel for
        for (int j = 0; j < nc; j += RR) {
            int k = j * mc;
            int nr = min(nc - j, RR);
            int ky = start_ky;
            int kx = start_kx;
            int c = start_c;
            // initial pixel positions
            int start_y = (start_col + j) % conv_params->owidth;
            int start_x = ((start_col + j) / conv_params->owidth) % conv_params->oheight;
            int start_b = ((start_col + j) / conv_params->owidth) / conv_params->oheight;
            for (int i = 0; i < mc; i++) {
                int y = start_y;
                int x = start_x;
                int b = start_b;
                int jj = 0;
                for (; jj < nr; jj++) {
                    // Mc[k] = Mcol(i,j+jj);
                    int ix = conv_params->vstride * x + conv_params->vdilation * kx - conv_params->vpadding;
                    int iy = conv_params->hstride * y + conv_params->hdilation * ky - conv_params->hpadding;
                    if (0 <= ix && ix < conv_params->height && 0 <= iy && iy < conv_params->width) {
                        Mc[k] = M[((b * conv_params->height + ix) * conv_params->width + iy) * conv_params->channels +
                                  c];
                    } else Mc[k] = 0.0;
                    k++;
                    // next pixel position
                    y++;
                    if (y >= conv_params->owidth) {
                        y = 0;
                        x++;
                        if (x >= conv_params->oheight) {
                            x = 0;
                            b++;
                        }
                    }
                }
                for (; jj < RR; jj++) {
                    Mc[k] = 0.0;
                    k++;
                }
                // k += (RR - nr);
                // next kernel position
                ky++;
                if (ky >= conv_params->kwidth) {
                    ky = 0;
                    kx++;
                    if (kx >= conv_params->kheight) {
                        kx = 0;
                        c++;
                    }
                }
            }
        }
    } else {
        int start_y = (start_row) % conv_params->owidth;
        int start_x = ((start_row) / conv_params->owidth) % conv_params->oheight;
        int start_b = ((start_row) / conv_params->owidth) / conv_params->oheight;
#pragma omp parallel for
        for (int j = 0; j < nc; j += RR) {
            int k = j * mc;
            int nr = min(nc - j, RR);
            int y = start_y;
            int x = start_x;
            int b = start_b;
            int start_ky = (start_col + j) % conv_params->kwidth;
            int start_kx = ((start_col + j) / conv_params->kwidth) % conv_params->kheight;
            int start_c = ((start_col + j) / conv_params->kwidth) / conv_params->kheight;
            for (int i = 0; i < mc; i++) {
                int ky = start_ky;
                int kx = start_kx;
                int c = start_c;
                int jj = 0;
                for (; jj < nr; jj++) {
                    // Mc[k] = Mcol(j+jj,i);
                    int ix = conv_params->vstride * x + conv_params->vdilation * kx - conv_params->vpadding;
                    int iy = conv_params->hstride * y + conv_params->hdilation * ky - conv_params->hpadding;
                    if (0 <= ix && ix < conv_params->height && 0 <= iy && iy < conv_params->width) {
                        Mc[k] = M[((b * conv_params->height + ix) * conv_params->width + iy) * conv_params->channels +
                                  c];
                    } else Mc[k] = 0.0;
                    k++;
                    // next kernel position
                    ky++;
                    if (ky >= conv_params->kwidth) {
                        ky = 0;
                        kx++;
                        if (kx >= conv_params->kheight) {
                            kx = 0;
                            c++;
                        }
                    }
                }
                for (; jj < RR; jj++) {
                    Mc[k] = 0.0;
                    k++;
                }
                // k += (RR - nr);
                // next pixel position
                y++;
                if (y >= conv_params->owidth) {
                    y = 0;
                    x++;
                    if (x >= conv_params->oheight) {
                        x = 0;
                        b++;
                    }
                }
            }
        }
    }
}

void im2row_nhwc(float *restrict rows, int ld, const float *restrict in, int batches,
                 int height, int width, int channels, int oheight, int owidth, int kheight, int kwidth, int vpadding,
                 int hpadding, int vstride, int hstride, int vdilation, int hdilation) {
#if 1
#pragma omp parallel for
    for (int b = 0; b < batches; b++)
        for (int x = 0; x < oheight; x++)
            for (int y = 0; y < owidth; y++) {
                int row = b * oheight * owidth + x * owidth + y;
                for (int kx = 0; kx < kheight; kx++) {
                    int ix = vstride * x + vdilation * kx - vpadding;
                    if (0 <= ix && ix < height)
                        for (int ky = 0; ky < kwidth; ky++) {
                            int iy = hstride * y + hdilation * ky - hpadding;
                            if (0 <= iy && iy < width)
                                for (int c = 0; c < channels; c++) {
                                    int col = c * kheight * kwidth + kx * kwidth + ky;
                                    rows[row * channels * kheight * kwidth + col] = in[
                                            ((b * height + ix) * width + iy) * channels + c];
                                }
                        }
                }
            }
#else
    assert(start_row < oheight * owidth * batches);
    assert(end_row <= oheight * owidth * batches);
    assert(start_col < channels * kheight * kwidth);
    assert(end_col <= channels * kheight * kwidth);
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
                rows[row * ld + col] = in[((b * height + ix) * width + iy) * channels + c];
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

void row2im_nhwc(int m, int n, const float *restrict rows, int ld, float *restrict out, int batches,
                 int height, int width, int channels, int oheight, int owidth, int kheight, int kwidth, int vpadding,
                 int hpadding, int vstride, int hstride, int vdilation, int hdilation) {
#pragma omp parallel for
    for (int b = 0; b < batches; b++)
        for (int x = 0; x < oheight; x++)
            for (int y = 0; y < owidth; y++) {
                int row = b * oheight * owidth + x * owidth + y;
                for (int kx = 0; kx < kheight; kx++) {
                    int ix = vstride * x + vdilation * kx - vpadding;
                    if (0 <= ix && ix < height)
                        for (int ky = 0; ky < kwidth; ky++) {
                            int iy = hstride * y + hdilation * ky - hpadding;
                            if (0 <= iy && iy < width)
                                for (int c = 0; c < channels; c++) {
                                    int col = c * kheight * kwidth + kx * kwidth + ky;
                                    // out[b, x_x, x_y, cc] += rows[row, col]
                                    out[((b * height + ix) * width + iy) * channels + c] += rows[
                                            row * channels * kheight * kwidth + col];
                                }
                        }
                }
            }
}

void post_row2im_nhwc(int m, int n, const float *restrict rows, int ldr, float beta, float *restrict out, int ldout,
                      const conv_p *conv_params, int start_row, int start_col, bool last) {
    /* int m = oheight * owidth * batches;
    int n = channels * kheight * kwidth; */

    // starting values for the first column
    // int col = (c * kheight + kx) * kwidth + ky;
    int start_ky = start_row % conv_params->kwidth;
    int start_kx = (start_row / conv_params->kwidth) % conv_params->kheight;
    int start_c = (start_row / conv_params->kwidth) / conv_params->kheight;

    for (int row = 0; row < n; row++) {
        // int row = (b * oheight + x) * owidth + y;
        int y = (start_col + row) % conv_params->owidth;
        int x = ((start_col + row) / conv_params->owidth) % conv_params->oheight;
        int b = ((start_col + row) / conv_params->owidth) / conv_params->oheight;
        for (int col = 0, c = start_c, kx = start_kx, ky = start_ky; col < m; col++) {
            int ix = conv_params->vstride * x + conv_params->vdilation * kx - conv_params->vpadding;
            int iy = conv_params->hstride * y + conv_params->hdilation * ky - conv_params->hpadding;
            if (0 <= ix && ix < conv_params->height && 0 <= iy && iy < conv_params->width) {
                // in[b, ix, iy, c] += rows[row, col]
#pragma omp atomic
                out[((b * conv_params->height + ix) * conv_params->width + iy) * conv_params->channels + c] += rows[
                        row * ldr + col];
            }
            ky++;
            if (ky >= conv_params->kwidth) {
                ky = 0;
                kx++;
                if (kx >= conv_params->kheight) {
                    kx = 0;
                    c++;
                }
            }
        }
    }
}

inline void add_bias_bn_relu_nhwc_inline(int mr, int nr, const float *restrict Cc, int ldCc, float beta,
                                         float *restrict C, int ldC, const conv_p *conv_params,
                                         int start_row, int start_col) {
    for (int j = 0; j < nr; j++) {
        const float *in = Cc + j * ldCc;
        float *out = C + start_row + (start_col + j) * ldC;
        for (int i = 0, ri = start_row; i < mr; i++, ri++) {
            float tmp = in[i];
            if (beta != 0.0) tmp += out[i];
            tmp += conv_params->bias_vector[ri]; // add bias
            tmp = (tmp - conv_params->running_mean[ri]) * conv_params->inv_std[ri]; // batchnorm
            tmp = (tmp * conv_params->gamma[ri]) + conv_params->beta[ri];
            if (tmp < 0) tmp = 0; // relu
            out[i] = tmp;
        }
    }
}

inline void add_bias_bn_nhwc_inline(int mr, int nr, const float *restrict Cc, int ldCc, float beta,
                                    float *restrict C, int ldC, const conv_p *conv_params,
                                    int start_row, int start_col) {
    for (int j = 0; j < nr; j++) {
        const float *in = Cc + j * ldCc;
        float *out = C + start_row + (start_col + j) * ldC;
        for (int i = 0, ri = start_row; i < mr; i++, ri++) {
            float tmp = in[i];
            if (beta != 0.0) tmp += out[i];
            tmp += conv_params->bias_vector[ri]; // add bias
            tmp = (tmp - conv_params->running_mean[ri]) * conv_params->inv_std[ri]; // batchnorm
            tmp = (tmp * conv_params->gamma[ri]) + conv_params->beta[ri];
            out[i] = tmp;
        }
    }
}

inline void add_bias_relu_nhwc_inline(int mr, int nr, const float *restrict Cc, int ldCc, float beta,
                                      float *restrict C, int ldC, const conv_p *conv_params,
                                      int start_row, int start_col) {
    for (int j = 0; j < nr; j++) {
        const float *in = Cc + j * ldCc;
        float *out = C + start_row + (start_col + j) * ldC;
        for (int i = 0, ri = start_row; i < mr; i++, ri++) {
            float tmp = in[i];
            if (beta != 0.0) tmp += out[i];
            tmp += conv_params->bias_vector[ri]; // add bias
            if (tmp < 0) tmp = 0; // relu
            out[i] = tmp;
        }
    }
}

inline void add_bias_nhwc_inline(int mr, int nr, const float *restrict Cc, int ldCc, float beta, float *restrict C,
                                 int ldC, const conv_p *conv_params, int start_row, int start_col) {
    for (int j = 0; j < nr; j++) {
        const float *in = Cc + j * ldCc;
        float *out = C + start_row + (start_col + j) * ldC;
        for (int i = 0, ri = start_row; i < mr; i++, ri++) {
            float tmp = in[i];
            if (beta != 0.0) tmp += out[i];
            tmp += conv_params->bias_vector[ri]; // add bias
            out[i] = tmp;
        }
    }
}

void add_bias_nhwc(int mr, int nr, const float *restrict Cc, int ldCc, float beta, float *restrict C, int ldC,
                   const conv_p *conv_params, int start_row, int start_col, bool last) {
    if (!last) {

        sxpbyM(mr, nr, Cc, ldCc, beta, C + start_row + start_col * ldC, ldC);

    } else if (conv_params->bias_vector && conv_params->running_mean &&
               conv_params->relu) { // fused convgemm + bn + relu

        if (beta == 0.0)
            add_bias_bn_relu_nhwc_inline(mr, nr, Cc, ldCc, 0.0, C, ldC, conv_params, start_row, start_col);
        else
            add_bias_bn_relu_nhwc_inline(mr, nr, Cc, ldCc, 1.0, C, ldC, conv_params, start_row, start_col);

    } else if (conv_params->bias_vector && conv_params->running_mean && !conv_params->relu) { // fused convgemm + bn

        if (beta == 0.0)
            add_bias_bn_nhwc_inline(mr, nr, Cc, ldCc, 0.0, C, ldC, conv_params, start_row, start_col);
        else
            add_bias_bn_nhwc_inline(mr, nr, Cc, ldCc, 1.0, C, ldC, conv_params, start_row, start_col);

    } else if (conv_params->bias_vector && !conv_params->running_mean && conv_params->relu) { // fused convgemm + relu

        if (beta == 0.0)
            add_bias_relu_nhwc_inline(mr, nr, Cc, ldCc, 0.0, C, ldC, conv_params, start_row, start_col);
        else
            add_bias_relu_nhwc_inline(mr, nr, Cc, ldCc, 1.0, C, ldC, conv_params, start_row, start_col);

    } else if (!conv_params->bias_vector && !conv_params->running_mean && !conv_params->relu) { // fused convgemm + bias

        if (beta == 0.0)
            add_bias_nhwc_inline(mr, nr, Cc, ldCc, 0.0, C, ldC, conv_params, start_row, start_col);
        else
            add_bias_nhwc_inline(mr, nr, Cc, ldCc, 1.0, C, ldC, conv_params, start_row, start_col);

    } else { // Unoptimized fallback
        add_bias_nhwc_inline(mr, nr, Cc, ldCc, beta, C, ldC, conv_params, start_row, start_col);
    }
}
