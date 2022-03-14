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

#include <stdlib.h>
#include <stdbool.h>

#include "gemm_blis.h"
#include "im2col_nchw.h"

/*
 * BLIS pack for M-->Mc using implicit im2col
*/
void pack_RB_nchw(char orderM, char transM, int mc, int nc, const float *restrict M, int ldM, float *restrict Mc,
                  int RR, const conv_p *conv_params, int start_i, int start_j) {
    if (((transM == 'N') && (orderM == 'C')) || ((transM == 'T') && (orderM == 'R'))) {
        // initial kernel positions
        int start_ky = start_j % conv_params->kwidth;
        int start_kx = (start_j / conv_params->kwidth) % conv_params->kheight;
        int start_c = (start_j / conv_params->kwidth) / conv_params->kheight;
#pragma omp parallel for
        for (int i = 0; i < mc; i += RR) {
            int k = i * nc;
            int rr = min(mc - i, RR);
            int ky = start_ky;
            int kx = start_kx;
            int c = start_c;
            // initial pixel positions
            int start_y = (start_i + i) % conv_params->owidth;
            int start_x = ((start_i + i) / conv_params->owidth) % conv_params->oheight;
            int start_b = ((start_i + i) / conv_params->owidth) / conv_params->oheight;
            for (int j = 0; j < nc; j++) {
                int y = start_y;
                int x = start_x;
                int b = start_b;
                int ii = 0;
                for (; ii < rr; ii++) {
                    // Mc[k] = Mcol(i+ii,j);
                    int ix = conv_params->vstride * x + conv_params->vdilation * kx - conv_params->vpadding;
                    int iy = conv_params->hstride * y + conv_params->hdilation * ky - conv_params->hpadding;
                    if (0 <= ix && ix < conv_params->height && 0 <= iy && iy < conv_params->width) {
                        Mc[k] = M[((b * conv_params->channels + c) * conv_params->height + ix) * conv_params->width + iy];
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
                for (; ii < RR; ii++) {
                    Mc[k] = 0.0;
                    k++;
                }
                // k += (RR-rr);
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
        int start_y = start_j % conv_params->owidth;
        int start_x = (start_j / conv_params->owidth) % conv_params->oheight;
        int start_b = (start_j / conv_params->owidth) / conv_params->oheight;
#pragma omp parallel for
        for (int i = 0; i < mc; i += RR) {
            int k = i * nc;
            int rr = min(mc - i, RR);
            int y = start_y;
            int x = start_x;
            int b = start_b;
            int start_ky = (start_i + i) % conv_params->kwidth;
            int start_kx = ((start_i + i) / conv_params->kwidth) % conv_params->kheight;
            int start_c = ((start_i + i) / conv_params->kwidth) / conv_params->kheight;
            for (int j = 0; j < nc; j++) {
                int ky = start_ky;
                int kx = start_kx;
                int c = start_c;
                int ii = 0;
                for (; ii < rr; ii++) {
                    // Mc[k] = Mcol(j,i+ii);
                    int ix = conv_params->vstride * x + conv_params->vdilation * kx - conv_params->vpadding;
                    int iy = conv_params->hstride * y + conv_params->hdilation * ky - conv_params->hpadding;
                    if (0 <= ix && ix < conv_params->height && 0 <= iy && iy < conv_params->width) {
                        Mc[k] = M[((b * conv_params->channels + c) * conv_params->height + ix) * conv_params->width + iy];
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
                for (; ii < RR; ii++) {
                    Mc[k] = 0.0;
                    k++;
                }
                // k += (RR-rr);
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

void im2col_nchw(float *restrict cols, int ld, const float *restrict in, int batches, int channels, int height, int width,
                 int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding, int vstride, int hstride,
                 int vdilation, int hdilation) {
#if 1
#pragma omp parallel for
    for (int b = 0; b < batches; b++)
        for (int c = 0; c < channels; c++)
            for (int kx = 0; kx < kheight; kx++)
                for (int ky = 0; ky < kwidth; ky++) {
                    int row = c * kheight * kwidth + kx * kwidth + ky;
                    for (int x = 0; x < oheight; x++) {
                        int ix = vstride * x + vdilation * kx - vpadding;
                        if (0 <= ix && ix < height)
                            for (int y = 0; y < owidth; y++) {
                                int iy = hstride * y + hdilation * ky - hpadding;
                                if (0 <= iy && iy < width) {
                                    int col = b * oheight * owidth + x * owidth + y;
                                    // cols[row, col] = in[b, c, ix, iy]
                                    cols[row * batches * oheight * owidth + col] = in[
                                            ((b * channels + c) * height + ix) * width + iy];
                                }
                            }
                    }
                }
#else
    assert(start_row < channels * kheight * kwidth);
    assert(end_row <= channels * kheight * kwidth);
    assert(start_col < oheight * owidth * batches);
    assert(end_col <= oheight * owidth * batches);
    // starting values for the first row
    // int row = (c * kheight + kx) * kwidth + ky;
    int ky =  start_row % kwidth;
    int kx = (start_row / kwidth) % kheight;
    int c  = (start_row / kwidth) / kheight;
    // starting values for the first column
    // int col = (b * oheight + x) * owidth + y;
    int start_y =  start_col % owidth;
    int start_x = (start_col / owidth) % oheight;
    int start_b = (start_col / owidth) / oheight;

    // #pragma omp parallel for
    for (int row = 0; row < end_row - start_row; row++) {
        for (int col = 0, b = start_b, x = start_x, y = start_y; col < end_col - start_col; col++) {
            int ix = vstride * x + vdilation * kx - vpadding;
            int iy = hstride * y + hdilation * ky - hpadding;
            if (0 <= ix && ix < height && 0 <= iy && iy < width) {
                // cols[row, col] = in[b, ix, iy, c]
                cols[row * ld + col] = in[((b * channels + c) * height + ix) * width + iy];
            } else cols[row * ld + col] = 0;
            y++; if (y >= owidth) { y = 0;
            x++; if (x >= oheight) { x = 0;
            b++; } }
        }
        ky++; if (ky >= kwidth) { ky = 0;
        kx++; if (kx >= kheight) { kx = 0;
        c++; } }
    }
#endif
}

void transpose_nchw(int rows, int cols, const float *restrict in, int ld, float beta, float *restrict out, int kn,
                    int ho, int wo, int start_row, int start_col) {
    // transpose first and second dimension
    int start_y = start_row % wo;
    int start_x = (start_row / wo) % ho;
    int start_b = (start_row / wo) / ho;
#pragma omp parallel for
    for (int j = 0; j < cols; j++) {
        int y = start_y;
        int x = start_x;
        int b = start_b;
        for (int i = 0; i < rows; i++) {
            // out[((b * kn + k) * ho + x) * wo + y] = in[((j * batches + b) * ho + x) * wo + y];
            int idx = ((b * kn + start_col + j) * ho + x) * wo + y;
            if (beta == 0.0) out[idx] = in[j * ld + i];
            else out[idx] = beta * out[idx] + in[j * ld + i];
            y++;
            if (y >= wo) {
                y = 0;
                x++;
                if (x >= ho) {
                    x = 0;
                    b++;
                }
            }
        }
    }
}

/*
 * BLIS pack for M-->Mc transposing first and second tensor dimensions
*/
void pack_CB_nchw_trans(char orderM, char transM, int mc, int nc, const float *restrict M, int ldM, float *restrict Mc,
                        int RR, const conv_p *conv_params, int start_row, int start_col) {
    if (((transM == 'N') && (orderM == 'C')) || ((transM == 'T') && (orderM == 'R'))) {
#pragma omp parallel for
        for (int j = 0; j < nc; j += RR) {
            int k = j * mc;
            int nr = min(nc - j, RR);
            for (int i = 0; i < mc; i++) {
                int jj = 0;
                for (; jj < nr; jj++) {
                    // Mc[k] = Mcol(i, j + jj);
                    // Mc[k] = M[start_row + i + (start_col + j + jj) * ldM];
                    // Mc[k] = M[(z * ho + x) * wo + y + (start_col + j + jj) * batches * ho * wo];
                    // Mc[k] = M[(((start_col + j + jj) * batches + z) * ho + x) * wo + y];
                    int y = (start_row + i) % conv_params->owidth;
                    int x = ((start_row + i) / conv_params->owidth) % conv_params->oheight;
                    int b = ((start_row + i) / conv_params->owidth) / conv_params->oheight;
                    Mc[k] = M[((b * conv_params->kn + (start_col + j + jj)) * conv_params->oheight + x) * conv_params->owidth + y];
                    k++;
                }
                for (; jj < RR; jj++) {
                    Mc[k] = 0.0;
                    k++;
                }
                // k += (RR-nr);
            }
        }
    } else {
        // TODO
        abort();
        for (int j = 0; j < nc; j += RR) {
            int k = j * mc;
            int nr = min(nc - j, RR);
            for (int i = 0; i < mc; i++) {
                int jj = 0;
                for (; jj < nr; jj++) {
                    Mc[k] = Mcol(j + jj, i);
                    k++;
                }
                for (; jj < RR; jj++) {
                    Mc[k] = (float) 0.0;
                    k++;
                }
                // k += (RR-nr);
            }
        }
    }
}

/*
 * BLIS pack for M-->Mc transposing first and second tensor dimensions
*/
void pack_RB_nchw_trans(char orderM, char transM, int mc, int nc, const float *restrict M, int ldM, float *restrict Mc,
                        int RR, const conv_p *conv_params, int start_row, int start_col) {
    if (((transM == 'N') && (orderM == 'C')) || ((transM == 'T') && (orderM == 'R'))) {
#pragma omp parallel for
        for (int i = 0; i < mc; i += RR) {
            int k = i * nc;
            int rr = min(mc - i, RR);
            for (int j = 0; j < nc; j++) {
                int ii = 0;
                for (; ii < rr; ii++) {
                    // Mc[k] = Mcol(i + ii, j);
                    // Mc[k] = M[start_row + i + ii + (start_col + j) * ldM];
                    int y = (start_row + i + ii) % conv_params->owidth;
                    int x = ((start_row + i + ii) / conv_params->owidth) % conv_params->oheight;
                    int b = ((start_row + i + ii) / conv_params->owidth) / conv_params->oheight;
                    Mc[k] = M[((b * conv_params->kn + (start_col + j)) * conv_params->oheight + x) * conv_params->owidth + y];
                    k++;
                }
                for (; ii < RR; ii++) {
                    Mc[k] = 0.0;
                    k++;
                }
                // k += (RR-rr);
            }
        }
    } else {
        // TODO
        abort();
        for (int i = 0; i < mc; i += RR) {
            int k = i * nc;
            int rr = min(mc - i, RR);
            for (int j = 0; j < nc; j++) {
                int ii = 0;
                for (; ii < rr; ii++) {
                    Mc[k] = Mcol(j, i + ii);
                    k++;
                }
                for (; ii < RR; ii++) {
                    Mc[k] = (float) 0.0;
                    k++;
                }
                // k += (RR-rr);
            }
        }
    }
}

void col2im_nchw(int m, int n, const float *restrict cols, int ld, float *restrict out, int batches, int channels,
                 int height, int width, int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding,
                 int vstride, int hstride, int vdilation, int hdilation) {
#pragma omp parallel for
    for (int b = 0; b < batches; b++)
        for (int c = 0; c < channels; c++)
            for (int kx = 0; kx < kheight; kx++)
                for (int ky = 0; ky < kwidth; ky++) {
                    int row = c * kheight * kwidth + kx * kwidth + ky;
                    for (int x = 0; x < oheight; x++) {
                        int ix = vstride * x + vdilation * kx - vpadding;
                        if (0 <= ix && ix < height)
                            for (int y = 0; y < owidth; y++) {
                                int iy = hstride * y + hdilation * ky - hpadding;
                                if (0 <= iy && iy < width) {
                                    int col = (b * oheight + x) * owidth + y;
                                    out[((b * channels + c) * height + ix) * width + iy] += cols[
                                            row * batches * oheight * owidth + col];
                                }
                            }
                    }
                }
}

void post_col2im_nchw(int n, int m, const float *restrict cols, int ldc, float beta, float *restrict out, int ldout,
                      const conv_p *d, int start_col, int start_row, bool last) {
    /* int m = channels * kheight * kwidth;
    int n = oheight * owidth * batches; */

    // starting values for the first column
    // int col = (b * oheight + x) * owidth + y;
    int start_y = start_col % d->owidth;
    int start_x = (start_col / d->owidth) % d->oheight;
    int start_b = (start_col / d->owidth) / d->oheight;

    for (int row = 0; row < m; row++) {
        // int row = (c * kheight + kx) * kwidth + ky;
        unsigned ky = (start_row + row) % d->kwidth;
        unsigned kx = ((start_row + row) / d->kwidth) % d->kheight;
        int c = ((start_row + row) / d->kwidth) / d->kheight;
        for (int col = 0, b = start_b, x = start_x, y = start_y; col < n; col++) {
            int ix = d->vstride * x + d->vdilation * kx - d->vpadding;
            int iy = d->hstride * y + d->hdilation * ky - d->hpadding;
            if (0 <= ix && ix < d->height && 0 <= iy && iy < d->width) {
#pragma omp atomic
                out[((b * d->channels + c) * d->height + ix) * d->width + iy] += cols[row * ldc + col];
            }
            y++;
            if (y >= d->owidth) {
                y = 0;
                x++;
                if (x >= d->oheight) {
                    x = 0;
                    b++;
                }
            }
        }
    }
}

static inline void
add_bias_transpose_nchw_inline(int mr, int nr, const float *restrict Cc, int ldCc, float beta, float *restrict C,
                               int ldC, const conv_p *conv_params, int start_row, int start_col, bool last, bool bias,
                               bool batchnorm, bool relu) {
    // transpose first and second dimension
    int start_y = start_row % conv_params->owidth;
    int start_x = (start_row / conv_params->owidth) % conv_params->oheight;
    int start_b = (start_row / conv_params->owidth) / conv_params->oheight;
    for (int j = 0; j < nr; j++) {
        int y = start_y;
        int x = start_x;
        int b = start_b;
        int k = start_col + j;
        float bv;
        if (last && bias) bv = conv_params->bias_vector[k];
        float rm, is, ga, be;
        if (last && batchnorm) {
            rm = conv_params->running_mean[k];
            is = conv_params->inv_std[k];
            ga = conv_params->gamma[k];
            be = conv_params->beta[k];
        }
        for (int i = 0; i < mr; i++) {
            // out[((b * kn + k) * ho + x) * wo + y] = in[((j * batches + b) * ho + x) * wo + y];
            int idx = ((b * conv_params->kn + k) * conv_params->oheight + x) * conv_params->owidth + y;
            float tmp = Cc[j * ldCc + i];
            if (beta != 0.0) tmp += C[idx];
            if (last) {
                if (bias) tmp += bv;
                if (batchnorm) {
                    tmp = (tmp - rm) * is;
                    tmp = (tmp * ga) + be;
                }
                if (relu && tmp < 0) tmp = 0;
            }
            C[idx] = tmp;
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

void add_bias_transpose_nchw(int mr, int nr, const float *restrict Cc, int ldCc, float beta, float *restrict C, int ldC,
                             const conv_p *conv_params, int start_row, int start_col, bool last) {
#if 1
    if (!last) {
        if (beta == 0.0) // first pass
            add_bias_transpose_nchw_inline(mr, nr, Cc, ldCc, 0.0, C, ldC, conv_params, start_row, start_col, false, false,
                                           false, false);
        else // most common case
            add_bias_transpose_nchw_inline(mr, nr, Cc, ldCc, 1.0, C, ldC, conv_params, start_row, start_col, false, false,
                                           false, false);

    } else if (conv_params->bias_vector && conv_params->running_mean && conv_params->relu) {
        // fused convgemm + bn + relu
        if (beta == 0.0)
            add_bias_transpose_nchw_inline(mr, nr, Cc, ldCc, 0.0, C, ldC, conv_params, start_row, start_col, true, true, true,
                                           true);
        else
            add_bias_transpose_nchw_inline(mr, nr, Cc, ldCc, 1.0, C, ldC, conv_params, start_row, start_col, true, true, true,
                                           true);

    } else if (conv_params->bias_vector && conv_params->running_mean && !conv_params->relu) {
        // fused convgemm + bn
        if (beta == 0.0)
            add_bias_transpose_nchw_inline(mr, nr, Cc, ldCc, 0.0, C, ldC, conv_params, start_row, start_col, true, true, true,
                                           false);
        else
            add_bias_transpose_nchw_inline(mr, nr, Cc, ldCc, 1.0, C, ldC, conv_params, start_row, start_col, true, true, true,
                                           false);

    } else if (conv_params->bias_vector && !conv_params->running_mean && conv_params->relu) {
        // fused convgemm + relu
        if (beta == 0.0)
            add_bias_transpose_nchw_inline(mr, nr, Cc, ldCc, 0.0, C, ldC, conv_params, start_row, start_col, true, true, false,
                                           true);
        else
            add_bias_transpose_nchw_inline(mr, nr, Cc, ldCc, 1.0, C, ldC, conv_params, start_row, start_col, true, true, false,
                                           true);

    } else {
        // Unoptimized fallback
        add_bias_transpose_nchw_inline(mr, nr, Cc, ldCc, beta, C, ldC, conv_params, start_row, start_col, true,
                                       conv_params->bias_vector, conv_params->running_mean, conv_params->relu);
    }
#else
    add_bias_transpose_nchw_inline(mr, nr, Cc, ldCc, beta, C, ldC, conv_params, start_row, start_col, last,
                                   conv_params->bias_vector, conv_params->running_mean, conv_params->relu);
#endif
}
