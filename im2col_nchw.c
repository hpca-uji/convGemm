#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

#include "convGemm.h"
#include "im2col_nchw.h"

#define min(a,b) (((a)<(b))?(a):(b))
#define Mcol(a1,a2)  M[ (a2)*(ldM)+(a1) ]

void pack_RB_nchw(char orderM, char transM, int mc, int nc, const float *M, int ldM, float *Mc, int RR, const convol_dim *d, int start_i, int start_j)
{
/*
  BLIS pack for M-->Mc using implicit im2col
*/
    if (((transM == 'N') && (orderM == 'C')) || ((transM == 'T') && (orderM == 'R')))
    {
        // initial kernel positions
        int start_ky =  start_j % d->kwidth;
        int start_kx = (start_j / d->kwidth) % d->kheight;
        int start_c  = (start_j / d->kwidth) / d->kheight;
        // #pragma omp parallel for
        for (int i = 0; i < mc; i += RR) {
            int k  = i * nc;
            int rr = min(mc - i, RR);
            int ky = start_ky;
            int kx = start_kx;
            int c  = start_c;
            // initial pixel positions
            int start_y  =  (start_i + i) % d->owidth;
            int start_x  = ((start_i + i) / d->owidth) % d->oheight;
            int start_b  = ((start_i + i) / d->owidth) / d->oheight;
            for (int j = 0; j < nc; j++) {
                int y  = start_y;
                int x  = start_x;
                int b  = start_b;
                int ii = 0;
                for ( ; ii < rr; ii++) {
                    // Mc[k] = Mcol(i+ii,j);
                    int ix = d->vstride * x + d->vdilation * kx - d->vpadding;
                    int iy = d->hstride * y + d->hdilation * ky - d->hpadding;
                    if (0 <= ix && ix < d->height && 0 <= iy && iy < d->width) {
                        Mc[k] = M[((b * d->channel + c) * d->height + ix) * d->width + iy];
                    } else Mc[k] = 0.0;
                    k++;
                    // next pixel position
                    y++; if (y >= d->owidth) { y = 0;
                    x++; if (x >= d->oheight) { x = 0;
                    b++; } }
                }
                for ( ; ii < RR; ii++) {
                    Mc[k] = 0.0;
                    k++;
                }
                // k += (RR-rr);
                // next kernel position
                ky++; if (ky >= d->kwidth) { ky = 0;
                kx++; if (kx >= d->kheight) { kx = 0;
                c++; } }
            }
        }
    } else {
        int start_y =  start_j % d->owidth;
        int start_x = (start_j / d->owidth) % d->oheight;
        int start_b = (start_j / d->owidth) / d->oheight;
        // #pragma omp parallel for
        for (int i = 0; i < mc; i += RR) {
            int k  = i*nc;
            int rr = min(mc - i, RR);
            int y  = start_y;
            int x  = start_x;
            int b  = start_b;
            int start_ky =  (start_i + i) % d->kwidth;
            int start_kx = ((start_i + i) / d->kwidth) % d->kheight;
            int start_c  = ((start_i + i) / d->kwidth) / d->kheight;
            for (int j = 0; j < nc; j++) {
                int ky = start_ky;
                int kx = start_kx;
                int c  = start_c;
                int ii = 0;
                for ( ; ii < rr; ii++) {
                    // Mc[k] = Mcol(j,i+ii);
                    int ix = d->vstride * x + d->vdilation * kx - d->vpadding;
                    int iy = d->hstride * y + d->hdilation * ky - d->hpadding;
                    if (0 <= ix && ix < d->height && 0 <= iy && iy < d->width) {
                        Mc[k] = M[((b * d->channel + c) * d->height + ix) * d->width + iy];
                    } else Mc[k] = 0.0;
                    k++;
                    // next kernel position
                    ky++; if (ky >= d->kwidth) { ky = 0;
                    kx++; if (kx >= d->kheight) { kx = 0;
                    c++; } }
                }
                for ( ; ii < RR; ii++) {
                    Mc[k] = 0.0;
                    k++;
                }
                // k += (RR-rr);
                // next pixel position
                y++; if (y >= d->owidth) { y = 0;
                x++; if (x >= d->oheight) { x = 0;
                b++; } }
            }
        }
    }
}

void im2col_nchw(float *cols, int ld, const float *in, int batch, int channel, int height, int width, int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation)
{
#if 1
    // #pragma omp parallel for
    for (int c = 0; c < channel; c++)
        for (int kx = 0; kx < kheight; kx++)
            for (int ky = 0; ky < kwidth; ky++) {
                int row = c * kheight * kwidth + kx * kwidth + ky;
                for (int b = 0; b < batch; b++)
                    for (int x = 0; x < oheight; x++) {
                        int ix = vstride * x + vdilation * kx - vpadding;
                        if (0 <= ix && ix < height)
                            for (int y = 0; y < owidth; y++) {
                                int iy = hstride * y + hdilation * ky - hpadding;
                                if (0 <= iy && iy < width) {
                                    int col = b * oheight * owidth + x * owidth + y;
                                    // cols[row, col] = in[b, c, ix, iy]
                                    cols[row * batch * oheight * owidth + col] = in[((b * channel + c) * height + ix) * width + iy];
                                }
                            }
                    }
            }
#else
    assert(start_row < channel * kheight * kwidth);
    assert(end_row <= channel * kheight * kwidth);
    assert(start_col < oheight * owidth * batch);
    assert(end_col <= oheight * owidth * batch);
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
                cols[row * ld + col] = in[((b * channel + c) * height + ix) * width + iy];
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

void transpose_nchw(int rows, int cols, const float *in, int ld, float beta, float *out, int kn, int ho, int wo, int start_row, int start_col)
{
    // transpose first and second dimension
    int start_y =  start_row % wo;
    int start_x = (start_row / wo) % ho;
    int start_b = (start_row / wo) / ho;
    for (int j = 0; j < cols; j++) {
        int y = start_y;
        int x = start_x;
        int b = start_b;
        for (int i = 0; i < rows; i++) {
            // out[((b * kn + k) * ho + x) * wo + y] = in[((j * batch + b) * ho + x) * wo + y];
            int idx = ((b * kn + start_col + j) * ho + x) * wo + y;
            if (beta == 0.0) out[idx] = in[j * ld + i];
            else out[idx] = beta * out[idx] + in[j * ld + i];
            y++; if (y >= wo) { y = 0;
            x++; if (x >= ho) { x = 0;
            b++; } }
        }
    }
}

void pack_CB_nchw_trans(char orderM, char transM, int mc, int nc, const float *M, int ldM, float *Mc, int RR, const convol_dim *d, int start_row, int start_col)
{
/*
  BLIS pack for M-->Mc transposing first and second tensor dimensions
*/
    if ( ((transM=='N')&&( orderM=='C')) || ((transM=='T')&&( orderM=='R')) ) {
        for (int j = 0; j < nc; j += RR) {
            int k = j * mc;
            int nr = min(nc - j, RR);
            for (int i = 0; i < mc; i++) {
                int jj = 0;
                for (; jj < nr; jj++) {
                    // Mc[k] = Mcol(i, j + jj);
                    // Mc[k] = M[start_row + i + (start_col + j + jj) * ldM];
                    // Mc[k] = M[(z * ho + x) * wo + y + (start_col + j + jj) * batch * ho * wo];
                    // Mc[k] = M[(((start_col + j + jj) * batch + z) * ho + x) * wo + y];
                    int y =  (start_row + i) % d->owidth;
                    int x = ((start_row + i) / d->owidth) % d->oheight;
                    int b = ((start_row + i) / d->owidth) / d->oheight;
                    Mc[k] = M[((b * d->kn + (start_col + j + jj)) * d->oheight + x) * d->owidth + y];
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
                    Mc[k] = 0.0;
                    k++;
                }
                // k += (RR-nr);
            }
        }
    }
}

void pack_RB_nchw_trans( char orderM, char transM, int mc, int nc, const float *M, int ldM, float *Mc, int RR, const convol_dim *d, int start_row, int start_col)
{
/*
  BLIS pack for M-->Mc transposing first and second tensor dimensions
*/
    if (((transM == 'N') && (orderM == 'C')) || ((transM == 'T') && (orderM == 'R'))) {
        for (int i = 0; i < mc; i += RR) {
            int k = i * nc;
            int rr = min(mc - i, RR);
            for (int j = 0; j < nc; j++) {
                int ii = 0;
                for (; ii < rr; ii++) {
                    // Mc[k] = Mcol(i + ii, j);
                    // Mc[k] = M[start_row + i + ii + (start_col + j) * ldM];
                    int y =  (start_row + i + ii) % d->owidth;
                    int x = ((start_row + i + ii) / d->owidth) % d->oheight;
                    int b = ((start_row + i + ii) / d->owidth) / d->oheight;
                    Mc[k] = M[((b * d->kn + (start_col + j)) * d->oheight + x) * d->owidth + y];
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
                    Mc[k] = 0.0;
                    k++;
                }
                // k += (RR-rr);
            }
        }
    }
}

void pack_transpose_nchw(int rows, int cols, const float *in, int ld, float *out, int b, int ho, int wo, int start_row, int start_col)
{
    // transpose first and second dimension
    int start_y =  start_row % wo;
    int start_x = (start_row / wo) % ho;
    int start_k = (start_row / wo) / ho;
    for (int j = 0; j < cols; j++) {
        int y = start_y;
        int x = start_x;
        int k = start_k;
        for (int i = 0; i < rows; i++) {
            // out[((b * kn + k) * ho + x) * wo + y] = in[((j * batch + b) * ho + x) * wo + y];
            int idx = ((k * b + start_col + j) * ho + x) * wo + y;
            out[j * ld + i] = in[idx];
            y++; if (y >= wo) { y = 0;
            x++; if (x >= ho) { x = 0;
            k++; } }
        }
    }
}

void col2im_nchw(int m, int n, const float *cols, int ld, float *out, int batch, int channel, int height, int width, int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation, int start_row, int start_col)
{
#if 0
    // #pragma omp parallel for
    for (int c = 0; c < channel; c++)
        for (int kx = 0; kx < kheight; kx++)
            for (int ky = 0; ky < kwidth; ky++) {
                int row = c * kheight * kwidth + kx * kwidth + ky;
                for (int b = 0; b < batch; b++)
                    for (int x = 0; x < oheight; x++) {
                        int ix = vstride * x + vdilation * kx - vpadding;
                        if (0 <= ix && ix < height)
                            for (int y = 0; y < owidth; y++) {
                                int iy = hstride * y + hdilation * ky - hpadding;
                                if (0 <= iy && iy < width) {
                                    int col = (b * oheight + x) * owidth + y;
                                    out[((b * channel + c) * height + ix) * width + iy] += cols[row * batch * oheight * owidth + col];
                                }
                            }
                    }
            }
#else
    /* int m = channel * kheight * kwidth;
    int n = oheight * owidth * batch;
    int start_row = 0;
    int start_col = 0; */
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
    for (int row = 0; row < m; row++) {
        for (int col = 0, b = start_b, x = start_x, y = start_y; col < n; col++) {
            int ix = vstride * x + vdilation * kx - vpadding;
            int iy = hstride * y + hdilation * ky - hpadding;
            if (0 <= ix && ix < height && 0 <= iy && iy < width) {
                out[((b * channel + c) * height + ix) * width + iy] += cols[row * ld + col];
            }
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

void add_bias_nchw(int mr, int nr, float *Cc, float beta, float *C, int ldC, const convol_dim *dim, const float *bias_vector, int start_row, int start_col, bool last)
{
    if (last && bias_vector) {
        for (int j = 0; j < nr; j++)
            for (int i = 0; i < mr; i++)
                Cc[j * mr + i] += bias_vector[start_col + j];
    }
    transpose_nchw(mr, nr, Cc, mr, beta, C, dim->kn, dim->oheight, dim->owidth, start_row, start_col);
}
