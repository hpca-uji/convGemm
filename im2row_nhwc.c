#include <stdio.h>
#include <assert.h>

#include "convGemm.h"
#include "im2row_nhwc.h"

#define min(a,b) (((a)<(b))?(a):(b))
#define Mcol(a1,a2)  M[ (a2)*(ldM)+(a1) ]

void pack_CB_nhwc(char orderM, char transM, int mc, int nc, const float *M, int ldM, float *Mc, int RR, const float *in, const convol_dim *d, int start_i, int start_j)
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
        // #pragma omp parallel for
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
                        Mc[k] = in[((b * d->height + ix) * d->width + iy) * d->channel + c];
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
                        Mc[k] = in[((b * d->height + ix) * d->width + iy) * d->channel + c];
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

void im2row_nhwc(float *rows, int ld, const float *in, int batch, int height, int width, int channel, int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation)
{
#if 1
    // #pragma omp parallel for
    for (int b = 0; b < batch; b++)
        for (int x = 0; x < oheight; x++)
            for (int y = 0; y < owidth; y++) {
                int row = b * oheight * owidth + x * owidth + y;
                for (int c = 0; c < channel; c++)
                    for (int kx = 0; kx < kheight; kx++) {
                        int ix = vstride * x + vdilation * kx - vpadding;
                        if (0 <= ix && ix < height)
                            for (int ky = 0; ky < kwidth; ky++) {
                                int iy = hstride * y + hdilation * ky - hpadding;
                                if (0 <= iy && iy < width) {
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

void row2im_nhwc(int m, int n, const float *rows, int ld, float *out, int batch, int height, int width, int channel, int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation, int start_row, int start_col)
{
#if 0
    // #pragma omp parallel for
    for (int b = 0; b < batch; b++)
        for (int x = 0; x < oheight; x++)
            for (int y = 0; y < owidth; y++) {
                int row = b * oheight * owidth + x * owidth + y;
                for (int c = 0; c < channel; c++)
                    for (int kx = 0; kx < kheight; kx++) {
                        int ix = vstride * x + vdilation * kx - vpadding;
                        if (0 <= ix && ix < height)
                            for (int ky = 0; ky < kwidth; ky++) {
                                int iy = hstride * y + hdilation * ky - hpadding;
                                if (0 <= iy && iy < width) {
                                    int col = c * kheight * kwidth + kx * kwidth + ky;
                                    // out[b, x_x, x_y, cc] += rows[row, col]
                                    out[((b  * height + ix) * width + iy) * channel + c] += rows[row * channel * kheight * kwidth + col];
                                }
                            }
                    }
            }
#else
    /* int m = oheight * owidth * batch;
    int n = channel * kheight * kwidth;
    int ld = channel * kheight * kwidth;
    int start_row = 0;
    int start_col = 0; */
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
    for (int row = 0; row < m; row++) {
        for (int col = 0, c = start_c, kx = start_kx, ky = start_ky; col < n; col++) {
            int ix = vstride * x + vdilation * kx - vpadding;
            int iy = hstride * y + hdilation * ky - hpadding;
            if (0 <= ix && ix < height && 0 <= iy && iy < width) {
                // in[b, ix, iy, c] += rows[row, col]
                out[((b * height + ix) * width + iy) * channel + c] += rows[row * ld + col];
            }
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
