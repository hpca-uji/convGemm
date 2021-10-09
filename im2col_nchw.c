#include <stdio.h>
#include <assert.h>

#include "im2col_nchw.h"

#define min(a,b) (((a)<(b))?(a):(b))
#define Mcol(a1,a2)  M[ (a2)*(ldM)+(a1) ]

void pack_RB_nchw(char orderM, char transM, int mc, int nc, const float *M, int ldM, float *Mc, int RR, const float *in, int batch, int channel, int height, int width, int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation, int start_i, int start_j)
{
/*
  BLIS pack for M-->Mc using implicit im2col
*/
    if (((transM == 'N') && (orderM == 'C')) || ((transM == 'T') && (orderM == 'R')))
    {
        // initial kernel positions
        int start_ky =  start_j % kwidth;
        int start_kx = (start_j / kwidth) % kheight;
        int start_c  = (start_j / kwidth) / kheight;
        // #pragma omp parallel for
        for (int i = 0; i < mc; i += RR) {
            int k  = i * nc;
            int rr = min(mc - i, RR);
            int ky = start_ky;
            int kx = start_kx;
            int c  = start_c;
            // initial pixel positions
            int start_y  =  (start_i + i) % owidth;
            int start_x  = ((start_i + i) / owidth) % oheight;
            int start_b  = ((start_i + i) / owidth) / oheight;
            for (int j = 0; j < nc; j++) {
                int y  = start_y;
                int x  = start_x;
                int b  = start_b;
                int ii = 0;
                for ( ; ii < rr; ii++) {
                    // Mc[k] = Mcol(i+ii,j);
                    int ix = vstride * x + vdilation * kx - vpadding;
                    int iy = hstride * y + hdilation * ky - hpadding;
                    if (0 <= ix && ix < height && 0 <= iy && iy < width) {
                        Mc[k] = in[((b * channel + c) * height + ix) * width + iy];
                    } else Mc[k] = 0.0;
                    k++;
                    // next pixel position
                    y++; if (y >= owidth) { y = 0;
                    x++; if (x >= oheight) { x = 0;
                    b++; } }
                }
                for ( ; ii < RR; ii++) {
                    Mc[k] = 0.0;
                    k++;
                }
                // k += (RR-rr);
                // next kernel position
                ky++; if (ky >= kwidth) { ky = 0;
                kx++; if (kx >= kheight) { kx = 0;
                c++; } }
            }
        }
    } else {
        int start_y =  start_j % owidth;
        int start_x = (start_j / owidth) % oheight;
        int start_b = (start_j / owidth) / oheight;
        // #pragma omp parallel for
        for (int i = 0; i < mc; i += RR) {
            int k  = i*nc;
            int rr = min(mc - i, RR);
            int y  = start_y;
            int x  = start_x;
            int b  = start_b;
            int start_ky =  (start_i + i) % kwidth;
            int start_kx = ((start_i + i) / kwidth) % kheight;
            int start_c  = ((start_i + i) / kwidth) / kheight;
            for (int j = 0; j < nc; j++) {
                int ky = start_ky;
                int kx = start_kx;
                int c  = start_c;
                int ii = 0;
                for ( ; ii < rr; ii++) {
                    // Mc[k] = Mcol(j,i+ii);
                    int ix = vstride * x + vdilation * kx - vpadding;
                    int iy = hstride * y + hdilation * ky - hpadding;
                    if (0 <= ix && ix < height && 0 <= iy && iy < width) {
                        Mc[k] = in[((b * channel + c) * height + ix) * width + iy];
                    } else Mc[k] = 0.0;
                    k++;
                    // next kernel position
                    ky++; if (ky >= kwidth) { ky = 0;
                    kx++; if (kx >= kheight) { kx = 0;
                    c++; } }
                }
                for ( ; ii < RR; ii++) {
                    Mc[k] = 0.0;
                    k++;
                }
                // k += (RR-rr);
                // next pixel position
                y++; if (y >= owidth) { y = 0;
                x++; if (x >= oheight) { x = 0;
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
            out[idx] = beta * out[idx] + in[j * ld + i];
            y++; if (y >= wo) { y = 0;
            x++; if (x >= ho) { x = 0;
            b++; } }
        }
    }
}

void col2im_nchw(const float *cols, int ld, float *out, int batch, int channel, int height, int width, int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation)
{
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
}
