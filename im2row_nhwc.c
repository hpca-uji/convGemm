#include <stdio.h>
#include <assert.h>

#define min(a,b) (((a)<(b))?(a):(b))
#define Mcol(a1,a2)  M[ (a2)*(ldM)+(a1) ]
#define Mrow(a1,a2)  M[ (a1)*(ldM)+(a2) ]

void pack_CB_nhwc( char orderM, char transM, int mc, int nc, float *M, int ldM, float *Mc, int RR, float *in, int batch, int height, int width, int channel, int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation, int start_i, int start_j)
{
/*
  BLIS pack for M-->Mc using implicit im2row
*/
    if (((transM == 'N') && (orderM == 'C')) || ((transM == 'T') && (orderM == 'R')))
    {
        // initial kernel positions
        int start_ky =  start_i % kwidth;
        int start_kx = (start_i / kwidth) % kheight;
        int start_c  = (start_i / kwidth) / kheight;
        #pragma omp parallel for
        for (int j = 0; j < nc; j += RR ) {
            int k = j * mc;
            int nr = min(nc - j, RR);
            int ky = start_ky;
            int kx = start_kx;
            int c  = start_c;
            // initial pixel positions
            int start_y  =  (start_j + j) % owidth;
            int start_x  = ((start_j + j) / owidth) % oheight;
            int start_b  = ((start_j + j) / owidth) / oheight;
            for (int i = 0; i < mc; i++) {
                int y = start_y;
                int x = start_x;
                int b = start_b;
                for (int jj=0; jj<nr; jj++ ) {
                    // Mc[k] = Mcol(i,j+jj);
                    int ix = vstride * x + vdilation * kx - vpadding;
                    int iy = hstride * y + hdilation * ky - hpadding;
                    if (0 <= ix && ix < height && 0 <= iy && iy < width) {
                        Mc[k] = in[b * height * width * channel +
                                   ix         * width * channel +
                                   iy                 * channel +
                                   c];
                    } else Mc[k] = 0.0;
                    k++;
                    // next pixel position
                    y++; if (y >= owidth) { y = 0;
                    x++; if (x >= oheight) { x = 0;
                    b++; } }
                }
                k += (RR-nr);
                // next kernel position
                ky++; if (ky >= kwidth) { ky = 0;
                kx++; if (kx >= kheight) { kx = 0;
                c++; } }
            }
        }
    } else {
        /* int start_y  =  (start_i) % owidth;
        int start_x  = ((start_i) / owidth) % oheight;
        int start_b  = ((start_i) / owidth) / oheight; */
        #pragma omp parallel for
        for (int j = 0; j < nc; j += RR) {
            int k = j * mc;
            int nr = min(nc - j, RR);
            /* int y = start_y;
            int x = start_x;
            int b = start_b;
            int start_ky =  (start_j + j) % kwidth;
            int start_kx = ((start_j + j) / kwidth) % kheight;
            int start_c  = ((start_j + j) / kwidth) / kheight; */
            for (int i = 0; i < mc; i++) {
                /* int ky = start_ky;
                int kx = start_kx;
                int c  = start_c; */
                for (int jj = 0; jj < nr; jj++) {
                    Mc[k] = Mcol(j+jj,i);
                    /* int ix = vstride * x + vdilation * kx - vpadding;
                    int iy = hstride * y + hdilation * ky - hpadding;
                    if (0 <= ix && ix < height && 0 <= iy && iy < width) {
                        Mc[k] = in[b * height * width * channel +
                                   ix         * width * channel +
                                   iy                 * channel +
                                   c];
                    } else Mc[k] = 0.0;*/
                    k++;
                    /* // next kernel position
                    ky++; if (ky >= kwidth) { ky = 0;
                    kx++; if (kx >= kheight) { kx = 0;
                    c++; } } */
                }
                k += (RR - nr);
                /* // next pixel position
                y++; if (y >= owidth) { y = 0;
                x++; if (x >= oheight) { x = 0;
                b++; } } */
            }
        }
    }
}

void im2row_nhwc(float *rows, int ld, const float *in, int batch, int height, int width, int channel, int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation, int start_row, int end_row, int start_col, int end_col)
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
                                    rows[row * channel * kheight * kwidth + col] = in[
                                        b * height * width * channel +
                                        ix      * width * channel +
                                        iy          * channel +
                                        c];
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
                // rows[row, col] = x[b, x_x, x_y, c]
                rows[row * ld + col] = in[
                    b * height * width * channel +
                    ix      * width * channel +
                    iy          * channel +
                    c];
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

