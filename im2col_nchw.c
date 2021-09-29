#include <stdio.h>

#include "im2col_nchw.h"

void im2col_nchw(float *cols, const float *in, int batch, int channel, int height, int width, int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation)
{
    // #pragma omp parallel for
    for (int c = 0; c < channel; c++)
        for (int kh = 0; kh < kheight; kh++)
            for (int kw = 0; kw < kwidth; kw++) {
                int row = c * kheight * kwidth + kh * kwidth + kw;
                for (int b = 0; b < batch; b++)
                    for (int x = 0; x < oheight; x++) {
                        int ix = vstride * x + vdilation * kh - vpadding;
                        if (0 <= ix && ix < height)
                            for (int y = 0; y < owidth; y++) {
                                int iy = hstride * y + hdilation * kw - hpadding;
                                if (0 <= iy && iy < width) {
                                    int col = b * oheight * owidth + x * owidth + y;
                                    // cols[row, col] = in[b, c, ix, iy]
                                    // printf("cols[%d, %d] = in[%d, %d, %d, %d]\n", row, col, b, c, ix, iy);
                                    // printf("in[] = %e\n", in[b * channel * height * width + c * height * width + ix * width + iy]);
                                    // printf("cols[] = %e\n", cols[row * batch * oheight * owidth + col]);
                                    cols[row * batch * oheight * owidth + col] = in[
                                        b * channel * height * width +
                                        c           * height * width +
                                        ix                   * width +
                                        iy];
                                }
                            }
                    }
            }
}
