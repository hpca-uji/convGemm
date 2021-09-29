void im2row_nhwc(float *rows, int ld, const float *in, int batch, int height, int width, int channel, int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation, int start_row, int end_row, int start_col, int end_col);

void pack_CB_nhwc(char orderM, char transM, int mc, int nc, float *M, int ldM, float *Mc, int RR, float *in, int batch, int height, int width, int channel, int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation, int start_row, int start_col);

void row2im_nhwc(const float *rows, float *x, int n, int h, int w, int c, int hh, int ww, int kh, int kw, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation);
