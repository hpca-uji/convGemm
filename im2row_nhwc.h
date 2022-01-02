void im2row_nhwc(float *rows, int ld, const float *in, int batch, int height, int width, int channel, int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation);

void pack_CB_nhwc(char orderM, char transM, int mc, int nc, const float *M, int ldM, float *Mc, int RR, const convol_dim *dim, int start_row, int start_col);

void row2im_nhwc(int m, int n, const float *rows, int ld, float *out, int batch, int height, int width, int channel, int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation);

void post_row2im_nhwc(int m, int n, const float *rows, int ldr, float beta, float *out, int ldout, const convol_dim *d, int start_row, int start_col, bool last);

void add_bias_nhwc(int mr, int nr, const float *Cc, int ldCc, float beta, float *C, int ldC, const convol_dim *dim, int start_row, int start_col, bool last);
