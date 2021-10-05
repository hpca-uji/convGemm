void im2col_nchw(float *cols, int ld, const float *in, int batch, int channel, int height, int width, int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation, int start_row, int end_row, int start_col, int end_col);

void pack_RB_nchw( char orderM, char transM, int mc, int nc, float *M, int ldM, float *Mc, int RR, float *in, int batch, int height, int width, int channel, int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation, int start_i, int start_j);

void transpose_nchw(int rows, int cols, float *in, int ld, float beta, float *out, int kn, int ho, int wo, int start_row, int start_col);
