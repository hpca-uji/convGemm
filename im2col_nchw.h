void im2col_nchw(float *cols, int ld, const float *in, int batch, int channel, int height, int width, int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation);

void pack_RB_nchw(char orderM, char transM, int mc, int nc, const float *M, int ldM, float *Mc, int RR, const convol_dim *dim, int start_i, int start_j);

void transpose_nchw(int rows, int cols, const float *in, int ld, float beta, float *out, int kn, int ho, int wo, int start_row, int start_col);

void pack_CB_nchw_trans(char orderM, char transM, int mc, int nc, const float *M, int ldM, float *Mc, int RR, const convol_dim *dim, int start_row, int start_col);

void pack_RB_nchw_trans(char orderM, char transM, int mc, int nc, const float *M, int ldM, float *Mc, int RR, const convol_dim *dim, int start_row, int start_col);

void col2im_nchw(int m, int n, const float *cols, int ld, float *out, int batch, int channel, int height, int width, int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation);

void post_col2im_nchw(int n, int m, const float *cols, int ldc, float beta, float *out, int ldout, const convol_dim *d, const float *bias_vector, const float *bn_running_mean, const float *bn_inv_std, const float *bn_gamma, const float *bn_beta, bool relu, int start_col, int start_row, bool last);

void add_bias_transpose_nchw(int mr, int nr, const float *Cc, int ldCc, float beta, float *C, int ldC, const convol_dim *dim, const float *bias_vector, const float *bn_running_mean, const float *bn_inv_std, const float *bn_gamma, const float *bn_beta, bool relu, int start_row, int start_col, bool last);
