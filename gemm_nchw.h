void gemm_nchw_B3A2C0( char orderA, char orderB, char orderC,
                       char transA, char transB,
                       int m, int n, int k,
                       float alpha, float *A, int ldA,
                                    float *B, int ldB,
                       float beta,  float *C, int ldC,
                       float *Ac, float *Bc, float *Cc, cntx_t *cnt,
                       float *in, int b, int h, int w, int c, int ho, int wo, int kh, int kw, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation, float *bias_vector);
