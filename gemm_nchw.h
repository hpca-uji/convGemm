void gemm_nchw_B3A2C0( char orderA, char orderB, char orderC,
                       char transA, char transB,
                       int m, int n, int k,
                       float alpha, const float *A, int ldA,
                                    const float *B, int ldB,
                       float beta,  float *C, int ldC,
                       float *Ac, float *Bc, float *Cc, cntx_t *cnt,
                       const float *in, int b, int c, int h, int w, int ho, int wo, int kh, int kw, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation, const float *bias_vector);
