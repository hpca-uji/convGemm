void gemm_nhwc_B3A2C0( char orderA, char orderB, char orderC,
                       char transA, char transB,
                       int m, int n, int k,
                       float alpha, const float *A, int ldA,
                                    const float *B, int ldB,
                       float beta,  float *C, int ldC,
                       float *Ac, float *Bc, float *Cc, cntx_t *cnt,
                       const float *in, const convol_dim *dim, const float *bias_vector);
