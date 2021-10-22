void gemm_nhwc_B3A2C0( char orderA, char orderB, char orderC,
                       char transA, char transB,
                       int m, int n, int k,
                       float alpha, const float *A, int ldA,
                                    const float *B, int ldB,
                       float beta,  float *C, int ldC,
                       float *Ac, pack_func pack_RB_func, float *Bc, pack_func pack_CB_func, float *Cc, cntx_t *cnt,
                       const convol_dim *dim, const float *bias_vector);
