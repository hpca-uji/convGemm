void gemm_nchw_B3A2C0( char orderA, char orderB, char orderC,
                       char transA, char transB,
                       int m, int n, int k,
                       float alpha, const float *A, int ldA,
                                    const float *B, int ldB,
                       float beta,  float *C, int ldC,
                       float *Ac, pack_func pack_RB, float *Bc, pack_func pack_CB, float *Cc, cntx_t *cnt,
                       const convol_dim *dim, const float *bias_vector);
