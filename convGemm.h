// void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, const float *a, int *lda, const float *b, int *ldb, float *beta, float *c, int *ldc);

static inline void sgemm(char transa, char transb, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb, float beta, float *c, int ldc) {
    sgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

int alloc_pack_buffs(float** Ac_pack, float** Bc_pack);

void sconvGemmNHWC(char trans,
                    unsigned kn, unsigned kh, unsigned kw, unsigned c,
                    float alpha, float *in,
                    unsigned b, unsigned h, unsigned w,
                    unsigned vpadding, unsigned hpadding,
                    unsigned vstride, unsigned hstride,
                    unsigned vdilation, unsigned hdilation,
                    float *x, float beta,
                    float *out, float *bias_vector,
                    float *ac_pack, float *bc_pack);

void sconvGemmNHWC_back(unsigned kn, unsigned kh, unsigned kw, unsigned c,
                        float alpha, float *weights,
                        unsigned b, unsigned h, unsigned w,
                        unsigned hstride, unsigned vstride,
                        unsigned hpadding, unsigned vpadding,
                        unsigned vdilation, unsigned hdilation,
                        float *dy, float *dx,
                        float *ac_pack, float *bc_pack, float *cc_pack);
