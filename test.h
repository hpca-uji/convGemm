static inline void sgemm(char transa, char transb, int m, int n, int k, float alpha, float *a, int lda, float *b, int ldb, float beta, float *c, int ldc) {
#if 0
    sgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#else
    bli_sgemm(transa == 'T' ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE,
              transb == 'T' ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE,
              m, n, k, &alpha, a, 1, lda, b, 1, ldb, &beta, c, 1, ldc);
#endif
}

static inline float *random_alloc(int n)
{
    float *a = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
        a[i] = (float)rand() / RAND_MAX;
    return a;
}

static inline bool check(int n, float *a, float *b)
{
    for (int i = 0; i < n; i++) {
        float d = fabsf((a[i] - b[i]) / a[i]);
        if (d > 1e-5) {
            printf(": %d %e %e", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

#define TEST_INIT \
    if (argc < 8 || argc > 15) { \
        printf("program params: rep b h w c kn kh kw [ vpadding hpadding vstride hstride vdilation hdilation]\n"); \
        return 1; \
    } \
    int rep = atoi(argv[1]); \
    int b  = atoi(argv[2]); \
    int h  = atoi(argv[3]); \
    int w  = atoi(argv[4]); \
    int c  = atoi(argv[5]); \
    int kn = atoi(argv[6]); \
    int kh = atoi(argv[7]); \
    int kw = atoi(argv[8]); \
    int vpadding  = argc >  9 ? atoi(argv[ 9]) : 1; \
    int hpadding  = argc > 10 ? atoi(argv[10]) : 1; \
    int vstride   = argc > 11 ? atoi(argv[11]) : 1; \
    int hstride   = argc > 12 ? atoi(argv[12]) : 1; \
    int vdilation = argc > 13 ? atoi(argv[13]) : 1; \
    int hdilation = argc > 14 ? atoi(argv[14]) : 1; \
    float alpha = 1.0; \
    float beta = 0.0; \
 \
    bli_init(); \
    cntx_t *cntx = bli_gks_query_cntx(); \
    bli_thread_set_num_threads(omp_get_max_threads()); \
    int ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) / vstride + 1; \
    int wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) / hstride + 1; \
    float *ac_pack, *bc_pack; \
    alloc_pack_buffs(ho * wo * b, kn, kh * kw * c, &ac_pack, &bc_pack); \
    printf("# %d %d %d %d %d %d %d %d %d %d %d %d %d\n", b, h, w, c, kn, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation); \
    convol_dim dim = { b, h, w, c, kn, kh, kw, vstride, hstride, vpadding, hpadding, vdilation, hdilation, ho, wo };
