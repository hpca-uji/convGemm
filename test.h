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
    if (argc < 7 || argc > 14) { \
        printf("program params: b h w c kn kh kw [ vpadding hpadding vstride hstride vdilation hdilation]\n"); \
        return 1; \
    } \
    int b  = atoi(argv[1]); \
    int h  = atoi(argv[2]); \
    int w  = atoi(argv[3]); \
    int c  = atoi(argv[4]); \
    int kn = atoi(argv[5]); \
    int kh = atoi(argv[6]); \
    int kw = atoi(argv[7]); \
    int vpadding  = argc >  8 ? atoi(argv[ 8]) : 1; \
    int hpadding  = argc >  9 ? atoi(argv[ 9]) : 1; \
    int vstride   = argc > 10 ? atoi(argv[10]) : 1; \
    int hstride   = argc > 11 ? atoi(argv[11]) : 1; \
    int vdilation = argc > 12 ? atoi(argv[12]) : 1; \
    int hdilation = argc > 13 ? atoi(argv[13]) : 1; \
    float alpha = 1.0; \
    float beta = 0.0; \
 \
    bli_init(); \
    cntx_t *cntx = bli_gks_query_cntx(); \
    int NC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NC, cntx); \
    int MC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MC, cntx); \
    int KC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_KC, cntx); \
    float *ac_pack = aligned_alloc(4096, MC * KC * sizeof(float)); \
    float *bc_pack = aligned_alloc(4096, KC * NC * sizeof(float)); \
    float *cc_pack = aligned_alloc(4096, MC * NC * sizeof(float)); \
    int ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) / vstride + 1; \
    int wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) / hstride + 1; \
    printf("%d %d %d %d %d %d %d %d %d %d %d %d %d ", b, h, w, c, kn, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation); \
