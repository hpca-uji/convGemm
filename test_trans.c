#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#include <blis.h>

#include "test.h"
#include "convGemm.h"
#include "gemm_blis.h"
#include "im2row_nhwc.h"
#include "im2col_nchw.h"

int main(int argc, char *argv[])
{
    TEST_INIT

    float *image = random_alloc(b * h * w * c);
    float *out = random_alloc(kn * ho * wo * b);

    float *aux_trans  = malloc(kn * ho * wo * b * sizeof(float));
    float *aux        = malloc(c * kh * kw * ho * wo * b * sizeof(float));
    float *kernel      = malloc(c * kh * kw * kn * sizeof(float));
    float *kernel_gemm = malloc(c * kh * kw * kn * sizeof(float));

    double t1 = get_time();
    memset(aux, 0, c * kh * kw * ho * wo * b * sizeof(float));
    im2row_nhwc(aux, c * kh * kw, image, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation);
    double t2 = get_time();
    double t_im2row = t2 - t1;

    t1 = get_time();
    sgemm('N', 'T', kn, kh * kw * c, ho * wo * b, alpha, out, kn, aux, kh * kw * c, beta, kernel_gemm, kn);
    t2 = get_time();
    gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'T', kn, kh * kw * c, ho * wo * b, alpha, out, kn, image, kh * kw * c, beta, kernel, kn, ac_pack, pack_RB, bc_pack, pack_CB_nhwc, cc_pack, NULL, cntx, &dim, NULL);
    double t3 = get_time();
    double t_gemm = t2 - t1;
    double t_nhwc = t3 - t2;
    printf("\t%e %e %e", t_im2row, t_gemm, t_nhwc);

    if (!check(c * kh * kw * kn, kernel_gemm, kernel)) {
        printf(" error in gemm_nhwc 'T' NHWC\n");
        return 2;
    }

    t1 = get_time();
    memset(aux, 0, c * kh * kw * ho * wo * b * sizeof(float));
    im2col_nchw(aux, b * ho * wo, image, b, c, h, w, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation);
    t2 = get_time();
    double t_im2col = t2 - t1;

    t1 = get_time();
    // transpose first and second dimension
    for (int i = 0; i < kn; i++)
        for (int j = 0; j < b; j++)
            for (int x = 0; x < ho; x++)
                for (int y = 0; y < wo; y++)
                    aux_trans[((i * b + j) * ho + x) * wo + y] = out[((j * kn + i) * ho + x) * wo + y];
    t2 = get_time();
    sgemm('T', 'N', kh * kw * c, kn, ho * wo * b, alpha, aux, ho * wo * b, aux_trans, ho * wo * b, beta, kernel_gemm, kh * kw * c);
    t3 = get_time();
    gemm_blis_B3A2C0('C', 'C', 'C', 'T', 'N', kh * kw * c, kn, ho * wo * b, alpha, image, ho * wo * b, out, ho * wo * b, beta, kernel, kh * kw * c, ac_pack, pack_RB_nchw, bc_pack, pack_CB_nchw_trans, cc_pack, NULL, cntx, &dim, NULL);
    double t4 = get_time();
    double t_extra = t2 - t1;
    t_gemm = t3 - t2;
    double t_nchw = t4 - t3;
    printf("\t%e %e %e %e", t_im2col, t_gemm, t_extra, t_nchw);

    if (!check(c * kh * kw * kn, kernel_gemm, kernel)) {
        printf(" error in gemm_nchw 'T' NCHW\n");
        return 3;
    }
    printf("\n");

    return 0;
}
