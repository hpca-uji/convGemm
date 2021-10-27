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
    float *kernel = random_alloc(kn * kh * kw * c);
    float *bias_vector = random_alloc(kn);

    float *out       = malloc(kn * ho * wo * b * sizeof(float));
    float *out_gemm  = malloc(kn * ho * wo * b * sizeof(float));
    float *aux_trans = malloc(kn * ho * wo * b * sizeof(float));
    float *aux       = malloc(c * kh * kw * ho * wo * b * sizeof(float));

    double t1 = get_time();
    memset(aux, 0, c * kh * kw * ho * wo * b * sizeof(float));
    im2row_nhwc(aux, c * kh * kw, image, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation);
    double t2 = get_time();
    double t_im2row = t2 - t1;

    t1 = get_time();
    sgemm('N', 'N', kn, ho * wo * b, kh * kw * c, alpha, kernel, kn, aux, kh * kw * c, beta, out_gemm, kn);
    t2 = get_time();
    for(int j = 0; j < ho * wo * b; j++) // add bias
        for(int i = 0; i < kn; i++)
            out_gemm[i + j * kn] += bias_vector[i];
    double t3 = get_time();
    gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'N', kn, ho * wo * b, kh * kw * c, alpha, kernel, kn, image, kh * kw * c, beta, out, kn, ac_pack, pack_RB, bc_pack, pack_CB_nhwc, cc_pack, add_bias_nhwc, cntx, &dim, bias_vector);
    double t4 = get_time();
    double t_gemm = t2 - t1;
    double t_extra = t3 - t2;
    double t_nhwc = t4 - t3;
    printf("\t%e %e %e %e", t_im2row, t_gemm, t_extra, t_nhwc);

    if (!check(kn * ho * wo * b, out_gemm, out)) {
        printf(" error in gemm_nhwc 'N' NHWC\n");
        return 2;
    }

    t1 = get_time();
    memset(aux, 0, c * kh * kw * ho * wo * b * sizeof(float));
    im2col_nchw(aux, b * ho * wo, image, b, c, h, w, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation);
    t2 = get_time();
    double t_im2col = t2 - t1;

    t1 = get_time();
    sgemm('N', 'N', ho * wo * b, kn, kh * kw * c, alpha, aux, ho * wo * b, kernel, kh * kw * c, beta, aux_trans, ho * wo * b);
    t2 = get_time();
    for(int i = 0; i < kn; i++) // add bias
        for(int j = 0; j < ho * wo * b; j++)
            aux_trans[i * ho * wo * b + j] += bias_vector[i];
    // transpose first and second dimension
    for (int i = 0; i < b; i++)
        for (int j = 0; j < kn; j++)
            for (int x = 0; x < ho; x++)
                for (int y = 0; y < wo; y++)
                    out_gemm[((i * kn + j) * ho + x) * wo + y] = aux_trans[((j * b + i) * ho + x) * wo + y];
    t3 = get_time();
    gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'N', ho * wo * b, kn, kh * kw * c, alpha, image, ho * wo * b, kernel, kh * kw * c, beta, out, ho * wo * b, ac_pack, pack_RB_nchw, bc_pack, pack_CB, cc_pack, add_bias_transpose_nchw, cntx, &dim, bias_vector);
    t4 = get_time();
    t_gemm = t2 - t1;
    t_extra = t3 - t2;
    double t_nchw = t4 - t3;
    printf("\t%e %e %e %e", t_im2col, t_gemm, t_extra, t_nchw);

    if (!check(kn * ho * wo * b, out_gemm, out)) {
        printf(" error in gemm_nchw 'N' NCHW\n");
        return 2;
    }
    printf("\n");

    return 0;
}
