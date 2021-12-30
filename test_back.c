#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#include <omp.h>
#include <blis.h>

#include "test.h"
#include "convGemm.h"
#include "gemm_blis.h"
#include "im2row_nhwc.h"
#include "im2col_nchw.h"

int main(int argc, char *argv[])
{
    TEST_INIT

    float *kernel = random_alloc(kn * kh * kw * c);
    float *out    = random_alloc(kn * ho * wo * b);

    float *aux        = malloc(c * kh * kw * ho * wo * b * sizeof(float));
    float *aux_trans  = malloc(kn * ho * wo * b * sizeof(float));
    float *image      = malloc(b * h * w * c * sizeof(float));
    float *image2     = malloc(b * h * w * c * sizeof(float));
    float *image_gemm = malloc(b * h * w * c * sizeof(float));

    for (int r = 0; r < rep; r++) {
    if (r > 0) printf("%d %d %d", c * kh * kw, ho * wo * b, kn);

    double t1 = get_time();
    sgemm('T', 'N', c * kh * kw, ho * wo * b, kn, alpha, kernel, kn, out, kn, 0.0, aux, c * kh * kw);
    double t2 = get_time();
    memset(image_gemm, 0, b * h * w * c * sizeof(float));
    row2im_nhwc(ho * wo * b, c * kh * kw, aux, c * kh * kw, image_gemm, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation);
    double t3 = get_time();
    memset(image, 0, b * h * w * c * sizeof(float));
    gemm_blis_B3A2C0('C', 'C', 'C', 'T', 'N', c * kh * kw, ho * wo * b, kn, alpha, kernel, kn, out, kn, 1.0, image, c * kh * kw, ac_pack, pack_RB, bc_pack, pack_CB, cc_pack, post_row2im_nhwc, cntx, &dim, NULL, NULL, NULL, NULL, NULL, false);
    double t4 = get_time();
    double t_gemm = t2 - t1;
    double t_row2im = t3 - t2;
    double t_nhwc = t4 - t3;
    if (r > 0) printf("\t%e %e %e", t_gemm, t_row2im, t_nhwc);

    if (!check(b * h * w * c, image_gemm, image)) {
        printf(" error in gemm_blis_B3A2C0 NHWC\n");
        return 2;
    }

    t1 = get_time();
    memset(image2, 0, b * h * w * c * sizeof(float));
    gemm_blis_A3B2C0('C', 'C', 'C', 'T', 'N', c * kh * kw, ho * wo * b, kn, alpha, kernel, kn, out, kn, 1.0, image2, c * kh * kw, ac_pack, pack_RB, bc_pack, pack_CB, cc_pack, post_row2im_nhwc, cntx, &dim, NULL);
    t2 = get_time();
    t_nhwc = t2 - t1;
    if (r > 0) printf(" %e", t_nhwc);

    if (!check(b * h * w * c, image_gemm, image2)) {
        printf(" error in gemm_blis_A3B2C0 NHWC\n");
        return 2;
    }

    t1 = get_time();
    // transpose first and second dimension
    transpose_nchw(kn * ho * wo, b, out, kn * ho * wo, 0.0, aux_trans, b, ho, wo, 0, 0);
    t2 = get_time();
    sgemm('N', 'T', b * ho * wo, c * kh * kw, kn, alpha, aux_trans, b * ho * wo, kernel, c * kh * kw, 0.0, aux, b * ho * wo);
    t3 = get_time();
    memset(image_gemm, 0, b * h * w * c * sizeof(float));
    col2im_nchw(c * kh * kw, b * ho * wo, aux, b * ho * wo, image_gemm, b, c, h, w, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation);
    t4 = get_time();
    memset(image, 0, b * h * w * c * sizeof(float));
    gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'T', b * ho * wo, c * kh * kw, kn, alpha, out, b * ho * wo, kernel, c * kh * kw, 1.0, image, b * ho * wo, ac_pack, pack_RB_nchw_trans, bc_pack, pack_CB, cc_pack, post_col2im_nchw, cntx, &dim, NULL, NULL, NULL, NULL, NULL, false);
    double t5 = get_time();
    double t_trans = t2 - t1;
    t_gemm = t3 - t2;
    double t_col2im = t4 - t3;
    double t_nchw = t5 - t4;
    if (r > 0) printf("\t%e %e %e %e", t_trans, t_gemm, t_col2im, t_nchw);

    if (!check(b * h * w * c, image_gemm, image)) {
        printf(" error in gemm_back NCHW\n");
        return 2;
    }

    t1 = get_time();
    memset(image2, 0, b * h * w * c * sizeof(float));
    gemm_blis_A3B2C0('C', 'C', 'C', 'N', 'T', b * ho * wo, c * kh * kw, kn, alpha, out, b * ho * wo, kernel, c * kh * kw, 1.0, image2, b * ho * wo, ac_pack, pack_RB_nchw_trans, bc_pack, pack_CB, cc_pack, post_col2im_nchw, cntx, &dim, NULL);
    t2 = get_time();
    t_nchw = t2 - t1;
    if (r > 0) printf(" %e", t_nchw);

    if (!check(b * h * w * c, image_gemm, image2)) {
        printf(" error in gemm_blis_A3B2C0 NCHW\n");
        return 2;
    }

    if (r > 0) printf("\n");

    }

    return 0;
}
