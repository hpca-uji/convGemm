/**
 * This file is part of convGemmNHWC
 *
 * Copyright (C) 2021-22 Universitat Politècnica de València and
 *                       Universitat Jaume I
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <blis/blis.h>

#include "convGemm.h"
#include "gemm_blis.h"
#include "im2row_nhwc.h"
#include "im2col_nchw.h"

int alloc_pack_buffs(float **Ac_pack, float **Bc_pack) {
    gemm_blis_init();
    int MC, NC, KC;
    gemm_blis_workspace(blis_cntx, 0, 0, 0, &MC, &NC, &KC);

    *Ac_pack = aligned_alloc(4096, omp_get_max_threads() * MC * KC * sizeof(float));
    *Bc_pack = aligned_alloc(4096, omp_get_max_threads() * KC * NC * sizeof(float));

    if (*Ac_pack == NULL || *Bc_pack == NULL) return 1;
    return 0;
}

/*
 * Computes
 *    out = in * im2row(x) + bias_vector
 * or
 *    out = in * transpose(im2row(x))
 */
void sconvGemmNHWC(char trans,
                   int b, int h, int w, int c,
                   int kn, int kh, int kw,
                   int vpadding, int hpadding,
                   int vstride, int hstride,
                   int vdilation, int hdilation,
                   const float *in,
                   const float *x,
                   float *out, const float *bias_vector,
                   const float *bn_running_mean, const float *bn_inv_std,
                   const float *bn_gamma, const float *bn_beta, bool relu,
                   float *ac_pack, float *bc_pack) {
    int ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) / vstride + 1;
    int wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) / hstride + 1;
    conv_p conv_params = {b, h, w, c, kn, kh, kw, vstride, hstride,
                          vpadding, hpadding, vdilation, hdilation, ho, wo,
                          bias_vector, bn_running_mean, bn_inv_std, bn_gamma, bn_beta, relu};
    if (trans == 'N') {
        gemm_blis_B3A2C0_orig('C', 'C', 'C', 'N', 'N', kn, ho * wo * b, kh * kw * c,
                              1.0, in, kn, x, kh * kw * c, 0.0,
                              out, kn, ac_pack, pack_RB, bc_pack, pack_CB_nhwc,
                              bias_vector == NULL && bn_running_mean == NULL && relu == false ? NULL : add_bias_nhwc,
                              blis_cntx, &conv_params);
    } else {
        gemm_blis_B3A2C0_orig('C', 'C', 'C', 'N', 'T', kn, kh * kw * c, ho * wo * b,
                              1.0, in, kn, x, kh * kw * c, 0.0,
                              out, kn, ac_pack, pack_RB, bc_pack, pack_CB_nhwc,
                              NULL, blis_cntx, &conv_params);
    }
}

/*
 * Computes dx = dx + row2im(transpose(weights) * dy)
 */
void sconvGemmNHWC_back(int b, int h, int w, int c,
                        int kn, int kh, int kw,
                        int vstride, int hstride,
                        int vpadding, int hpadding,
                        int vdilation, int hdilation,
                        const float *weights,
                        const float *dy, float *dx,
                        float *ac_pack, float *bc_pack) {
    int ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) / vstride + 1;
    int wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) / hstride + 1;
    conv_p conv_params = {b, h, w, c, kn, kh, kw, vstride, hstride,
                          vpadding, hpadding, vdilation, hdilation, ho, wo};
    gemm_blis_B3A2C0_orig('C', 'C', 'C', 'T', 'N', c * kh * kw, ho * wo * b, kn,
                          1.0, weights, kn, dy, kn, 1.0, dx, c * kh * kw, ac_pack, pack_RB, bc_pack, pack_CB,
                          post_row2im_nhwc, blis_cntx, &conv_params);
}

void sconvGemmNCHW(char trans,
                   int b, int c, int h, int w,
                   int kn, int kh, int kw,
                   int vpadding, int hpadding,
                   int vstride, int hstride,
                   int vdilation, int hdilation,
                   const float *in,
                   const float *x,
                   float *out, const float *bias_vector,
                   const float *bn_running_mean, const float *bn_inv_std,
                   const float *bn_gamma, const float *bn_beta, bool relu,
                   float *ac_pack, float *bc_pack) {
    int ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) / vstride + 1;
    int wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) / hstride + 1;
    conv_p conv_params = {b, h, w, c, kn, kh, kw,
                          vstride, hstride, vpadding, hpadding, vdilation, hdilation, ho, wo, bias_vector,
                          bn_running_mean, bn_inv_std, bn_gamma, bn_beta, relu};
    if (trans == 'N') {
        gemm_blis_B3A2C0('C', 'C', 'C', 'N', 'N', ho * wo * b, kn, kh * kw * c, 1.0, x, ho * wo * b, in, kh * kw * c,
                         0.0, out, ho * wo * b, ac_pack, pack_RB_nchw, bc_pack, pack_CB,
                         add_bias_transpose_nchw, blis_cntx, &conv_params);
    } else {
        gemm_blis_B3A2C0('C', 'C', 'C', 'T', 'N', kh * kw * c, kn, ho * wo * b, 1.0, x, ho * wo * b, in, ho * wo * b,
                         0.0, out, kh * kw * c, ac_pack, pack_RB_nchw, bc_pack, pack_CB_nchw_trans,
                         NULL, blis_cntx, &conv_params);
    }
}

/*
 * Computes dx = col2im(dy * transpose(weights))
 */
void sconvGemmNCHW_back(int b, int c, int h, int w,
                        int kn, int kh, int kw,
                        int vstride, int hstride,
                        int vpadding, int hpadding,
                        int vdilation, int hdilation,
                        const float *weights,
                        const float *dy, float *dx,
                        float *ac_pack, float *bc_pack) {
    int ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) / vstride + 1;
    int wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) / hstride + 1;
    conv_p conv_params = {b, h, w, c, kn, kh, kw,
                          vstride, hstride, vpadding, hpadding, vdilation, hdilation, ho, wo};
    gemm_blis_B3A2C0_orig('C', 'C', 'C', 'N', 'T', b * ho * wo, c * kh * kw, kn,
                          1.0, dy, b * ho * wo, weights, c * kh * kw, 1.0, dx, b * ho * wo,
                          ac_pack, pack_RB_nchw_trans, bc_pack, pack_CB,
                          post_col2im_nchw, blis_cntx, &conv_params);
}
