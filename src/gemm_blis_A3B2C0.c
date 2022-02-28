/**
 * This file is part of convGemm
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
#include <blis/blis.h>

#include "gemm_blis.h"

/*
 * Computes the GEMM C := beta * C + alpha * A * B  following the BLIS approach
*/
void gemm_blis_A3B2C0(char orderA, char orderB, char orderC,
                      char transA, char transB,
                      int m, int n, int k,
                      float alpha, const float *A, int ldA,
                      const float *B, int ldB,
                      float beta, float *C, int ldC,
                      float *Ac, pack_func pack_RB,
                      float *Bc, pack_func pack_CB,
                      post_func postprocess,
                      cntx_t *cntx, const conv_p *conv_params) {

    // Test the input parameters
#if defined(CHECK)
#include "check_params.h"
#endif

    // Quick return if possible
    float zero = (float) 0.0, one = (float) 1.0;
    if ((m == 0) || (n == 0) || (((alpha == zero) || (k == 0)) && (beta == one)))
        return;

    // Get Gemm BLIS blocks sizes
    int MR_bs, NR_bs, MC_bs, NC_bs, KC_bs;
    gemm_blis_blocks_sizes(m, n, k, &MR_bs, &NR_bs, &MC_bs, &NC_bs, &KC_bs);

#include "quick_gemm.h"

    for (int ic = 0; ic < m; ic += MC_bs) {
        int mc = min(m - ic, MC_bs);

        for (int pc = 0; pc < k; pc += KC_bs) {
            int kc = min(k - pc, KC_bs);
            bool last = (pc + KC_bs) >= k;

            pack_RB(orderA, transA, mc, kc, A, ldA, Ac, MR_bs, conv_params, ic, pc);

            float betaI = (pc == 0) ? beta : 1.0;

#pragma omp parallel for
            for (int jc = 0; jc < n; jc += NC_bs) {
                int nc = min(n - jc, NC_bs);

                int tid = omp_get_thread_num();

                pack_CB(orderB, transB, kc, nc, B, ldB, Bc + tid * NC_bs * KC_bs, NR_bs, conv_params, pc, jc);

                for (int ir = 0; ir < mc; ir += MR_bs) {
                    for (int jr = 0; jr < nc; jr += NR_bs) {

                        int mr = min(mc - ir, MR_bs);
                        int nr = min(nc - jr, NR_bs);

                        float *Cptr = (orderC == 'C') ? &Ccol(ic + ir, jc + jr) : &Crow(ic + ir, jc + jr);
                        float Clocal[MR_bs * NR_bs];
                        auxinfo_t aux = {0};
                        bli_auxinfo_set_next_a(&Ac[(ir + MR_bs) * kc], &aux);
                        bli_auxinfo_set_next_b(&Bc[tid * NC_bs * KC_bs + (jr + NR_bs) * kc], &aux);

#if BLIS_ABI_VERSION == 3
                        if (postprocess == NULL && nr == NR_bs && mr == MR_bs) { // don't use buffer
#elif BLIS_ABI_VERSION == 4
                        if (postprocess == NULL) { // don't use buffer
#else
#pragma message "Specified BLIS_ABI_VERSION not supported!"
#endif
                            gemm_kernel(mr, nr,
                                        kc, &alpha, &Ac[ir * kc], &Bc[tid * NC_bs * KC_bs + jr * kc], &betaI, Cptr, 1, ldC,
                                        &aux, cntx);
                        } else { // use buffer for border elements (BLIS3) or postprocessing
                            gemm_kernel(mr, nr,
                                        kc, &alpha, &Ac[ir * kc], &Bc[tid * NC_bs * KC_bs + jr * kc], &zero, Clocal, 1, MR_bs,
                                        &aux, cntx);
#if BLIS_ABI_VERSION == 3
                            if (postprocess == NULL) {
                                sxpbyM(mr, nr, Clocal, MR_bs, betaI, Cptr, ldC);
                            } else {
#endif
                            postprocess(mr, nr, Clocal, MR_bs, betaI, C, ldC, conv_params, ic + ir, jc + jr, last);
#if BLIS_ABI_VERSION == 3
                            }
#endif
                        }
                    }
                }
            }
        }
    }
}
