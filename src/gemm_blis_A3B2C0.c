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
    int MR, NR, MC, NC, KC;
    gemm_blis_blocks_sizes(m, n, k, &MR, &NR, &MC, &NC, &KC);

#include "quick_gemm.h"

    for (int ic = 0; ic < m; ic += MC) {
        int mc = min(m - ic, MC);

        for (int pc = 0; pc < k; pc += KC) {
            int kc = min(k - pc, KC);
            bool last = (pc + KC) >= k;

            pack_RB(orderA, transA, mc, kc, A, ldA, Ac, MR, conv_params, ic, pc);

            float betaI = (pc == 0) ? beta : 1.0;

#pragma omp parallel for
            for (int jc = 0; jc < n; jc += NC) {
                int nc = min(n - jc, NC);

                int tid = omp_get_thread_num();

                pack_CB(orderB, transB, kc, nc, B, ldB, Bc + tid * NC * KC, NR, conv_params, pc, jc);

                for (int ir = 0; ir < mc; ir += MR) {
                    for (int jr = 0; jr < nc; jr += NR) {

                        int mr = min(mc - ir, MR);
                        int nr = min(nc - jr, NR);

                        float *Cptr = (orderC == 'C') ? &Ccol(ic + ir, jc + jr) : &Crow(ic + ir, jc + jr);
                        float Clocal[MR * NR];
                        auxinfo_t aux = {0};
                        bli_auxinfo_set_next_a(&Ac[(ir + MR) * kc], &aux);
                        bli_auxinfo_set_next_b(&Bc[tid * NC * KC + (jr + NR) * kc], &aux);

                        if (postprocess == NULL && nr == NR && mr == MR) { // don't use buffer
                            gemm_kernel(mr, nr,
                                        kc, &alpha, &Ac[ir * kc], &Bc[tid * NC * KC + jr * kc], &betaI, Cptr, 1, ldC,
                                        &aux, cntx);
                        } else { // use buffer for border elements or postprocessing
                            gemm_kernel(mr, nr,
                                        kc, &alpha, &Ac[ir * kc], &Bc[tid * NC * KC + jr * kc], &zero, Clocal, 1, mr,
                                        &aux, cntx);
                            if (postprocess == NULL) {
                                sxpbyM(mr, nr, Clocal, mr, betaI, Cptr, ldC);
                            } else {
                                postprocess(mr, nr, Clocal, mr, betaI, C, ldC, conv_params, ic + ir, jc + jr, last);
                            }
                        }
                    }
                }
            }
        }
    }
}
