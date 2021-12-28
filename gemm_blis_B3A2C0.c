/* 
   GEMM FLAVOURS

   -----

   GEMM FLAVOURS is a family of algorithms for matrix multiplication based
   on the BLIS approach for this operation: https://github.com/flame/blis

   -----

   This program is free software: you can redistribute it and/or modify it under
   the terms of the GNU General Public License as published by the Free Software
   Foundation, either version 3 of the License, or (at your option) any later
   version.

   This program is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
   You should have received a copy of the GNU General Public License along with
   this program. If not, see <http://www.gnu.org/licenses/>.

   -----

   author    = "Enrique S. Quintana-Orti"
   contact   = "quintana@disca.upv.es"
   copyright = "Copyright 2021, Universitat Politecnica de Valencia"
   license   = "GPLv3"
   status    = "Production"
   version   = "1.1"
*/

#include <stdio.h>
#include <stdlib.h>
// #include <arm_neon.h>

#include <blis.h>

#include "convGemm.h"
#include "gemm_blis.h"

void gemm_blis_B3A2C0(char orderA, char orderB, char orderC,
                      char transA, char transB,
                      int m, int n, int k,
                      float alpha, const float *A, int ldA,
                      const float *B, int ldB,
                      float beta, float *C, int ldC,
                      float *Ac, pack_func pack_RB,
                      float *Bc, pack_func pack_CB,
                      float *Cc, post_func postprocess,
                      cntx_t * cntx, const convol_dim * dim,
                      const float *bias_vector)
{
    float zero = 0.0, one = 1.0;

    sgemm_ukr_ft gemm_kernel = bli_cntx_get_l3_nat_ukr_dt(BLIS_FLOAT, BLIS_GEMM, cntx);
    int MR = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MR, cntx);
    int NR = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NR, cntx);
    int MC, NC, KC;
    gemm_blis_workspace(cntx, m, n, k, &MC, &NC, &KC);
    MC -= MC % MR;
    NC -= NC % NR;
/* 
  Computes the GEMM C := beta * C + alpha * A * B
  following the BLIS approach
*/

/*
*     Test the input parameters.
*/
#if defined(CHECK)
#include "check_params.h"
#endif

    // Quick return if possible
    if ((m == 0) || (n == 0) || (((alpha == zero) || (k == 0)) && (beta == one)))
        return;

#include "quick_gemm.h"

    for (int jc = 0; jc < n; jc += NC) {
        int nc = min(n - jc, NC);

        for (int pc = 0; pc < k; pc += KC) {
            int kc = min(k - pc, KC);

            pack_CB(orderB, transB, kc, nc, B, ldB, Bc, NR, dim, pc, jc);

            float betaI = (pc == 0) ? beta : 1.0;

#pragma omp parallel for
            for (int ic = 0; ic < m; ic += MC) {
                int mc = min(m - ic, MC);

                int tid = omp_get_thread_num();

                pack_RB(orderA, transA, mc, kc, A, ldA, Ac + tid * MC * KC, MR, dim, ic, pc);

// #pragma omp parallel for collapse(2)
                for (int jr = 0; jr < nc; jr += NR) {
                    for (int ir = 0; ir < mc; ir += MR) {

                        int mr = min(mc - ir, MR);
                        int nr = min(nc - jr, NR);

                        float *Cptr = (orderC == 'C') ? &Ccol(ic + ir, jc + jr) : &Crow(ic + ir, jc + jr);
                        float Clocal[MR * NR];
                        auxinfo_t aux = { 0 };
                        bli_auxinfo_set_next_a(&Ac[tid * MC * KC + (ir + MR) * kc], &aux);
                        bli_auxinfo_set_next_b(&Bc[(jr + NR) * kc], &aux);

                        if (postprocess == NULL && nr == NR && mr == MR) { // don't use buffer
                                gemm_kernel(kc, &alpha, &Ac[tid * MC * KC + ir * kc], &Bc[jr * kc], &betaI, Cptr, 1, ldC, &aux, cntx);
                        } else { // use buffer for border elements or postprocessing
                            gemm_kernel(kc, &alpha, &Ac[tid * MC * KC + ir * kc], &Bc[jr * kc], &zero, Clocal, 1, MR, &aux, cntx);
                            if (postprocess == NULL) {
                                sxpbyM(mr, nr, Clocal, MR, betaI, Cptr, ldC);
                            } else {
                                postprocess(mr, nr, Clocal, MR, betaI, C, ldC, dim, bias_vector, ic + ir, jc + jr, pc == 0);
                            }
                        }
                    }
                }
            }
        }
    }
}
