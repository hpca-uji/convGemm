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

void gemm_blis_A3B2C0( char orderA, char orderB, char orderC,
                       char transA, char transB, 
                       int m, int n, int k, 
                       float alpha, const float *A, int ldA, 
		                    const float *B, int ldB, 
		       float beta, float *C, int ldC, 
                       float *Ac, pack_func pack_RB,  float *Bc, pack_func pack_CB, float *Cc,
                       post_func postprocess, cntx_t *cntx,
                       const convol_dim *dim, const float *bias_vector)
{
  int    ic, jc, pc, mc, nc, kc, ir, jr, mr, nr; 
  float  zero = 0.0, one = 1.0, betaI; 
  float *Cptr;

  sgemm_ukr_ft gemm_kernel = bli_cntx_get_l3_nat_ukr_dt(BLIS_FLOAT, BLIS_GEMM, cntx);
  int MR = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MR, cntx);
  int NR = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NR, cntx);
  int NC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NC, cntx);
  int MC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MC, cntx);
  int KC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_KC, cntx);

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
  if ( (m==0)||(n==0)||(((alpha==zero)||(k==0))&&(beta==one)) )
    return;

  #include "quick_gemm.h"

  for ( ic=0; ic<m; ic+=MC ) {
    mc = min(m-ic, MC); 

    for ( pc=0; pc<k; pc+=KC ) {
      kc = min(k-pc, KC); 

      BEGIN_TIMER
      pack_RB( orderA, transA, mc, kc, A, ldA, Ac, MR, dim, ic, pc);
      END_TIMER(t_pack)

      if ( pc==0 )
        betaI = beta;
      else
        betaI = one;

      for ( jc=0; jc<n; jc+=NC ) {
        nc = min(n-jc, NC); 

        BEGIN_TIMER
        pack_CB( orderB, transB, kc, nc, B, ldB, Bc, NR, dim, pc, jc);
        END_TIMER(t_pack)

        for ( ir=0; ir<mc; ir+=MR ) {
          mr = min(mc-ir, MR); 

          for ( jr=0; jr<nc; jr+=NR ) {
            nr = min(nc-jr, NR); 

            if ( orderC=='C' )
              Cptr = &Ccol(ic+ir,jc+jr);
            else
              Cptr = &Crow(ic+ir,jc+jr);
            if (postprocess == NULL) {
                BEGIN_TIMER
                if (nr == NR && mr == MR) { // don't use buffer
                    gemm_kernel(kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &betaI, Cptr, 1, ldC, NULL, cntx);
                } else { // use buffer for border elements
                    gemm_kernel(kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &zero, Cc, 1, mr, NULL, cntx);
                    sxpbyM(mr, nr, Cc, mr, betaI, Cptr, ldC);
                }
                END_BEGIN_TIMER(t_kernel)
            } else { // use buffer for postprocessing
                BEGIN_TIMER
                gemm_kernel(kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &zero, Cc, 1, mr, NULL, cntx);
                END_BEGIN_TIMER(t_kernel)
                postprocess(mr, nr, Cc, betaI, C, ldC, dim, bias_vector, ic + ir, jc + jr, pc == 0);
                END_TIMER(t_generic)
              // gemm_base_Cresident( orderC, mr, nr, kc, alpha, &Ac[ir*kc], MR, &Bc[jr*kc], NR, betaI, Cptr, ldC );
	    // gemm_microkernel_Cresident_neon_4x4_prefetch( orderC, mr, nr, kc, alpha, &Ac[ir*kc], &Bc[jr*kc], betaI, Cptr, ldC );
            }
          }
        }
      }
    }
  }
}

