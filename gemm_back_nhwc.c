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
#include "gemm_back_nhwc.h"
#include "im2row_nhwc.h"

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

#define Acol(a1,a2)  A[ (a2)*(ldA)+(a1) ]
#define Bcol(a1,a2)  B[ (a2)*(ldB)+(a1) ]
#define Ccol(a1,a2)  C[ (a2)*(ldC)+(a1) ]
#define Mcol(a1,a2)  M[ (a2)*(ldM)+(a1) ]

#define Arow(a1,a2)  A[ (a1)*(ldA)+(a2) ]
#define Brow(a1,a2)  B[ (a1)*(ldB)+(a2) ]
#define Crow(a1,a2)  C[ (a1)*(ldC)+(a2) ]
#define Mrow(a1,a2)  M[ (a1)*(ldM)+(a2) ]

void gemm_back_nhwc_B3A2C0( char orderA, char orderB, char orderC,
                       char transA, char transB, 
                       int m, int n, int k, 
                       float alpha, const float *A, int ldA, 
		                    const float *B, int ldB, 
		       float beta, float *C, int ldC, 
                       float *Ac, float *Bc, float *Cc, cntx_t *cntx,
                       float *dx, int b, int h, int w, int c, int ho, int wo, int kh, int kw, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation)
{
    convol_dim dim = { b, h, w, c, k /* kn */, kh, kw, vstride, hstride, vpadding, hpadding, vdilation, hdilation, ho, wo };
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

  for ( jc=0; jc<n; jc+=NC ) {
    nc = min(n-jc, NC); 

    for ( pc=0; pc<k; pc+=KC ) {
      kc = min(k-pc, KC); 

      BEGIN_TIMER
      pack_CB( orderB, transB, kc, nc, B, ldB, Bc, NR, &dim, pc, jc);
      END_TIMER(t_pack)

      /* if ( pc==0 )
        betaI = beta;
      else
        betaI = one; */

      for ( ic=0; ic<m; ic+=MC ) {
        mc = min(m-ic, MC); 

        BEGIN_TIMER
        pack_RB( orderA, transA, mc, kc, A, ldA, Ac, MR, &dim, ic, pc);
        END_TIMER(t_pack)
        
        for ( jr=0; jr<nc; jr+=NR ) {
          nr = min(nc-jr, NR); 

          for ( ir=0; ir<mc; ir+=MR ) {
            mr = min(mc-ir, MR); 

            if ( orderC=='C' )
              Cptr = &Ccol(ic+ir,jc+jr);
            else
              Cptr = &Crow(ic+ir,jc+jr);
            /* if (nr == NR && mr == MR) {
              BEGIN_TIMER
              gemm_kernel(kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &betaI, Cptr, 1, ldC, NULL, cntx);
              END_TIMER(t_kernel)
            } else */ {
              BEGIN_TIMER
              gemm_kernel(kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &zero, Cc, 1, MR, NULL, cntx);
              END_BEGIN_TIMER(t_kernel)
              row2im_nhwc(nr, mr, Cc, MR, dx, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation, jc + jr, ic + ir);
              // gemm_base_Cresident( orderC, mr, nr, kc, alpha, &Ac[ir*kc], MR, &Bc[jr*kc], NR, betaI, Cptr, ldC );
	    // gemm_microkernel_Cresident_neon_4x4_prefetch( orderC, mr, nr, kc, alpha, &Ac[ir*kc], &Bc[jr*kc], betaI, Cptr, ldC );
              END_TIMER(t_generic)
            }
          }
        }
      }
    }
  }
}
