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

void gemm_blis_B3A2C0( char orderA, char orderB, char orderC,
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

  for ( jc=0; jc<n; jc+=NC ) {
    nc = min(n-jc, NC); 

    for ( pc=0; pc<k; pc+=KC ) {
      kc = min(k-pc, KC); 

      BEGIN_TIMER
      pack_CB( orderB, transB, kc, nc, B, ldB, Bc, NR, dim, pc, jc);
      END_TIMER(t_pack)

      if ( pc==0 )
        betaI = beta;
      else
        betaI = one;

      for ( ic=0; ic<m; ic+=MC ) {
        mc = min(m-ic, MC); 

        BEGIN_TIMER
        pack_RB( orderA, transA, mc, kc, A, ldA, Ac, MR, dim, ic, pc);
        END_TIMER(t_pack)
        
        for ( jr=0; jr<nc; jr+=NR ) {
          nr = min(nc-jr, NR); 

          for ( ir=0; ir<mc; ir+=MR ) {
            mr = min(mc-ir, MR); 

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

void pack_RB( char orderM, char transM, int mc, int nc, const float *M, int ldM, float *Mc, int RR, const convol_dim *d, int start_row, int start_col ){
/*
  BLIS pack for M-->Mc
*/
  int    i, j, ii, k, rr;

  if ( (transM=='N')&&(orderM=='C') )
    M = &Mcol(start_row, start_col);
  else if ( (transM=='N')&&(orderM=='R') )
    M = &Mrow(start_row, start_col);
  else if ( (transM=='T')&&(orderM=='C') )
    M = &Mcol(start_col, start_row);
  else
    M = &Mrow(start_col, start_row);

  if ( ((transM=='N')&&( orderM=='C'))||
       ((transM=='T')&&( orderM=='R')) )
    // #pragma omp parallel for private(j, ii, rr, k)
    for ( i=0; i<mc; i+=RR ) { 
      k = i*nc;
      rr = min( mc-i, RR );
      for ( j=0; j<nc; j++ ) {
        for ( ii=0; ii<rr; ii++ ) {
          Mc[k] = Mcol(i+ii,j);
          k++;
        }
        for ( ; ii<RR; ii++ ) {
          Mc[k] = 0.0;
          k++;
        }
        // k += (RR-rr);
      }
    }
  else
    // #pragma omp parallel for private(j, ii, rr, k)
    for ( i=0; i<mc; i+=RR ) { 
      k = i*nc;
      rr = min( mc-i, RR );
      for ( j=0; j<nc; j++ ) {
        for ( ii=0; ii<rr; ii++ ) {
          Mc[k] = Mcol(j,i+ii);
          k++;
        }
        for ( ; ii<RR; ii++ ) {
          Mc[k] = 0.0;
          k++;
        }
        // k += (RR-rr);
      }
    }
}

void pack_CB( char orderM, char transM, int mc, int nc, const float *M, int ldM, float *Mc, int RR, const convol_dim *dim, int start_row, int start_col ){
/*
  BLIS pack for M-->Mc
*/
  int    i, j, jj, k, nr;

  if ( (transM=='N')&&(orderM=='C') )
    M = &Mcol(start_row, start_col);
  else if ( (transM=='N')&&(orderM=='R') )
    M = &Mrow(start_row, start_col);
  else if ( (transM=='T')&&(orderM=='C') )
    M = &Mcol(start_col, start_row);
  else
    M = &Mrow(start_col, start_row);

  k = 0;
  if ( ((transM=='N')&&( orderM=='C'))||
       ((transM=='T')&&( orderM=='R')) )
    // #pragma omp parallel for private(i, jj, nr, k)
    for ( j=0; j<nc; j+=RR ) { 
      k = j*mc;
      nr = min( nc-j, RR );
      for ( i=0; i<mc; i++ ) {
        for ( jj=0; jj<nr; jj++ ) {
          Mc[k] = Mcol(i,j+jj);
          k++;
        }
        for ( ; jj<RR; jj++ ) {
          Mc[k] = 0.0;
          k++;
        }
        // k += (RR-nr);
      }
    }
  else
    // #pragma omp parallel for private(i, jj, nr, k)
    for ( j=0; j<nc; j+=RR ) { 
      k = j*mc;
      nr = min( nc-j, RR );
      for ( i=0; i<mc; i++ ) {
        for ( jj=0; jj<nr; jj++ ) {
          Mc[k] = Mcol(j+jj,i);
          k++;
        }
        for ( ; jj<RR; jj++ ) {
          Mc[k] = 0.0;
          k++;
        }
        // k += (RR-nr);
      }
    }
}

void unpack_RB( char orderM, char transM, int mc, int nc, float *M, int ldM, const float *Mc, int RR ){
/*
  BLIS unpack for M-->Mc
*/
  int    i, j, ii, k, rr;

  if ( ((transM=='N')&&( orderM=='C'))||
       ((transM=='T')&&( orderM=='R')) )
    #pragma omp parallel for private(j, ii, rr, k)
    for ( i=0; i<mc; i+=RR ) { 
      k = i*nc;
      rr = min( mc-i, RR );
      for ( j=0; j<nc; j++ ) {
        for ( ii=0; ii<rr; ii++ ) {
          Mcol(i+ii,j) = Mc[k];
          k++;
        }
        k += (RR-rr);
      }
    }
  else
    #pragma omp parallel for private(j, ii, rr, k)
    for ( i=0; i<mc; i+=RR ) { 
      k = i*nc;
      rr = min( mc-i, RR );
      for ( j=0; j<nc; j++ ) {
        for ( ii=0; ii<rr; ii++ ) {
          Mcol(j,i+ii) = Mc[k];
          k++;
        }
        k += (RR-rr);
      }
    }
}

void unpack_CB( char orderM, char transM, int mc, int nc, float *M, int ldM, const float *Mc, int RR ){
/*
  BLIS unpack for M-->Mc
*/
  int    i, j, jj, k, nr;

  k = 0;
  if ( ((transM=='N')&&( orderM=='C'))||
       ((transM=='T')&&( orderM=='R')) )
    #pragma omp parallel for private(i, jj, nr, k)
    for ( j=0; j<nc; j+=RR ) { 
      k = j*mc;
      nr = min( nc-j, RR );
      for ( i=0; i<mc; i++ ) {
        for ( jj=0; jj<nr; jj++ ) {
          Mcol(i,j+jj) = Mc[k];
          k++;
        }
        k += (RR-nr);
      }
    }
  else
    #pragma omp parallel for private(i, jj, nr, k)
    for ( j=0; j<nc; j+=RR ) { 
      k = j*mc;
      nr = min( nc-j, RR );
      for ( i=0; i<mc; i++ ) {
        for ( jj=0; jj<nr; jj++ ) {
          Mcol(j+jj,i) = Mc[k];
          k++;
        }
        k += (RR-nr);
      }
    }
}

void gemm_base_Cresident( char orderC, int m, int n, int k, 
                          float alpha, const float *A, int ldA, 
                                       const float *B, int ldB, 
                          float beta,  float *C, int ldC ){
/*
  Baseline micro-kernel 
  Replace with specialized micro-kernel where C-->m x n is resident in registers
*/
  int    i, j, p;
  float  zero = 0.0, one = 1.0, tmp;

  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ ) {
      tmp = 0.0; 
      for ( p=0; p<k; p++ ) 
        tmp += Acol(i,p) * Brow(p,j);

      if ( beta==zero ) {
        if ( orderC=='C' )
          Ccol(i,j) = alpha*tmp;
        else
          Crow(i,j) = alpha*tmp;
      }
      else {
        if ( orderC=='C' )
          Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
        else
          Crow(i,j) = alpha*tmp + beta*Crow(i,j);
      }
    }
}

void sxpbyM(int m, int n, const float *X, int ldx, float beta, float *Y, int ldy)
{
    if (beta == 0.0) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                Y[j * ldy + i] = X[j * ldx + i];
    } else if (beta = 1.0) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                Y[j * ldy + i] += X[j * ldx + i];
    } else {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                Y[j * ldy + i] = beta * Y[j * ldy + i] + X[j * ldx + i];
    }
}
