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

int print_matrix(char *, char, int, int, float *, int);

#ifdef BENCHMARK
double t_pack = 0.0, t_kernel = 0.0, t_generic = 0.0;
#define BEGIN_TIMER { double t1 = get_time();
#define END_TIMER(t) double t2 = get_time(); t += t2 - t1; }
#else
#define BEGIN_TIMER
#define END_TIMER(t)
#endif

void gemm_blis_B3A2C0( char orderA, char orderB, char orderC,
                       char transA, char transB, 
                       int m, int n, int k, 
                       float alpha, float *A, int ldA, 
		                    float *B, int ldB, 
		       float beta,  float *C, int ldC, 
		       float *Ac, float *Bc, float *Cc, cntx_t *cntx ){
  int    ic, jc, pc, mc, nc, kc, ir, jr, mr, nr; 
  float  zero = 0.0, one = 1.0, betaI; 
  float  *Aptr, *Bptr, *Cptr;

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

      if ( (transB=='N')&&(orderB=='C') )
        Bptr = &Bcol(pc,jc);
      else if ( (transB=='N')&&(orderB=='R') )
        Bptr = &Brow(pc,jc);
      else if ( (transB=='T')&&(orderB=='C') )
        Bptr = &Bcol(jc,pc);
      else
        Bptr = &Brow(jc,pc);
      BEGIN_TIMER
      pack_CB( orderB, transB, kc, nc, Bptr, ldB, Bc, NR);
      END_TIMER(t_pack)

      if ( pc==0 )
        betaI = beta;
      else
        betaI = one;

      for ( ic=0; ic<m; ic+=MC ) {
        mc = min(m-ic, MC); 

        if ( (transA=='N')&&(orderA=='C') )
          Aptr = &Acol(ic,pc);
        else if ( (transA=='N')&&(orderA=='R') )
          Aptr = &Arow(ic,pc);
        else if ( (transA=='T')&&(orderA=='C') )
          Aptr = &Acol(pc,ic);
        else
          Aptr = &Arow(pc,ic);
        BEGIN_TIMER
        pack_RB( orderA, transA, mc, kc, Aptr, ldA, Ac, MR);
        END_TIMER(t_pack)
        
        for ( jr=0; jr<nc; jr+=NR ) {
          nr = min(nc-jr, NR); 

          for ( ir=0; ir<mc; ir+=MR ) {
            mr = min(mc-ir, MR); 

            if ( orderC=='C' )
              Cptr = &Ccol(ic+ir,jc+jr);
            else
              Cptr = &Crow(ic+ir,jc+jr);
            if (nr == NR && mr == MR) {
              BEGIN_TIMER
              gemm_kernel(kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &betaI, Cptr, 1, ldC, NULL, cntx);
              END_TIMER(t_kernel)
            } else {
              BEGIN_TIMER
              gemm_kernel(kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &zero, Cc, 1, MR, NULL, cntx);
              for (int j = 0; j < nr; j++)
                  for (int i = 0; i < mr; i++)
                      Cptr[j * ldC + i] = betaI * Cptr[j * ldC + i] + Cc[j * MR + i];
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

void pack_RB( char orderM, char transM, int mc, int nc, float *M, int ldM, float *Mc, int RR ){
/*
  BLIS pack for M-->Mc
*/
  int    i, j, ii, k, rr;

  if ( ((transM=='N')&&( orderM=='C'))||
       ((transM=='T')&&( orderM=='R')) )
    #pragma omp parallel for private(j, ii, rr, k)
    for ( i=0; i<mc; i+=RR ) { 
      k = i*nc;
      rr = RR; // min( mc-i, RR );
      for ( j=0; j<nc; j++ ) {
        for ( ii=0; ii<rr; ii++ ) {
          if (ii < mc - i)
            Mc[k] = Mcol(i+ii,j);
          else
            Mc[k] = 0.0;
          k++;
        }
        k += (RR-rr);
      }
    }
  else
    #pragma omp parallel for private(j, ii, rr, k)
    for ( i=0; i<mc; i+=RR ) { 
      k = i*nc;
      rr = RR; // min( mc-i, RR );
      for ( j=0; j<nc; j++ ) {
        for ( ii=0; ii<rr; ii++ ) {
          if (ii < mc - i)
            Mc[k] = Mcol(j,i+ii);
          else
            Mc[k] = 0.0;
          k++;
        }
        k += (RR-rr);
      }
    }
}

void pack_CB( char orderM, char transM, int mc, int nc, float *M, int ldM, float *Mc, int RR ){
/*
  BLIS pack for M-->Mc
*/
  int    i, j, jj, k, nr;

  k = 0;
  if ( ((transM=='N')&&( orderM=='C'))||
       ((transM=='T')&&( orderM=='R')) )
    #pragma omp parallel for private(i, jj, nr, k)
    for ( j=0; j<nc; j+=RR ) { 
      k = j*mc;
      nr = RR; // min( nc-j, RR );
      for ( i=0; i<mc; i++ ) {
        for ( jj=0; jj<nr; jj++ ) {
          if (jj < nc -j)
            Mc[k] = Mcol(i,j+jj);
          else
            Mc[k] = 0.0;
          k++;
        }
        k += (RR-nr);
      }
    }
  else
    #pragma omp parallel for private(i, jj, nr, k)
    for ( j=0; j<nc; j+=RR ) { 
      k = j*mc;
      nr = RR; // min( nc-j, RR );
      for ( i=0; i<mc; i++ ) {
        for ( jj=0; jj<nr; jj++ ) {
          if (jj < nc -j)
            Mc[k] = Mcol(j+jj,i);
          else
            Mc[k] = 0.0;
          k++;
        }
        k += (RR-nr);
      }
    }
}

void unpack_RB( char orderM, char transM, int mc, int nc, float *M, int ldM, float *Mc, int RR ){
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

void unpack_CB( char orderM, char transM, int mc, int nc, float *M, int ldM, float *Mc, int RR ){
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
                          float alpha, float *A, int ldA, 
                                       float *B, int ldB, 
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
