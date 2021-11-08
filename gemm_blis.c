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

void pack_RB( char orderM, char transM, int mc, int nc, const float *restrict M, int ldM, float *restrict Mc, int RR, const convol_dim *d, int start_row, int start_col ){
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
    #pragma omp parallel for private(i, j, ii, rr, k)
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
    #pragma omp parallel for private(i, j, ii, rr, k)
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

void pack_CB( char orderM, char transM, int mc, int nc, const float *restrict M, int ldM, float *restrict Mc, int RR, const convol_dim *dim, int start_row, int start_col ){
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
    #pragma omp parallel for private(i, j, jj, nr, k)
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
    #pragma omp parallel for private(i, j, jj, nr, k)
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

void sxpbyM(int m, int n, const float *restrict X, int ldx, float beta, float *restrict Y, int ldy)
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
