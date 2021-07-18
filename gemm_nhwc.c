#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "blis.h"
#include "gemm_blis.h"
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

void gemm_nhwc_B3A2C0( char orderA, char orderB, char orderC,
                       char transA, char transB,
                       int m, int n, int k,
                       float alpha, float *A, int ldA,
                                    float *B, int ldB,
                       float beta,  float *C, int ldC,
                       float *Ac, float *Bc,
                       float *in, int b, int h, int w, int c, int ho, int wo, int kh, int kw, int vpadding, int hpadding, int vstride, int hstride, int vdilation, int hdilation)
{
  int    ic, jc, pc, mc, nc, kc, ir, jr, mr, nr;
  float  zero = 0.0, one = 1.0, betaI;
  float  *Aptr, *Bptr, *Cptr;
/*
  Computes the GEMM C := beta * C + alpha * A * B
  following the BLIS approach
*/

  // Quick return if possible
  if ( (m==0)||(n==0)||(((alpha==zero)||(k==0))&&(beta==one)) )
    return;

  #include "quick_gemm.h"

  for ( jc=0; jc<n; jc+=NC ) {
    nc = min(n-jc, NC);

    for ( pc=0; pc<k; pc+=KC ) {
      kc = min(k-pc, KC);

      if ( (transB=='N')&&(orderB=='C') ) {
        // printf("%d %d %d %d\n", k, n, kh * kw * c, ho * wo * b);
        Bptr = &Bcol(pc,jc); // B[pc+jc*ldB]
        pack_CB_nhwc( orderB, transB, kc, nc, Bptr, ldB, Bc, NR, in, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation, pc, jc);
#if 0
        float *aux = malloc(kc * nc * sizeof(float));
        printf("im2row %d %d %d\n", kc, nc, NR);
        im2row_nhwc('N', aux, kc, in, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation, jc, jc + nc, pc, pc + kc);
        if (kc == 384)
        for (int j = 0; j < nc; j++) {
            for (int i = 0; i < kc; i++) {
                float a = Bcol(pc+i,jc+j);
                // float b = aux[j * kc + i];
                float b = Bptr[j * kc + i];
                /* float d = fabs(a - b);
                if (d > 1e-5) */ printf("%d %d %e %e\n", pc+i, jc+j, a, b);
            }
        }
        free(aux);
#endif
      } else if ( (transB=='T')&&(orderB=='C') ) {
        Bptr = &Bcol(jc,pc); // B[pc*ldB+jc]
        pack_CB_nhwc( orderB, transB, kc, nc, Bptr, ldB, Bc, NR, in, b, h, w, c, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation, pc, jc);
      } else { if ( (transB=='N')&&(orderB=='R') )
        Bptr = &Brow(pc,jc); // B[pc*ldB+jc]
      else
        Bptr = &Brow(jc,pc); // B[pc+jc*ldB]
      pack_CB( orderB, transB, kc, nc, Bptr, ldB, Bc, NR); }

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
        pack_RB( orderA, transA, mc, kc, Aptr, ldA, Ac, MR);

        for ( jr=0; jr<nc; jr+=NR ) {
          nr = min(nc-jr, NR);

          for ( ir=0; ir<mc; ir+=MR ) {
            mr = min(mc-ir, MR);

            if ( orderC=='C' )
              Cptr = &Ccol(ic+ir,jc+jr);
            else
              Cptr = &Crow(ic+ir,jc+jr);
            gemm_base_Cresident( orderC, mr, nr, kc, alpha, &Ac[ir*kc], MR, &Bc[jr*kc], NR, betaI, Cptr, ldC );
            // gemm_microkernel_Cresident_neon_4x4_prefetch( orderC, mr, nr, kc, alpha, &Ac[ir*kc], &Bc[jr*kc], betaI, Cptr, ldC );
          }
        }
      }
    }
  }
}
