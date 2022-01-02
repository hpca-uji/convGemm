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

#ifdef BENCHMARK
extern double t_pack, t_kernel, t_generic;
#define BEGIN_TIMER { double t1 = get_time();
#define END_TIMER(t) double t2 = get_time(); t += t2 - t1; }
#define END_BEGIN_TIMER(t) { double t3 = get_time(); t += t3 - t1; t1 = t3; }
#else
#define BEGIN_TIMER
#define END_TIMER(t)
#define END_BEGIN_TIMER(t)
#endif

void gemm_blis_B3A2C0( char, char, char, char, char, int, int, int, float, const float *, int, const float *, int, float, float *, int,
                       float *, pack_func, float *, pack_func, float *, post_func, cntx_t *, const convol_dim *);
void gemm_blis_A3B2C0( char, char, char, char, char, int, int, int, float, const float *, int, const float *, int, float, float *, int,
                       float *, pack_func, float *, pack_func, float *, post_func, cntx_t *, const convol_dim *, const float *);
void gemm_base_Cresident( char, int, int, int, float, const float *, int, const float *, int, float, float *, int );
void pack_RB(char, char, int, int, const float *, int, float *, int, const convol_dim *, int, int);
void pack_CB(char, char, int, int, const float *, int, float *, int, const convol_dim *, int, int);
void unpack_RB( char, char, int, int, float *, int, const float *, int );
void unpack_CB( char, char, int, int, float *, int, const float *, int );
void sxpbyM(int, int, const float *, int, float, float *, int);

int alloc_pack_buffs(int m, int n, int k, float** Ac_pack, float** Bc_pack);

inline static double model_level(double NL, double CL, double WL, double Sdata, double m, double n)
{
/*
  Purpose
    Estimate the dimension of the panels that will fit into a given level of the cache hierarchy following the
    principles in the paper "Analytical modeling is enough for enough for high performance BLIS" by
    T. M. Low et al, 2016

  Inputs:
     NL:    Number of sets
     CL:    Bytes per line
     WL:    Associativity degree
     (m,n): Dimensions of block in higher level of cache
     Sdata: Bytes per element (e.g., 8 for FP64)

  Output
     k:     Determines that a block of size k x n stays in this level of the cache

  Rule of thumb: subtract 1 line from WL (associativity), which is dedicated to the
  operand which does not reside in the cache, and distribute the rest between the two
  other operands proportionaly to the ratio n/m
  For example, with the conventional algorithm B3A2B1C0 and the L1 cache,
  1 line is dedicated to Cr (non-resident in cache) while the remaining lines are distributed
  between Ar (mr x kc) and Br (kc x nr) proportionally to the ratio nr/mr to estimate kc
*/
    double CAr = floor( ( WL - 1 ) / (1 + n/m) ); // Lines of each set for Ar
    if (CAr==0) { // Special case
        CAr = 1;
        // CBr = WL-2;
    } else {
        // CBr = ceil( ( n / m ) * CAr ); // Lines of each set for Br
    }
    return floor( CAr * NL * CL / (m * Sdata) );
}

inline static void gemm_blis_workspace(cntx_t *cntx, int m, int n, int k, int *MC, int *NC, int *KC)
{
#if 1
    *MC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MC, cntx);
    *NC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NC, cntx);
    *KC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_KC, cntx);
#else
    // *NC = 3072;
    // *KC = 368; //640
    // *MC = 560; //120
    int MR = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MR, cntx);
    int NR = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NR, cntx);
    int Sdata=4;
    int SL1=64*1024, WL1=4, NL1 = 256, CL1 = SL1 / (WL1 * NL1);
    int SL2=1*1024*1024, WL2=16, NL2 = 2048, CL2 = SL2 / (WL2 * NL2);
    int SL3=4*1024*1024, WL3=16, NL3 = 4096, CL3 = SL3 / (WL3 * NL3);

    *KC = model_level(NL1, CL1, WL1, Sdata, MR, NR);
    if (k > 0 && *KC > k) *KC = k;

    *MC = model_level(NL2, CL2, WL2, Sdata, *KC, NR);
    if (m > 0 && *MC > m) *MC = m;
    *MC = *MC - *MC % MR;
    if (*MC < MR) *MC = MR;

    *NC = model_level(NL3, CL3, WL3, Sdata, *KC, *MC);
    // if (n > 0 && *NC > n) *NC = n;
    // *MC = 448; *NC = 1020; *KC = 512;
    // printf("m=%d n=%d k=%d MC=%d NC=%d KC=%d\n", m, n, k, *MC, *NC, *KC);
#endif
}
