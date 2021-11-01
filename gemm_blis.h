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
                       float *, pack_func, float *, pack_func, float *, post_func, cntx_t *, const convol_dim *, const float *);
void gemm_blis_A3B2C0( char, char, char, char, char, int, int, int, float, const float *, int, const float *, int, float, float *, int,
                       float *, pack_func, float *, pack_func, float *, post_func, cntx_t *, const convol_dim *, const float *);
void gemm_base_Cresident( char, int, int, int, float, const float *, int, const float *, int, float, float *, int );
void pack_RB(char, char, int, int, const float *, int, float *, int, const convol_dim *, int, int);
void pack_CB(char, char, int, int, const float *, int, float *, int, const convol_dim *, int, int);
void unpack_RB( char, char, int, int, float *, int, const float *, int );
void unpack_CB( char, char, int, int, float *, int, const float *, int );
void sxpbyM(int, int, const float *, int, float, float *, int);
