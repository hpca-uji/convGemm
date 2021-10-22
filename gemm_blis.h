static inline double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

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
                       float *, float *, float *, cntx_t* );
void gemm_base_Cresident( char, int, int, int, float, const float *, int, const float *, int, float, float *, int );
void pack_RB(char, char, int, int, const float *, int, float *, int, const convol_dim *, int, int);
void pack_CB(char, char, int, int, const float *, int, float *, int, const convol_dim *, int, int);
void unpack_RB( char, char, int, int, float *, int, const float *, int );
void unpack_CB( char, char, int, int, float *, int, const float *, int );
