static inline double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

extern double t_pack, t_kernel, t_generic;

void gemm_blis_B3A2C0( char, char, char, char, char, int, int, int, float, const float *, int, const float *, int, float, float *, int,
                       float *, float *, float *, cntx_t* );
void gemm_base_Cresident( char, int, int, int, float, const float *, int, const float *, int, float, float *, int );
void pack_RB( char, char, int, int, const float *, int, float *, int );
void pack_CB( char, char, int, int, const float *, int, float *, int );
void unpack_RB( char, char, int, int, float *, int, const float *, int );
void unpack_CB( char, char, int, int, float *, int, const float *, int );
