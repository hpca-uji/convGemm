void gemm_blis_B3A2C0( char, char, char, char, char, int, int, int, float, float *, int, float *, int, float, float *, int, 
                       float *, float *);
void gemm_base_Cresident( char, int, int, int, float, float *, int, float *, int, float, float *, int );
void pack_RB( char, char, int, int, float *, int, float *, int );
void pack_CB( char, char, int, int, float *, int, float *, int );
void unpack_RB( char, char, int, int, float *, int, float *, int );
void unpack_CB( char, char, int, int, float *, int, float *, int );
