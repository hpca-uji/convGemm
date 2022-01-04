/*
 * convgemm dynamic library interface (used by PyDTNN)
 */

int alloc_pack_buffs(float** Ac_pack, float** Bc_pack);

void sconvGemmNHWC(char trans,
                    unsigned b, unsigned h, unsigned w, unsigned c,
                    unsigned kn, unsigned kh, unsigned kw,
                    unsigned vpadding, unsigned hpadding,
                    unsigned vstride, unsigned hstride,
                    unsigned vdilation, unsigned hdilation,
                    const float *in,
                    const float *x,
                    float *out, const float *bias_vector,
                    const float *bn_running_mean, const float *bn_inv_std,
                    const float *bn_gamma, const float *bn_beta, bool relu,
                    float *ac_pack, float *bc_pack);

void sconvGemmNHWC_back(unsigned b, unsigned h, unsigned w, unsigned c,
                        unsigned kn, unsigned kh, unsigned kw,
                        unsigned vstride, unsigned hstride,
                        unsigned vpadding, unsigned hpadding,
                        unsigned vdilation, unsigned hdilation,
                        const float *weights,
                        const float *dy, float *dx,
                        float *ac_pack, float *bc_pack);

void sconvGemmNCHW(char trans,
                    unsigned b, unsigned c, unsigned h, unsigned w,
                    unsigned kn, unsigned kh, unsigned kw,
                    unsigned vpadding, unsigned hpadding,
                    unsigned vstride, unsigned hstride,
                    unsigned vdilation, unsigned hdilation,
                    const float *in,
                    const float *x,
                    float *out, const float *bias_vector,
                    const float *bn_running_mean, const float *bn_inv_std,
                    const float *bn_gamma, const float *bn_beta, bool relu,
                    float *ac_pack, float *bc_pack);

void sconvGemmNCHW_back(unsigned b, unsigned c, unsigned h, unsigned w,
                        unsigned kn, unsigned kh, unsigned kw,
                        unsigned vstride, unsigned hstride,
                        unsigned vpadding, unsigned hpadding,
                        unsigned vdilation, unsigned hdilation,
                        const float *weights,
                        const float *dy, float *dx,
                        float *ac_pack, float *bc_pack);
