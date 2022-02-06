/**
 * This file is part of convGemmNHWC
 *
 * Copyright (C) 2021-22 Universitat Politècnica de València and
 *                       Universitat Jaume I
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

int alloc_pack_buffs(float **Ac_pack, float **Bc_pack);

void sconvGemmNHWC(char trans,
                   int b, int h, int w, int c,
                   int kn, int kh, int kw,
                   int vpadding, int hpadding,
                   int vstride, int hstride,
                   int vdilation, int hdilation,
                   const float *in,
                   const float *x,
                   float *out, const float *bias_vector,
                   const float *bn_running_mean, const float *bn_inv_std,
                   const float *bn_gamma, const float *bn_beta, bool relu,
                   float *ac_pack, float *bc_pack);

void sconvGemmNHWC_back(int b, int h, int w, int c,
                        int kn, int kh, int kw,
                        int vstride, int hstride,
                        int vpadding, int hpadding,
                        int vdilation, int hdilation,
                        const float *weights,
                        const float *dy, float *dx,
                        float *ac_pack, float *bc_pack);

void sconvGemmNCHW(char trans,
                   int b, int c, int h, int w,
                   int kn, int kh, int kw,
                   int vpadding, int hpadding,
                   int vstride, int hstride,
                   int vdilation, int hdilation,
                   const float *in,
                   const float *x,
                   float *out, const float *bias_vector,
                   const float *bn_running_mean, const float *bn_inv_std,
                   const float *bn_gamma, const float *bn_beta, bool relu,
                   float *ac_pack, float *bc_pack);

void sconvGemmNCHW_back(int b, int c, int h, int w,
                        int kn, int kh, int kw,
                        int vstride, int hstride,
                        int vpadding, int hpadding,
                        int vdilation, int hdilation,
                        const float *weights,
                        const float *dy, float *dx,
                        float *ac_pack, float *bc_pack);
