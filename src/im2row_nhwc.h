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

void im2row_nhwc(float *rows, int ld, const float *in, int batches, int height, int width, int channels,
                 int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding,
                 int vstride, int hstride, int vdilation, int hdilation);

void pack_CB_nhwc(char orderM, char transM, int mc, int nc, const float *M, int ldM, float *Mc, int RR,
                  const conv_p *conv_params, int start_row, int start_col);

void row2im_nhwc(int m, int n, const float *rows, int ld, float *out, int batches, int height, int width, int channels,
                 int oheight, int owidth, int kheight, int kwidth, int vpadding, int hpadding,
                 int vstride, int hstride, int vdilation, int hdilation);

void post_row2im_nhwc(int m, int n, const float *rows, int ldr, float beta, float *out, int ldout,
                      const conv_p *conv_params, int start_row, int start_col, bool last);

void add_bias_nhwc(int mr, int nr, const float *Cc, int ldCc, float beta, float *C, int ldC,
                   const conv_p *conv_params, int start_row, int start_col, bool last);
