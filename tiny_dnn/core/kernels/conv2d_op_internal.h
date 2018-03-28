/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#ifdef QUANT
#include "tiny_dnn/core/kernels/tiny_quantization_kernel.h"
#include "tiny_dnn/core/kernels/tiny_quantized_conv2d_kernel.h"
#endif

namespace tiny_dnn {
namespace kernels {

inline void conv2d_op_internal(const tensor_t &in_data,
                               const vec_t &W,
                               const vec_t &bias,
                               tensor_t &out_data,
                               const core::conv_params &params,
                               const bool parallelize) {
  for_(parallelize, 0u, in_data.size(),
      [&](const blocked_range &r) {
        size_t out_area    = params.out.area();
        size_t iw          = params.in_padded.width_;
        size_t ih          = params.in_padded.height_;
        size_t id          = params.in.depth_;
        size_t ow          = params.out.width_;
        size_t oh          = params.out.height_;
        size_t od          = params.out.depth_;
        size_t kw          = params.weight.width_;
        size_t kh          = params.weight.height_;
        size_t elem_stride = params.w_stride;
        size_t line_stride = iw * params.h_stride;
        for (size_t sample = r.begin(); sample < r.end(); sample++) {
          const vec_t &in = in_data[sample];
          vec_t &a        = out_data[sample];

          #ifndef QUANT
          for (size_t o = 0; o < od; o++) {
            float_t *pa = &a[params.out.get_index(0, 0, o)];
            for (size_t inc = 0; inc < id; inc++) {
              if (!params.tbl.is_connected(o, inc)) continue;
              size_t idx;
              idx                = params.weight.get_index(0, 0, id * o + inc);
              const float_t *pw  = &W[idx];
              idx                = params.in_padded.get_index(0, 0, inc);
              const float_t *pin = &in[idx];
              float_t *pout      = pa;
              for (size_t y = 0; y < oh; y++) {
                const float_t *pin_line = pin;
                for (size_t x = 0; x < ow; x++) {
                  const float_t *pin_element = pin_line;
                  const float_t *pw_element  = pw;
                  float_t sum{0};
                  // should be optimized for small kernel(3x3,5x5)
                  for (size_t wy = 0; wy < kh; wy++) {    // NOLINT
                    for (size_t wx = 0; wx < kw; wx++) {  // NOLINT
                      sum += pw_element[wx] * pin_element[wx];
                    }
                    pw_element += kw;
                    pin_element += iw;
                  }
                  pout[x] += sum;
                  pin_line += elem_stride;
                }
                pout += ow;
                pin += line_stride;
              }
            }
            if (params.has_bias) {
              vectorize::add(bias[o], out_area, pa);
            }
          }
          #else
          //tiny_dnn::core::kernels::tiny_quantized_conv2d_kernel(params, in, W, bias, a, parallelize);
          using namespace tiny_dnn::core::kernels;
          float_t min_input(in[0]);
          float_t max_input(in[0]);

          for(size_t inc = 0; inc < id; inc++){
            for(size_t ins = 0; ins < ih * ih; ins++){
              size_t idx = params.in_padded.get_index(0, 0, inc);
              min_input = std::min(min_input, (&in[idx])[ins]);
              max_input = std::max(max_input, (&in[idx])[ins]);
            }             
          }

          float_t min_filter(W[0]);
          float_t max_filter(W[0]);

          for(size_t inc = 0; inc < id; inc++){
            for(size_t ins = 0; ins < kh * kh; ins++){
              size_t idx = params.in_padded.get_index(0, 0, inc);
              min_filter = std::min(min_filter, (&W[idx])[ins]);
              max_filter = std::max(max_filter, (&W[idx])[ins]);
            }             
          }

          if (min_filter == max_filter) {
            max_filter = W[0] + 1e-3f;
            min_filter = W[0] - 1e-3f;
          }

          float_t min_bias(0);
          float_t max_bias(0);

          if(params.has_bias){
            for(size_t inc = 0; inc < od; inc++){
              min_bias = std::min(min_bias, bias[inc]);
              max_bias = std::max(max_bias, bias[inc]);
            }
            if (min_bias == max_bias) {
              max_bias = bias[0] + 1e-3f;
              min_bias = bias[0] - 1e-3f;
            }
          }
          
          float_t min_output_value;
          float_t max_output_value;
          quantization_range_for_multiplication<uint8_t, uint8_t, int32_t>(
            min_input, max_input, min_filter, max_filter, &min_output_value,
            &max_output_value);
          
          std::vector<uint8_t> in_quantized =
            float_tensor_to_quantized<uint8_t>(in, min_input, max_input);

          std::vector<uint8_t> W_quantized =
            float_tensor_to_quantized<uint8_t>(W, min_filter, max_filter);

          std::vector<uint8_t> bias_quantized;
          if (params.has_bias) {
            bias_quantized =
              float_tensor_to_quantized<uint8_t>(bias, min_bias, max_bias);
          }
          
          std::vector<int32_t> a_quantized(a.size(), static_cast<int32_t>(0));  
          
          // calculating offset
          const int32_t offset_input = int64_to_int32(
            float_to_quantized_unclamped<uint8_t>(0.0f, min_input, max_input));
          const int32_t offset_filter = int64_to_int32(
            float_to_quantized_unclamped<uint8_t>(0.0f, min_filter, max_filter));
          const int32_t zero_in_total_space = int64_to_int32(
            float_to_quantized<int32_t>(0.0f, min_output_value, max_output_value));

          for_i(parallelize, od, [&](size_t o){
            for(size_t inc = 0; inc < id; inc++){
              if (!params.tbl.is_connected(o, inc)) continue;

              size_t idx        = 0;
              idx               = id * o + inc;
              idx               = params.weight.get_index(0, 0, idx);
              const uint8_t *pw = &W_quantized[idx];

              idx               = params.in_padded.get_index(0, 0, inc);
              const uint8_t *pi = &in_quantized[idx];

              idx                   = params.out.get_index(0, 0, o);
              int32_t *pa_quantized = &a_quantized[idx];

              for (size_t y = 0; y < oh; y++) {
                for (size_t x = 0; x < ow; x++) {
                  const uint8_t *ppw = pw;
                  const uint8_t *ppi = pi + iw * (y * params.h_stride) +
                                      x * params.w_stride;
                  int32_t sum = 0;

                  // should be optimized for small kernel(3x3,5x5)
                  for (size_t wy = 0; wy < kh; wy++) {
                    for (size_t wx = 0; wx < kw; wx++) {
                      idx = wy * iw + wx;
                      sum += (static_cast<int32_t>(*ppw++) - offset_filter) *
                            (static_cast<int32_t>(ppi[idx]) - offset_input);
                    }
                  }
                  pa_quantized[y * ow + x] += sum;
                }
              }
            }
            if (params.has_bias) {
              int32_t *pa_quantized = &a_quantized[params.out.get_index(0, 0, o)];
              int32_t *paa_quantized =
                pa_quantized + ow * oh;
              std::for_each(pa_quantized, paa_quantized, [&](int32_t &f) {
                f += (bias_quantized[o] - zero_in_total_space);
              });
            }
          });
          float_t min_output_requantized;
          float_t max_output_requantized;
          std::vector<uint8_t> a_requantized(a_quantized.size(),
                                            static_cast<uint8_t>(0));

          // Requantize from 32bits to 8 bits for next layer
          quantize_down_and_shrink_range<int32_t, uint8_t>(
            a_quantized, min_output_value, max_output_value, &min_output_requantized,
            &max_output_requantized, &a_requantized);

          // dequantize to flaot, this could be removed within concatenated quantized
          // network
          a = quantized_tensor_to_float<uint8_t>(a_requantized, min_output_requantized,
                                                max_output_requantized);
          
          #endif 
        }
      },
      0u);
}

/******************************************************************/

template <typename tensor_t, typename vec_t>
void conv2d_op_internal(const tensor_t &prev_out,
                        const vec_t &W,
                        tensor_t &dW,
                        tensor_t &db,
                        tensor_t &curr_delta,
                        tensor_t &prev_delta,
                        const core::conv_params &params,
                        const bool parallelize) {
  typedef typename vec_t::value_type float_t;

  for_i(parallelize, prev_out.size(), [&](size_t sample) {
    #ifndef QUANT
    // propagate delta to previous layer
    for (size_t inc = 0; inc < params.in.depth_; inc++) {
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        if (!params.tbl.is_connected(outc, inc)) continue;

        size_t idx        = 0;
        idx               = params.in.depth_ * outc + inc;
        idx               = params.weight.get_index(0, 0, idx);
        const float_t *pw = &W[idx];

        idx                       = params.out.get_index(0, 0, outc);
        const float_t *pdelta_src = &curr_delta[sample][idx];

        idx = params.in_padded.get_index(0, 0, inc);
        // float_t* pdelta_dst = &(*prev_delta)[sample][idx];
        float_t *pdelta_dst = &prev_delta[sample][idx];

        for (size_t y = 0; y < params.out.height_; y++) {
          for (size_t x = 0; x < params.out.width_; x++) {
            const float_t *ppw = pw;

            idx                       = y * params.out.width_ + x;
            const float_t ppdelta_src = pdelta_src[idx];

            float_t *ppdelta_dst =
              pdelta_dst + y * params.h_stride * params.in_padded.width_ +
              x * params.w_stride;

            for (size_t wy = 0; wy < params.weight.height_; wy++) {   // NOLINT
              for (size_t wx = 0; wx < params.weight.width_; wx++) {  // NOLINT
                idx = wy * params.in_padded.width_ + wx;
                ppdelta_dst[idx] += *ppw++ * ppdelta_src;
              }
            }
          }
        }
      }
    }

    // accumulate dw
    for (size_t inc = 0; inc < params.in.depth_; inc++) {
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        if (!params.tbl.is_connected(outc, inc)) continue;

        for (size_t wy = 0; wy < params.weight.height_; wy++) {
          for (size_t wx = 0; wx < params.weight.width_; wx++) {
            float_t dst{0};

            size_t idx           = 0;
            idx                  = params.in_padded.get_index(wx, wy, inc);
            const float_t *prevo = &prev_out[sample][idx];

            idx                  = params.out.get_index(0, 0, outc);
            const float_t *delta = &curr_delta[sample][idx];

            if (params.w_stride > 1) {
              for (size_t y = 0; y < params.out.height_; y++) {
                size_t prevo_idx =
                  y * params.in_padded.width_ * params.h_stride;
                size_t delta_idx = y * params.out.width_;

                for (size_t x = 0; x < params.out.width_; x++) {
                  dst += prevo[prevo_idx + x * params.w_stride] *
                         delta[delta_idx + x];
                }
              }
            } else {
              for (size_t y = 0; y < params.out.height_; y++) {
                dst += vectorize::dot(
                  prevo + y * params.in_padded.width_ * params.h_stride,
                  delta + y * params.out.width_, params.out.width_);
              }
            }

            idx = params.in.depth_ * outc + inc;
            dW[sample][params.weight.get_index(wx, wy, idx)] += dst;
          }
        }
      }
    }

    // accumulate db
    if (params.has_bias) {
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        size_t idx            = params.out.get_index(0, 0, outc);
        const float_t *delta  = &curr_delta[sample][idx];
        const float_t *deltaa = delta + params.out.width_ * params.out.height_;
        db[sample][outc] += std::accumulate(delta, deltaa, float_t{0});
      }
    }
    #else
    tiny_dnn::core::kernels::tiny_quantized_conv2d_back_kernel(
      params, prev_out[sample], W, dW[sample], db[sample],
      curr_delta[sample], &prev_delta[sample]);
    #endif
  });
}

}  // namespace kernels
}  // namespace tiny_dnn
