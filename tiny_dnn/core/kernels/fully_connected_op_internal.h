/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/params/fully_params.h"

#ifdef QUANT
#include "tiny_dnn/core/kernels/tiny_quantization_kernel.h"
#include "tiny_dnn/core/kernels/tiny_quantized_fully_connected_kernel.h"
#endif

namespace tiny_dnn {
namespace kernels {

inline void fully_connected_op_internal(const tensor_t &in_data,
                                        const vec_t &W,
                                        const vec_t &bias,
                                        tensor_t &out_data,
                                        const core::fully_params &params,
                                        const bool layer_parallelize) {
  for_i(layer_parallelize, in_data.size(), [&](size_t sample) {
    const vec_t &in = in_data[sample];
    vec_t &out      = out_data[sample];

    #ifndef QUANT
    for (size_t i = 0; i < params.out_size_; i++) {
      out[i] = float_t{0};
      for (size_t c = 0; c < params.in_size_; c++) {
        out[i] += W[c * params.out_size_ + i] * in[c];
      }

      if (params.has_bias_) {
        out[i] += bias[i];
      }
    }

    #else
    //tiny_dnn::core::kernels::tiny_quantized_fully_connected_kernel(
    //  params, in, W, bias, out, layer_parallelize);
    using namespace tiny_dnn::core::kernels;

    float_t min_input(in[0]);
    float_t max_input(in[0]);

    for (size_t c = 0; c < params.in_size_; c++) {
      min_input = std::min(min_input, in[c]);
      max_input = std::max(max_input, in[c]);
    }

    std::vector<uint8_t> in_quantized =
      float_tensor_to_quantized<uint8_t>(in, min_input, max_input);
    
    float_t min_filter(W[0]);
    float_t max_filter(W[0]);

    for (size_t c = 0; c < W.size(); c++) {
      min_filter = std::min(min_filter, W[c]);
      max_filter = std::max(max_filter, W[c]);
    }

    if (min_filter == max_filter) {
      max_filter = W[0] + 1e-3f;
      min_filter = W[0] - 1e-3f;
    }

    std::vector<uint8_t> W_quantized =
      float_tensor_to_quantized<uint8_t>(W, min_filter, max_filter);
    
    float_t min_output_value;
    float_t max_output_value;
    
    quantization_range_for_multiplication<uint8_t, uint8_t, int32_t>(
      min_input, max_input, min_filter, max_filter, &min_output_value,
      &max_output_value);
    
    float_t min_bias(0);
    float_t max_bias(0);
    
    std::vector<uint8_t> bias_quantized;
    
    if (params.has_bias_) {
      for (size_t inc = 0; inc < bias.size(); inc++) {
        min_bias = std::min(min_bias, bias[inc]);
        max_bias = std::max(max_bias, bias[inc]);
      }
      if (min_bias == max_bias) {
        max_bias = bias[0] + 1e-3f;
        min_bias = bias[0] - 1e-3f;
      }
      bias_quantized = float_tensor_to_quantized<uint8_t>(bias, min_bias, max_bias);
    }
    
    min_output_value += min_bias;
    max_output_value += max_bias;

    std::vector<int32_t> out_quantized(out.size(), static_cast<int32_t>(0));

    // calculating offset
    const int32_t offset_input =
      float_to_quantized_unclamped<uint8_t>(0.0f, min_input, max_input);
    
    const int32_t offset_filter =
      float_to_quantized_unclamped<uint8_t>(0.0f, min_filter, max_filter);
  
    const int32_t zero_in_total_space =
      float_to_quantized<int32_t>(0.0f, min_output_value, max_output_value);

    for_i(layer_parallelize, params.out_size_, [&](size_t i) {
      for (size_t c = 0; c < params.in_size_; c++) {
        out_quantized[i] +=
          static_cast<int32_t>(W_quantized[c * params.out_size_ + i] -
                                offset_filter) *
          static_cast<int32_t>(in_quantized[c] - offset_input);
      }
      if (params.has_bias_) {
        out_quantized[i] += (bias_quantized[i] - zero_in_total_space);
      }
    });

    float_t min_output_requantized;
    float_t max_output_requantized;
    
    std::vector<uint8_t> out_requantized(out_quantized.size(),
                                        static_cast<uint8_t>(0));

    // Requantize from 32bits to 8 bits for next layer
    quantize_down_and_shrink_range<int32_t, uint8_t>(
      out_quantized, min_output_value, max_output_value, &min_output_requantized,
      &max_output_requantized, &out_requantized);

    out = quantized_tensor_to_float<uint8_t>(
      out_requantized, min_output_requantized, max_output_requantized);
    #endif
  });
}


inline void fully_connected_op_internal(const tensor_t &prev_out,
                                        const vec_t &W,
                                        tensor_t &dW,
                                        tensor_t &db,
                                        tensor_t &curr_delta,
                                        tensor_t &prev_delta,
                                        const core::fully_params &params,
                                        const bool layer_parallelize) {
  
  for (size_t sample = 0; sample < prev_out.size(); sample++) {
    #ifndef QUANT
    for (size_t c = 0; c < params.in_size_; c++) {
      // propagate delta to previous layer
      // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
      prev_delta[sample][c] += vectorize::dot(
        &curr_delta[sample][0], &W[c * params.out_size_], params.out_size_);
    }

    for_(layer_parallelize, 0, params.out_size_, [&](const blocked_range &r) {
      // accumulate weight-step using delta
      // dW[c * out_size + i] += current_delta[i] * prev_out[c]
      for (size_t c = 0; c < params.in_size_; c++) {
        vectorize::muladd(&curr_delta[sample][r.begin()], prev_out[sample][c],
                          r.end() - r.begin(),
                          &dW[sample][c * params.out_size_ + r.begin()]);
      }

      if (params.has_bias_) {
        // vec_t& db = *in_grad[2];
        for (size_t i = r.begin(); i < r.end(); i++) {
          db[sample][i] += curr_delta[sample][i];
        }
      }
    });
    #else
    //tiny_dnn::core::kernels::tiny_quantized_fully_connected_back_kernel(
    //  params, prev_out[sample], W, dW[sample], prev_delta[sample], 
    //  curr_delta[sample], db[sample], layer_parallelize);
    using namespace tiny_dnn::core::kernels;

    // previous output quantization
    float_t min_prev_out(prev_out[sample][0]);
    float_t max_prev_out(prev_out[sample][0]);
    for (size_t inc = 0; inc < prev_out[sample].size(); inc++) {
      min_prev_out = std::min(min_prev_out, prev_out[sample][inc]);
      max_prev_out = std::max(min_prev_out, prev_out[sample][inc]);
    }
    std::vector<uint8_t> prev_out_quantized =
      float_tensor_to_quantized<uint8_t>(prev_out[sample], min_prev_out, max_prev_out);

    // filter quantization
    float_t min_filter(W[0]);
    float_t max_filter(W[0]);
    for (size_t c = 0; c < W.size(); c++) {
      min_filter = std::min(min_filter, W[c]);
      max_filter = std::max(max_filter, W[c]);
    }
    if (min_filter == max_filter) {
      max_filter = W[0] + 1e-3f;
      min_filter = W[0] - 1e-3f;
    }
    std::vector<uint8_t> W_quantized =
      float_tensor_to_quantized<uint8_t>(W, min_filter, max_filter);

    // current delta quantization
    float_t min_curr_delta(curr_delta[sample][0]);
    float_t max_curr_delta(curr_delta[sample][0]);
    for (size_t inc = 0; inc < curr_delta[sample].size(); inc++) {
      min_curr_delta = std::min(min_curr_delta, curr_delta[sample][inc]);
      max_curr_delta = std::max(max_curr_delta, curr_delta[sample][inc]);
    }
    std::vector<uint8_t> curr_delta_quantized =
      float_tensor_to_quantized<uint8_t>(curr_delta[sample], min_curr_delta,
                                          max_curr_delta);

    // output range for previous delta
    float_t min_prev_delta_value;
    float_t max_prev_delta_value;
    quantization_range_for_multiplication<uint8_t, uint8_t, int32_t>(
      min_curr_delta, max_curr_delta, min_filter, max_filter,
      &min_prev_delta_value, &max_prev_delta_value);

    std::vector<int32_t> prev_delta_quantized(prev_delta[sample].size(),
                                              static_cast<int32_t>(0));

    // output range for dW
    float_t min_dW_value;
    float_t max_dW_value;
    quantization_range_for_multiplication<uint8_t, uint8_t, int32_t>(
      min_curr_delta, max_curr_delta, min_prev_out, max_prev_out, &min_dW_value,
      &max_dW_value);

    std::vector<int32_t> dW_quantized(dW[sample].size(), static_cast<int32_t>(0));

    // calculating offset
    const int32_t offset_prev_out =
      float_to_quantized_unclamped<uint8_t>(0.0f, min_prev_out, max_prev_out);
    const int32_t offset_filter =
      float_to_quantized_unclamped<uint8_t>(0.0f, min_filter, max_filter);
    const int32_t offset_curr_delta =
      float_to_quantized_unclamped<uint8_t>(0.0f, min_curr_delta, max_curr_delta);
    // const int32_t zero_in_prev_delta =
    //    float_to_quantized<int32_t>(0.0f, min_prev_delta_value,
    //    max_prev_delta_value);

    for (size_t c = 0; c < params.in_size_; c++) {
      // propagate delta to previous layer
      // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
      for (size_t io = 0; io < params.out_size_; io++) {
        prev_delta_quantized[c] +=
          (static_cast<int32_t>(curr_delta_quantized[io]) - offset_curr_delta) *
          (static_cast<int32_t>(W_quantized[c * params.out_size_ + io]) -
            offset_filter);
      }
    }

    float_t min_prev_delta_requantized;
    float_t max_prev_delta_requantized;
    std::vector<uint8_t> prev_delta_requantized(prev_delta_quantized.size(),
                                                static_cast<uint8_t>(0));

    // Requantize from 32bits to 8 bits for next layer
    quantize_down_and_shrink_range<int32_t, uint8_t>(
      prev_delta_quantized, min_prev_delta_value, max_prev_delta_value,
      &min_prev_delta_requantized, &max_prev_delta_requantized,
      &prev_delta_requantized);

    // dequantize to flaot, this could be removed within concatenated quantized
    // network
    prev_delta[sample] = quantized_tensor_to_float<uint8_t>(prev_delta_requantized,
                                                    min_prev_delta_requantized,
                                                    max_prev_delta_requantized);

    for_(layer_parallelize, 0, size_t(params.out_size_),
          [&](const blocked_range &r) {
            // accumulate weight-step using delta
            // dW[c * out_size + i] += current_delta[i] * prev_out[c]
            for (size_t c = 0; c < params.in_size_; c++) {
              for (size_t io = 0; io < params.out_size_; io++) {
                dW_quantized[c * params.out_size_ + io] +=
                  (static_cast<int32_t>(curr_delta_quantized[io]) -
                  offset_curr_delta) *
                  (static_cast<int32_t>(prev_out_quantized[c]) - offset_prev_out);
              }
            }

            if (params.has_bias_) {
              // vec_t& db = *in_grad[2];
              for (size_t i = r.begin(); i < r.end(); i++) {
                db[sample][i] += curr_delta[sample][i];
              }
            }
          });

    float_t min_dW_requantized;
    float_t max_dW_requantized;
    std::vector<uint8_t> dW_requantized(dW_quantized.size(),
                                        static_cast<uint8_t>(0));

    // requantize from 32bits to 8 bits for next layer
    quantize_down_and_shrink_range<int32_t, uint8_t>(
      dW_quantized, min_dW_value, max_dW_value, &min_dW_requantized,
      &max_dW_requantized, &dW_requantized);

    // dequantize to flaot, this could be removed within concatenated quantized
    // network
    dW[sample] = quantized_tensor_to_float<uint8_t>(dW_requantized, min_dW_requantized,
                                            max_dW_requantized);
    #endif
  }
}

}  // namespace kernels
}  // namespace tiny_dnn
