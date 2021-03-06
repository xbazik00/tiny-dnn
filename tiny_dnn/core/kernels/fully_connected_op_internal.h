/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/params/fully_params.h"
#include "tiny_dnn/core/fixed.h"

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

    #if defined(FIXED)
    for (size_t i = 0; i < params.out_size_; i++) {
      fixed_t fixed_out = fixed_t{0};
      for (size_t c = 0; c < params.in_size_; c++) {
        fixed_out += static_cast<fixed_t>(W[c * params.out_size_ + i]) * static_cast<fixed_t>(in[c]);
      }

      if (params.has_bias_) {
        fixed_out += static_cast<fixed_t>(bias[i]);
      }
      out[i] = static_cast<float_t>(fixed_out);
    }

    #elif defined(QUANT)
    tiny_dnn::core::kernels::tiny_quantized_fully_connected_kernel(
      params, in, W, bias, out, layer_parallelize);

    #else
    for (size_t i = 0; i < params.out_size_; i++) {
      out[i] = float_t{0};
      for (size_t c = 0; c < params.in_size_; c++) {
        out[i] += W[c * params.out_size_ + i] * in[c];
      }

      if (params.has_bias_) {
        out[i] += bias[i];
      }
    }
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
    #if defined(FIXED)
    
    for (size_t c = 0; c < params.in_size_; c++) {
      fixed_t prev_delta_fixed{0};
      for (size_t io = 0; io < params.out_size_; io++) {
        prev_delta_fixed += static_cast<fixed_t>(curr_delta[sample][io]) * static_cast<fixed_t>(W[c * params.out_size_ + io]);
      }
      prev_delta[sample][c] += static_cast<float_t>(prev_delta_fixed);
    }

    for_(layer_parallelize, 0, size_t(params.out_size_),
       [&](const blocked_range &r) {
      for (size_t c = 0; c < params.in_size_; c++) {
        for (size_t io = 0; io < params.out_size_; io++) {
          dW[sample][c * params.out_size_ + io] = static_cast<fixed_t>(curr_delta[sample][io]) * static_cast<fixed_t>(prev_out[sample][c]);
        } 
      }
      if (params.has_bias_) {
        for (size_t i = r.begin(); i < r.end(); i++) {
          db[sample][i] = static_cast<fixed_t>(db[sample][i]) + static_cast<fixed_t>(curr_delta[sample][i]);
        }
      }
    });


    #elif defined(QUANT)
    tiny_dnn::core::kernels::tiny_quantized_fully_connected_back_kernel(
      params, prev_out[sample], W, dW[sample], prev_delta[sample], 
      curr_delta[sample], db[sample], layer_parallelize);

    #else
    for (size_t c = 0; c < params.in_size_; c++) {
      // propagate delta to previous layer
      // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
 
      prev_delta[sample][c] += vectorize::dot(
        &curr_delta[sample][0], &W[c * params.out_size_], params.out_size_);
    }

    for_(layer_parallelize, 0, params.out_size_, [&](const blocked_range &r) {
      // accumulate weight-step using delta
      // dW[c * out_size + i] += current_delta[i] * prev_out[c]curr_delta
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
    #endif
  }
}

}  // namespace kernels
}  // namespace tiny_dnn
