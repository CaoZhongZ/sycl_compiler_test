/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#include "compatible.h"
#include "conversion_utils.h"
#include "memory_access_utils.h"
#include "reduction_utils.h"
#include "load_store.hpp"

using rop = reduce::ROpType;

namespace ln {
constexpr int granularity = 16;
} // namespace ln

/*
Primary layer norm implementation. Assumes elems_per_row % 8
is equal to 0.

Args:
    output: buffer for output data
    vals: buffer for input data
    gamma: gain for normalization
    beta: bias for normalization
    epsilon: numeric stability
    elems_per_row: number of elements each block will normalize
*/
template <typename T, int unRoll, int threadsPerGroup, int maxThreads>
class load_func {
  T *output;
  const T *vals;
  const T *gamma;
  const T *beta;
  float epsilon;
  int elems_per_row;
  sycl::stream out;

public:
  load_func(T *output, const T *vals, const T *gamma, const T *beta,
           float epsilon, int elems_per_row, sycl::stream out)
      : output(output), vals(vals), gamma(gamma), beta(beta), epsilon(epsilon),
        elems_per_row(elems_per_row), out(out) {};

  void operator()(sycl::nd_item<2> pos) const {
    constexpr int T_per_load = ln::granularity / sizeof(T);

    auto warp = sycl::ext::oneapi::this_sub_group();
    auto tb = pos.get_group();
    const int block_offset =
        (tb.get_group_id(1) * (maxThreads / threadsPerGroup) * elems_per_row) +
        (tb.get_local_id(0) * elems_per_row);
    const int thread_offset = tb.get_local_id(1) * T_per_load;
    const int base_offset = block_offset + thread_offset;
    const int stride = tb.get_local_linear_range() * T_per_load;
    // if (pos.get_group_linear_id() == 127) {
    //   out << "<const: > "<< block_offset << " " << thread_offset << " " << base_offset << '\n';
    // }

    const T *input_base = vals + base_offset;

    T local_buffer[unRoll * T_per_load];
    T *block_output = output + block_offset;

#pragma unroll (unRoll)
    for (int i = 0; i < unRoll; i++) {
      T *iteration_buffer = local_buffer + i * T_per_load;
      
      const int iter_idx = thread_offset + i * stride;
      const bool do_loads = iter_idx < elems_per_row;


      mem_access::load_global<ln::granularity>(
          iteration_buffer, input_base + i * stride, thread_offset + i * stride < elems_per_row);
      
      // if (pos.get_group_linear_id() == 127) {
      //   const T* origin_ = input_base + i * stride;
      //   out << "<load> "<< conversion::to<float>(origin_[1]) << " " << " " << conversion::to<float>(iteration_buffer[1]) <<'\n';
      // }
      
      if (do_loads) {
        mem_access::store_global<ln::granularity>(block_output + iter_idx,
                                                  iteration_buffer);
      }
      
      // if (pos.get_group_linear_id() == 127) {
      //   const T* origin_ = block_output + iter_idx;
      //   out << "<store> "<< conversion::to<float>(origin_[1]) << " " << " " << conversion::to<float>(iteration_buffer[1]) <<'\n';
      // }
      
    }
    
  };
};

#define _disableLAUNCH_LOAD_FUNC(unRollFactor, threadsPerGroup, maxThreads)             \
  {                                                                            \
    load_func<T, unRollFactor, threadsPerGroup, maxThreads> fn(                 \
        output, vals, gamma, beta, epsilon, elems_per_row);                    \
    stream.submit([&](sycl::handler &cmd_list) {                               \
      cmd_list.parallel_for(sycl::nd_range<2>{grid, block}, fn);               \
    });                                                                        \
  }

#define LAUNCH_LOAD_FUNC(unRollFactor, threadsPerGroup, maxThreads)             \
  {                                                                            \
    stream.submit([&](sycl::handler &cmd_list) {                               \
      sycl::stream out(0x100000, 8192, cmd_list);                              \
      load_func<T, unRollFactor, threadsPerGroup, maxThreads> fn(               \
          output, vals, gamma, beta, epsilon, elems_per_row, out);             \
      cmd_list.parallel_for(sycl::nd_range<2>{grid, block}, fn);               \
    });                                                                        \
  }

template <typename T>
void launch_load_func(T *output, const T *vals, const T *gamma, const T *beta,
                     float epsilon, int rows, int elems_per_row,
                     sycl::queue stream) {
  // 8 for sycl::half, 4 for float
  constexpr int T_per_load = ln::granularity / sizeof(T);

  constexpr int maxThreads = 256;

  // For Flaoat, unRoll 4, for sycl::half, unRoll 2
  constexpr int internal_unRoll = sizeof(T) == 4 ? 4 : 2;

  const bool is_subblock_schedule = (elems_per_row <= 128) ? true : false;
  const int h_per_step =
      is_subblock_schedule ? T_per_load : T_per_load * internal_unRoll;

  // Scheduling concern: may be slightly faster for some inputs to assign
  // multiple stages of warp-sized blocks rather than stepping up to 64/96
  // threads
  const int one_step_threads =
      next_pow2((elems_per_row + h_per_step - 1) / h_per_step);
  const int threadsPerGroup =
      (one_step_threads < maxThreads) ? one_step_threads : maxThreads;

  const int groups_per_block_max =
      is_subblock_schedule
          ? (maxThreads + threadsPerGroup - 1) / threadsPerGroup
          : 1;
  const int groups_per_block =
      (rows < groups_per_block_max) ? rows : groups_per_block_max;
  const int groups_launch = (groups_per_block + rows - 1) / groups_per_block;

  sycl::range<2> block {(size_t)groups_per_block, (size_t)threadsPerGroup};
  sycl::range<2> grid {(size_t)groups_per_block, (size_t)(threadsPerGroup * groups_launch)};

  const int elems_per_step = threadsPerGroup * h_per_step;
  const int external_unRoll =
      (elems_per_row + elems_per_step - 1) / elems_per_step;

  if (is_subblock_schedule) {
    // <=128
    if (threadsPerGroup == 1) {
      LAUNCH_LOAD_FUNC(1, 1, maxThreads);
    } else if (threadsPerGroup == 2) {
      LAUNCH_LOAD_FUNC(1, 2, maxThreads);
    } else if (threadsPerGroup == 4) {
      LAUNCH_LOAD_FUNC(1, 4, maxThreads);
    } else if (threadsPerGroup == 8) {
      LAUNCH_LOAD_FUNC(1, 8, maxThreads);
    } else if (threadsPerGroup == 16) {
      LAUNCH_LOAD_FUNC(1, 16, maxThreads);
    }
  } else if (external_unRoll == 1) {
    // 129 - 4096 elems
    // (this can launch with 1-7 warps as well)
    LAUNCH_LOAD_FUNC(1 * internal_unRoll, maxThreads, maxThreads);
  } else if (external_unRoll == 2) {
    // 4097 - 8192 elems
    LAUNCH_LOAD_FUNC(2 * internal_unRoll, maxThreads, maxThreads);
  } else if (external_unRoll == 3) {
    // 8193 - 12288 elems
    LAUNCH_LOAD_FUNC(3 * internal_unRoll, maxThreads, maxThreads);
  } else if (external_unRoll == 4) {
    // 12289 - 16384 elems
    LAUNCH_LOAD_FUNC(4 * internal_unRoll, maxThreads, maxThreads);
  }
}

template void launch_load_func(sycl::half *, const sycl::half *,
                              const sycl::half *, const sycl::half *, float,
                              int, int, sycl::queue);

template void launch_load_func(float *, const float *, const float *,
                              const float *, float, int, int, sycl::queue);


