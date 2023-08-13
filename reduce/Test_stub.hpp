#pragma once

#include <CL/sycl.hpp>
#include "Reduce.hpp"

#include <ATen/ATen.h>
#include <ATen/core/Array.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

using namespace porting;
using at::detail::Array;

namespace test_stub {
 
template <typename T>
using __slm__
= sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local>;

template<typename out_scalar_t, typename scalar_t, typename R>
class test_input_vectorized_item_reduce_impl {
public:
  test_input_vectorized_item_reduce_impl(out_scalar_t* out, const scalar_t* in, R reduction)
    :out_(out), in_(in), reduction_(reduction) {}

  void operator() (sycl::nd_item<2> pos) const {
    out_[pos.get_local_id(1)] = reduction_.input_vectorized_item_reduce_impl(pos, in_);
  }
private:
  const scalar_t *in_;
  out_scalar_t *out_;
  R reduction_;
};

template<
  typename out_scalar_t,
  typename scalar_t,
  typename R,
  typename C,
  typename offset_calc_t>
class test_item_reduce_impl {
public:
  test_item_reduce_impl(
      out_scalar_t* out,
      const scalar_t* in,
      R reduction, C config, offset_calc_t calc, int output_vec_size)
    :out_(out), in_(in), reduction_(reduction), config_(config), calc_(calc),
    output_vec_size_(output_vec_size) {
  }

  void operator() (sycl::nd_item<2> pos) const {
    switch (output_vec_size_) {
    case 4: {
        using vec_o_t = at::detail::Array<scalar_t, 4>;
        auto* out = reinterpret_cast<vec_o_t (*)[pos.get_local_range(1)]>(out_);
        out[pos.get_local_id(0)][pos.get_local_id(1)]
          = reduction_.template item_reduce_impl<4>(pos, in_, calc_);
      }
      break;
    case 2: {
        using vec_o_t = at::detail::Array<scalar_t, 2>;
        auto* out = reinterpret_cast<vec_o_t (*)[pos.get_local_range(1)]>(out_);
        out[pos.get_local_id(0)][pos.get_local_id(1)]
          = reduction_.template item_reduce_impl<2>(pos, in_, calc_);
      }
      break;
    case 1: {
        using vec_o_t = at::detail::Array<scalar_t, 1>;
        auto* out = reinterpret_cast<vec_o_t (*)[pos.get_local_range(1)]>(out_);
        out[pos.get_local_id(0)][pos.get_local_id(1)]
          = reduction_.template item_reduce_impl<1>(pos, in_, calc_);
      }
      break;
    default:
      assert(false && "Can't be here");
      break;
    }
  }
private:
  const scalar_t *in_;
  out_scalar_t *out_;
  R reduction_;
  C config_;
  offset_calc_t calc_;
  int output_vec_size_;
};

template <typename scalar_t, typename out_scalar_t>
ReduceConfig genConfig(int num_outputs, int num_inputs) {
  ReduceConfig config(sizeof(scalar_t), num_outputs, num_inputs);
  config.output_vec_size = 1;
  auto dim0 = num_inputs;
  auto dim1 = num_outputs;

  config.set_group_dimension<scalar_t>(dim0, dim1);
  return config;
}

// Pointer is explained inside R
// Sample reduce [*] -> []
// Item behavior only
template <
  typename scalar_t,
  typename out_scalar_t,
  typename ops_t,
  typename ident_t>
static void launch_test_kernel1(
    sycl::queue q,
    out_scalar_t* out,
    const scalar_t* in,
    int num_outputs,
    int num_inputs, ops_t ops, ident_t ident) {

  auto config = genConfig<scalar_t, out_scalar_t>(num_outputs, num_inputs);

  auto group_sz = config.group_sz();
  auto global_sz = config.global_sz();

  // 8 EUs * 8 threads * 16 SIMD = 1024 max-lane per-SS
  const auto max_group_per_ss = 1024 / (config.group_width * config.group_height);
  const auto num_ss = 64; /* clinfo is not accurate */
  const auto target_global_sz = num_ss * max_group_per_ss;

  // split inputs amount a group
  config.input_mult[0] = config.split_input(config.group_width);
  config.input_mult[1] = config.split_input(config.group_height);

  // config.input_mult[2] = config.split_input(1);
  // aplit output amount a group
  config.output_mult[0] = config.split_output(1);

  // Mocking runtime
  auto m_self = at::empty({num_inputs},
      at::TensorOptions().dtype<scalar_t>().memory_format(c10::MemoryFormat::Contiguous));
  auto m_result = at::empty({num_outputs},
      at::TensorOptions().dtype<out_scalar_t>().memory_format(c10::MemoryFormat::Contiguous));

  at::TensorIterator iter = at::meta::make_reduction_from_out_ty(
      m_self, m_result, {}, false, m_result.scalar_type());

  auto output_calc = make_output_calculator<uint32_t>(iter);
  auto input_calc = make_input_calculator<uint32_t>(iter);

  int64_t base_idx = 0;

  auto reduce = ReduceOp<scalar_t, ops_t, uint32_t, out_scalar_t, 4>(
    ops,
    config,
    input_calc,
    output_calc,
    in,
    (char *)out,
    nullptr,
    nullptr,
    nullptr,
    nullptr /*(int*)semaphores.get()*/,
    ident,
    num_outputs,
    base_idx);

  std::cout<<"Launch kernel in: "<<group_sz[0]<<", "<<group_sz[1]<<"."<<std::endl<<std::flush;
  auto e = q.submit([&](sycl::handler &h) {
    h.parallel_for(
      sycl::nd_range<2>(group_sz, group_sz),

      test_input_vectorized_item_reduce_impl<out_scalar_t, scalar_t, decltype(reduce)>(out, in, reduce)
    );
  });
}

// Sample reduce [128, 1024] -> [128, 1]
// Item behavior only
template <
  typename scalar_t,
  typename out_scalar_t,
  typename ops_t,
  typename ident_t>
static void launch_test_kernel2(
    sycl::queue q,
    out_scalar_t* out,
    const scalar_t* in,
    int num_outputs,
    int num_inputs, ops_t ops, ident_t ident) {

  auto config = genConfig<scalar_t, out_scalar_t>(num_outputs, num_inputs);

  auto group_sz = config.group_sz();
  auto global_sz = config.global_sz();

  // 8 EUs * 8 threads * 16 SIMD = 1024 max-lane per-SS
  const auto max_group_per_ss = 1024 / (config.group_width * config.group_height);
  const auto num_ss = 64; /* clinfo is not accurate */
  const auto target_global_sz = num_ss * max_group_per_ss;

  config.input_mult[0] = config.split_input(config.group_width);
  config.input_mult[1] = config.split_input(config.group_height);
  // config.input_mult[2] = config.split_input(1);

  // Mocking runtime
  auto m_self = num_outputs == 1 ? at::empty({num_inputs},
      at::TensorOptions().dtype<scalar_t>().memory_format(c10::MemoryFormat::Contiguous)) :
    at::empty({num_outputs, num_inputs},
      at::TensorOptions().dtype<scalar_t>().memory_format(c10::MemoryFormat::Contiguous));

  auto m_result = at::empty({num_outputs},
      at::TensorOptions().dtype<out_scalar_t>().memory_format(c10::MemoryFormat::Contiguous));

  auto ndim = num_outputs == 1 ?  std::vector<int64_t>() : std::vector<int64_t>({1});
  at::TensorIterator iter = at::meta::make_reduction_from_out_ty(
      m_self, m_result, ndim, false, m_result.scalar_type());

  auto output_calc = make_output_calculator<uint32_t>(iter);
  auto input_calc = make_input_calculator<uint32_t>(iter);

  int64_t base_idx = 0;

  auto reduce = ReduceOp<scalar_t, ops_t, uint32_t, out_scalar_t, 4>(
    ops,
    config,
    input_calc,
    output_calc,
    in,
    (char *)out,
    nullptr,
    nullptr,
    nullptr,
    nullptr /*(int*)semaphores.get()*/,
    ident,
    num_outputs,
    base_idx);

  uint32_t element_stride = input_calc.strides_[0][0] / sizeof(scalar_t);
  bool is_contiguous = (input_calc.dims == 1 && element_stride == 1);

  int output_vec_size = 1;
  // !reduction_on_fastest_striding_dimension
  // output_vec_size = get_output_vec_size(iter);

  std::cout<<"Launch kernel in: "<<group_sz[0]<<", "<<group_sz[1]<<"."<<std::endl<<std::flush;

  if (is_contiguous) {
    auto calc = [](uint32_t idx) { return idx; };

    auto test_config = test_item_reduce_impl<
        out_scalar_t, scalar_t, decltype(reduce), decltype(config), decltype(calc)>(
            out, in, reduce, config, calc, output_vec_size);

    q.submit([&](sycl::handler &h) {
      h.parallel_for(
        sycl::nd_range<2>(group_sz, group_sz), test_config);
    });
  } else if (input_calc.dims == 1) {
    auto idx_calc = [=](uint32_t idx) {
      return idx * element_stride;
    };

    auto test_config = test_item_reduce_impl<
        out_scalar_t, scalar_t, decltype(reduce), decltype(config), decltype(idx_calc)>(
            out, in, reduce, config, idx_calc, output_vec_size);

    q.submit([&](sycl::handler &h) {
      h.parallel_for(
        sycl::nd_range<2>(group_sz, group_sz),
        test_config);
    });
  } else {
    auto calc = [=](uint32_t idx) {
          return input_calc.get(idx)[0] / sizeof(scalar_t); };

    auto test_config = test_item_reduce_impl<
        out_scalar_t, scalar_t, decltype(reduce), decltype(config), decltype(calc)>(
            out, in, reduce, config, calc, output_vec_size);

    q.submit([&](sycl::handler &h) {
      h.parallel_for(
        sycl::nd_range<2>(group_sz, group_sz),
        test_config);
    });
  }
}

// Sample reduce [128, 1024] -> [1, 1024]
// Item behavior only
template <
  typename scalar_t,
  typename out_scalar_t,
  typename ops_t,
  typename ident_t>
static void launch_test_kernel3(
    sycl::queue q,
    out_scalar_t* out,
    const scalar_t* in,
    int num_outputs,
    int num_inputs, ops_t ops, ident_t ident) {

  auto config = genConfig<scalar_t, out_scalar_t>(num_outputs, num_inputs);

  auto group_sz = config.group_sz();
  auto global_sz = config.global_sz();

  // 8 EUs * 8 threads * 16 SIMD = 1024 max-lane per-SS
  const auto max_group_per_ss = 1024 / (config.group_width * config.group_height);
  const auto num_ss = 64; /* clinfo is not accurate */
  const auto target_global_sz = num_ss * max_group_per_ss;

  config.input_mult[0] = config.split_input(config.group_width);
  config.input_mult[1] = config.split_input(config.group_height);
  config.input_mult[2] = config.split_input(1);

  // Mocking runtime
  auto m_self = num_outputs == 1 ? at::empty({num_inputs},
      at::TensorOptions().dtype<scalar_t>().memory_format(c10::MemoryFormat::Contiguous)) :
    at::empty({num_outputs, num_inputs},
      at::TensorOptions().dtype<scalar_t>().memory_format(c10::MemoryFormat::Contiguous));

  auto m_result = at::empty({num_outputs},
      at::TensorOptions().dtype<out_scalar_t>().memory_format(c10::MemoryFormat::Contiguous));

  at::TensorIterator iter = at::meta::make_reduction_from_out_ty(
      m_self, m_result, {}, false, m_result.scalar_type());

  auto output_calc = make_output_calculator<uint32_t>(iter);
  auto input_calc = make_input_calculator<uint32_t>(iter);

  int64_t base_idx = 0;

  auto reduce = ReduceOp<scalar_t, ops_t, uint32_t, out_scalar_t, 4>(
    ops,
    config,
    input_calc,
    output_calc,
    in,
    (char *)out,
    nullptr,
    nullptr,
    nullptr,
    nullptr /*(int*)semaphores.get()*/,
    ident,
    num_outputs,
    base_idx);

  uint32_t element_stride = input_calc.strides_[0][0] / sizeof(scalar_t);
  bool is_contiguous = (input_calc.dims == 1 && element_stride == 1);

  int output_vec_size = 1;
  // !reduction_on_fastest_striding_dimension
  // output_vec_size = get_output_vec_size(iter);

  std::cout<<"Launch kernel in: "<<group_sz[0]<<", "<<group_sz[1]<<"."<<std::endl<<std::flush;

  if (is_contiguous) {
    auto calc = [](uint32_t idx) { return idx; };

    auto test_config = test_item_reduce_impl<
        out_scalar_t, scalar_t, decltype(reduce), decltype(config), decltype(calc)>(
            out, in, reduce, config, calc, output_vec_size);

    q.submit([&](sycl::handler &h) {
      h.parallel_for(
        sycl::nd_range<2>(group_sz, group_sz), test_config);
    });
  } else if (input_calc.dims == 1) {
    auto idx_calc = [=](uint32_t idx) {
      return idx * element_stride;
    };

    auto test_config = test_item_reduce_impl<
        out_scalar_t, scalar_t, decltype(reduce), decltype(config), decltype(idx_calc)>(
            out, in, reduce, config, idx_calc, output_vec_size);

    q.submit([&](sycl::handler &h) {
      h.parallel_for(
        sycl::nd_range<2>(group_sz, group_sz),
        test_config);
    });
  } else {
    auto calc = [=](uint32_t idx) {
          return input_calc.get(idx)[0] / sizeof(scalar_t); };

    auto test_config = test_item_reduce_impl<
        out_scalar_t, scalar_t, decltype(reduce), decltype(config), decltype(calc)>(
            out, in, reduce, config, calc, output_vec_size);

    q.submit([&](sycl::handler &h) {
      h.parallel_for(
        sycl::nd_range<2>(group_sz, group_sz),
        test_config);
    });
  }
}

}
