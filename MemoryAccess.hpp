#pragma once

#include <cstdint>
#include <type_traits>
#include "OffsetCalculator.hpp"

#include <tuple>

// References:
// https://devblogs.nvidia.com/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

namespace porting { namespace memory {

namespace detail {

// What does the `static_unroll` do?
//
// We want to do something like:
//
//    using args_t = typename traits::ArgsTuple;
//    args_t args;
//    #pragma unroll
//    for (int i = 0; i < traits::arity; i++) {
//      std::get<i>(args) = ....
//    }
//
// but unfortunately the above code does not work because
// the template argument has to be a compile time constant
// so `static_unroll` is created to simulate `#pragma unroll`
// using template metaprogramming.

template<template<int i> typename func, int end, int current=0>
struct static_unroll {
  template<typename... Args>
  static inline void with_args(Args&&... args) {
    func<current>::apply(std::forward<Args>(args)...);
    static_unroll<func, end, current+1>::with_args(args...);
  }
};

template<template<int i> typename func, int end>
struct static_unroll<func, end, end> {
  template<typename... Args>
  static inline void with_args(Args... args) {}
};

// helper structs to be used with static_unroll to load arguments
// one by one

template<int arg_index>
struct vectorized_load_helper {
  template <typename args_t, typename policy_t>
  static void apply(sycl::nd_item<1> pos, policy_t &self, args_t *args, int idx) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    // `data` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a +1 offset to get the input
    auto ptr = reinterpret_cast<arg_t *>(self.data[arg_index + 1]) + group_work_size() * idx;
    auto args_accessor = [&args] (int item_unroll_idx) -> arg_t & { return std::get<arg_index>(args[item_unroll_idx]); };
    self.load_single_arg(pos, args_accessor, ptr);
  }
};

template<int arg_index>
struct unroll_load_helper {
  template <typename args_t, typename policy_t, typename offset_t, typename loader_t>
  static void apply(policy_t &self, args_t *args, offset_t offset, loader_t loader, int j, int num_outputs) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    // `data` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a +1 offset to get the input
    std::get<arg_index>(args[j]) = loader.template load<arg_t>(self.data[arg_index + num_outputs], offset[arg_index], arg_index);
  }
};

template <int current>
struct multi_outputs_store_helper {
  template<int ntensors, int num_outputs, typename ...Args>
  static void apply(
      std::array<char*, ntensors> data,
      std::array<uint32_t, num_outputs> offsets,
      std::tuple<Args...> ret) {
    using T = typename std::tuple_element<current, std::tuple<Args...>>::type;
    T *to = reinterpret_cast<T *>(data[current]) + offsets[current];
    *to = std::get<current>(ret);
  }
};

}  // namespace detail

struct LoadWithoutCast {
  template<typename scalar_t>
  scalar_t load(char *base_ptr, uint32_t offset, int arg) {
    return *(reinterpret_cast<scalar_t *>(base_ptr) + offset);
  }
};

struct StoreWithoutCast {
  template<typename scalar_t>
  void store(scalar_t value, char *base_ptr, uint32_t offset) {
    *(reinterpret_cast<scalar_t *>(base_ptr) + offset) = value;
  }
};

#if 1
// aligned vector generates vectorized load/store on CUDA
template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
  scalar_t& operator[] (size_t index) {
    return val[index];
  }
  const scalar_t& operator[] (size_t index) const {
    return val[index];
  }
};
#else
template <typename scalar_t, int vec_size>
using aligned_vector = sycl::vec<scalar_t, vec_size>;
#endif

namespace policies {

// Assumption:
// all tensors are contiguous, that is: stride == sizeof(type) for all tensors
template<typename data_t, typename inp_calc_t, typename out_calc_t, typename loader_t, typename storer_t, int num_outputs = 1>
struct unroll {

  data_t& data;
  int& remaining;
  inp_calc_t& input_offset_calculator;
  out_calc_t& output_offset_calculator;
  loader_t& loader;
  storer_t& storer;

  unroll(data_t& data, int& remaining, inp_calc_t& ic, out_calc_t& oc, loader_t& l, storer_t& s):
    data(data), remaining(remaining), input_offset_calculator(ic), output_offset_calculator(oc), loader(l), storer(s) {}

  inline bool check_inbounds(int local_id, int item_work_elem) {
    return ((local_id  + item_work_elem*group_size()) < remaining);
  }

  template<typename args_t>
  inline void load(sycl::nd_item<1> pos, args_t *args, int idx) {
    constexpr int arity = std::tuple_size<args_t>::value;
    int item_idx = pos.get_local_id();
    #pragma unroll (item_work_size())
    for (int i = 0; i < item_work_size(); i++) {
      if (item_idx >= remaining) {
        return;
      }
      int linear_idx = item_idx + group_work_size() * idx;
      auto offset = input_offset_calculator.get(linear_idx);
      detail::static_unroll<detail::unroll_load_helper, arity>::with_args(*this, args, offset, loader, i, num_outputs);
      item_idx += group_size();
    }
  }

  template<typename scalar_t>
  inline void store(sycl::nd_item<1> pos, scalar_t *from, int idx) {
    int item_idx = pos.get_local_id();
    scalar_t *to = reinterpret_cast<scalar_t *>(data[0]) + group_work_size() * idx;
    #pragma unroll (item_work_size())
    for (int i = 0; i < item_work_size(); i++) {
      if (item_idx >= remaining) {
        return;
      }
      int linear_idx = item_idx + group_work_size() * idx;
      int offset = output_offset_calculator.get(linear_idx)[0];
      storer.store(from[i], data[0], offset);
      item_idx += group_size();
    }
  }
};

// Assumption:
// all tensors are contiguous, that is: stride == sizeof(type) for all tensors
// Note:
// Functions in vectorized policy does not do boundary check. It assumes the whole block
// has its job to do. So the reminders should be handled by the the caller manually.
template <int vec_size, typename data_t>  // vec_size: number of scalars, can be 1, 2, or 4.
struct vectorized {

  static_assert(item_work_size() % vec_size == 0, "The workload per thread must be a multiple of vec_size");
  static constexpr int loop_size = item_work_size() / vec_size;

  data_t data;

  vectorized(data_t data) : data(data) {}

  inline constexpr bool check_inbounds(int local_id, int thread_work_elem) {
    return true;
  }

  template<typename accessor_t, typename scalar_t>
  inline void load_single_arg(sycl::nd_item<1> pos, accessor_t to, scalar_t *from) {
    using vec_t = aligned_vector<scalar_t, vec_size>;
    vec_t *from_ = reinterpret_cast<vec_t *>(from);
    int item_idx = pos.get_local_id();
    #pragma unroll (loop_size)
    for (int i = 0; i < loop_size; i++) {
      int index = item_idx + i * group_size();
      vec_t v = from_[index];
      #pragma unroll (vec_size)
      for (int j = 0; j < vec_size; j++) {
        to(vec_size * i + j) = v[j];
      }
    }
  }

  template<typename args_t>
  inline void load(sycl::nd_item<1> pos, args_t *args, int idx) {
    constexpr int arity = std::tuple_size<args_t>::value;
    detail::static_unroll<detail::vectorized_load_helper, arity>::with_args(pos, *this, args, idx);
  }

  template<typename scalar_t>
  inline void store(sycl::nd_item<1> pos, scalar_t *from, int idx) {
    using vec_t = aligned_vector<scalar_t, vec_size>;
    scalar_t *to = reinterpret_cast<scalar_t *>(data[0]) + group_work_size() * idx;
    vec_t *to_ = reinterpret_cast<vec_t *>(to);
    int item_idx = pos.get_local_id();
    #pragma unroll (loop_size)
    for (int i = 0; i < loop_size; i++) {
      int index = item_idx + i * group_size();
      vec_t v;
      #pragma unroll (vec_size)
      for (int j = 0; j < vec_size; j++) {
        v[j] = from[vec_size * i + j];
      }
      to_[index] = v;
    }
  }
};

template <typename data_t, typename inp_calc_t, typename out_calc_t, int num_outputs>
struct multi_outputs_unroll {
  //multi_outputs_unroll struct members and check_inbounds and load methods are copypasted from unroll struct
  //we don't use inheritance because of compiler bug in cuda 10.2+
  data_t data;
  int remaining;
  inp_calc_t input_offset_calculator;
  out_calc_t output_offset_calculator;
  LoadWithoutCast loader;
  StoreWithoutCast storer;

  multi_outputs_unroll(data_t data, int remaining, inp_calc_t ic, out_calc_t oc):
  data(data), remaining(remaining), input_offset_calculator(ic), output_offset_calculator(oc) {}

  inline bool check_inbounds(int local_id, int thread_work_elem) {
    return ((local_id + thread_work_elem*group_size()) < remaining);
  }

  template<typename args_t>
  inline void load(sycl::nd_item<1> pos, args_t *args, int idx) {
    constexpr int arity = std::tuple_size<args_t>::value;
    int item_idx = pos.get_local_id();
    #pragma unroll (item_work_size())
    for (int i = 0; i < item_work_size(); i++) {
      if (item_idx >= remaining) {
        return;
      }
      int linear_idx = item_idx + group_work_size() * idx;
      auto offset = input_offset_calculator.get(linear_idx);
      detail::static_unroll<detail::unroll_load_helper, arity>::with_args(*this, args, offset, loader, i, num_outputs);
      item_idx += group_size();
    }
  }


  template <typename return_t>
  inline void store(sycl::nd_item<1> pos, return_t *from, int idx) {
    int item_idx = pos.get_local_id();
    #pragma unroll (item_work_size())
    for (int i = 0; i < item_work_size(); i++) {
      if (item_idx >= this->remaining) {
        return;
      }
      int linear_idx = item_idx + group_work_size() * idx;
      auto offsets = this->output_offset_calculator.get(linear_idx);
      memory::detail::static_unroll<detail::multi_outputs_store_helper, num_outputs>::with_args(this->data, offsets, from[i]);
      item_idx += group_size();
    }
  }
};

}  // namespace policies

// This is only used in host, but we will wrap this into some templates
// which is C10_HOST_DEVICE, so we have to make this C10_HOST_DEVICE
// in order to compile
template<typename scalar_t>
inline int can_vectorize_up_to(char *pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec2_alignment = std::alignment_of<aligned_vector<scalar_t, 2>>::value;
  constexpr int vec4_alignment = std::alignment_of<aligned_vector<scalar_t, 4>>::value;
  if (address % vec4_alignment == 0) {
    return 4;
  } else if (address % vec2_alignment == 0) {
    return 2;
  }
  return 1;
}

template<int i>
struct can_vectorize_up_to_helper {
  template <typename array_t, typename traits>
  static void apply(int &result, array_t pointers, traits _) {
    using arg_t = typename traits::template arg<i>::type;
    // `pointers` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a +1 offset to get the input
    result = std::min<int>(result, can_vectorize_up_to<arg_t>(pointers[i + 1]));
  }
};

template<typename func_t, typename array_t>
inline int can_vectorize_up_to(array_t pointers) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  constexpr int arity = traits::arity;
  int result = can_vectorize_up_to<return_t>(pointers[0]);
  // We need to get the type for each argument of `func_t`, this can only
  // be done at compile time.
  detail::static_unroll<can_vectorize_up_to_helper, arity>::with_args(result, pointers, traits());
  return result;
}

// jitted version of the above
// See Note [Jiterator], this relies on the assumptions enumerated there
template<typename result_type, typename common_type, int arity, typename array_t>
inline int jitted_can_vectorize_up_to(array_t pointers) {
  // Deals with output
  int result = can_vectorize_up_to<result_type>(pointers[0]);

  // Incorporates input(s)
  for (auto i = decltype(arity){1}; i < (arity + 1); ++i) {
    result = std::min<int>(result, can_vectorize_up_to<common_type>(pointers[i]));
  }

  return result;
}

}}
