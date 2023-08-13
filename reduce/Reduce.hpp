#pragma once

#include <assert.h>
#include "IntegerDivider.hpp"
#include "OffsetCalculator.hpp"
#include <ATen/core/Array.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>

#include <cstdint>
#include <functional>
#include <utility>
#include <type_traits>

namespace porting {

using at::detail::Array;
template <typename T>
using local_accessor = sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local>;

template <typename T>
class atomic_ref : public sycl::ext::oneapi::atomic_ref<T, sycl::ext::oneapi::memory_order::relaxed,
                       sycl::memory_scope::device, sycl::access::address_space::global_space> {
public:
  atomic_ref(T &r)
    : sycl::ext::oneapi::atomic_ref<T, sycl::ext::oneapi::memory_order::relaxed,
        sycl::memory_scope::device, sycl::access::address_space::global_space>(r) {}
};
static inline int64_t div_up(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

// returns floor(log2(n))
static inline int last_pow2(int n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::max(1, n - (n >> 1));
}

// Warning: 64-bit integer loop inside device
static void reduce_fraction(size_t &numerator, size_t &denominator) {
  // get GCD of num and denom using Euclid's algorithm.
  // Can replace this with std::gcd if we ever support c++17.
  size_t a = denominator;
  size_t b = numerator;
  while (b != 0) {
    a %= b;
    // swap(a, b)
    size_t tmp = a;
    a = b;
    b = tmp;
  }

  // a is now the GCD
  numerator /= a;
  denominator /= a;
}

//template for changing MAX_NUM_ITEMS based on op dtype
template <typename T>
struct mni_wrapper {
  static constexpr int MAX_NUM_ITEMS = 512;
};

// template <>
// struct mni_wrapper <c10::complex<double>> {
//   static constexpr int MAX_NUM_ITEMS = 256;
// }

struct ReduceConfig {
  static constexpr int GROUP_X = 0;
  static constexpr int GROUP_Y = 1;
  static constexpr int GROUP = 2;

  static constexpr int input_vec_size = 4;

  ReduceConfig(int element_size_bytes, int num_outputs, int num_inputs)
    : element_size_bytes(element_size_bytes), num_inputs(num_inputs), num_outputs(num_outputs) {}

  int element_size_bytes;
  int num_inputs;
  int num_outputs;
  int step_input = 1;
  int step_output = 1;
  int group_per_output = 1;
  int input_mult[3] = {0, 0, 0};
  int output_mult[2] = {0, 0};

  // Change terms in accordance with SYCL
  // Common pattern: width is equal to sub-group size
  int group_width;
  int group_height;
  int num_items; /* workgroup size */

  bool vectorize_input = false;
  int output_vec_size = 1;

  template <typename T>
  void set_group_dimension(int64_t dim0, int64_t dim1) {
    const int max_num_items = mni_wrapper<T>::MAX_NUM_ITEMS / output_vec_size;
    int dim0_pow2 = dim0 < max_num_items ? static_cast<int>(last_pow2(dim0)) : max_num_items;
    int dim1_pow2 = dim1 < max_num_items ? static_cast<int>(last_pow2(dim1)) : max_num_items;
    group_width = std::min(dim0_pow2, int(32)); /* suggested sub-group */
    group_height = std::min(dim1_pow2, int(max_num_items / group_width));
    group_width = std::min(dim0_pow2, int(max_num_items / group_height));
    num_items = group_width * group_height;

    if (num_items < 32) group_width = 32;
  }

  int split_input(int parallelism) {
    int step = step_input;
    step_input *= parallelism;
    return step;
  }

  int split_output(int parallelism) {
    int step = step_output;
    step_output *= parallelism;
    return step;
  }

  // Becareful of the geometry order
  sycl::range<2> group_sz() const { return {(size_t)group_height, (size_t)group_width}; }
  sycl::range<2> global_sz() const {
    return {(size_t)(group_per_output * group_height),
        (size_t)(group_width * div_up(num_outputs / output_vec_size, step_output))};
  }
  sycl::range<2> n_groups() const {
    return {
        (size_t)(group_per_output), (size_t)(div_up(num_outputs / output_vec_size, step_output))};
  }

  bool should_group_x_reduce() const { return input_mult[GROUP_X] != 0; }

  bool should_group_y_reduce() const { return input_mult[GROUP_Y] != 0; }

  bool should_global_reduce() const { return input_mult[GROUP] != 0; }

  bool should_store(sycl::nd_item<2> pos, int output_idx) const {
    return output_idx < num_outputs && (!should_group_x_reduce() || pos.get_local_id(1) == 0)
        && (!should_group_y_reduce() || pos.get_local_id(0) == 0);
  }

  bool should_reduce_tail(sycl::nd_item<2> pos) const {
    return (!should_group_y_reduce() || pos.get_local_id(0) == 0)
        && (!should_global_reduce() || pos.get_group(0) == 0);
  }

  int input_idx(sycl::nd_item<2> pos) const {
    int lane = pos.get_local_id(1);
    int thread = pos.get_local_id(0);
    int group_y = pos.get_group(0);
    return (
        lane * input_mult[GROUP_X] + thread * input_mult[GROUP_Y] + group_y * input_mult[GROUP]);
  }

  template <int output_vec_size>
  int output_idx(sycl::nd_item<2> pos) const {
    int lane = pos.get_local_id(1);
    int thread = pos.get_local_id(0);
    int group_x = pos.get_group(1);
    return (lane * output_mult[GROUP_X] + thread * output_mult[GROUP_Y] + group_x * step_output)
        * output_vec_size;
  }

  int slm_offset(sycl::nd_item<2> pos, int offset) const {
    return pos.get_local_id(1) + (pos.get_local_id(0) + offset) * pos.get_local_range(1);
  }

  int staging_memory_offset(sycl::nd_item<2> pos, int wg_y) const {
    int offset = wg_y + pos.get_group(1) * pos.get_group_range(0);
    if (!should_group_x_reduce()) { offset = pos.get_local_id(1) + offset * pos.get_group(1); }
    return offset;
  }

  int slm_size() const {
    if (!should_group_y_reduce() && (!should_group_x_reduce() || group_width <= 32)) { return 0; }
    return element_size_bytes * num_items * output_vec_size;
  }

  int64_t global_memory_size() const {
    if (!should_global_reduce()) { return 0; }
    auto size = (int64_t)element_size_bytes * num_outputs * group_per_output;
    if (!should_group_x_reduce()) { size *= group_sz()[1] * output_vec_size; }
    return size;
  }

  int semaphore_size() const {
    if (!should_global_reduce()) { return 0; }
    return sizeof(int) * global_sz()[1];
  }

  int values_per_item() const { return div_up(num_inputs, step_input); }
};

std::ostream &operator<<(std::ostream &out, const ReduceConfig &config);

template <int nitems, int output_vec_size, typename R>
// No launch bounds
class reduce_kernel {
public:
  reduce_kernel(
      R reduction,
      local_accessor<char> shared,
      local_accessor<bool> finished)
    : reduction(reduction), shared(shared), finished(finished) {}

  void operator ()(sycl::nd_item<2> pos) const {
    reduction.template run<output_vec_size>(pos, shared, finished);
  }
private:
  R reduction;
  local_accessor<char> shared;
  local_accessor<bool> finished;
};

template <typename index_t>
static OffsetCalculator<2, index_t> make_output_calculator(const at::TensorIterator &iter) {
  int num_reduce_dims = iter.num_reduce_dims();
  int num_output_dims = iter.ndim() - num_reduce_dims;
  int input_index = iter.ntensors() - 1;
  int output_index = 0;
  std::array<const int64_t *, 2> strides = {
      iter.strides(output_index).data() + num_reduce_dims,
      iter.strides(input_index).data() + num_reduce_dims,
  };
  auto shape = iter.shape().data() + num_reduce_dims;
  return OffsetCalculator<2, index_t>(num_output_dims, shape, strides.data());
}

template <typename index_t>
static OffsetCalculator<1, index_t> make_input_calculator(const at::TensorIterator &iter) {
  int num_reduce_dims = iter.num_reduce_dims();
  int input_index = iter.ntensors() - 1;
  std::array<const int64_t *, 1> strides = {
      iter.strides(input_index).data(),
  };
  return OffsetCalculator<1, index_t>(num_reduce_dims, iter.shape().data(), strides.data());
}

template <typename out_scalar_t, typename func_t>
struct func_wrapper_t {
  using arg_t = typename binary_function_traits<func_t>::arg1_t;
  using scalar_t = typename binary_function_traits<func_t>::arg2_t;

  func_t combine;
  static inline out_scalar_t project(arg_t arg) { return (out_scalar_t)arg; }

  /* Sycl already have handy shuffle down template
  static inline arg_t sub_group_shfl_down(sycl::sub_group sg, arg_t arg, int offset) {
    return sg.shuffle_down(arg, offset);
  } */

  static inline arg_t translate_idx(arg_t acc, int64_t /*idx*/) { return acc; }

  func_wrapper_t(const func_t &op) : combine(op) {}

  arg_t reduce(arg_t acc, scalar_t val, int64_t idx) const { return combine(acc, val); }
};

template <typename scalar_t, typename func_t>
func_wrapper_t<scalar_t, func_t> func_wrapper(const func_t &op) {
  return func_wrapper_t<scalar_t, func_t> {op};
}

template <typename scalar_t, typename ops_t, typename index_t, typename out_scalar_t = scalar_t,
    int vt0 = 4>
struct ReduceOp {
  using traits = function_traits<decltype(&ops_t::reduce)>;
  using arg_t = typename std::decay<typename traits::template arg<0>::type>::type;

  using InputCalculator = OffsetCalculator<1, index_t>;
  using OutputCalculator = OffsetCalculator<2, index_t>;

  static constexpr bool can_accumulate_in_output = std::is_convertible<arg_t, out_scalar_t>::value
      && std::is_convertible<out_scalar_t, arg_t>::value;

  static constexpr float acc_buffer_multiplier = (float)sizeof(arg_t) / sizeof(out_scalar_t);

  static constexpr int input_vec_size = ReduceConfig::input_vec_size;

  ops_t ops;
  arg_t ident;
  ReduceConfig config;
  InputCalculator input_calc;
  OutputCalculator output_calc;
  const void *src;
  const char *dst[2]; //it accepts at most two destinations
  // acc_buf used for accumulation among sub Tensor Iterator when accumulation on
  // output is not permissible
  void *acc_buf;
  // workgroup buf used for accumulation between workgrouops during global reduction
  void *group_buf;
  int *semaphores;
  int64_t base_idx;
  bool accumulate;
  bool final_output;
  int noutputs;

  ReduceOp(ops_t ops, ReduceConfig config, InputCalculator input_calc, OutputCalculator output_calc,
      const void *src, char *dst0, std::optional<char *> dst1, void *acc_buf, void *group_buf,
      int *semaphores, arg_t ident, int noutputs, int64_t base_idx)
    : ops(ops)
    , ident(ident)
    , config(config)
    , input_calc(input_calc)
    , output_calc(output_calc)
    , src(src)
    , acc_buf(acc_buf)
    , group_buf(group_buf)
    , semaphores(semaphores)
    , base_idx(base_idx)
    , noutputs(noutputs) {
    dst[0] = dst0;
    if (dst1.has_value()) { dst[1] = dst1.value(); }
  }

  // Where shared local memory should be materialize
  template <int output_vec_size>
  void run(sycl::nd_item<2> pos,
      sycl::local_ptr<char> shared,
      sycl::local_ptr<bool> finished) const {
    index_t output_idx = config.output_idx<output_vec_size>(pos);
    index_t input_idx = config.input_idx(pos);
    auto base_offsets1 = output_calc.get(output_idx)[1];

    using arg_vec_t = at::detail::Array<arg_t, output_vec_size>;
    arg_vec_t value;

    if (output_idx < config.num_outputs && input_idx < config.num_inputs) {
      const scalar_t *input_slice = (const scalar_t *)((const char *)src + base_offsets1);
      value = item_reduce<output_vec_size>(input_slice);
    }

    if (config.should_group_y_reduce()) {
      value = group_y_reduce<output_vec_size>(value, shared);
    }
    if (config.should_group_x_reduce()) {
      value = group_x_reduce<output_vec_size>(value, shared);
    }

    using out_ptr_vec_t = at::detail::Array<out_scalar_t *, output_vec_size>;
    using offset_vec_t = at::detail::Array<index_t, output_vec_size>;
    offset_vec_t base_offsets;
    out_ptr_vec_t out;

#   pragma unroll
    for (int i = 0; i < output_vec_size; i++) {
      base_offsets[i] = output_calc.get(output_idx + i)[0];
      out[i] = (out_scalar_t *)((char *)dst[0] + base_offsets[i]);
    }

    arg_vec_t *acc = nullptr;
    if (acc_buf != nullptr) {
      size_t numerator = sizeof(arg_t);
      size_t denominator = sizeof(out_scalar_t);
      reduce_fraction(numerator, denominator);
      acc = (arg_vec_t *)((char *)acc_buf + (base_offsets[0] * numerator / denominator));
    }

    if (config.should_global_reduce()) {
      value = global_reduce<output_vec_size>(pos, value, acc, shared, finished);
    } else if (config.should_store(output_idx)) {
      if (accumulate) {
#       pragma unroll
        for (int i = 0; i < output_vec_size; i++) {
          value[i] = ops.translate_idx(value[i], base_idx);
        }
      }

      if (acc == nullptr) {
        if (accumulate) {
          value = accumulate_in_output<output_vec_size, can_accumulate_in_output>(out, value);
        }
        if (final_output) {
          set_results_to_output<output_vec_size>(value, base_offsets);
        } else {
#         pragma unroll
          for (int i = 0; i < output_vec_size; i++) {
            *(out[i]) = get_accumulated_output<can_accumulate_in_output>(out[i], value[i]);
          }
        }
      } else {
        if (accumulate) {
#         pragma unroll
          for (int i = 0; i < output_vec_size; i++) {
            value[i] = ops.combine((*acc)[i], value[i]);
          }
        }
        if (final_output) {
          set_results_to_output<output_vec_size>(value, base_offsets);
        } else {
          *acc = value;
        }
      }
    }
  }

  template <int output_vec_size>
  at::detail::Array<arg_t, output_vec_size> item_reduce(
      sycl::nd_item<2> pos, sycl::local_ptr<char> slm, const scalar_t *data) const {
    if (config.vectorize_input) {
      assert(output_vec_size == 1);
      // reduce at the header of input_slice where memory is not aligned,
      // so that group_reduce will have an aligned memory to work on.
      return {input_vectorized_item_reduce_impl(data)};
    } else {
      index_t element_stride = input_calc.strides_[0][0] / sizeof(scalar_t);
      bool is_contiguous = (input_calc.dims == 1 && element_stride == 1);
      if (is_contiguous) {
        return item_reduce_impl<output_vec_size>(data, [](index_t idx) { return idx; });
      } else if (input_calc.dims == 1) {
        return item_reduce_impl<output_vec_size>(
            data, [&](index_t idx) { return idx * element_stride; });
      } else {
        return item_reduce_impl<output_vec_size>(
            data, [&](index_t idx) { return input_calc.get(idx)[0] / sizeof(scalar_t); });
      }
    }
  }

  arg_t input_vectorized_item_reduce_impl(sycl::nd_item<2> pos, const scalar_t *data) const {
    index_t end = config.num_inputs;
    // Handle the head of input slice where data is not aligned
    arg_t value = ident;
    // sycl provided the composition so we don't need to rewrite it.
    constexpr int align_bytes = alignof(sycl::vec<scalar_t, input_vec_size>);
    constexpr int align_elements = align_bytes / sizeof(scalar_t);
    int shift = ((uint64_t)data) % align_bytes / sizeof(scalar_t);
    if (shift > 0) {
      auto idx = pos.get_local_id(1);
      if (idx >= shift && idx < align_elements && config.should_reduce_tail(pos)) {
        value = ops.reduce(value, data[pos.get_local_id(1) - shift], pos.get_local_id(1) - shift);
      }
      // align data to vector start
      data += align_elements - shift;
      if (end > align_elements - shift)
        end -= align_elements - shift; /* warning: end flip */
      else
        end = 0;
      shift = align_elements - shift;
    }

    // Do the vectorized reduction
    using load_t = sycl::vec<scalar_t, input_vec_size>;

    index_t idx = config.input_idx(pos);
    const index_t stride = config.step_input;

    // multiple registers
    arg_t value_list[input_vec_size];
    value_list[0] = value;

#   pragma unroll
    for (int i = 1; i < input_vec_size; ++i) {
      value_list[i] = ident;
    }

    load_t values;

    while (idx * input_vec_size + input_vec_size - 1 < end) {
      values = reinterpret_cast<const load_t *>(data)[idx];
#     pragma unroll
      for (index_t i = 0; i < input_vec_size; ++i) {
        value_list[i] = ops.reduce(value_list[i], values[i], shift + idx * input_vec_size + i);
      }
      idx += stride;
    }

    // tail
    index_t tail_start = end - end % input_vec_size;
    if (config.should_reduce_tail(pos)) {
      int idx = tail_start + pos.get_local_id(1);
      if (idx < end) { value_list[0] = ops.reduce(value_list[0], data[idx], idx + shift); }
    }

    // registers accumulation
#   pragma unroll
    for (int i = 1; i < input_vec_size; ++i) {
      value_list[0] = ops.combine(value_list[0], value_list[i]);
    }
    return value_list[0];
  }

  template <int output_vec_size, typename offset_calc_t>
  at::detail::Array<arg_t, output_vec_size> item_reduce_impl(
      sycl::nd_item<2> pos, const scalar_t *data_, offset_calc_t calc) const {
    index_t idx = config.input_idx(pos);
    const index_t end = config.num_inputs;
    const index_t stride = config.step_input;

    using arg_vec_t = at::detail::Array<arg_t, output_vec_size>;
    using load_t = sycl::vec<scalar_t, output_vec_size>;
    const load_t *data = reinterpret_cast<const load_t *>(data_);

    arg_vec_t value_list[vt0];

#   pragma unroll
    for (int i = 0; i < vt0; i++) {
#     pragma unroll
      for (int j = 0; j < output_vec_size; ++j) {
        value_list[i][j] = ident;
      }
    }

    load_t values[vt0];

    while (idx + (vt0 - 1) * stride < end) {
#     pragma unroll
      for (index_t i = 0; i < vt0; ++i) {
        values[i] = data[calc(idx + i * stride) / output_vec_size];
      }
#     pragma unroll
      for (index_t i = 0; i < vt0; ++i) {
#       pragma unroll
        for (index_t j = 0; j < output_vec_size; ++j) {
          value_list[i][j] = ops.reduce(value_list[i][j], values[i][j], idx + i * stride);
        }
      }
      idx += stride * vt0;
    }

    // tail
    int idx_ = idx;
#   pragma unroll
    for (index_t i = 0; i < vt0; ++i) {
      if (idx >= end) { break; }
      values[i] = data[calc(idx) / output_vec_size];
      idx += stride;
    }
    idx = idx_;
#   pragma unroll
    for (index_t i = 0; i < vt0; ++i) {
      if (idx >= end) { break; }
#     pragma unroll
      for (index_t j = 0; j < output_vec_size; ++j) {
        value_list[i][j] = ops.reduce(value_list[i][j], values[i][j], idx);
      }
      idx += stride;
    }

    // combine accumulators
#   pragma unroll
    for (int i = 1; i < vt0; ++i) {
#     pragma unroll
      for (index_t j = 0; j < output_vec_size; ++j) {
        value_list[0][j] = ops.combine(value_list[0][j], value_list[i][j]);
      }
    }
    return value_list[0];
  }

  template <int output_vec_size>
  at::detail::Array<arg_t, output_vec_size> group_x_reduce(sycl::nd_item<2> pos,
      at::detail::Array<arg_t, output_vec_size> value, sycl::local_ptr<char> shared_memory) const {
    using args_vec_t = at::detail::Array<arg_t, output_vec_size>;
    auto l_x = pos.get_local_id(1), l_y = pos.get_local_id(0);
    auto gp_x = pos.get_local_range(1);

    int dim_x = gp_x;
    auto shared = reinterpret_cast<sycl::local_ptr<args_vec_t>>(shared_memory);
    auto sg = pos.get_sub_group();
    uint32_t sbgrpSize = sg.get_local_range()[0];
    if (dim_x > sbgrpSize) {
      int address_base = l_x + l_y * gp_x;
      shared[address_base] = value;
      for (int offset = dim_x / 2; offset >= sbgrpSize; offset >>= 1) {
        pos.barrier(sycl::access::fence_space::local_space);
        if (l_x < offset && l_x + offset < gp_x /* redundant??? */) {
          args_vec_t other = shared[address_base + offset];
#         pragma unroll
          for (int i = 0; i < output_vec_size; ++i) {
            value[i] = ops.combine(value[i], other[i]);
          }
          shared[address_base] = value;
        }
      }
    }

    pos.barrier(sycl::access::fence_space::local_space);

    // sub-group reduction
    for (int offset = 1; offset < sbgrpSize; offset <<= 1) {
#     pragma unroll
      for (int i = 0; i < output_vec_size; ++i) {
        arg_t other = sg.shuffle_down(value[i], offset);
        value[i] = ops.combine(value[i], other);
      }
    }
    return value;
  }

  template <int output_vec_size>
  at::detail::Array<arg_t, output_vec_size> group_y_reduce(sycl::nd_item<2> pos,
      at::detail::Array<arg_t, output_vec_size> value, sycl::local_ptr<char> shared_memory) const {
    using args_vec_t = at::detail::Array<arg_t, output_vec_size>;
    auto shared = reinterpret_cast<sycl::local_ptr<args_vec_t>>(shared_memory);
    shared[config.slm_offset(pos, 0)] = value;

    auto l_y = pos.get_local_id(0);
    auto dim_y = pos.get_local_range(0);
    for (int offset = dim_y / 2; offset > 0; offset >>= 1) {
      pos.barrier(sycl::access::fence_space::local_space);
      if (l_y < offset && l_y + offset < dim_y /* redundant ??? */) {
        args_vec_t other = shared[config.slm_offset(pos, offset)];
#       pragma unroll
        for (int i = 0; i < output_vec_size; ++i) {
          value[i] = ops.combine(value[i], other[i]);
        }
        shared[config.slm_offset(pos, 0)] = value;
      }
    }
    return value;
  }

  // In/out from slm pointers
  void mark_group_finished(
      sycl::nd_item<2> pos,
      sycl::local_ptr<bool> is_last_group_done_shared)
  const {
    // Force forward progress
    sycl::group_barrier(pos.get_group(), sycl::memory_scope::work_group);

    if (pos.get_local_linear_id() == 0) {
      auto flag = atomic_ref(semaphores[pos.get_group(1)]);
      int prev_groups_finished = flag++;
      is_last_group_done_shared[0] = (prev_groups_finished == (pos.get_global_range(0) - 1));
    }
    // Force forward progress even in loop
    sycl::group_barrier(pos.get_group(), sycl::memory_scope::work_group);
  }

  template <int output_vec_size, bool can_acc>
  at::detail::Array<arg_t, output_vec_size> accumulate_in_output(
      at::detail::Array<out_scalar_t*, output_vec_size> out,
      at::detail::Array<arg_t, output_vec_size> value,
      typename std::enable_if<can_acc>::type * = nullptr
  ) const {
    at::detail::Array<arg_t, output_vec_size> ret;
#   pragma unroll
    for (int i = 0; i < output_vec_size; ++i) {
      ret[i] = ops.combine(*(out[i]), value[i]);
    }
    return ret;
  }

  template <bool can_acc>
  out_scalar_t get_accumulated_output(
      out_scalar_t *out, arg_t value, typename std::enable_if<can_acc>::type * = nullptr) const {
    assert(!final_output);
    return (out_scalar_t)value;
  }

  // This function should never be called --
  // It's the version of `accumulate_in_output`
  // when accumulation in the output is not possible.
  template <int output_vec_size, bool can_acc>
  at::detail::Array<arg_t, output_vec_size> accumulate_in_output(
      at::detail::Array<out_scalar_t *, output_vec_size>, at::detail::Array<arg_t, output_vec_size>,
      typename std::enable_if<!can_acc>::type * = nullptr) const {
    assert(false);
    return arg_t {};
  }

  // This function should never be called --
  // it's the version of `get_accumulated_output`
  // when accumulation in the output is not possible.
  template <bool can_acc>
  out_scalar_t get_accumulated_output(
      out_scalar_t *out, arg_t value, typename std::enable_if<!can_acc>::type * = nullptr) const {
    assert(false);
    return *out;
  }

  template <class T>
  void set_results(const T x, const index_t base_offset) const {
    assert(noutputs == 1);
    auto res = (out_scalar_t *)((char *)dst[0] + base_offset);
    *res = x;
  }

  //Currently implemented for max of two outputs
  template <class T1, class T2>
  void set_results(const std::pair<T1, T2> x, const index_t base_offset) const {
    if (noutputs >= 1) {
      auto res0 = (T1 *)((char *)dst[0] + base_offset);
      *res0 = x.first;
    }
    if (noutputs >= 2) {
      // base offset is computed assuming element size being sizeof(T1), so we need to make a
      // correction to obtain the correct base offset
      auto res1 = (T2 *)((char *)dst[1] + base_offset / sizeof(T1) * sizeof(T2));
      *res1 = x.second;
    }
  }

  template <int output_vec_size>
  void set_results_to_output(at::detail::Array<arg_t, output_vec_size> value,
      at::detail::Array<index_t, output_vec_size> base_offset) const {
    assert(final_output);
#   pragma unroll
    for (int i = 0; i < output_vec_size; ++i) {
      set_results(ops.project(value[i]), base_offset[i]);
    }
  }

  template <int output_vec_size>
  at::detail::Array<arg_t, output_vec_size> global_reduce(
      sycl::nd_item<2> pos,
      at::detail::Array<arg_t, output_vec_size> value,
      at::detail::Array<arg_t, output_vec_size> *acc,
      sycl::local_ptr<char> shared_memory,
      sycl::local_ptr<bool> is_last_group_done) const {
    using arg_vec_t = at::detail::Array<arg_t, output_vec_size>;
    using out_ptr_vec_t = at::detail::Array<out_scalar_t *, output_vec_size>;
    using offset_vec_t = at::detail::Array<index_t, output_vec_size>;

    arg_vec_t *reduce_buffer = (arg_vec_t *)group_buf;
    index_t output_idx = config.output_idx<output_vec_size>();
    offset_vec_t base_offsets;
    out_ptr_vec_t out;

#   pragma unroll
    for (int i = 0; i < output_vec_size; ++i) {
      base_offsets[i] = output_calc.get(output_idx + i)[0];
      out[i] = (out_scalar_t *)((char *)dst[0] + base_offsets[i]);
    }

    bool should_store = config.should_store(output_idx);
    if (should_store) {
      index_t offset = config.staging_memory_offset(pos, pos.get_group(0));
      reduce_buffer[offset] = value;
    }

    pos.barrier(sycl::access::fence_space::global_and_local);
    mark_group_finished(pos, is_last_group_done);

    if (is_last_group_done[0]) {
      value = ident;
      if (config.should_group_x_reduce()) {
        index_t input_offset = pos.get_local_id(1) + pos.get_local_id(0) * pos.get_group_range(1);
        index_t step = pos.get_group_range(0) * pos.get_group_range(1);
        for (; input_offset < config.group_per_output; input_offset += step) {
          index_t idx = config.staging_memory_offset(pos, input_offset);
          arg_vec_t next = reduce_buffer[idx];
#         pragma unroll
          for (int i = 0; i < output_vec_size; ++i) {
            value[i] = ops.combine(value[i], next[i]);
          }
        }
      } else {
        index_t input_offset = pos.get_local_id(0);
        index_t step = pos.get_group_range(0);
        for (; input_offset < config.group_per_output; input_offset += step) {
          index_t idx = config.staging_memory_offset(pos, input_offset);
          arg_vec_t next = reduce_buffer[idx];
#         pragma unroll
          for (int i = 0; i < output_vec_size; ++i) {
            value[i] = ops.combine(value[i], next[i]);
          }
        }
      }
      value = group_y_reduce(value, shared_memory);
      if (config.should_group_x_reduce()) {
        value = group_x_reduce<output_vec_size>(value, shared_memory);
      }
      if (should_store) {
        if (accumulate) {
#         pragma unroll
          for (int i = 0; i < output_vec_size; ++i) {
            value[i] = ops.translate_idx(value[i], base_idx);
          }
        }

        if (acc == nullptr) {
          if (final_output) {
            set_results_to_output<output_vec_size>(value, base_offsets);
          } else {
#           pragma unroll
            for (int i = 0; i < output_vec_size; ++i) {
              *(out[i]) = get_accumulated_output<can_accumulate_in_output>(out[i], value[i]);
            }
          }
        } else {
          if (accumulate) {
#           pragma unroll
            for (int i = 0; i < output_vec_size; ++i) {
              value[i] = ops.combine((*acc)[i], value[i]);
            }
          }
          if (final_output) {
            set_results_to_output<output_vec_size>(value, base_offsets);
          } else {
            *acc = value;
          }
        }
      }
    }
    return value;
  }
};

extern sycl::queue getDefaultQueue();

template<int max_threads, typename R>
static void launch_reduce_kernel(const ReduceConfig& config, const R& reduction) {
  auto group = config.group_sz();
  auto global = config.global_sz();

  auto q = getDefaultQueue();
  sycl::range<1> slm_sz {static_cast<uint32_t>(config.slm_size())};

  switch(config.output_vec_size) {
  case 4:
    q.submit([&](sycl::handler &cgh) {
      local_accessor<char> shared (slm_sz, cgh);
      local_accessor<bool> finished ({1}, cgh);
      cgh.parallel_for(
          sycl::nd_range<2>(global, group),
          reduce_kernel<max_threads / 4, 4, R>(reduction, shared, finished)
      );
    });
    break;
  case 2:
    q.submit([&](sycl::handler &cgh) {
      local_accessor<char> shared (slm_sz, cgh);
      local_accessor<bool> finished ({1}, cgh);
      cgh.parallel_for(
          sycl::nd_range<2>(global, group),
          reduce_kernel<max_threads / 2, 2, R>(reduction, shared, finished)
      );
    });
    break;
  default:
    q.submit([&](sycl::handler &cgh) {
      local_accessor<char> shared (slm_sz, cgh);
      local_accessor<bool> finished ({1}, cgh);
      cgh.parallel_for(
          sycl::nd_range<2>(global, group),
          reduce_kernel<max_threads, 1, R>(reduction, shared, finished)
      );
    });
    break;
  }
}

class AccumulationBuffer {
public:
  AccumulationBuffer() = default;
  AccumulationBuffer(
      size_t acc_t_size, size_t out_t_size, char* out_ptr, int64_t size) {
    out_ptr_ = out_ptr;
    if (out_t_size >= acc_t_size) {
      // reusing output buffer for accumulation.
      acc_ptr_ = out_ptr;
      numerator_ = 1;
      denominator_ = 1;
    } else {
      auto allocator
        = sycl::usm_allocator<char, sycl::usm::alloc::shared>(getDefaultQueue());
      buffer_ = at::DataPtr(allocator.allocate(size), at::DeviceType::CPU);
      acc_ptr_ = (char*)buffer_.get();
      numerator_ = acc_t_size;
      denominator_ = out_t_size;
      reduce_fraction(numerator_, denominator_);
    }
  }

  char* get_acc_slice(char* out_ptr) {
    return acc_ptr_ == nullptr ? nullptr
      : acc_ptr_ + ((out_ptr - out_ptr_) * numerator_ / denominator_);
  }
private:
  char* acc_ptr_ = nullptr;
  char* out_ptr_ = nullptr;
  size_t numerator_;
  size_t denominator_;
  at::DataPtr buffer_;
};

template <typename scalar_t>
int get_output_vec_size(at::TensorIterator &iter) {
  int vec_size = 4;
  auto update_vec_size = [&vec_size](uint64_t n) {
    while(n % vec_size != 0) {
      vec_size /= 2;
    }
  };

  uint64_t base_address = reinterpret_cast<uint64_t>(
      iter.data_ptr(iter.noutputs())) / sizeof(scalar_t);
  update_vec_size(base_address);

  const int output_index = iter.num_reduce_dims();
  update_vec_size(iter.shape()[output_index]);

  int j = 0;
  for (auto i : iter.strides(iter.noutputs())) {
    if (j != output_index) {
      update_vec_size(i / sizeof(scalar_t));
    }
    j ++;
  }
  return vec_size;
}

template <
  typename scalar_t,
  typename out_scalar_t,
  int vt0=4, typename ops_t,
  typename ident_t = double>
inline void gpu_reduce_kernel(
    at::TensorIterator& iter, const ops_t& ops, ident_t ident=0,
    AccumulationBuffer* acc_buf_ptr=nullptr, int64_t base_idx=0) {
  AT_ASSERT(iter.numel() > 0 && iter.ntensors() - iter.noutputs() == 1
      && iter.noutputs() >= 1);

  using traits = function_traits<decltype(&ops_t::reduce)>;
  using arg_t = typename traits::template arg<0>::type;
  static constexpr bool can_accumulate_in_output =
    std::is_convertible<arg_t, out_scalar_t>::value;

  bool can_use_32bit_indexing = iter.can_use_32bit_indexing();
  std::unique_ptr<AccumulationBuffer> owned_buf_ptr;

  // The acc_buf_ptr is a shared pointer. It is create at the first entrance and
  // resued by all recursive function calls.
}

} // namespace porting
