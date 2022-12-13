#include <CL/sycl.hpp>

#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdlib.h>
#include <time.h>

#include "dev_queue.hpp"
#include "TorchCompact.hpp"

#include "FunctionTraits.h"
#include "cxxopts.hpp"
#include "Loops.hpp"

#include "runtime.hpp"

static auto timeit(cl::sycl::event &event) {
  auto submit_time = event.get_profiling_info<cl::sycl::info::event_profiling::command_submit>();
  auto start_time = event.get_profiling_info<cl::sycl::info::event_profiling::command_start>();
  auto end_time = event.get_profiling_info<cl::sycl::info::event_profiling::command_end>();

  auto submission_time = (start_time - submit_time) / 1000.0f;
  auto execution_time = (end_time - start_time) / 1000.0f;

  //printf("submission lag = %f us, execution durationg = %f us\n",
  //    submission_time, execution_time);
  //
  return execution_time;
}

template <typename T>
struct AddFunctor {
  AddFunctor(T alpha) : alpha_(alpha) {}
  T alpha_;
  inline T operator()(T a, T b) const {
    return a + b * alpha_;
  }
};

struct test_drive : public at::TensorIterator {
  using at::TensorIterator::TensorIterator;

  at::Tensor operator () (const float alpha) {
    // const auto& the_type = this->common_dtype();
    // at::ScalarType _st = ::detail::scalar_type(the_type);
    // switch(_st) {
    // case at::ScalarType::Float: {
        using opmath_t = /*at::opmath_type<*/float/*>*/;
        porting::opmath_gpu_kernel_with_scalars<float>(*this, AddFunctor<opmath_t>(alpha));
    //   }
    //   break;
    // default:
    //   // c10::detail::AT_ERROR("test_add_meta", " not implemented for '", toString(_st), "'");
    //   throw std::exception();
    //   break;
    // }

    return std::move(this->tensors_[0]);
  }
};

template <typename T>
void test_add(int64_t x, int64_t y, uint32_t iters, sycl::queue q) {
  at::Tensor m_input1 = at::create_tensor<T>(q, {x, y}, {y, 1});
  at::Tensor m_input2 = at::create_tensor<T>(q, {x, y}, {y, 1});
  at::Tensor m_result = at::create_tensor<T>(q, {x, y}, {y, 1});

  test_drive drv(m_result, m_input1, m_input2);

  for (int i = 0; i < iters; ++i)
    drv(1.0);

  auto *o = (char*)sycl::malloc_host(x * y * sizeof(T), q);
  q.memcpy(o, m_result.storage_.get(), x * y * sizeof(T));
  q.wait();
  sycl::free(o, q);
}

int main(int argc, char **argv) {
  cxxopts::Options opts ("Reduce porting test", "Test Porting of PyTorch Reduce");
  opts.allow_unrecognised_options();
  opts.add_options()
    ("d,dev", "Device", cxxopts::value<uint32_t>()->default_value("1"))
    ("p,partition", "Partition of GPU", cxxopts::value<uint32_t>()->default_value("0"))
    ("x,x_axis", "Length of x", cxxopts::value<int64_t>()->default_value("1024"))
    ("c,iters", "Iterations", cxxopts::value<uint32_t>()->default_value("1"))
    ("y,y_axis", "Length of y", cxxopts::value<int64_t>()->default_value("1024"))
    ;

  auto parsed_opts = opts.parse(argc, argv);

  // static global variable in runtime.hpp
  g_dev_num = parsed_opts["dev"].as<uint32_t>();
  auto x = parsed_opts["x_axis"].as<int64_t>();
  auto y = parsed_opts["y_axis"].as<int64_t>();
  auto iters = parsed_opts["iters"].as<uint32_t>();

  test_add<float>(x, y, iters, currentQueue());

  return 0;
}
