#include <iostream>
#include <CL/sycl.hpp>

#include <algorithm>
#include <numeric>
#include <random>
#include <stdlib.h>
#include <time.h>

#include "common/common.hpp"
#include "common/cxxopts.hpp"
#include "sycl/device.hpp"
#include "sycl/utils.hpp"

#include "IntegerDivider.hpp"
#include "Reduce.hpp"
#include "Test_stub.hpp"

void RandomInit(int32_t *arr, size_t nelems) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> rg(1, 64);
  std::generate(arr, arr + nelems, [&] { return rg(gen); });
}

void SeqInit(int32_t *arr, size_t nelems) {
  for (size_t i = 0; i < nelems; ++ i) {
    arr[i] = i + 4;
  }
}

void ZeroInit(int32_t *arr, size_t nelems) {
  for (size_t i = 0; i < nelems; ++ i) {
    arr[i] = 0;
  }
}

// We always use aligned array to simulate unalign case
template <typename T>
void test_item_reduce_vec(sycl::queue q, T* out, T* in, size_t nelems) {
  test_stub::launch_test_kernel1( q, out, in, 1, nelems,
    porting::func_wrapper<T>([] (T a, T b) {return a + b;}), (T)0
  );
}

template <typename T>
void test_item_reduce_general(sycl::queue q, T* out, T* in, size_t n_out, size_t nelems) {
  test_stub::launch_test_kernel2( q, out, in, n_out, nelems,
    porting::func_wrapper<T>([] (T a, T b) {return a + b;}), (T)0
  );
}

// Simulate algorithm using single worker
template <typename T, int vec_sz = 4>
void verify_item_reduce_vec(T *host_o, T* host_in, size_t ninputs, int off = 2, int group_sz = 512) {
  for (int i = 0; i < group_sz; ++i) {
    auto compare = 0;
    auto end = off == 0 ? ninputs : ninputs > vec_sz ? ninputs - vec_sz : 0;

    auto host_head = host_in;

    if ( off > 0 && i >= off && i < vec_sz) {
      compare += host_head[i];
    }
    auto host_align = off == 0 ? host_in : host_in + vec_sz;
    auto idx = i;

    while (idx * vec_sz + vec_sz -1 < end) {
      auto idx_4 = idx * vec_sz;
      compare += host_align[idx_4] + host_align[idx_4+1] + host_align[idx_4+2] + host_align[idx_4+3];
      idx += group_sz;
    }

    // Always 4 elments at the tail
    auto tail_start = end - end % vec_sz;
    idx = tail_start + i;

    if ( idx < end )
      compare += host_align[idx];

    if (host_o[i] != compare) {
      std::cout<<"Errors occur @"<<i<<": "<<host_o[i]<<" vs. "<<compare<<" expected"<<std::endl;
    }
  }
}

template <typename T, int vt0 = 4>
void verify_item_reduce_general(
    T *host_o, T* host_in, size_t noutputs, size_t ninputs,
    int output_vec_size = 1, int group_sz = 512) {
  auto end = ninputs;
  auto stride = group_sz;

  for (int i = 0; i < group_sz; ++i) {
    int compare = 0;
    auto idx = i;

    while (idx + (vt0 - 1) * stride < end) {
      for (int j = 0; j < vt0; ++ j) {
        compare += host_in[(idx + j * stride) / output_vec_size];
      }
      idx += stride * vt0;
    }

    for (int j = 0; j < vt0; ++ j) {
      if (idx >= end) {
        break;
      }
      compare += host_in[idx / output_vec_size];
      idx += stride;
    }

    if (host_o[i] != compare) {
      std::cout<<"Errors occur @"<<i<<": "<<host_o[i]<<" vs. "<<compare<<" expected"<<std::endl;
    }
  }
}


void test_reduction_on_fastest_striding_dimension(
    sycl::queue q, int noutputs, int ninputs, int iters) {
  auto nelems = noutputs * ninputs;

  auto *p = reinterpret_cast<int32_t *>(
    sycl::aligned_alloc_host(4096, sizeof(int32_t) * nelems, q));
  auto *in = reinterpret_cast<int32_t *>(
    sycl::aligned_alloc_device(4096, sizeof(int32_t) * nelems, q));
  auto *o = reinterpret_cast<int32_t *>(
    sycl::aligned_alloc_device(4096, sizeof(int32_t) * nelems, q));
  auto *h = reinterpret_cast<int32_t *>(
    sycl::aligned_alloc_host(4096, sizeof(int32_t) * nelems, q));

  SeqInit(p, nelems);
  ZeroInit(h, nelems);
  
  double usec;
  for (int off = 0; off < 4; ++ off) {
    q.memcpy(in, p, sizeof(int32_t) * nelems);
    q.memcpy(o, h, sizeof(int32_t) * nelems);

    auto start = std::chrono::steady_clock::now();
    // test_item_reduce_vec(q, o, in, ninputs, off);
    test_item_reduce_general(q, o, in + off, noutputs, ninputs - off);

    auto duration = std::chrono::steady_clock::now() - start;
    usec =
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / iters;

    q.memcpy(h, o, sizeof(uint32_t) * nelems);
    q.wait();
    print_indicator(usec / 1000.0f, nelems * sizeof(float));

    verify_item_reduce_general(h, p + off, noutputs, ninputs - off);
  }

  if ( noutputs == 1) {
    for (int off = 0; off < 4; ++ off) {
      q.memcpy(in, p, sizeof(int32_t) * nelems);
      q.memcpy(o, h, sizeof(int32_t) * nelems);

      auto start = std::chrono::steady_clock::now();
      test_item_reduce_vec(q, o, in + off, ninputs - off);

      auto duration = std::chrono::steady_clock::now() - start;
      usec =
        std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / iters;

      q.memcpy(h, o, sizeof(uint32_t) * nelems);
      q.wait();
      print_indicator(usec / 1000.0f, nelems * sizeof(float));

      verify_item_reduce_vec(h, p, ninputs, off);
    }
  }


  sycl::free(p, q);
  sycl::free(in, q);
  sycl::free(o, q);
  sycl::free(h, q);
}

int main(int argc, char **argv) {
  codex::device_manager dm(codex::engine_kind_t::gpu);
  cxxopts::Options opts ("div_const", "Test constant division");
  opts.allow_unrecognised_options();
  opts.add_options()
    ("d,dev", "Device", cxxopts::value<uint32_t>()->default_value("1"))
    ("n,nelems", "Number of inputs", cxxopts::value<size_t>()->default_value("1024"))
    ("c,iters", "Iterations", cxxopts::value<uint32_t>()->default_value("10"))
    ("o,outs", "Number of outputs", cxxopts::value<size_t>()->default_value("1"))
    ;

  auto parsed_opts = opts.parse(argc, argv);
  auto dev_num = parsed_opts["dev"].as<uint32_t>();
  auto nelems = parsed_opts["nelems"].as<size_t>();
  auto nouts = parsed_opts["outs"].as<size_t>();
  auto iters = parsed_opts["iters"].as<uint32_t>();

  cl::sycl::queue q = dm.make_sycl_queue(dev_num, defaultPropList);
  codex::device_info(q.get_device());

  test_reduction_on_fastest_striding_dimension(q, nouts, nelems, iters);

  return 0;
}
