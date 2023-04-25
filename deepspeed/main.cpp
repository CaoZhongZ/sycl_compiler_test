#include <CL/sycl.hpp>

#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../runtime.hpp"
#include "conversion_utils.h"
#include "softmax.hpp"
#include "layer_norm.hpp"

static constexpr int group_size = 16;
#define MAX_DIMS 32

template <typename T> void fill_array(T* array, T c, size_t n_elem) {
  for (size_t i = 0; i < n_elem; ++ i) {
    array[i] = c;
  }
}

template <typename T>
void test_softmax(int batch, int heads, int num_seq, int soft_seq) {
  auto q = currentQueue();

  auto elem = batch * heads * num_seq * soft_seq;
  auto size = elem * sizeof(T);

  T* vals = (T *)sycl::malloc_device(size, q);
  T* mask = (T *)sycl::malloc_device(size, q);
  T* alibi = (T *)sycl::malloc_device(size, q);
  T* host_in = (T *)sycl::malloc_host(size, q);
  T* host_o = (T *)sycl::malloc_host(size, q);

  fill_array<T>(host_in, T(1.0), elem);

  q.memcpy(vals, host_in, size);
  q.memset(mask, T(0.0), size);
  q.memset(alibi, T(0.0), size);

  launch_attn_softmax_v2(vals, mask, alibi, 1.0, false,
      false, false, 1, batch, heads, num_seq, soft_seq, 0, 0, 1, q);

  q.memcpy(host_o, vals, size);
  q.wait();

  sycl::free(vals, q);
  sycl::free(mask, q);
  sycl::free(alibi, q);
  sycl::free(host_in, q);
  sycl::free(host_o, q);
}

template <typename T>
void test_layernorm(int rows, int elems_per_row) {
  auto q = currentQueue();

  auto elem = rows * elems_per_row;
  auto size = elem * sizeof(T);
  auto row_sz = elems_per_row * sizeof(T);

  auto* vals = (T *)sycl::malloc_device(size, q);
  auto* output = (T *)sycl::malloc_device(size, q);

  auto* gamma = (T *)sycl::malloc_device(row_sz, q);
  auto* beta = (T *)sycl::malloc_device(row_sz, q);

  auto* host_in = (T *)sycl::malloc_host(size, q);
  auto* host_o = (T *)sycl::malloc_host(size, q);

  fill_array<T>(host_in, T(1.0), elem);

  q.memcpy(vals, host_in, size);
  q.memset(gamma, T(1.0), row_sz);
  q.memset(beta, T(0.0), row_sz);

  launch_fused_ln(output, vals, gamma, beta, 0.00001, rows, elems_per_row, q);

  q.memcpy(host_o, output, size);
  q.wait();

  sycl::free(vals, q);
  sycl::free(output, q);
  sycl::free(gamma, q);
  sycl::free(beta, q);
  sycl::free(host_in, q);
  sycl::free(host_o, q);
}


template <typename T, typename F>
T test_conversion(F val) {
  return conversion::to<T>(val);
}

int main(int argc, char ** argv) {
  if (argc < 4) {
    printf("Usage: small <batch> <heads> <seq> <seq>\n");
    return -1;
  }
  int batch_size = std::stoi(argv[1]);
  int heads = std::stoi(argv[2]);
  int num_seq = std::stoi(argv[3]);
  int soft_seq = std::stoi(argv[4]);

  sycl::half standard(1.0);
  float standard_float = standard;
  float standard_converted = conversion::to<float>(standard);

  float f_standard = 1.0;
  sycl::half converted_f_standard = conversion::to<sycl::half>(f_standard);

  test_layernorm<sycl::half>(128, 1024);
  // test_softmax<sycl::half>(batch_size, heads, num_seq, soft_seq);
  // test_softmax<bf16>(batch_size, heads, num_seq, soft_seq);

  return 0;
}
