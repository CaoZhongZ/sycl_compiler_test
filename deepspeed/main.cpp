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

static constexpr int group_size = 16;
#define MAX_DIMS 32

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

  for (size_t i = 0; i < elem; ++ i) {
    host_in[i] = T(1.0);
  }

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

  test_softmax<sycl::half>(batch_size, heads, num_seq, soft_seq);

  return 0;
}
