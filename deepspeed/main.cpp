#include <CL/sycl.hpp>

#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../runtime.hpp"

static constexpr int group_size = 16;
#define MAX_DIMS 32

void test_float(int batch, int heads, int num_seq, int soft_seq) {
  auto q = currentQueue();

  auto size = batch * heads * num_seq * soft_seq;

  float* vals = sycl::malloc_device(size * sizeof(float), q);
  float* mask = sycl::malloc_device(size * sizeof(float), q);
  float* alibi = sycl::malloc_device(size * sizeof(float), q);
  float* host_m = sycl::malloc_host(size * sizeof(float), q);

  q.memset(val, 1.0);
  q.memset(mask, 0.0);
  q.memset(alibi, 0.0);

  launch_attn_softmax_v2(vals, mask, alibi, 1.0, true,
      false, false, 1, batch, heads, num_seq, soft_seq, 0, 0, 1, q);

  q.memcpy(host_m, vals, size * sizeof(float));
  q.wait();
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

  test_float(batch_size, heads, num_seq, soft_seq);

  return 0;
}
