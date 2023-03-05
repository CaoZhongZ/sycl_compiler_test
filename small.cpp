#include <CL/sycl.hpp>

#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "runtime.hpp"

static constexpr int group_size = 16;
#define MAX_DIMS 32

struct captures {
  void operator () (sycl::nd_item<1> pos) const {
    auto group_id = pos.get_group().get_group_id();
    auto local = pos.get_local_id();

    auto linear_id = group_id * group_size + local;
    int sum = 0;

#   pragma unroll (MAX_DIMS)
    for (int dim = 0; dim < MAX_DIMS; ++ dim) {
      if (dim == dims) {
        break;
      }
      sum += table[dim];
    }
    usm[linear_id] = sum + (int)local;
  }

  captures(int dims, int *usm) : dims(dims), usm(usm) {}

  int table[MAX_DIMS];
  int dims;
  int *usm;
};

int main(int argc, char ** argv) {
  if (argc < 4) {
    printf("Usage: small <size> <dims> <value>... (dims <value>)"
        "size must be multiply of 16, dim must be smaller than %d\n", MAX_DIMS);
    return -1;
  }
  int size = std::stoi(argv[1]);
  int dims = std::stoi(argv[2]);

  if (argc - 3 < dims) {
    printf("Usage: small <size> <dims> <value>... (dims <value>)"
        "number of <value> should be equal to dims\n");
    return -1;
  }
  auto q = currentQueue();

  auto* usm = sycl::malloc_device(size * sizeof(int), q);

  auto captured = captures(dims, (int *)usm);

  for (int i = 0; i < dims; ++ i) {
    captured.table[i] = std::stoi(argv[3+i]);
  }

  auto grid = size / group_size;
  sycl::buffer params(const_cast<const decltype(captured) *>(&captured), sycl::range<1>(1));

  auto e = q.submit([&] (sycl::handler &cgh) {
#if defined(SYCL_BUFFER_PARAMS_WRAPPER)
    auto device_captured = params.template get_access<sycl::access_mode::read, sycl::target::constant>(cgh);

    cgh.parallel_for(sycl::nd_range<1>({grid * group_size, group_size}),
      [=] (sycl::nd_item<1> pos) {
        device_captured[0](pos);
      });
#else
    cgh.parallel_for(sycl::nd_range<1>({grid * group_size, group_size}), captured);
#endif
  });

  e.wait();
  auto* host_m = sycl::malloc_host(size * sizeof(int), q);
  q.memcpy(host_m, usm, size * sizeof(int)).wait();

  return 0;
}
