#include <iostream>
#include <sycl/sycl.hpp>

#include "cxxopts.hpp"
#include "sycl_misc.hpp"

template <typename T, int Unroll>
struct group_copy {
  static inline void run(sycl::nd_item<1> pos, T* dst, const T* src, size_t elems) {
    auto grp = pos.get_group();
    auto n_grps = grp.get_group_linear_range();
    auto grp_sz = grp.get_local_linear_range();

    auto grp_id = grp.get_group_id(0);
    auto loc_id = grp.get_local_id(0);
    auto slice = elems * Unroll / n_grps;
    auto base_off = grp_id * slice;

    for (auto off = base_off + loc_id; off < base_off + slice; off += grp_sz * Unroll) {
#     pragma unroll
      for (int i = 0; i < Unroll; ++ i) {
        if (off + i * grp_sz < elems)
          dst[off + i * grp_sz] = src[off + i * grp_sz];
      }
    }
  }
};

template <typename T, int Unroll>
struct asm_copy {
  static inline void run(sycl::nd_item<1> pos, T* dst, const T* src, size_t elems) {
    auto grp = pos.get_group();
    auto n_grps = grp.get_group_linear_range();
    auto grp_sz = grp.get_local_linear_range();

    auto grp_id = grp.get_group_id(0);
    auto loc_id = grp.get_local_id(0);
    auto slice = elems * Unroll / n_grps;
    auto base_off = grp_id * slice;

    for (auto off = base_off + loc_id; off < base_off + slice; off += grp_sz * Unroll) {
#     pragma unroll
      for (int i = 0; i < Unroll; ++ i) {
#if defined(__SYCL_DEVICE_ONLY__)
        T tmp;

        if constexpr (sizeof(T) == 4) {
          asm volatile ("lsc_load.ugm (M1, 16) %0:d32 flat[%1]:a64\n" : "=rw"(tmp) : "rw"(src + off + i * grp_sz));
        } else if constexpr (sizeof(T) == 8) {
          asm volatile ("lsc_load.ugm (M1, 16) %0:d32x2 flat[%1]:a64\n" : "=rw"(tmp) : "rw"(src + off + i * grp_sz));
        } else if constexpr (sizeof(T) == 16) {
          asm volatile ("lsc_load.ugm (M1, 16) %0:d32x4 flat[%1]:a64\n" : "=rw"(tmp) : "rw"(src + off + i * grp_sz));
        }
#else
        if (off + i * grp_sz < elems)
          dst[off + i * grp_sz] = src[off + i * grp_sz];
#endif
      }
    }
  }
};

template <typename T, int Unroll>
struct group_copy_acquire {
  static inline void run(sycl::nd_item<1> pos, T* dst, const T* src, size_t elems) {
    auto grp = pos.get_group();
    auto n_grps = grp.get_group_linear_range();
    auto grp_sz = grp.get_local_linear_range();

    auto grp_id = grp.get_group_id(0);
    auto loc_id = grp.get_local_id(0);
    auto slice = elems * Unroll / n_grps;
    auto base_off = grp_id * slice;

    for (auto off = base_off + loc_id; off < base_off + slice; off += grp_sz * Unroll) {
#     pragma unroll
      for (int i = 0; i < Unroll; ++ i) {
        if (off + i * grp_sz < elems)
          dst[off + i * grp_sz] = src[off + i * grp_sz];
      }
      sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
    }
  }
};

template <typename T, int Unroll>
struct group_copy_release {
  static inline void run(sycl::nd_item<1> pos, T* dst, const T* src, size_t elems) {
    auto grp = pos.get_group();
    auto n_grps = grp.get_group_linear_range();
    auto grp_sz = grp.get_local_linear_range();

    auto grp_id = grp.get_group_id(0);
    auto loc_id = grp.get_local_id(0);
    auto slice = elems * Unroll / n_grps;
    auto base_off = grp_id * slice;

    for (auto off = base_off + loc_id; off < base_off + slice; off += grp_sz * Unroll) {
#     pragma unroll
      for (int i = 0; i < Unroll; ++ i) {
        if (off + i * grp_sz < elems)
          dst[off + i * grp_sz] = src[off + i * grp_sz];
      }
      sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
    }
  }
};

template <typename T, int lane_v, int Unroll, template <typename, int> class copy_policy>
struct copy_persist : copy_policy<sycl::vec<T, lane_v/sizeof(T)>, Unroll> {
  using v_T = sycl::vec<T, lane_v/sizeof(T)>;

  void operator() [[sycl::reqd_sub_group_size(16)]] (sycl::nd_item<1> pos) const {
    this->run(pos, dst, src, vu_nelems);
  }

  copy_persist(T* dst, const T* src, size_t elems) :
    src(reinterpret_cast<const v_T *>(src)),
    dst(reinterpret_cast<v_T *>(dst)),
    vu_nelems(elems/v_T::size()/Unroll) {}

  static sycl::event launch(sycl::queue queue, T* dst, const T* src, size_t nelems,
      size_t max_group =64, size_t local_size = 1024, uint32_t repeat = 1) {
    if (nelems < v_T::size() || nelems % v_T::size() != 0)
      throw std::logic_error("Vectorize can't be satisfied");

    auto v_nelems = nelems / v_T::size();

    if (v_nelems % Unroll != 0)
      throw std::logic_error("Unroll can't be satisfied");

    auto vu_nelems = v_nelems / Unroll;

    size_t required_groups = (vu_nelems + local_size -1)/local_size;
    auto group_num = std::min(required_groups, max_group);
    size_t global_size = group_num * local_size;

    printf("Launch copy_persist (%zu, %zu) with unroll %d\n", group_num, local_size, Unroll);

    auto e = queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>({global_size}, {local_size}),
              copy_persist(dst, src, nelems));
    });

    for (int i = 1; i < repeat; ++ i) {
      e = queue.submit([&](sycl::handler &cgh) {
          cgh.parallel_for(sycl::nd_range<1>({global_size}, {local_size}),
                copy_persist(dst, src, nelems));
      });
    }
    return e;
  }

  const v_T* src;
  v_T* dst;
  size_t vu_nelems;
};

template <template <typename, int, int, template <typename, int> class> class copy,
         typename T, int lane_v, template <typename, int> class copy_policy,
         typename ... Args>
static sycl::event launch(sycl::queue queue, int unroll, Args&& ... args) {
#define CASE(Unroll)  \
  case Unroll: \
    { \
      return copy<T, lane_v, Unroll, copy_policy>::launch(queue, \
          std::forward<Args>(args)...); \
    } \
    break;

  switch (unroll) {
  CASE(1);
  CASE(2);
  CASE(4);
  CASE(8);
  default:
    throw std::length_error("Unsupported unroll.");
    break;
  }
#undef CASE
}

template <template <typename, int, int, template <typename, int> class> class copy,
         typename T, template <typename, int> class copy_policy,
         typename ... Args>
static sycl::event launch(sycl::queue queue, int v_lane, int unroll, Args&& ... args) {
#define CASE(LaneV)  \
  case LaneV: \
    { \
      return launch<copy, T, LaneV, copy_policy>( \
          queue, unroll, std::forward<Args>(args)...); \
    } \
    break;

  switch (v_lane) {
    CASE(2);
    CASE(4);
    CASE(8);
    CASE(16);
  default:
    throw std::length_error("Unsupported lane width.");
    break;
  }
#undef CASE
}

size_t parse_nelems(const std::string& nelems_string) {
  size_t base = 1;
  size_t pos = nelems_string.rfind("K");
  if (pos != std::string::npos) {
    base = 1024ull;
  } else {
    pos = nelems_string.rfind("M");
    if (pos != std::string::npos)
      base = 1024 * 1024ull;
    else {
      pos = nelems_string.rfind("G");
      if (pos != std::string::npos)
        base = 1024 * 1024 * 1024ull;
    }
  }

  return stoull(nelems_string) * base;
}

template <typename T>
void fill_sequential(void *p, int rank, size_t size) {
  auto typed_sz = size / sizeof(T);
  auto *p_t = reinterpret_cast<T *>(p);

  for (size_t i = 0; i < typed_sz; ++ i) {
    p_t[i] = i + rank;
  }
}

template <typename T>
double bandwidth_from_event(sycl::event e, size_t nelems) {
  e.wait();
  auto start = e.template get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end = e.template get_profiling_info<sycl::info::event_profiling::command_end>();

  // since timestamp is in the unit of ns, then the bandwidth is GB/s in unit
  return (double)(nelems * sizeof(T) * 2) / (double)(end - start);
}

double time_from_event(sycl::event e) {
  e.wait();
  auto start = e.template get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end = e.template get_profiling_info<sycl::info::event_profiling::command_end>();

  return (double)(end -start);
}

int main(int argc, char *argv[]) {
  cxxopts::Options opts("Copy", "Copy baseline for performance");
  opts.allow_unrecognised_options();
  opts.add_options()
    ("n,nelems", "Number of elements", cxxopts::value<std::string>()->default_value("16MB"))
    ("u,unroll", "Unroll request", cxxopts::value<uint32_t>()->default_value("1"))
    ("g,groups", "Max Group Size", cxxopts::value<size_t>()->default_value("64"))
    ("l,local", "Local size", cxxopts::value<size_t>()->default_value("512"))
    ("s,sequential", "Sequential Unroll", cxxopts::value<bool>()->default_value("false"))
    ("f,fence", "Atomic Fence at the end of bulk", cxxopts::value<int>()->default_value("0"))
    ("t,tile", "On which tile to deploy the test", cxxopts::value<uint32_t>()->default_value("0"))
    ("v,lanev", "Vecterize amoung SIMD lane", cxxopts::value<int>()->default_value("16"))
    ;

  auto parsed_opts = opts.parse(argc, argv);
  auto nelems_string = parsed_opts["nelems"].as<std::string>();
  auto unroll =  parsed_opts["unroll"].as<uint32_t>();
  auto local = parsed_opts["local"].as<size_t>();
  auto max_groups = parsed_opts["groups"].as<size_t>();
  auto seq = parsed_opts["sequential"].as<bool>();
  auto tile = parsed_opts["tile"].as<uint32_t>();
  auto v_lane = parsed_opts["lanev"].as<int>();
  auto fence = parsed_opts["fence"].as<int>();

  auto nelems = parse_nelems(nelems_string);
  using test_type = sycl::half;
  size_t alloc_size = nelems * sizeof(test_type);

  auto queue = currentQueue(tile >> 1, tile & 1);

  auto* src = (test_type *)sycl::malloc_device(alloc_size, queue);
  auto* dst = (test_type *)sycl::malloc_device(alloc_size, queue);
  auto* b_host = sycl::malloc_host(alloc_size, queue);
  auto* b_check = sycl::malloc_host(alloc_size, queue);

  release_guard __guard([&]{
    sycl::free(src, queue);
    sycl::free(dst, queue);
    sycl::free(b_host, queue);
    sycl::free(b_check, queue);
  });

  fill_sequential<test_type>(b_host, 0, alloc_size);

  queue.memcpy(src, b_host, alloc_size);
  queue.wait();

  sycl::event e;
  (void)seq; (void)fence;

  /*
  if (seq) {
    switch (fence) {
    default:
      e = launch<copy_persist, test_type, group_copy>(
          queue, v_lane, unroll, dst, src, nelems, max_groups, local);
      break;
    case 1:
      e = launch<copy_persist, test_type, group_copy_release>(
          queue, v_lane, unroll, dst, src, nelems, max_groups, local);
      break;
    case 2:
      e = launch<copy_persist, test_type, group_copy_acquire>(
          queue, v_lane, unroll, dst, src, nelems, max_groups, local);
      break;
    }
  } else {
    switch (fence) {
    default:
    */
      e = launch<copy_persist, test_type, asm_copy>(
          queue, v_lane, unroll, dst, src, nelems, max_groups, local);
      /*
      break;
    case 1:
      e = launch<copy_persist, test_type, group_copy_release>(
          queue, v_lane, unroll, dst, src, nelems, max_groups, local);
      break;
    case 2:
      e = launch<copy_persist, test_type, group_copy_acquire>(
          queue, v_lane, unroll, dst, src, nelems, max_groups, local);
      break;
    }
  } */

  // auto e = queue.memcpy(dst, src, alloc_size);
  auto bandwidth = bandwidth_from_event<test_type>(e, nelems);
  auto time = time_from_event(e);
  printf("Copy %zu half in %fns, bandwidth: %fGB/s\n", alloc_size, time, bandwidth);

  queue.memcpy(b_check, dst, alloc_size);
  queue.wait();

  if (memcmp(b_check, b_host, alloc_size) == 0)
    printf("Verified\n");
}
