#pragma once

#if defined(__SYCL_DEVICE_ONLY__)
#include <CL/sycl.hpp>
#endif

#include <assert.h>
#include <cstdint>

// Porting same functionality from PyTorch/Eigen GPU code.
namespace porting {

template <typename T>
struct DivMod {
  T div, mod;
  DivMod(T div, T mod) : div(div), mod(mod) { }
};

template <typename T>
struct IntDivider {
  IntDivider() = default;
  IntDivider(T d) : divisor(d) { }

  inline T div(T n) const { return n / divisor; }
  inline T mod(T n) const { return n % divisor; }
  inline DivMod<T> divmod(T n) const {
    return DivMod<T>(n / divisor, n % divisor);
  }

  T divisor;
};

template <>
struct IntDivider<uint32_t> {
  static_assert(sizeof(unsigned int) == 4, "Assumes 32-bit unsigned int.");

  IntDivider() = default;

  IntDivider(uint32_t d) : divisor(d) {
    assert(divisor >= 1 && divisor <= INT32_MAX);

    for (shift = 0; shift < 32; ++ shift) if ((1U << shift) >= divisor) break;
    uint64_t one = 1;
    uint64_t magic = ((one <<32) * ((one << shift) - divisor)) / divisor + 1;
    m1 = magic;
    assert(m1 > 0 && m1 == magic);  // m1 must fit in 32 bits.
  }

  inline uint32_t div(uint32_t n) const {
#if defined(__SYCL_DEVICE_ONLY__)
    uint32_t t = cl::sycl::mul_hi(n, static_cast<uint32_t>(m1));
#else
    uint64_t t = ((uint64_t) n * m1) >> 32;
#endif
    return (t + n) >> shift;
  }

  inline uint32_t mod(uint32_t n) const {
    return n - div(n) * divisor;
  }

  inline DivMod<uint32_t> divmod(uint32_t n) const {
    uint32_t q = div(n);
    return DivMod<uint32_t>(q, n - q * divisor);
  }

  uint32_t divisor;
  uint32_t m1;
  uint32_t shift;
};

}
