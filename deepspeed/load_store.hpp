#pragma once
#include "compatible.h"

template <typename T>
void  launch_load_func(T *output, const T *vals, const T *gamma, const T *beta,
                     float epsilon, int rows, int elems_per_row,
                     sycl::queue stream);
