#include <iostream>

// original CUDA:
/*
template <typename T, typename U, typename... ArgTypes>
__global__ void multi_tensor_apply_kernel(
  int chunk_size, volatile int* noop_flag, T tl, U callable, ArgTypes... args) {
  callable(chunk_size, noop_flag, tl, args...);
}
*/

//
// Expected transfer, without lexical scope issue and prepare for size limitation
//

template <typename T, typename U, typename... ArgTypes>
class multi_tensor_apply_kernel {
public:
  //
  // XXX: Parameters list should be IDENTICAL to original __global__ function
  //
  multi_tensor_apply_kernel(
      int chunk_size, volatile int* noop_flag, T tl, U callable, ArgTypes... args)
    : chunk_size(chunk_size), noop_flag(noop_flag), tl(tl), callable(callable), args(args...)
  {}

  // XXX: This static function should be IDENTICAL to original __global__ function
  static void inline __global__function(
      int chunk_size, volatile int* noop_flag, T tl, U callable, ArgTypes... args
  ) {
    callable(chunk_size, noop_flag, tl, args...);
  }

  // If global function template contains *parameter pack*,
  // (we only deal with parameter pack at the end of template parameter list)
  //
  // Note that we can't use std::apply because function pointer constraint
  //
  // XXX: Parameters list should be IDENTICAL to original __global__ function except
  // the last parameter pack for further processing.
  //
  template <typename Tuple, std::size_t... I>
  static void inline __tuple_expand_driver(
      int chunk_size, volatile int* noop_flag,
      T tl, U callable, /* ->*/Tuple args, std::index_sequence<I...> /*<--*/
  ) {
    __global__function(chunk_size, noop_flag, tl, callable, std::get<I>(args)...);
  }

  //
  // Because __global__ function can't really use any reference types, we can sure that args
  // are all good behaviors
  //
  // we use free function for coordinates, so, no sycl::nd_item object needed.
  //
  void operator() (sycl::nd_item<3>) const {
    __tuple_expand_driver(chunk_size, noop_flag, tl, callable, args,
        std::make_index_sequence<sizeof ...(ArgTypes)>());
  }

private:
  int chunk_size;
  volatile int* noop_flag;
  T tl;
  U callable;
  std::tuple<ArgTypes...> args;
};

template <typename... ArgTypes>
struct dummy {
  void inline operator()(ArgTypes ... args) {
    std::cout<<"Get "<<sizeof...(args)<<" args, but lazy ass does nothing"<<std::endl;
  }
};

int main() {
  volatile int flag;

  auto return_flag_address = [&] {
    return &flag;
  };

  multi_tensor_apply_kernel offload(
      1024, return_flag_address(), 10,
      dummy<int, volatile int*, int, int, int, int>(), 1, 1, 1
  );

  offload();

// Launching site will become:
  // XXX: CUDA:
  multi_tensor_apply_kernel<<<loc_block_info, block_size, 0, stream>>>(
      chunk_size, noop_flag.DATA_PTR<int>(), tl, callable, args...);

  // XXX: SYCL (with 2048 parameters' size choice):
  if constexpr (sizeof multi_tensor_apply_kernel(
       chunk_size, noop_flag.DATA_PTR<int>(), tl, callable, args...) < 2048 ) {
    queue.parallel_for(
      sycl::nd_range<3>(...loc_block_info, block_size, 0...),
      multi_tensor_apply_kernel(
         chunk_size, noop_flag.DATA_PTR<int>(), tl, callable, args...));
  } else {
    auto capture = multi_tensor_apply_kernel(
         chunk_size, noop_flag.DATA_PTR<int>(), tl, callable, args...);
    sycl::buffer params(const_cast<const decltype(capture) *>(&capture, sycl::range<1>(1)));

    stream.submit([&] (sycl::handler &cgh) {
      auto device_params = params.template get_access<
        sycl::access_mode::read, sycl::target::constant_buffer>(cgh);
      cgh.parallel_for(
        sycl::nd_range<3>(...loc_block_info, block_size, 0...),
        [=] (sycl::nd_item<3> item) {
          device_params[0](item);
        });
    });
  }
}
