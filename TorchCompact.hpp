#pragma once

#include <CL/sycl.hpp>
#include <vector>

namespace at {
class Tensor {
public:
  Tensor(
      void* p,
      sycl::queue& q,
      const std::vector<int64_t>& sizes,
      const std::vector<int64_t>& strides)
    : sizes_(sizes), strides_(strides), dims_(sizes.size()),
    storage_((char*)p, [&q](void *p){
        sycl::free(p,q);
        /*std::cout<<"free"<<std::endl;*/
      })
  {}
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  int dims_;
  std::shared_ptr<char> storage_;
};

class TensorIteratorBase {
public:
  TensorIteratorBase(Tensor& output, Tensor& input1, Tensor& input2)
    : tensors_({output, input1, input2}) {}

  int64_t numel() const {
    auto shape = tensors_[0].sizes_;
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
  }
  bool is_cpu_scalar(int) const {return false;}
  bool is_contiguous() const {return false;} // Force slow path
  void *data_ptr(int idx) const {return tensors_[idx].storage_.get();}
  bool can_use_32bit_indexing() const {return true;}
  int noutputs() const {return 1;}
  int ninputs() const {return 2;}
  int ntensors() const {return 3;}
  int element_size(int index) const {return 4;}
  int ndim() const {return output().dims_;}
  Tensor& output() {return tensors_[0];}
  const Tensor& output() const {return tensors_[0];}
  std::vector<int64_t>& strides(int idx) {
    return tensors_[idx].strides_;
  }
  const std::vector<int64_t>& strides(int idx) const {
    return tensors_[idx].strides_;
  }
  std::vector<int64_t>& shape() {
    return tensors_[0].sizes_;
  }
  const std::vector<int64_t>& shape() const {
    return tensors_[0].sizes_;
  }
  std::vector<Tensor> tensors_;
};

class TensorIterator : public TensorIteratorBase {
public:
  TensorIterator(Tensor& output, Tensor& input1, Tensor& input2)
    : TensorIteratorBase(output, input1, input2) {}
};

void RandomInit(int32_t *arr, size_t nelems) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> rg(1, 64);
  std::generate(arr, arr + nelems, [&] { return rg(gen); });
}

template <typename T>
void SeqInit(T *arr, size_t nelems) {
  for (size_t i = 0; i < nelems; ++ i) {
    arr[i] = i + 4;
  }
}

template <typename integer>
void ZeroInit(integer *arr, size_t nelems) {
  for (size_t i = 0; i < nelems; ++ i) {
    arr[i] = 0;
  }
}

template<typename T>
Tensor create_tensor(sycl::queue& q, char *host, const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides) {
  auto size = std::accumulate(
          sizes.begin(), sizes.end(), sizeof(T), std::multiplies<T>());
  auto* buffer = sycl::malloc_device(size, q);
  q.memcpy(buffer, host, size);
  return at::Tensor(buffer, q, sizes, strides);
}

template<typename T>
Tensor create_tensor(sycl::queue& q, const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides) {
  auto size = std::accumulate(
          sizes.begin(), sizes.end(), sizeof(T), std::multiplies<T>());
  void* buffer = sycl::malloc_device(size, q);
  void* host = sycl::malloc_host(size, q);
  SeqInit<T>((T*)host, size/sizeof(T));
  q.memcpy(buffer, host, size);
  return at::Tensor(buffer, q, sizes, strides);
}

}
