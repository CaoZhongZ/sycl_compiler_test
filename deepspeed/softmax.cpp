/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#include <limits>

#include <cstdio>
#include <cstdlib>
#include <ctime>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
using namespace cl;
#else
#error "Unsupported compiler"
#endif

#include "compatible.h"
#include "conversion_utils.h"
#include "softmax.hpp"

#define MAX_REG_SIZE 8
#define minus_infinity -10000.0

template <typename T> class attn_softmax_v2 {

private:
  T *vals;
  T *mask;
  T *alibi;
  float layer_scale;
  bool triangular;
  bool recompute;
  bool local_attention;
  int window_size;
  int total_count;
  int heads;
  int sequence_length;
  int num_seq;
  int head_offset;
  int mask_stride;
  int mp_size;
  int iterations;
  int reduceWidth;

public:
  attn_softmax_v2(T *vals, T *mask, T *alibi, float layer_scale,
                  bool triangular, bool recompute, bool local_attention,
                  int window_size, int total_count, int heads,
                  int sequence_length, int num_seq, int head_offset,
                  int mask_stride, int mp_size, int iterations, int reduceWidth)
      : vals(vals), mask(mask), alibi(alibi), layer_scale(layer_scale),
        triangular(triangular), recompute(recompute),
        local_attention(local_attention), window_size(window_size),
        total_count(total_count), heads(heads),
        sequence_length(sequence_length), num_seq(num_seq),
        head_offset(head_offset), mask_stride(mask_stride), mp_size(mp_size),
        iterations(iterations), reduceWidth(reduceWidth){};

  void operator() [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1>) const {
    /* cg::thread_block b = cg::this_thread_block(); */
    /* cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);
     */

    auto b = sycl::ext::oneapi::experimental::this_group<1>();
    auto g = sycl::ext::oneapi::experimental::this_sub_group();
    auto pos = sycl::ext::oneapi::experimental::this_nd_item<1>();

    float2 low_data[MAX_REG_SIZE];
    float2 high_data[MAX_REG_SIZE];
    const T zero_h = conversion::to<T>(0.f);

    auto tid = pos.get_local_id(0);
    int wid = tid >> 5;
    int lane = tid & 0x1f;
    int warp_num = pos.get_local_range(0) >> 5;

    int reduce_blocks = reduceWidth >> 5;
    int seq_lane = tid % reduceWidth;

    /* __shared__ float partialSum[MAX_WARP_NUM]; */
    /* auto partialSum =
     * local_ptr<float>(local_ptr<void>(sycl::ext::oneapi::group_local_memory<float[MAX_WARP_NUM]>(b).get()));
     */
    auto partialSum = __group_local_memory<float[MAX_WARP_NUM]>(b);

    int iter_offset =
        pos.get_group(0) * (warp_num / reduce_blocks) + (wid / reduce_blocks);
    int batch_idx = iter_offset / (num_seq * heads);
    int alibi_offset = batch_idx * heads * mp_size + head_offset;
    int mask_offset = batch_idx * mask_stride + (iter_offset % mask_stride);

    T* rvals = vals;

    if (iter_offset < total_count) {
      rvals += (iter_offset * sequence_length);

      alibi_offset =
          (alibi_offset + ((iter_offset / num_seq) % heads)) * sequence_length;
      mask_offset = mask_offset * sequence_length;
      int seq_id = iter_offset % num_seq;
      int seq_id4 = seq_id >> 2;

      int real_seq_id =
          seq_id + (num_seq == sequence_length ? 0 : sequence_length);
      int window_stride4 =
          (local_attention && (real_seq_id >> 2) > (window_size >> 2))
              ? (real_seq_id >> 2) - (window_size >> 2)
              : 0;
      int window_stride = (local_attention && real_seq_id >= window_size)
                              ? real_seq_id - window_size
                              : -1;

      float max_val = minus_infinity;
      // if (lane == 0) printf("%d, %d: %d \n", wid, blockIdx.x(), mask_offset);
      for (int i = 0; i < iterations; i++) {
        int data_id = i * (reduceWidth << 2) + (seq_lane << 2);
        if ((!triangular || ((data_id >> 2) <= seq_id4)) &&
            (data_id >> 2) >= window_stride4 && data_id < sequence_length) {
          if ((sequence_length - data_id) >= 4) {
            low_data[i].x() =
                data_id > window_stride
                    ? conversion::to<float>(rvals[data_id]) * layer_scale
                    : minus_infinity;
            low_data[i].y() =
                ((!triangular || ((data_id + 1) <= seq_id)) &&
                 (data_id + 1) > window_stride)
                    ? conversion::to<float>(rvals[data_id + 1]) * layer_scale
                    : minus_infinity;
            high_data[i].x() =
                ((!triangular || ((data_id + 2) <= seq_id)) &&
                 (data_id + 2) > window_stride)
                    ? conversion::to<float>(rvals[data_id + 2]) * layer_scale
                    : minus_infinity;
            high_data[i].y() =
                ((!triangular || ((data_id + 3) <= seq_id)) &&
                 (data_id + 3) > window_stride)
                    ? conversion::to<float>(rvals[data_id + 3]) * layer_scale
                    : minus_infinity;
            if (alibi) {
              low_data[i].x() =
                  low_data[i].x() +
                  conversion::to<float>(alibi[data_id + alibi_offset]);
              low_data[i].y() =
                  low_data[i].y() +
                  conversion::to<float>(alibi[data_id + alibi_offset + 1]);
              high_data[i].x() =
                  high_data[i].x() +
                  conversion::to<float>(alibi[data_id + alibi_offset + 2]);
              high_data[i].y() =
                  high_data[i].y() +
                  conversion::to<float>(alibi[data_id + alibi_offset + 3]);
            }
            if (mask) {
              low_data[i].x() +=
                  conversion::to<float>(mask[data_id + mask_offset]);
              low_data[i].y() +=
                  conversion::to<float>(mask[data_id + mask_offset + 1]);
              high_data[i].x() +=
                  conversion::to<float>(mask[data_id + mask_offset + 2]);
              high_data[i].y() +=
                  conversion::to<float>(mask[data_id + mask_offset + 3]);
            }
          } else {
            low_data[i].x() =
                data_id > window_stride
                    ? conversion::to<float>(rvals[data_id]) * layer_scale
                    : minus_infinity;
            low_data[i].y() =
                (((!triangular || (data_id + 1) <= seq_id) &&
                  (data_id + 1) > window_stride) &&
                 (data_id + 1) < sequence_length)
                    ? conversion::to<float>(rvals[data_id + 1]) * layer_scale
                    : minus_infinity;
            high_data[i].x() =
                (((!triangular || (data_id + 2) <= seq_id) &&
                  (data_id + 2) > window_stride) &&
                 (data_id + 2) < sequence_length)
                    ? conversion::to<float>(rvals[data_id + 2]) * layer_scale
                    : minus_infinity;
            if (alibi) {
              low_data[i].x() =
                  low_data[i].x() +
                  conversion::to<float>(alibi[data_id + alibi_offset]);
              if ((data_id + 1) < sequence_length)
                low_data[i].y() =
                    low_data[i].y() +
                    conversion::to<float>(alibi[data_id + alibi_offset + 1]);
              if ((data_id + 2) < sequence_length)
                high_data[i].x() =
                    high_data[i].x() +
                    conversion::to<float>(alibi[data_id + alibi_offset + 2]);
            }
            high_data[i].y() = minus_infinity;
            if (mask) {
              low_data[i].x() +=
                  conversion::to<float>(mask[data_id + mask_offset]);
              if ((data_id + 1) < sequence_length)
                low_data[i].y() +=
                    conversion::to<float>(mask[data_id + mask_offset + 1]);
              if ((data_id + 2) < sequence_length)
                high_data[i].x() +=
                    conversion::to<float>(mask[data_id + mask_offset + 2]);
            }
          }
          // if(lane == 0) printf("%f , %d, %d \n", low_data[i].x, data_id,
          // seq_id);
          max_val = (low_data[i].x() > max_val ? low_data[i].x() : max_val);
          max_val = (low_data[i].y() > max_val ? low_data[i].y() : max_val);
          max_val = (high_data[i].x() > max_val ? high_data[i].x() : max_val);
          max_val = (high_data[i].y() > max_val ? high_data[i].y() : max_val);
        } else {
          low_data[i].x() = minus_infinity;
          low_data[i].y() = minus_infinity;
          high_data[i].x() = minus_infinity;
          high_data[i].y() = minus_infinity;
        }
      }

      for (int i = 1; i < WARP_SIZE; i *= 2) {
        auto temp = g.shuffle_xor(max_val, i);
        max_val = (temp > max_val ? temp : max_val);
      }

      if (reduceWidth > WARP_SIZE) {
        if (lane == 0)
          partialSum[wid] = max_val;
        /* b.sync(); */
        sycl::group_barrier(b, b.fence_scope);

        if (lane < warp_num)
          max_val = partialSum[lane];

        /* b.sync(); */
        sycl::group_barrier(b, b.fence_scope);

        for (int i = 1; i < reduce_blocks; i *= 2) {
          auto temp = g.shuffle_xor(max_val, i);
          max_val = (temp > max_val ? temp : max_val);
        }

        max_val = g.shuffle(max_val, tid / WARP_SIZE);
      }
      float sum = 0;
      for (int i = 0; i < iterations; i++) {
        low_data[i].x() = sycl::exp(low_data[i].x() - max_val);
        low_data[i].y() = sycl::exp(low_data[i].y() - max_val);
        high_data[i].x() = sycl::exp(high_data[i].x() - max_val);
        high_data[i].y() = sycl::exp(high_data[i].y() - max_val);

        sum += (low_data[i].x() + low_data[i].y() + high_data[i].x() +
                high_data[i].y());
      }

      for (int i = 1; i < WARP_SIZE; i *= 2)
        sum += g.shuffle_xor(sum, i);

      if (reduceWidth > WARP_SIZE) {
        if (lane == 0)
          partialSum[wid] = sum;
        /* b.sync(); */
        sycl::group_barrier(b, b.fence_scope);

        if (lane < warp_num)
          sum = partialSum[lane];

        /* b.sync(); */
        sycl::group_barrier(b, b.fence_scope);

        for (int i = 1; i < reduce_blocks; i *= 2) {
          sum += g.shuffle_xor(sum, i);
        }

        sum = g.shuffle(sum, tid / WARP_SIZE);
      }
      sum += 1e-6;
      for (int i = 0; i < iterations; i++) {
        int data_id = i * (reduceWidth << 2) + (seq_lane << 2);

        if (data_id < sequence_length) {
          if ((sequence_length - data_id) >= 4) {
            rvals[data_id] = conversion::to<T>(low_data[i].x() / sum);
            rvals[data_id + 1] = conversion::to<T>(low_data[i].y() / sum);
            rvals[data_id + 2] = conversion::to<T>(high_data[i].x() / sum);
            rvals[data_id + 3] = conversion::to<T>(high_data[i].y() / sum);
          } else {
            rvals[data_id] = conversion::to<T>(low_data[i].x() / sum);
            if ((data_id + 1) < sequence_length)
              rvals[data_id + 1] = conversion::to<T>(low_data[i].y() / sum);
            if ((data_id + 2) < sequence_length)
              rvals[data_id + 2] = conversion::to<T>(high_data[i].x() / sum);
          }
        }
      }
    }
  };
};

template <> class attn_softmax_v2<float> {

private:
  float *vals;
  float *attn_mask;
  float *alibi;
  float layer_scale;
  bool triangular;
  bool recompute;
  bool local_attention;
  int window_size;
  int total_count;
  int heads;
  int sequence_length;
  int num_seq;
  int head_offset;
  int mask_stride;
  int mp_size;
  int iterations;
  int reduceWidth;

public:
  attn_softmax_v2(float *vals, float *attn_mask, float *alibi,
                  float layer_scale, bool triangular, bool recompute,
                  bool local_attention, int window_size, int total_count,
                  int heads, int sequence_length, int num_seq, int head_offset,
                  int mask_stride, int mp_size, int iterations, int reduceWidth):
      vals(vals),
      attn_mask(attn_mask), alibi(alibi), layer_scale(layer_scale),
      triangular(triangular), recompute(recompute),
      local_attention(local_attention), window_size(window_size),
      total_count(total_count), heads(heads), sequence_length(sequence_length),
      num_seq(num_seq), head_offset(head_offset), mask_stride(mask_stride),
      mp_size(mp_size), iterations(iterations), reduceWidth(reduceWidth){};

  void operator() [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1>) const {

    /* cg::thread_block b = cg::this_thread_block(); */
    /* cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);
     */

    auto b = sycl::ext::oneapi::experimental::this_group<1>();
    auto g = sycl::ext::oneapi::experimental::this_sub_group();
    auto pos = sycl::ext::oneapi::experimental::this_nd_item<1>();

    float4 data[MAX_REG_SIZE];

    auto tid = pos.get_local_id(0);
    int wid = tid >> 5;
    int lane = tid & 0x1f;
    int warp_num = pos.get_local_range(0) >> 5;

    int reduce_blocks = reduceWidth >> 5;
    int seq_lane = tid % reduceWidth;

    /* __shared__ float partialSum[MAX_WARP_NUM]; */
    auto partialSum = __group_local_memory<float[MAX_WARP_NUM]>(b);

    int iter_offset =
        pos.get_group(0) * (warp_num / reduce_blocks) + (wid / reduce_blocks);

    float* rvals = vals;
    if (iter_offset < total_count) {
      rvals += (iter_offset * sequence_length);

      int batch_idx = iter_offset / (num_seq * heads);
      int alibi_offset = batch_idx * heads * mp_size + head_offset;
      int mask_offset = batch_idx * mask_stride + (iter_offset % mask_stride);
      mask_offset = mask_offset * sequence_length;
      int seq_id = iter_offset % num_seq;
      int seq_id4 = seq_id >> 2;

      int real_seq_id =
          seq_id + (num_seq == sequence_length ? 0 : sequence_length);
      int window_stride4 =
          (local_attention && (real_seq_id >> 2) > (window_size >> 2))
              ? (real_seq_id >> 2) - (window_size >> 2)
              : 0;
      int window_stride = (local_attention && real_seq_id >= window_size)
                              ? real_seq_id - window_size
                              : -1;

      float max_val = minus_infinity;

      for (int i = 0; i < iterations; i++) {
        int data_id = i * (reduceWidth << 2) + (seq_lane << 2);
        if ((!triangular || ((data_id >> 2) <= seq_id4)) &&
            (data_id >> 2) >= window_stride4 && data_id < sequence_length) {
          if ((sequence_length - data_id) >= 4) {
            data[i].x() =
                (data_id > window_stride ? rvals[data_id] : minus_infinity);
            data[i].y() = ((!triangular || ((data_id + 1) <= seq_id)) &&
                           (data_id + 1) > window_stride)
                              ? rvals[data_id + 1]
                              : minus_infinity;
            data[i].z() = ((!triangular || ((data_id + 2) <= seq_id)) &&
                           (data_id + 2) > window_stride)
                              ? rvals[data_id + 2]
                              : minus_infinity;
            data[i].w() = ((!triangular || ((data_id + 3) <= seq_id)) &&
                           (data_id + 3) > window_stride)
                              ? rvals[data_id + 3]
                              : minus_infinity;
            if (attn_mask) {
              data[i].x() += attn_mask[data_id + mask_offset];
              data[i].y() += attn_mask[data_id + mask_offset + 1];
              data[i].z() += attn_mask[data_id + mask_offset + 2];
              data[i].w() += attn_mask[data_id + mask_offset + 3];
            }
          } else {
            data[i].x() =
                data_id > window_stride ? rvals[data_id] : minus_infinity;
            data[i].y() = (((!triangular || (data_id + 1) <= seq_id)) &&
                           (data_id + 1) > window_stride &&
                           (data_id + 1) < sequence_length)
                              ? (rvals[data_id + 1])
                              : minus_infinity;
            data[i].z() = (((!triangular || (data_id + 2) <= seq_id)) &&
                           (data_id + 2) > window_stride &&
                           (data_id + 2) < sequence_length)
                              ? (rvals[data_id + 2])
                              : minus_infinity;
            data[i].w() = minus_infinity;
            if (attn_mask) {
              data[i].x() += attn_mask[data_id + mask_offset];
              if ((data_id + 1) < sequence_length)
                data[i].y() += attn_mask[data_id + mask_offset + 1];
              if ((data_id + 2) < sequence_length)
                data[i].z() += attn_mask[data_id + mask_offset + 2];
            }
          }
          max_val = (data[i].x() > max_val ? data[i].x() : max_val);
          max_val = (data[i].y() > max_val ? data[i].y() : max_val);
          max_val = (data[i].z() > max_val ? data[i].z() : max_val);
          max_val = (data[i].w() > max_val ? data[i].w() : max_val);
        } else {
          data[i].x() = minus_infinity;
          data[i].y() = minus_infinity;
          data[i].z() = minus_infinity;
          data[i].w() = minus_infinity;
        }
      }

      for (int i = 1; i < WARP_SIZE; i *= 2) {
        auto temp = g.shuffle_xor(max_val, i);
        max_val = (temp > max_val ? temp : max_val);
      }

      if (reduceWidth > WARP_SIZE) {
        if (lane == 0)
          partialSum[wid] = max_val;
        /* b.sync(); */
        sycl::group_barrier(b, b.fence_scope);

        if (lane < warp_num)
          max_val = partialSum[lane];

        /* b.sync(); */
        sycl::group_barrier(b, b.fence_scope);

        for (int i = 1; i < reduce_blocks; i *= 2) {
          auto temp = g.shuffle_xor(max_val, i);
          max_val = (temp > max_val ? temp : max_val);
        }

        max_val = g.shuffle(max_val, tid / WARP_SIZE);
      }

      float sum = 0;
      for (int i = 0; i < iterations; i++) {
        data[i].x() = sycl::exp(data[i].x() - max_val);
        data[i].y() = sycl::exp(data[i].y() - max_val);
        data[i].z() = sycl::exp(data[i].z() - max_val);
        data[i].w() = sycl::exp(data[i].w() - max_val);

        sum += (data[i].x() + data[i].y() + data[i].z() + data[i].w());
      }

      for (int i = 1; i < WARP_SIZE; i *= 2)
        sum += g.shuffle_xor(sum, i);

      if (reduceWidth > WARP_SIZE) {
        if (lane == 0)
          partialSum[wid] = sum;
        /* b.sync(); */
        sycl::group_barrier(b, b.fence_scope);

        if (lane < warp_num)
          sum = partialSum[lane];

        /* b.sync(); */
        sycl::group_barrier(b, b.fence_scope);

        for (int i = 1; i < reduce_blocks; i *= 2) {
          sum += g.shuffle_xor(sum, i);
        }

        sum = g.shuffle(sum, tid / WARP_SIZE);
      }
      sum += 1e-6;

      for (int i = 0; i < iterations; i++) {
        int data_id = i * (reduceWidth << 2) + (seq_lane << 2);

        if (data_id < sequence_length) {
          if ((sequence_length - data_id) >= 4) {
            rvals[data_id] = data[i].x() / sum;
            rvals[data_id + 1] = data[i].y() / sum;
            rvals[data_id + 2] = data[i].z() / sum;
            rvals[data_id + 3] = data[i].w() / sum;
          } else {
            rvals[data_id] = data[i].x() / sum;
            if ((data_id + 1) < sequence_length)
              rvals[data_id + 1] = data[i].y() / sum;
            if ((data_id + 2) < sequence_length)
              rvals[data_id + 2] = data[i].z() / sum;
          }
        }
      }
    }
  };
};

template <typename T>
void launch_attn_softmax_v2(T *vals, T *mask, T *alibi, float layer_scale,
                            bool triangular, bool recompute,
                            bool local_attention, int window_size,
                            int batch_size, int heads, int num_seq,
                            int sequence_length, int head_offset,
                            int mask_stride, int mp_size, sycl::queue stream) {
  const int total_count = batch_size * heads * num_seq;

  // Scheduling Overview
  // 4 element unroll with power of 2 `reduce_width` threads to a ceiling of
  // `attn_threads` Each block should be partitioned into as many `reduce_width`
  // blocks as can be fit.
  constexpr int attn_threads = 256;
  constexpr int min_reduce_width = hw_warp_size;
  constexpr int internal_unroll = 4;

  // Handle internal unroll then round to next power of 2. Bump up to minimum
  // granularity.
  const int thread_steps_rounded =
      next_pow2((sequence_length + internal_unroll - 1) / internal_unroll);
  const int thread_steps_schedule = (thread_steps_rounded < min_reduce_width)
                                        ? min_reduce_width
                                        : thread_steps_rounded;
  // Bound reduce width to the number of threads
  const int reduce_width = (thread_steps_schedule < attn_threads)
                               ? thread_steps_schedule
                               : attn_threads;
  // Scale for the excess
  const int iterations = thread_steps_schedule / reduce_width;
  // Should be safe since reduce_width is capped to attn_threads
  const int partitions = attn_threads / reduce_width;

  // Launch params
  /* dim3 grid((total_count + partitions - 1) / partitions); */
  /* dim3 block(attn_threads); */
  // TODO: change grid number
  // range<3> grid_dim(1, heads * sequence_length)
  sycl::range<1> block(attn_threads);
  sycl::range<1> grid(((total_count + partitions - 1) / partitions) * attn_threads);

  if (sequence_length <= 32768) {
    // TODO: grid and block
    attn_softmax_v2<T> fn(vals, mask, alibi, layer_scale, triangular, recompute,
                          local_attention, window_size, total_count, heads,
                          sequence_length, num_seq, head_offset, mask_stride,
                          mp_size, iterations, reduce_width);

    stream.submit([&](sycl::handler &cmd_list) {
      cmd_list.parallel_for(sycl::nd_range<1>{grid, block}, fn);
    });
  } else
    throw std::runtime_error("Unsupport Seq_Length!");
}

template void launch_attn_softmax_v2(float *vals, float *mask, float *alibi,
                                     float layer_scale, bool triangular,
                                     bool recompute, bool local_attention,
                                     int window_size, int batch_size, int heads,
                                     int num_seq, int sequence_length,
                                     int head_offset, int mask_stride,
                                     int mp_size, sycl::queue);

template void launch_attn_softmax_v2(bf16 *vals, bf16 *mask, bf16 *alibi,
                                     float layer_scale, bool triangular,
                                     bool recompute, bool local_attention,
                                     int window_size, int batch_size, int heads,
                                     int num_seq, int sequence_length,
                                     int head_offset, int mask_stride,
                                     int mp_size, sycl::queue);

template void launch_attn_softmax_v2(sycl::half *vals, sycl::half *mask, sycl::half *alibi,
                                     float layer_scale, bool triangular,
                                     bool recompute, bool local_attention,
                                     int window_size, int batch_size, int heads,
                                     int num_seq, int sequence_length,
                                     int head_offset, int mask_stride,
                                     int mp_size, sycl::queue);

template class attn_softmax_v2<sycl::half>;

template class attn_softmax_v2<bf16>;
