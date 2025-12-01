#pragma once

#include "pytorch_extension_utils.h"

#include <cuda_runtime.h>
#include <optional>
#include <raft/matrix/detail/select_k.cuh>

using namespace raft::matrix::detail::select::radix::impl;

namespace sspec::topk {
template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
RAFT_KERNEL BatchTopkKernel(
    const T *in_val, const IdxT *in_idx, const IdxT *in_val_start_pos,
    const IdxT *in_idx_start_pos, const IdxT *in_lens, const IdxT *ks,
    T *out_val, IdxT *out_idx, char *bufs, const size_t num_requests,
    const size_t batch_size, const size_t MAX_TOTAL_VERIFY_KV_LEN,
    const size_t MAX_DRAFT_KV_LEN, const bool select_min, const size_t offset) {
  // This kernel performs individual top-k on each request with each CTA
  // Use size_t to avoid multiplication overflow
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  __shared__ Counter<T, IdxT> counter;
  __shared__ IdxT histogram[num_buckets];

  // unroll for [num_requests, batch_size]
  // batch-size major to increase cache locality
  const size_t task_id = blockIdx.x + offset;
  const size_t batch_id = task_id % batch_size;
  const size_t request_id = task_id / batch_size;

  if (batch_id >= batch_size || request_id >= num_requests) {
    return;
  }

  IdxT l_len = in_lens[request_id];
  const IdxT k = ks[request_id];

  if (k == 0) {
    // skip topk
    return;
  }

  // add offsets to the pointers
  // in_val [batch_size, MAX_TOTAL_VERIFY_KV_LEN]; last dimension in CSR
  // in_idx CSR format
  in_val += batch_id * MAX_TOTAL_VERIFY_KV_LEN + in_val_start_pos[request_id];
  in_idx += in_idx_start_pos[request_id];
  out_idx += (request_id * batch_size + batch_id) * MAX_DRAFT_KV_LEN;
  if (out_val) {
    // [num_requests, batch_size, MAX_DRAFT_KV_LEN]
    out_val += (request_id * batch_size + batch_id) * MAX_DRAFT_KV_LEN;
  }

  // calculate buf_len, assure same size for all CTAs
  const IdxT buf_len = calc_buf_len<T, IdxT, unsigned>(MAX_DRAFT_KV_LEN);
  bufs += blockIdx.x * buf_len * 2 * (sizeof(T) + sizeof(IdxT));

  if (l_len <= k) {
    if (out_val) {
      copy_in_val(out_val, in_val, l_len, k, select_min);
    }
    copy_in_idx(out_idx, in_idx, l_len);
    __syncthreads();
    return;
  }

  if (threadIdx.x == 0) {
    counter.k = k;
    counter.len = l_len;
    counter.previous_len = l_len;
    counter.kth_value_bits = 0;
    counter.out_cnt = 0;
    counter.out_back_cnt = 0;
  }
  __syncthreads();

  constexpr int num_passes = calc_num_passes<T, BitsPerPass>();
  for (int pass = 0; pass < num_passes; ++pass) {
    const T *in_buf;
    const IdxT *in_idx_buf;
    T *out_buf;
    IdxT *out_idx_buf;
    set_buf_pointers(in_val, in_idx, bufs, buf_len, pass, in_buf, in_idx_buf,
                     out_buf, out_idx_buf);

    const IdxT current_len = counter.len;
    const IdxT current_k = counter.k;
    IdxT previous_len = counter.previous_len;
    if (previous_len > buf_len) {
      in_buf = in_val;
      in_idx_buf = in_idx;
      previous_len = l_len;
    }
    if (current_len > buf_len) {
      // so "out_buf==nullptr" denotes skipping writing buffer in current pass
      out_buf = nullptr;
      out_idx_buf = nullptr;
    }

    // in case we have individual len for each query defined we want to make
    // sure that we only iterate valid elements.
    const IdxT max_len = max(l_len, k);
    if (max_len < previous_len)
      previous_len = max_len;

    filter_and_histogram_for_one_block<T, IdxT, BitsPerPass>(
        in_buf, in_idx_buf, out_buf, out_idx_buf, out_val, out_idx,
        previous_len, &counter, histogram, select_min, pass);
    __syncthreads();

    scan<IdxT, BitsPerPass, BlockSize>(histogram);
    __syncthreads();

    choose_bucket<T, IdxT, BitsPerPass>(&counter, histogram, current_k, pass);
    if (threadIdx.x == 0) {
      counter.previous_len = current_len;
    }
    __syncthreads();

    if (counter.len == counter.k || pass == num_passes - 1) {
      last_filter<T, IdxT, BitsPerPass>(out_buf ? out_buf : in_val,
                                        out_buf ? out_idx_buf : in_idx, out_val,
                                        out_idx, out_buf ? current_len : l_len,
                                        k, &counter, select_min, pass);
      break;
    }
  }
}
} // namespace sspec::topk

/*!
 * Select Top-k value in a batched tensor. This function is a wrapper
 * around RAFT's implementation on topk selection. Official documentation
 * reference:
 * https://github.com/rapidsai/cuvs/blob/daede3ea7f43a9849d6ee8dd3633344ff1c466b4/cpp/src/selection/select_k.cuh
 * This is an implementation from SC'21:
 * https://dl.acm.org/doi/pdf/10.1145/3581784.3607062
 */
template <typename DType, typename IdType>
void BatchTopkDispatch(const DType *in_val, const IdType *in_idx,
                       const IdType *in_val_start_pos,
                       const IdType *in_idx_start_pos, const IdType *in_lens,
                       const IdType *ks, DType *out_val, IdType *out_idx,
                       char *bufs, const size_t buf_size,
                       const size_t num_requests, const size_t batch_size,
                       const size_t MAX_TOTAL_VERIFY_KV_LEN,
                       const size_t MAX_DRAFT_KV_LEN, cudaStream_t stream) {
  // Hyperparatmer selection:
  // https://github.com/rapidsai/raft/blob/72b43bc0a8d305f83a507745798c4f4c0d68331f/cpp/include/raft/matrix/detail/select_k-inl.cuh#L47
  constexpr int BitsPerPass = 8;
  constexpr int BlockSize = 512;
  // Fused since softmaxed data is skewed
  // https://github.com/rapidsai/raft/blob/72b43bc0a8d305f83a507745798c4f4c0d68331f/cpp/include/raft/matrix/detail/select_radix.cuh#L1262
  constexpr bool select_min = false; // Select top-k

  // K quite small, use single threadblock
  // https://github.com/rapidsai/raft/blob/72b43bc0a8d305f83a507745798c4f4c0d68331f/cpp/include/raft/matrix/detail/select_radix.cuh#L1310
  auto kernel =
      sspec::topk::BatchTopkKernel<DType, IdType, BitsPerPass, BlockSize>;

  int sm_cnt;
  {
    int dev;
    (cudaGetDevice(&dev));
    (cudaDeviceGetAttribute(&sm_cnt, cudaDevAttrMultiProcessorCount, dev));
  }

  // Each request will have `batch_size` identical works
  // ref:
  // https://github.com/rapidsai/raft/blob/82fb37a42b226c8908a32fb847bde4bfa47e4804/cpp/include/raft/matrix/detail/select_radix.cuh#L814
  const size_t num_total_works = batch_size * num_requests;
  const size_t max_chunk_size = calc_chunk_size<DType, IdType, BlockSize>(
      num_total_works, MAX_TOTAL_VERIFY_KV_LEN, sm_cnt, kernel, true);

  // check buf size
  // ref:
  // https://github.com/rapidsai/raft/blob/82fb37a42b226c8908a32fb847bde4bfa47e4804/cpp/include/raft/matrix/detail/select_radix.cuh#L115
  const IdType _buf_size_per_cta =
      calc_buf_len<DType, IdType, unsigned>(MAX_DRAFT_KV_LEN);
  if (buf_size < max_chunk_size * _buf_size_per_cta * 2 *
                     (sizeof(DType) + sizeof(IdType))) {
    TORCH_CHECK(false, "Buf size is too small");
  }

  for (size_t offset = 0; offset < num_total_works; offset += max_chunk_size) {
    int chunk_size = std::min(max_chunk_size, num_total_works - offset);
    kernel<<<chunk_size, BlockSize, 0, stream>>>(
        in_val, in_idx, in_val_start_pos, in_idx_start_pos, in_lens, ks,
        out_val, out_idx, bufs, num_requests, batch_size,
        MAX_TOTAL_VERIFY_KV_LEN, MAX_DRAFT_KV_LEN, select_min, offset);
  }
}

/*!
 * \brief
 * \param in_val [batch_size, MAX_TOTAL_VERIFY_KV_LEN] lst dimension in CSR
 * \param in_idx CSR format
 * \param in_val_start_pos [num_requests]
 * \param in_idx_start_pos [num_requests]
 * \param in_lens [num_requests]
 * \param ks [num_requests] k used for each sequence
 * \param num_requests number of requests
 * \param out_val [num_requests, batch_size, MAX_DRAFT_KV_LEN]
 * \param out_idx [num_requests, batch_size, MAX_DRAFT_KV_LEN]
 * \param buf temporary working space
 */
void batch_topk(at::Tensor in_val, at::Tensor in_idx,
                at::Tensor in_val_start_pos, at::Tensor in_idx_start_pos,
                at::Tensor in_lens, at::Tensor ks, int64_t num_requests,
                std::optional<at::Tensor> out_val, at::Tensor out_idx,
                at::Tensor buf) {
  CHECK_DIM(2, in_val);
  CHECK_DIM(1, in_idx);
  CHECK_DIM(1, in_val_start_pos);
  CHECK_DIM(1, in_idx_start_pos);
  CHECK_DIM(1, in_lens);
  CHECK_DIM(3, out_idx);
  CHECK_DIM(1, ks);

  TORCH_CHECK(num_requests == ks.size(0));
  TORCH_CHECK(num_requests == in_val_start_pos.size(0));
  TORCH_CHECK(num_requests == in_idx_start_pos.size(0));
  TORCH_CHECK(num_requests == in_lens.size(0));
  TORCH_CHECK(num_requests == out_idx.size(0));
  TORCH_CHECK(in_val.size(0) == out_idx.size(1));

  if (out_val) {
    CHECK_DIM(3, (*out_val));
    TORCH_CHECK(out_val->size(0) == num_requests);
  }

  const size_t batch_size = in_val.size(0);
  const size_t MAX_TOTAL_VERIFY_KV_LEN = in_val.size(1);
  const size_t MAX_DRAFT_KV_LEN = out_idx.size(2);
  const size_t buf_size = buf.numel() * buf.element_size();

  const c10::cuda::OptionalCUDAGuard device_guard(in_val.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  bool success =
      DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(in_val.scalar_type(), DType, [&] {
        return DISPATCH_PYTORCH_IDTYPE_TO_CTYPE(
            in_idx.scalar_type(), IdType, [&] {
              BatchTopkDispatch<DType, IdType>(
                  static_cast<DType *>(in_val.data_ptr()),
                  static_cast<IdType *>(in_idx.data_ptr()),
                  static_cast<IdType *>(in_val_start_pos.data_ptr()),
                  static_cast<IdType *>(in_idx_start_pos.data_ptr()),
                  static_cast<IdType *>(in_lens.data_ptr()),
                  static_cast<IdType *>(ks.data_ptr()),
                  out_val ? static_cast<DType *>(out_val->data_ptr()) : nullptr,
                  static_cast<IdType *>(out_idx.data_ptr()),
                  static_cast<char *>(buf.data_ptr()), buf_size, num_requests,
                  batch_size, MAX_TOTAL_VERIFY_KV_LEN, MAX_DRAFT_KV_LEN,
                  stream);
              return cudaPeekAtLastError() == cudaSuccess;
            });
      });

  // print error message
  if (!success) {
    auto err = cudaGetLastError();
    TORCH_CHECK(false, "Batch Top-K launch failed with error: ",
                cudaGetErrorString(err));
  }
}
