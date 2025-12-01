import array

import flashinfer
import numpy as np
import torch
import triton
import triton.language as tl
from flashinfer import get_batch_indices_positions
from numba import njit

from serve.attention.backend import BatchAttention, BatchPrefillWithPagedKVCacheWrapper
from serve.model.model import KVPool
from serve.utils import dtype_to_bytes


@njit
def convert_physical_to_virtual(
    selected_indices_buffer,
    indices,
    len_selected_indices,
    indices_cpu_len,
    length,
):
    _, num_layers, num_kv_heads = selected_indices_buffer.shape
    for i in range(num_layers):
        for j in range(num_kv_heads):
            idx = -1
            for k in range(20, -1, -1):
                step = 1 << k
                if (
                    idx + step < len_selected_indices
                    and selected_indices_buffer[idx + step, i, j] < 0
                ):
                    idx += step
            idx += 1

            for k in range(length):
                if idx >= len_selected_indices:
                    break
                if selected_indices_buffer[idx, i, j] == indices[-(k + 1)]:
                    selected_indices_buffer[idx, i, j] = -(
                        indices_cpu_len - (length - (k + 1))
                    )
                    idx += 1
    return selected_indices_buffer


@triton.jit
def batch_copy_tensor_kernel(
    src_tensor_ptrs,
    tgt_tensor_ptrs,
    n_lens,
    n_stride,
    BLOCK_SIZE: tl.constexpr,
):
    req_id = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    n_len = tl.load(n_lens + req_id) * n_stride
    src_tensor_ptr = tl.load(src_tensor_ptrs + req_id)
    tgt_tensor_ptr = tl.load(tgt_tensor_ptrs + req_id)

    # Note that copied tensor should be tl.float32 for now
    src_addr = src_tensor_ptr.to(tl.pointer_type(tl.float32))
    tgt_addr = tgt_tensor_ptr.to(tl.pointer_type(tl.float32))

    start = worker_id * BLOCK_SIZE
    step = num_workers * BLOCK_SIZE

    for base in range(start, n_len, step):
        offs = base + tl.arange(0, BLOCK_SIZE)
        src_vals = tl.load(src_addr + offs, mask=offs < n_len, other=0)
        tl.store(tgt_addr + offs, src_vals, mask=offs < n_len)


class FullCachePtr:
    # Used for attention metadata
    _MAX_BATCH_SIZE = 4 * 1024
    _MAX_NNZ = 8192 * 4
    _MAX_KV_CONTEXT_LEN = 2 * 1024 * 1024  # 2M tokens

    # must be init before usage
    _INITIALIZED = 0

    def __init__(
        self,
        kv_pool: KVPool,
        indices: array.array = None,
        len_kv_cache: int = 0,
    ):
        self.kv_pool = kv_pool
        self.indices: array.array = (
            array.array("i", indices) if indices is not None else array.array("i", [])
        )
        self.indices_cpu = array.array("i", [])
        self.len_kv_cache: int = len_kv_cache
        self.page_size = kv_pool.page_size

    def __del__(self):
        for page_index in self.indices:
            self.kv_pool.free_page(page_index)

    @property
    def last_page_offset(self) -> int:
        return (self.len_kv_cache - 1) % self.page_size + 1

    def allocate_tokens(self, num_tokens: int, restore=False) -> int:
        assert (
            0 < num_tokens <= self.kv_pool.num_free_pages * self.page_size
        ), f"Cannot allocate {num_tokens} tokens (exceeds available memory)"

        num_new_pages = (num_tokens + self.last_page_offset - 1) // self.page_size
        for _ in range(num_new_pages):
            self.indices.append(self.kv_pool.alloc_page())

        self.len_kv_cache += num_tokens
        return num_new_pages

    def retract_tokens(self, num_tokens: int) -> None:
        assert (
            0 <= num_tokens <= self.len_kv_cache
        ), f"Cannot retract {num_tokens} tokens (only {self.len_kv_cache} exist)"

        for _ in range(num_tokens):
            if self.last_page_offset == 1:
                self.kv_pool.free_page(self.indices.pop())
            self.len_kv_cache -= 1

    def offload(self, length):
        assert self.page_size == 1
        assert length > 0

        length = min(length, len(self.indices))
        for _ in range(length):
            self.indices_cpu.append(self.kv_pool.alloc_page_cpu())

        self.kv_pool.offload(self.indices[-length:], self.indices_cpu[-length:][::-1])
        self.retract_tokens(length)

        return len(self.indices)

    def restore(self, length):
        assert self.page_size == 1
        assert length > 0

        length = min(length, len(self.indices_cpu))

        import torch.cuda.nvtx as nvtx

        nvtx.range_push("Allocate tokens")
        self.allocate_tokens(length, restore=True)
        nvtx.range_pop()

        nvtx.range_push("Restore KV cache")
        self.kv_pool.restore(self.indices_cpu[-length:][::-1], self.indices[-length:])
        nvtx.range_pop()

        nvtx.range_push("Free pages")
        for _ in range(length):
            self.kv_pool.free_page_cpu(self.indices_cpu.pop())
        nvtx.range_pop()

        return len(self.indices_cpu)

    @classmethod
    def initialize_class_metadata(cls, **kwargs):
        assert cls._INITIALIZED == 0
        cls._INITIALIZED = 1

    @classmethod
    def dispatch_wrappers_and_metadata(
        cls,
        kv_layout: str,
        page_size: int,
        enable_vanilla_backend: bool,
        device: str,
    ):
        """
        This function is for modularizing the dispatching of different attention.
        Called by kv-pool.
        """
        # step 1. initialize backend attention wrappers
        if not enable_vanilla_backend:
            wrapper = BatchAttention(
                kv_layout=kv_layout,
            )
        else:
            # use vanilla prefill kernel
            _float_buf = torch.empty(
                512 * 1024 * 1024, dtype=torch.uint8, device=device
            )
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                float_workspace_buffer=_float_buf,
                kv_layout=kv_layout,
                use_cuda_graph=False,
                backend="fa2",
            )

        # step 2. initialize all attention metadata (this is for cuda graph compatible)
        metadata = {}
        metadata["batch_size_host"] = 0
        metadata["nnz_host"] = 0
        metadata["batch_size_tensor"] = torch.zeros(
            [1], dtype=torch.int32, device=device
        )
        metadata["nnz_tensor"] = torch.zeros([1], dtype=torch.int32, device=device)
        metadata["kv_indices"] = torch.empty(
            cls._MAX_KV_CONTEXT_LEN, dtype=torch.int32, device=device
        )
        metadata["batch_indices"] = torch.empty(
            cls._MAX_NNZ, dtype=torch.int32, device=device
        )
        metadata["positions"] = torch.empty(
            cls._MAX_NNZ, dtype=torch.int32, device=device
        )
        metadata["pos_offsets"] = torch.empty(
            cls._MAX_BATCH_SIZE, dtype=torch.int32, device=device
        )
        metadata["qo_indptr"] = torch.empty(
            cls._MAX_BATCH_SIZE + 1, dtype=torch.int32, device=device
        )
        metadata["kv_indptr"] = torch.empty(
            cls._MAX_BATCH_SIZE + 1, dtype=torch.int32, device=device
        )
        metadata["all_kv_indptr"] = torch.empty(
            cls._MAX_BATCH_SIZE + 1, dtype=torch.int32, device=device
        )  # used for kv-append in `PillarCachePtr`
        metadata["kv_len_arr"] = torch.empty(
            cls._MAX_BATCH_SIZE, dtype=torch.int32, device=device
        )
        metadata["kv_last_page_len"] = torch.ones(
            cls._MAX_BATCH_SIZE, dtype=torch.int32, device=device
        )

        return [wrapper], metadata

    @classmethod
    def dispatch_attn_plan(
        cls,
        wrappers,
        attn_fwd_metadata: dict,
        kv_pool: KVPool,  # read-only; for parameters
        **kwargs,
    ):
        """
        Dispatch to differen attention backend. Use `kwargs` for variable arguments.
        """
        # instantiate arguments
        # using name from prepare_forward_inputs
        ids = kwargs["token_ids"]
        qo_indptr = kwargs["token_indptr"]
        kv_indptr = kwargs["kv_indptr"]
        kv_indices = kwargs["kv_indices"]
        kv_len_arr = kwargs["kv_len_arr"]
        pos_offsets = kwargs["pos_offsets"]
        wrapper = wrappers[0]  # attention wrapper is first

        # Use CUDA Graph buf for kv_indices
        assert attn_fwd_metadata["kv_indices"].shape[0] >= kv_indices.numel()
        assert kv_pool.capacity >= kv_indices.numel()
        attn_fwd_metadata["kv_indices"][: kv_indices.numel()] = kv_indices.flatten()

        nnz = ids.size(0)
        batch_size = qo_indptr.size(0) - 1

        if not kv_pool.vanilla_backend:
            # Call customized
            wrapper.plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                kv_len_arr=kv_len_arr,
                num_qo_heads=kv_pool.num_qo_heads,
                num_kv_heads=kv_pool.num_kv_heads,
                head_dim_qk=kv_pool.head_dim,
                head_dim_vo=kv_pool.head_dim,
                page_size=kv_pool.page_size,
                causal=True,
                q_data_type=kv_pool._physical_mem_gpu.dtype,
                kv_data_type=kv_pool._physical_mem_gpu.dtype,
                use_profiler=False,
            )
        else:
            wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=kv_indptr,
                paged_kv_indices=attn_fwd_metadata["kv_indices"],
                paged_kv_last_page_len=attn_fwd_metadata["kv_last_page_len"][
                    :batch_size
                ],
                num_qo_heads=kv_pool.num_qo_heads,
                num_kv_heads=kv_pool.num_kv_heads,
                head_dim_qk=kv_pool.head_dim,
                page_size=kv_pool.page_size,
                causal=True,
                q_data_type=kv_pool._physical_mem_gpu.dtype,
                kv_data_type=kv_pool._physical_mem_gpu.dtype,
            )

        if cls.__name__ == "PillarCachePtr":
            # use all-kv-indices, all-kv-len-arr, and all-kv-indptr
            # to index through the page-table
            batch_indices, positions = get_batch_indices_positions(
                append_indptr=qo_indptr,
                seq_lens=kwargs["all_kv_len_arr"],
                nnz=nnz,
            )
            # NOTE (Yilong): this is for kv page append kernel in `PillarCachePtr`
            all_kv_indptr = kwargs["all_kv_indptr"]
            attn_fwd_metadata["all_kv_indptr"][: all_kv_indptr.shape[0]] = all_kv_indptr
        else:
            batch_indices, positions = get_batch_indices_positions(
                append_indptr=qo_indptr,
                seq_lens=kv_len_arr,
                nnz=nnz,
            )

        assert attn_fwd_metadata["batch_indices"].shape[0] > batch_indices.shape[0]
        assert attn_fwd_metadata["positions"].shape[0] > positions.shape[0]
        assert attn_fwd_metadata["pos_offsets"].shape[0] > pos_offsets.shape[0]
        assert attn_fwd_metadata["qo_indptr"].shape[0] > qo_indptr.shape[0]
        assert attn_fwd_metadata["kv_indptr"].shape[0] > kv_indptr.shape[0]

        attn_fwd_metadata["batch_indices"][: batch_indices.shape[0]] = batch_indices
        attn_fwd_metadata["positions"][: positions.shape[0]] = positions
        attn_fwd_metadata["pos_offsets"][: pos_offsets.shape[0]] = pos_offsets
        attn_fwd_metadata["qo_indptr"][: qo_indptr.shape[0]] = qo_indptr
        attn_fwd_metadata["kv_indptr"][: kv_indptr.shape[0]] = kv_indptr
        attn_fwd_metadata["batch_size_host"] = batch_size
        attn_fwd_metadata["nnz_host"] = nnz
        attn_fwd_metadata["batch_size_tensor"][0] = batch_size
        attn_fwd_metadata["nnz_tensor"][0] = nnz

    @classmethod
    def dispatch_attn_fwd(
        cls,
        wrappers,
        attn_fwd_metadata: dict,
        kv_pool: KVPool,  # read-only; for parameters
        **kwargs,
    ):
        """
        Dispatch to differen attention backend. Use `kwargs` for variable arguments.
        """
        # instantiate arguments
        q = kwargs["q"]
        layer_idx = kwargs["layer_idx"]
        wrapper = wrappers[0]  # attention wrapper is first

        if not kv_pool.vanilla_backend:
            out, _ = wrapper.run(
                q,
                kv_cache=kv_pool._physical_mem_gpu[layer_idx],
                kv_indices=attn_fwd_metadata["kv_indices"],
            )
        else:
            out, _ = wrapper.run(
                q, paged_kv_cache=kv_pool._physical_mem_gpu[layer_idx], return_lse=True
            )

        return out

    @classmethod
    def dispatch_append_kv_cache(
        cls,
        k,
        v,
        layer_idx,
        kv_pool,  # read-only; for parameters
    ):
        _meta = kv_pool.attn_metadata
        if kv_pool.vanilla_backend:
            # not cuda graph compatible
            flashinfer.append_paged_kv_cache(
                append_key=k,
                append_value=v,
                batch_indices=_meta["batch_indices"][: _meta["nnz_host"]],
                positions=_meta["positions"][: _meta["nnz_host"]],
                paged_kv_cache=kv_pool._physical_mem_gpu[layer_idx],
                kv_indices=_meta["kv_indices"],
                kv_indptr=_meta["kv_indptr"][: _meta["batch_size_host"] + 1],
                kv_last_page_len=_meta["kv_last_page_len"][: _meta["batch_size_host"]],
                kv_layout=kv_pool._kv_layout,
            )
        else:
            # customized cuda-graph compatiable version
            flashinfer.append_paged_kv_cache_graph(
                append_key=k,
                append_value=v,
                batch_indices=_meta["batch_indices"][: _meta["nnz_host"]],
                positions=_meta["positions"][: _meta["nnz_host"]],
                nnz=_meta["nnz_tensor"],
                paged_kv_cache=kv_pool._physical_mem_gpu[layer_idx],
                kv_indices=_meta["kv_indices"],
                kv_indptr=_meta["kv_indptr"][: _meta["batch_size_host"] + 1],
                kv_last_page_len=_meta["kv_last_page_len"][: _meta["batch_size_host"]],
                kv_layout=kv_pool._kv_layout,
            )

    @classmethod
    def dispatch_output_snapshot(
        cls,
        wrappers,
        attn_fwd_metadata,
        input_metadata,
    ) -> dict:
        return {}

    @staticmethod
    def scatter_token_probs(
        src_tensor: torch.Tensor,  # [num_tokens, vocab_size]
        src_start_pos: np.ndarray,  # [batch_size]
        tgt_tensor_ptrs_host: np.ndarray,  # [batch_size]
        token_num_list: np.ndarray,  # [batch_size]
    ):
        """
        This function scatters the draft token probabilities to the local buffer of each request,
        from the large `token-probs` tensor from model forward.
        """
        assert src_tensor.dtype == torch.float32
        assert src_tensor.ndim == 2
        assert tgt_tensor_ptrs_host.dtype == np.int64

        src_start_addr = src_tensor.data_ptr()
        n_stride = src_tensor.stride(0)
        batch_size = len(src_start_pos)

        assert batch_size == len(tgt_tensor_ptrs_host)
        assert batch_size == len(token_num_list)
        assert batch_size > 0

        src_tensor_ptrs_host = np.full_like(
            src_start_pos, src_start_addr, dtype=np.int64
        )
        src_tensor_ptrs_host += (
            src_start_pos * dtype_to_bytes(src_tensor.dtype) * n_stride
        )

        src_tensor_ptrs_device = torch.from_numpy(src_tensor_ptrs_host).to(
            src_tensor.device, non_blocking=True
        )
        tgt_tensor_ptrs_device = torch.from_numpy(tgt_tensor_ptrs_host).to(
            src_tensor.device, non_blocking=True
        )
        n_lens = torch.from_numpy(token_num_list).to(
            src_tensor.device, non_blocking=True
        )

        grid = (batch_size, 4)
        batch_copy_tensor_kernel[grid](
            src_tensor_ptrs_device,
            tgt_tensor_ptrs_device,
            n_lens,
            n_stride,
            BLOCK_SIZE=1024,
        )

    @staticmethod
    def gather_token_probs(
        src_tensor_ptrs_host: np.ndarray,  # [batch_size]
        tgt_tensor: torch.Tensor,  # [num_tokens, vocab_size]
        tgt_start_pos: np.ndarray,  # [batch_size]
        token_num_list: np.ndarray,  # [batch_size]
    ):
        """
        This function gathers the draft token probabilities from the local buffer of each request,
        to the large `token-probs` which is then fed into the `RejectionSampler`.
        """
        assert tgt_tensor.dtype == torch.float32
        assert tgt_tensor.ndim == 2
        assert src_tensor_ptrs_host.dtype == np.int64
        assert np.sum(token_num_list) == tgt_tensor.shape[0]

        tgt_start_addr = tgt_tensor.data_ptr()
        n_stride = tgt_tensor.stride(0)
        batch_size = len(tgt_start_pos)

        assert batch_size == len(src_tensor_ptrs_host)
        assert batch_size == len(token_num_list)
        assert batch_size > 0

        tgt_tensor_ptrs_host = np.full_like(
            tgt_start_pos, tgt_start_addr, dtype=np.int64
        )
        tgt_tensor_ptrs_host += (
            tgt_start_pos * dtype_to_bytes(tgt_tensor.dtype) * n_stride
        )

        src_tensor_ptrs_device = torch.from_numpy(src_tensor_ptrs_host).to(
            tgt_tensor.device, non_blocking=True
        )
        tgt_tensor_ptrs_device = torch.from_numpy(tgt_tensor_ptrs_host).to(
            tgt_tensor.device, non_blocking=True
        )
        n_lens = torch.from_numpy(token_num_list).to(
            tgt_tensor.device, non_blocking=True
        )

        # calculate num_workers to increase parallelism
        BLOCK_SIZE = 1024
        num_workers = min(32, n_stride // BLOCK_SIZE)
        num_workers = max(1, num_workers)
        assert num_workers * BLOCK_SIZE < n_stride

        grid = (batch_size, num_workers)
        batch_copy_tensor_kernel[grid](
            src_tensor_ptrs_device,
            tgt_tensor_ptrs_device,
            n_lens,
            n_stride,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    @staticmethod
    def copy_token_probs(
        src_tensor: torch.Tensor,  # [num_tokens, vocab_size]
        src_start_pos: np.ndarray,  # [batch_size]
        tgt_tensor: torch.Tensor,  # [num_tokens, vocab_size]
        tgt_start_pos: np.ndarray,  # [batch_size]
        token_num_list: np.ndarray,  # [batch_size]
    ):
        """
        This function gathers the draft token probabilities from the local buffer of each request,
        to the large `token-probs` which is then fed into the `RejectionSampler`.
        """
        assert tgt_tensor.dtype == torch.float32
        assert src_tensor.dtype == torch.float32
        assert tgt_tensor.ndim == 2
        assert src_tensor.ndim == 2
        assert src_tensor.stride() == tgt_tensor.stride()

        n_stride = tgt_tensor.stride(0)
        batch_size = len(tgt_start_pos)

        assert batch_size == len(token_num_list)
        assert batch_size == len(src_start_pos)
        assert batch_size > 0

        src_start_addr = src_tensor.data_ptr()
        tgt_start_addr = tgt_tensor.data_ptr()

        src_tensor_ptrs_host = np.full_like(
            src_start_pos, src_start_addr, dtype=np.int64
        )
        src_tensor_ptrs_host += (
            src_start_pos * dtype_to_bytes(src_tensor.dtype) * n_stride
        )
        tgt_tensor_ptrs_host = np.full_like(
            tgt_start_pos, tgt_start_addr, dtype=np.int64
        )
        tgt_tensor_ptrs_host += (
            tgt_start_pos * dtype_to_bytes(tgt_tensor.dtype) * n_stride
        )

        src_tensor_ptrs_device = torch.from_numpy(src_tensor_ptrs_host).to(
            tgt_tensor.device, non_blocking=True
        )
        tgt_tensor_ptrs_device = torch.from_numpy(tgt_tensor_ptrs_host).to(
            tgt_tensor.device, non_blocking=True
        )
        n_lens = torch.from_numpy(token_num_list).to(
            tgt_tensor.device, non_blocking=True
        )

        # calculate num_workers to increase parallelism
        BLOCK_SIZE = 1024
        num_workers = min(32, n_stride // BLOCK_SIZE)
        num_workers = max(1, num_workers)
        assert num_workers * BLOCK_SIZE < n_stride

        grid = (batch_size, num_workers)
        batch_copy_tensor_kernel[grid](
            src_tensor_ptrs_device,
            tgt_tensor_ptrs_device,
            n_lens,
            n_stride,
            BLOCK_SIZE=1024,
        )
