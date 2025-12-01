import array

import numpy as np
import torch
import torch.cuda.nvtx as nvtx
import triton
import triton.language as tl

from serve.model.model import KVPool
from serve.request.kv_cache_ptr.base import FullCachePtr, convert_physical_to_virtual
from serve.request.request import Request


class StreamingCachePtr(FullCachePtr):
    # Maximum length of selected KV cache
    # used for the sparse kv-cache metadata buf allocation
    MAX_LEN_SELECTED_KV_CACHE = 6 * 1024
    MAX_TOTAL_DRAFT_KV_LEN = 128 * 1024 * 4

    # for streaming cache, all heads and layers are identical
    NUM_LAYERS = 1
    NUM_KV_HEADS = 1

    def __init__(self, kv_pool: KVPool, indices: array.array, len_kv_cache: int):
        super().__init__(kv_pool, indices, len_kv_cache)

        # length of the **actual loading** indices
        # i.e., selected indices + recent tokens generated in drafting stage
        self.len_loading_indices: int = 0

        # length of the selected indices (on device)
        # i.e., Top-K important tokens that are already verified
        self.len_selected_indices_gpu: int = 0

        # length of the selected indices (offloaded on host)
        # Note that (host+device) == actual
        self.len_selected_indices_cpu: int = 0

        # Selected indices pool on GPU
        self.selected_indices_buffer_gpu: torch.Tensor = torch.zeros(
            (self.NUM_LAYERS, self.NUM_KV_HEADS, self.MAX_LEN_SELECTED_KV_CACHE),
            dtype=torch.int32,
            device=kv_pool.device,
        )
        # pointer to the selected indices memory on GPU
        self.selected_indices_ptr = self.selected_indices_buffer_gpu.data_ptr()

        # **virtual** indices on cpu
        self.selected_indices_buffer_cpu = np.ndarray(
            (self.MAX_LEN_SELECTED_KV_CACHE, self.NUM_LAYERS, self.NUM_KV_HEADS),
            dtype=np.int32,
        )

    def __del__(self):
        super().__del__()
        del self.selected_indices_buffer_gpu
        del self.selected_indices_buffer_cpu
        del self.selected_indices_ptr

    @classmethod
    def initialize_class_metadata(cls, **kwargs):
        """
        Initialize the class metadata.
        """
        super().initialize_class_metadata(**kwargs)

        # hardcode for streaming cache
        num_kv_heads = 1
        num_layers = 1

        capcity = kwargs["capacity"]
        spec_cache_args = kwargs["spec_cache_args"]

        # this method should only be called once
        assert not hasattr(cls, "kv_indices_tensor_buf")
        _max_buf_ratio = (
            max(
                spec_cache_args["budget_ratio"],
                spec_cache_args["num_min_budget"]
                * spec_cache_args["max_batch_size"]
                / capcity,
            )
            * 4
        )
        cls.NUM_KV_HEADS = num_kv_heads
        cls.NUM_LAYERS = num_layers
        cls.kv_indices_tensor_buf = torch.empty(
            (cls.NUM_LAYERS * cls.NUM_KV_HEADS * cls.MAX_TOTAL_DRAFT_KV_LEN),
            dtype=torch.int32,
            device=kwargs["device"],
        )

    def allocate_tokens(self, num_tokens: int, restore=False) -> None:
        assert num_tokens == 1 or restore == True

        num_new_pages = super().allocate_tokens(num_tokens)
        assert self.selected_indices_buffer_gpu is not None

        if restore == False:
            """
            Update indices to include recent generated tokens.
            Offloaded requests maintain its own indices.
            """
            self.len_loading_indices += num_new_pages

    def retract_tokens(self, num_tokens: int) -> None:
        """Release Device Memory for the tokens."""
        super().retract_tokens(num_tokens)

    def offload(self, length):
        length = min(length, len(self.indices))
        selected_length = min(length, self.len_selected_indices_gpu)

        assert self.page_size == 1
        assert length > 0

        nvtx.range_push("Allocate pages on CPU")
        for _ in range(length):
            self.indices_cpu.append(self.kv_pool.alloc_page_cpu())
        nvtx.range_pop()

        nvtx.range_push("Offload KV cache")
        self.kv_pool.offload(self.indices[-length:], self.indices_cpu[-length:][::-1])
        nvtx.range_pop()

        if selected_length > 0:
            nvtx.range_push("Copy selected indices from GPU to CPU")
            # Copy the selected indices from GPU to CPU
            start_idx_cpu = self.len_selected_indices_cpu
            end_idx_gpu = self.len_selected_indices_gpu
            self.selected_indices_buffer_cpu[
                start_idx_cpu : start_idx_cpu + selected_length, ...
            ] = (
                self.selected_indices_buffer_gpu[
                    ..., end_idx_gpu - selected_length : end_idx_gpu
                ]
                .flip(dims=[-1])
                .permute(2, 0, 1)
                .cpu()
                .numpy()
            )
            self.len_selected_indices_cpu += selected_length
            self.len_selected_indices_gpu -= selected_length
            nvtx.range_pop()

        # Convert the physical indices of the selected indices on-device to virtual
        # need be called each time to **amortize the overhead**
        nvtx.range_push("Convert physical to virtual indices")
        self.selected_indices_buffer_cpu = convert_physical_to_virtual(
            self.selected_indices_buffer_cpu,
            self.indices,
            self.len_selected_indices_cpu,
            len(self.indices_cpu),
            length,
        )
        nvtx.range_pop()

        nvtx.range_push("Retract tokens")
        self.retract_tokens(length)
        nvtx.range_pop()

        return len(self.indices)

    def restore(self, length):
        length = min(length, len(self.indices_cpu))
        selected_length = min(length, self.len_selected_indices_cpu)

        assert self.page_size == 1
        assert length > 0

        # First reallocate required kv-tokens on GPU
        # which enables virtual to physical
        nvtx.range_push("Allocate pages on GPU")
        self.allocate_tokens(length, restore=True)
        nvtx.range_pop()

        nvtx.range_push("Restore KV cache")
        self.kv_pool.restore(self.indices_cpu[-length:][::-1], self.indices[-length:])
        nvtx.range_pop()

        if selected_length > 0:
            nvtx.range_push("Copy selected indices from CPU to GPU")
            start_idx_gpu = self.len_selected_indices_gpu
            end_idx_cpu = self.len_selected_indices_cpu
            self.selected_indices_buffer_gpu[
                ..., start_idx_gpu : start_idx_gpu + selected_length
            ] = (
                torch.from_numpy(
                    self.selected_indices_buffer_cpu[
                        end_idx_cpu - selected_length : end_idx_cpu, ...
                    ]
                )
                .to(device=self.kv_pool.device, non_blocking=True)
                .permute(1, 2, 0)
                .flip(dims=[-1])
            )
            self.len_selected_indices_gpu += selected_length
            self.len_selected_indices_cpu -= selected_length
            nvtx.range_pop()

        nvtx.range_push("Free CPU indices")
        for _ in range(length):
            self.kv_pool.free_page_cpu(self.indices_cpu.pop())
        nvtx.range_pop()

        nvtx.range_push("Convert virtual address to physical address")
        # virtual to physical indices
        # NOTE (Yilong): this could be perf bottleneck
        if len(self.indices_cpu) == 0:
            selected_indices_buffer_gpu = self.selected_indices_buffer_gpu[
                ..., : self.len_selected_indices_gpu
            ]
            mask = selected_indices_buffer_gpu < 0
            indices_gpu = torch.frombuffer(self.indices, dtype=torch.int32).to(
                device=self.kv_pool.device, non_blocking=True
            )
            selected_indices_buffer_gpu[mask] = indices_gpu[
                selected_indices_buffer_gpu[mask]
            ]
        nvtx.range_pop()

        return len(self.indices_cpu)

    def get_select_metadata(self, **config) -> None:
        """
        This function is the entry point to **update** the sparsity patterns,
        which is frequently called by `scheduler`.
        Note that `len_loading_indices` will be updated during drafting.
        """
        # below is streaming cache
        min_token_budget = config.get("num_min_budget", 128)
        sink_token_count = config.get("num_sink_tokens", 16)
        memory_budget_ratio = config.get("budget_ratio", 0.05)

        token_budget = max(
            min_token_budget, int(self.len_kv_cache * memory_budget_ratio)
        )
        assert token_budget < self.MAX_LEN_SELECTED_KV_CACHE

        len_indices = len(self.indices)
        assert self.page_size == 1
        assert len_indices == self.len_kv_cache

        # Update length of loading indices and selected indices
        # Return metadata for updating the selected indices of all requests
        # at once later to avoid frequent **HtoD overhead**
        if token_budget >= self.len_kv_cache:
            self.len_loading_indices = len_indices
            self.len_selected_indices_gpu = len_indices

            return {
                "ptr": self.selected_indices_ptr,
                "sink": 0,
                "recent": len_indices,
            }
        else:
            # Update length of loading indices and selected indices
            # NOTE (Yilong): recent is still included by the selected_indices_gpu
            # i.e., sink+recent == StreamingLLM
            recent_token_count = token_budget - sink_token_count
            self.len_loading_indices = token_budget
            self.len_selected_indices_gpu = token_budget

            return {
                "ptr": self.selected_indices_ptr,
                "sink": sink_token_count,
                "recent": recent_token_count,
            }

    @classmethod
    def from_kv_cache(cls, source_cache: FullCachePtr, **config):
        assert isinstance(source_cache, FullCachePtr)

        assert source_cache is not None and source_cache.len_kv_cache > 0
        assert len(source_cache.indices) == source_cache.len_kv_cache
        assert source_cache.kv_pool is not None

        streaming_cache = cls(
            source_cache.kv_pool,
            source_cache.indices,
            source_cache.len_kv_cache,
        )
        selected_indices_metadata = streaming_cache.get_select_metadata(**config)

        # avoid automatically free pages
        source_cache.indices = []
        del source_cache

        return streaming_cache, selected_indices_metadata

    @classmethod
    def dispatch_prepare_fwd_metadata(
        cls,
        **kwargs,
    ):
        """
        This function updates the `input_metadata` and `input_metadata_tensors` for model forward,
        specialized for different attention backend.
        """
        input_metadata = kwargs["input_metadata"]
        input_metadata_tensors = kwargs["input_metadata_tensors"]
        active_requests = kwargs["active_requests"]
        device = kwargs["device"]

        # additional metadata
        kv_indptr = array.array("i", [0])
        for request in active_requests:
            kv_indptr.append(kv_indptr[-1] + request.len_loading_kv_cache)

        # append extra metadata
        additional_metadata = {
            "kv_indptr": kv_indptr,
        }
        additional_metadata_tensors = dict()
        for key, arr in additional_metadata.items():
            tensor_data_pinned = torch.frombuffer(arr, dtype=torch.int32)
            tensor_data = tensor_data_pinned.to(device, non_blocking=True)
            additional_metadata_tensors[key] = tensor_data

        # append to input_metadata and input_metadata_tensors
        # this is in-place update
        input_metadata.update(additional_metadata)
        input_metadata_tensors.update(additional_metadata_tensors)

        # calculate concat indices
        # Concatenate [[[selected indices], [recent indices]] for request in active_requests]
        # Draft: [[Selected indices], [Recent indices generated in draft stage]]
        # Normal and Verification: [[None], [All indices]]
        # The first part will be extracted from selected indices pool
        # The second part will be extracted from all kv indices
        nvtx.range_push("Get loading indices on GPU")
        kv_indices_tensor = cls.concat_selected_indices(
            input_metadata,
            input_metadata_tensors,
            active_requests,
            cls.kv_indices_tensor_buf,
            cls.NUM_LAYERS,
            cls.NUM_KV_HEADS,
        )
        nvtx.range_pop()

        # append indices
        input_metadata_tensors["kv_indices"] = kv_indices_tensor

    @classmethod
    def dispatch_update_selected_indices(
        cls,
        wrappers,
        selected_indices_metadatas,
        spec_cache_args,  # current stage
        input_metadata,  # last stage
        device,
    ):
        """
        Update the selected indices of requests whose selected indices need to be updated at once (to avoid HtoD overhead)
        Action: fill the selected indices pools for these requests with the physical memory address of selected kv cache
        Input:
            - selected_indices_metadatas: list of metadata for updating the selected indices of all requests
                - idx of requests whose selected indices need to be updated
                - ptr of its selected indices pool on GPU
                - cache specified information: sink, recent
                - physical memory address of all kv indices on GPU
            - all_kv_indices_gpu: total kv indices on device
            - all_kv_indptr_cpu: indptr of all kv indices on CPU
        """
        if len(selected_indices_metadatas) == 0:
            return

        stride = cls.MAX_LEN_SELECTED_KV_CACHE
        num_layers = cls.NUM_LAYERS
        num_kv_heads = cls.NUM_KV_HEADS
        assert num_layers == 1 and num_kv_heads == 1

        # extract args from `spec_cache_args`
        all_kv_indices_gpu = spec_cache_args["all_kv_indices_gpu"]
        all_kv_indptr_cpu = spec_cache_args["all_kv_indptr_cpu"]

        # offsets of sinks start and recent starts
        sinks_offsets = np.empty(len(selected_indices_metadatas), dtype=np.int32)
        recent_offsets = np.empty(len(selected_indices_metadatas), dtype=np.int32)

        # len of sinks and recent
        len_sinks = np.empty((len(selected_indices_metadatas),), dtype=np.int32)
        len_recents = np.empty((len(selected_indices_metadatas),), dtype=np.int32)

        # ptrs of selected indices pool
        selected_indices_ptrs = np.empty(
            len(selected_indices_metadatas), dtype=np.int64
        )

        nvtx.range_push("Prepare selected indices")
        for i, metadata in enumerate(selected_indices_metadatas):
            idx = metadata["idx"]
            start_idx = all_kv_indptr_cpu[idx]
            end_idx = start_idx + metadata["len_kv_indices"]

            sinks_offsets[i] = start_idx
            recent_offsets[i] = end_idx - metadata["recent"]
            len_sinks[i] = metadata["sink"]
            len_recents[i] = metadata["recent"]
            selected_indices_ptrs[i] = metadata["ptr"]
        nvtx.range_pop()

        nvtx.range_push("Concatenate selected indices")
        sinks_offsets = torch.from_numpy(sinks_offsets).to(device=device)
        recent_offsets = torch.from_numpy(recent_offsets).to(device=device)
        len_sinks = torch.from_numpy(len_sinks).to(device=device)
        len_recents = torch.from_numpy(len_recents).to(device=device)
        selected_indices_ptrs = torch.from_numpy(selected_indices_ptrs).to(
            device=device
        )

        # Concatenate the selected indices
        grid = (
            num_layers * num_kv_heads,
            len(selected_indices_metadatas),
            2 if (num_layers == 1) else 1,
        )
        cls.update_selected_indices_kernel[grid](
            all_kv_indices_gpu,
            sinks_offsets,
            len_sinks,
            recent_offsets,
            len_recents,
            selected_indices_ptrs,
            stride,
        )
        nvtx.range_pop()

    @staticmethod
    def concat_selected_indices(
        input_metadata_host,
        input_metadata_device,
        requests,
        o_kv_indices_buf,  # output concated indices
        num_layers,
        num_kv_heads,
    ):
        assert num_layers == 1 and num_kv_heads == 1
        assert o_kv_indices_buf.ndim == 1

        nvtx.range_push("Prepare loading indices")
        num_requests = len(requests)

        selected_kv_indices_ptrs = np.empty(num_requests, dtype=np.int64)
        len_selected_kv_indices = np.empty(num_requests, dtype=np.int32)
        all_kv_indices_offsets = np.empty(num_requests, dtype=np.int32)
        len_all_kv_indices = np.empty(num_requests, dtype=np.int32)

        all_kv_indptr = input_metadata_host["all_kv_indptr"]
        all_kv_indices_tensor = input_metadata_device["all_kv_indices"]
        kv_indptr_tensor = input_metadata_device["kv_indptr"]

        for idx, request in enumerate(requests):
            if request.status == Request.ReqExecType.DRAFT:
                assert request.kv_cache_ptr.len_selected_indices_cpu == 0
                selected_kv_indices_ptrs[idx] = (
                    request.kv_cache_ptr.selected_indices_ptr
                )
                len_selected_kv_indices[idx] = (
                    request.kv_cache_ptr.len_selected_indices_gpu
                )
                # Recent tokens appended in draft stage
                # Note: `recent` specifically denotes these candidate tokens
                len_recent_indices = (
                    request.kv_cache_ptr.len_loading_indices
                    - request.kv_cache_ptr.len_selected_indices_gpu
                )
                assert len_recent_indices >= 0
                all_kv_indices_offsets[idx] = (
                    all_kv_indptr[idx + 1] - len_recent_indices
                )
                len_all_kv_indices[idx] = len_recent_indices
            elif request.status == Request.ReqExecType.NORMAL:
                selected_kv_indices_ptrs[idx] = 0
                len_selected_kv_indices[idx] = 0
                all_kv_indices_offsets[idx] = all_kv_indptr[idx]
                len_all_kv_indices[idx] = all_kv_indptr[idx + 1] - all_kv_indptr[idx]
            elif request.status == Request.ReqExecType.STALL:
                assert request.kv_cache_ptr.len_selected_indices_cpu == 0
                selected_kv_indices_ptrs[idx] = 0
                len_selected_kv_indices[idx] = 0
                all_kv_indices_offsets[idx] = 0
                len_all_kv_indices[idx] = 0
            elif request.status == Request.ReqExecType.VERIFY:
                assert request.kv_cache_ptr.len_selected_indices_cpu == 0
                selected_kv_indices_ptrs[idx] = 0
                len_selected_kv_indices[idx] = 0
                all_kv_indices_offsets[idx] = all_kv_indptr[idx]
                len_all_kv_indices[idx] = all_kv_indptr[idx + 1] - all_kv_indptr[idx]
            else:
                raise ValueError(f"Unknown request status: {request.status}")
        nvtx.range_pop()

        len_total_kv_indices = np.sum(len_all_kv_indices) + np.sum(
            len_selected_kv_indices
        )
        assert (
            len_total_kv_indices * num_layers * num_kv_heads <= o_kv_indices_buf.numel()
        )
        # slice the output buffer with desired shape
        o_kv_indices = o_kv_indices_buf[
            : len_total_kv_indices * num_layers * num_kv_heads
        ].view(num_layers * num_kv_heads, -1)

        nvtx.range_push("Concatenate loading indices")
        selected_kv_indices_stride = StreamingCachePtr.MAX_LEN_SELECTED_KV_CACHE
        selected_kv_indices_ptrs = torch.from_numpy(selected_kv_indices_ptrs).to(
            device=kv_indptr_tensor.device
        )
        all_kv_indices_offsets = torch.from_numpy(all_kv_indices_offsets).to(
            device=kv_indptr_tensor.device
        )
        len_selected_kv_indices = torch.from_numpy(len_selected_kv_indices).to(
            device=kv_indptr_tensor.device
        )
        len_all_kv_indices = torch.from_numpy(len_all_kv_indices).to(
            device=kv_indptr_tensor.device
        )

        grid = (
            num_requests,
            num_layers * num_kv_heads,
        )
        StreamingCachePtr.concat_kv_indices_contiguous_kernel[grid](
            selected_kv_indices_ptrs,
            selected_kv_indices_stride,
            len_selected_kv_indices,
            all_kv_indices_tensor,
            all_kv_indices_offsets,
            len_all_kv_indices,
            o_kv_indices,
            o_kv_indices.stride(0),
            kv_indptr_tensor,
            num_requests=num_requests,
            BLOCK_SIZE=128,
        )
        nvtx.range_pop()

        return o_kv_indices

    @staticmethod
    @triton.jit
    def update_selected_indices_kernel(
        all_kv_indices,
        sinks_indices_offsets,
        len_sinks_indices,
        recent_indices_offsets,
        len_recent_indices,
        selected_indices_ptrs,
        selected_indices_ptrs_stride,
    ):
        head_id = tl.program_id(0)
        req_id = tl.program_id(1)

        l_sinks = tl.load(len_sinks_indices + req_id)
        l_recent = tl.load(len_recent_indices + req_id)
        total = l_sinks + l_recent

        sinks_off = tl.load(sinks_indices_offsets + req_id)
        recent_off = tl.load(recent_indices_offsets + req_id)

        out_addr = tl.load(selected_indices_ptrs + req_id)
        out_ptr = out_addr.to(tl.pointer_type(tl.int32))

        pid_inner = tl.program_id(2)
        stride = tl.num_programs(2)
        for i in range(pid_inner, total, stride):
            val = (
                tl.load(all_kv_indices + sinks_off + i)
                if i < l_sinks
                else tl.load(all_kv_indices + recent_off + (i - l_sinks))
            )
            tl.store(out_ptr + head_id * selected_indices_ptrs_stride + i, val)

    @staticmethod
    @triton.jit
    def concat_kv_indices_contiguous_kernel(
        selected_indices_ptrs,
        selected_indices_ptrs_stride,
        len_selected_indices,
        all_indices_ptr,
        all_indices_offsets,
        len_all_indices,
        loading_indices_ptr,
        loading_indices_ptr_stride,
        loading_indices_offsets,
        num_requests,
        BLOCK_SIZE: tl.constexpr,
    ):
        # add `num-head` dimention to increase occupancy
        head = tl.program_id(1)
        for req_id in range(tl.program_id(0), num_requests, tl.num_programs(0)):
            l_sel = tl.load(len_selected_indices + req_id)
            l_all = tl.load(len_all_indices + req_id)

            sel_addr = tl.load(selected_indices_ptrs + req_id)
            base_sel_ptr = sel_addr.to(tl.pointer_type(tl.int32))

            all_off = tl.load(all_indices_offsets + req_id)
            out_off = tl.load(loading_indices_offsets + req_id)

            head_sel_ptr = base_sel_ptr + head * selected_indices_ptrs_stride
            head_out_ptr = (
                loading_indices_ptr + head * loading_indices_ptr_stride + out_off
            )

            for i in range(0, l_sel, BLOCK_SIZE):
                offs = i + tl.arange(0, BLOCK_SIZE)
                sel_vals = tl.load(head_sel_ptr + offs, mask=offs < l_sel, other=0)
                tl.store(head_out_ptr + offs, sel_vals, mask=offs < l_sel)

            for i in range(0, l_all, BLOCK_SIZE):
                offs = i + tl.arange(0, BLOCK_SIZE)
                all_vals = tl.load(
                    all_indices_ptr + all_off + offs, mask=offs < l_all, other=0
                )
                tl.store(head_out_ptr + offs + l_sel, all_vals, mask=offs < l_all)
