import array
import math

import flashinfer
import numpy as np
import torch
import torch.cuda.nvtx as nvtx
import triton
import triton.language as tl

from serve.attention.backend import BatchTopKAttention, BatchTopKWrapper
from serve.model.model import KVPool
from serve.request.kv_cache_ptr.base import FullCachePtr
from serve.request.kv_cache_ptr.streaming import StreamingCachePtr
from serve.request.request import Request


class PillarCachePtr(StreamingCachePtr):
    # these constants are used for jit compile
    MAX_TOTAL_DRAFT_KV_LEN = 128 * 1024
    MAX_TOTAL_VERIFY_KV_LEN = 256 * 1024
    MAX_LEN_SELECTED_KV_CACHE = 6 * 1024
    MAX_SPEC_STRIDE = 16
    NUM_QO_HEADS = 1
    HEAD_DIM = 128
    DTYPE = torch.float16

    def __init__(self, kv_pool: KVPool, indices: array.array, len_kv_cache: int):
        assert self._INITIALIZED == 1, "kv_cache_class not initialized"
        super().__init__(kv_pool, indices, len_kv_cache)

    @classmethod
    def initialize_class_metadata(cls, **kwargs):
        # call base init
        super(StreamingCachePtr, cls).initialize_class_metadata(**kwargs)

        # init metadata
        spec_cache_args = kwargs["spec_cache_args"]
        cls.NUM_QO_HEADS = kwargs["num_qo_heads"]
        cls.NUM_KV_HEADS = kwargs["num_kv_heads"]
        cls.NUM_LAYERS = kwargs["num_layers"]
        cls.DTYPE = kwargs["dtype"]
        cls.HEAD_DIM = kwargs["head_dim"]
        cls.MAX_SPEC_STRIDE = spec_cache_args["spec_stride"] + 1  # bonus token

        # init buffers
        # these are double buffer used to avoid race condition w/ cuda graph
        cls.draft_kv_indices_tensor_buf = torch.empty(
            (cls.NUM_LAYERS, cls.NUM_KV_HEADS, cls.MAX_TOTAL_DRAFT_KV_LEN),
            dtype=torch.int32,
            device=kwargs["device"],
        )
        cls.dump_logits_snapshot_tensor_buf = torch.empty(
            (cls.NUM_LAYERS, cls.NUM_KV_HEADS, cls.MAX_TOTAL_VERIFY_KV_LEN),
            dtype=torch.float16,
            device=kwargs["device"],
        )

    def get_select_metadata(self, **config) -> None:
        """
        This function is the entry point to **update** the sparsity patterns,
        which is frequently called by `scheduler`.
        Note that `len_loading_indices` will be updated during drafting.
        """
        # below is streaming cache
        min_token_budget = config.get("num_min_budget", 128)
        memory_budget_ratio = config.get("budget_ratio", 0.05)

        # the right triangle in prefill requests doesn't dump logits
        # thus forcefully included in recent tokens
        num_accepted_tokens = config["num_accepted_tokens"] + 1
        token_budget = max(
            min_token_budget, math.ceil(self.len_kv_cache * memory_budget_ratio)
        )
        len_indices = len(self.indices)

        assert token_budget < self.MAX_LEN_SELECTED_KV_CACHE
        assert self.page_size == 1
        assert len_indices == self.len_kv_cache

        token_budget = min(token_budget, len_indices)
        num_selected_tokens = max(0, token_budget - num_accepted_tokens)
        self.len_loading_indices = token_budget
        self.len_selected_indices_gpu = num_selected_tokens

        return {
            "ptr": self.selected_indices_ptr,
            "num_selected_tokens": num_selected_tokens,
            "num_effective_kv_len": max(0, len_indices - num_accepted_tokens),
        }

    @classmethod
    def from_kv_cache(cls, source_cache: FullCachePtr, **config):
        assert isinstance(source_cache, FullCachePtr)

        assert source_cache is not None and source_cache.len_kv_cache > 0
        assert len(source_cache.indices) == source_cache.len_kv_cache
        assert source_cache.kv_pool is not None

        pillar_cache = cls(
            source_cache.kv_pool,
            source_cache.indices,
            source_cache.len_kv_cache,
        )

        # NOTE(Yilong): this is a cold start
        # forcefully setting all token_budget into recent windows
        pillar_cache.len_loading_indices = source_cache.len_kv_cache
        pillar_cache.len_selected_indices_gpu = 0
        selected_indices_metadata = {
            "ptr": pillar_cache.selected_indices_ptr,
            "num_selected_tokens": pillar_cache.len_selected_indices_gpu,
        }

        # avoid automatically free pages
        source_cache.indices = []
        del source_cache

        return pillar_cache, selected_indices_metadata

    @classmethod
    def dispatch_wrappers_and_metadata(
        cls,
        kv_layout: str,
        page_size: int,
        enable_vanilla_backend: bool,
        device: str,
    ):
        assert not enable_vanilla_backend, "vanilla backend not supported"
        _, metadata = super().dispatch_wrappers_and_metadata(
            kv_layout, page_size, enable_vanilla_backend, device
        )

        # step 1. construct wrappers
        attn_core_wrapper = BatchTopKAttention(
            kv_layout=kv_layout,
            device=device,
            num_kv_heads=cls.NUM_KV_HEADS,
            num_qo_heads=cls.NUM_QO_HEADS,
            head_dim=cls.HEAD_DIM,
            q_data_type=cls.DTYPE,
            kv_data_type=cls.DTYPE,
            MAX_TOTAL_DRAFT_KV_LEN=cls.MAX_TOTAL_DRAFT_KV_LEN,
            MAX_TOTAL_VERIFY_KV_LEN=cls.MAX_TOTAL_VERIFY_KV_LEN,
            MAX_SPEC_STRIDE=cls.MAX_SPEC_STRIDE,
        )
        topk_wrapper = BatchTopKWrapper(
            # NOTE(Yilong): no reuse for race condition
            # buf for topk is quite small (~32MB)
            float_workspace_buffer=None,
            MAX_TOTAL_DRAFT_KV_LEN=cls.MAX_TOTAL_DRAFT_KV_LEN,
            MAX_TOTAL_VERIFY_KV_LEN=cls.MAX_TOTAL_VERIFY_KV_LEN,
            MAX_DRAFT_KV_LEN=cls.MAX_LEN_SELECTED_KV_CACHE,
        )

        # step 2. construct persistent buffers for cuda graph
        # [request_type, verify_kv_indptr, dump_logits, flatten_indices]
        metadata["request_type"] = torch.empty(
            cls._MAX_BATCH_SIZE, dtype=torch.int32, device=device
        )
        metadata["verify_kv_indptr"] = torch.empty(
            cls._MAX_BATCH_SIZE + 1, dtype=torch.int32, device=device
        )
        metadata["dump_logits"] = torch.empty(
            (cls.NUM_LAYERS, cls.NUM_KV_HEADS, cls.MAX_TOTAL_VERIFY_KV_LEN),
            dtype=torch.float16,
            device=device,
        )
        metadata["flatten_indices"] = torch.empty(
            (cls.NUM_LAYERS, cls.NUM_KV_HEADS, cls.MAX_TOTAL_DRAFT_KV_LEN),
            dtype=torch.int32,
            device=device,
        )

        return [attn_core_wrapper, topk_wrapper], metadata

    @classmethod
    def dispatch_attn_plan(
        cls,
        wrappers,
        attn_fwd_metadata: dict,
        kv_pool: KVPool,  # read-only; for parameters
        **kwargs,
    ):
        # call base impl
        super().dispatch_attn_plan(
            wrappers=wrappers,
            attn_fwd_metadata=attn_fwd_metadata,
            kv_pool=kv_pool,
            **kwargs,
        )

        # setup remaining parameters
        # i.e., [request_type, verify_kv_indptr, flatten_indices]
        # skip `dump_logits` as it is output
        request_type = kwargs["request_type"]
        verify_kv_indptr = kwargs["verify_kv_indptr"]
        flatten_indices = kwargs["flatten_indices"]
        flatten_indices_len = kwargs["flatten_indices_len"]

        assert attn_fwd_metadata["request_type"].shape[0] > request_type.shape[0]
        assert (
            attn_fwd_metadata["verify_kv_indptr"].shape[0] > verify_kv_indptr.shape[0]
        )
        assert attn_fwd_metadata["flatten_indices"].shape == flatten_indices.shape
        assert attn_fwd_metadata["flatten_indices"].shape[-1] > flatten_indices_len

        attn_fwd_metadata["request_type"][: request_type.shape[0]] = request_type
        attn_fwd_metadata["verify_kv_indptr"][
            : verify_kv_indptr.shape[0]
        ] = verify_kv_indptr

        # use stride memory copy to avoid extra memory access
        # layout [num_layers, num_kv_heads, n_len] : [,n_len,1]
        attn_fwd_metadata["flatten_indices"][..., :flatten_indices_len] = (
            flatten_indices[..., :flatten_indices_len]
        )

    @classmethod
    def dispatch_attn_fwd(
        cls,
        wrappers,
        attn_fwd_metadata: dict,
        kv_pool: KVPool,  # read-only; for parameters
        **kwargs,
    ):
        assert not kv_pool.vanilla_backend, "vanilla backend not supported"

        # get args
        attn_core_wrapper = wrappers[0]
        q = kwargs["q"]
        layer_idx = kwargs["layer_idx"]

        # step 1. call attention core
        out, lse = attn_core_wrapper.run_attention(
            q,
            kv_pool._physical_mem_gpu[layer_idx],
            attn_fwd_metadata["kv_indices"],
            attn_fwd_metadata["flatten_indices"][layer_idx],
            attn_fwd_metadata["request_type"],
        )

        # step 2. rematerialize scores
        attn_core_wrapper.rematerialize_scores(
            q,
            out,
            lse,
            kv_pool._physical_mem_gpu[layer_idx],
            attn_fwd_metadata["kv_indices"],
            attn_fwd_metadata["request_type"],
            attn_fwd_metadata["verify_kv_indptr"],
            attn_fwd_metadata["dump_logits"][layer_idx],
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
                kv_indptr=_meta["all_kv_indptr"][: _meta["batch_size_host"] + 1],
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
                kv_indptr=_meta["all_kv_indptr"][: _meta["batch_size_host"] + 1],
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
        """
        This function copies `dump_logits` to avoid race condition when enable async.
        """
        dump_logits_len = input_metadata["dump_logits_len"]
        if dump_logits_len > 0:
            assert (
                cls.dump_logits_snapshot_tensor_buf.shape
                == attn_fwd_metadata["dump_logits"].shape
            )
            assert cls.dump_logits_snapshot_tensor_buf.shape[-1] > dump_logits_len
            cls.dump_logits_snapshot_tensor_buf[..., :dump_logits_len] = (
                attn_fwd_metadata["dump_logits"][..., :dump_logits_len]
            )
        return {
            "dump_logits": cls.dump_logits_snapshot_tensor_buf,
        }

    @classmethod
    def dispatch_prepare_fwd_metadata(
        cls,
        **kwargs,
    ):
        """
        Prepare tensor [request_type, verify_kv_indptr, flatten_indices] + indices + indptr
        , skip `dump_logits`. Also prepare scalar `dump_logits_len` and `flatten_indices_len`.
        """
        input_metadata = kwargs["input_metadata"]
        input_metadata_tensors = kwargs["input_metadata_tensors"]
        active_requests = kwargs["active_requests"]
        device = kwargs["device"]

        # existing metadata
        token_indptr_host = input_metadata["token_indptr"]
        all_kv_indptr_host = input_metadata["all_kv_indptr"]
        all_kv_indices_device = input_metadata_tensors["all_kv_indices"]

        # additional metadata
        kv_indptr_host = array.array("i", [])
        request_type_host = array.array("i", [])
        verify_kv_indptr_host = array.array("i", [])

        dump_logits_len: int = 0  # verify indices len
        flatten_indices_len: int = 0  # draft indices len
        nvtx.range_push("Metadata collect")
        for idx, request in enumerate(active_requests):
            request_type_host.append(request.status.value)

            if request.status == Request.ReqExecType.DRAFT:
                kv_indptr_host.append(flatten_indices_len)
                flatten_indices_len += request.len_loading_kv_cache
            else:
                # all other requests do sparse loading from original `all-kv-indices`
                kv_indptr_host.append(all_kv_indptr_host[idx])

            if request.status == Request.ReqExecType.VERIFY:
                if (
                    token_indptr_host[idx + 1] - token_indptr_host[idx]
                ) * cls.NUM_QO_HEADS // cls.NUM_KV_HEADS > 64:
                    # This warning is kernel implementation limit
                    print(
                        f"[Warning]: packed qo-len exceeds 64, which will be truncated"
                    )

                verify_kv_indptr_host.append(dump_logits_len)
                dump_logits_len += request.len_loading_kv_cache
            else:
                verify_kv_indptr_host.append(dump_logits_len)
        nvtx.range_pop()

        # append the last element to align w/ StreamingCachePtr
        kv_indptr_host.append(
            kv_indptr_host[-1] + active_requests[-1].len_loading_kv_cache
        )

        # append extra metadata
        additional_metadata = {
            "kv_indptr": kv_indptr_host,
            "verify_kv_indptr": verify_kv_indptr_host,
            "request_type": request_type_host,
        }
        additional_metadata_tensors = dict()
        nvtx.range_push("Metadata to GPU")
        for key, arr in additional_metadata.items():
            tensor_data_pinned = torch.frombuffer(arr, dtype=torch.int32)
            tensor_data = tensor_data_pinned.to(device, non_blocking=True)
            additional_metadata_tensors[key] = tensor_data
        nvtx.range_pop()

        # append to input_metadata and input_metadata_tensors
        # this is in-place update
        additional_metadata.update(
            {
                "dump_logits_len": dump_logits_len,
                "flatten_indices_len": flatten_indices_len,
            }
        )
        additional_metadata_tensors.update(
            {
                "kv_indices": all_kv_indices_device,  # directly use
                "dump_logits_len": dump_logits_len,  # also send into plan
                "flatten_indices_len": flatten_indices_len,
            }
        )
        input_metadata.update(additional_metadata)
        input_metadata_tensors.update(additional_metadata_tensors)

        # calculate concat draft-indices
        # Concatenate [[[selected indices], [recent indices]] for request in active_requests]
        # Draft: [[Selected indices], [Recent indices generated in draft stage]]
        # Normal and Verification: [[None], [All indices]]
        # The first part will be extracted from selected indices pool
        # The second part will be extracted from all kv indices
        nvtx.range_push("Get draft-flatten-indices on GPU")
        flatten_indices_tensor = cls.concat_selected_indices(
            input_metadata,
            input_metadata_tensors,
            active_requests,
            cls.draft_kv_indices_tensor_buf,
            cls.NUM_LAYERS,
            cls.NUM_KV_HEADS,
        )
        nvtx.range_pop()

        # append flatten indices
        input_metadata_tensors["flatten_indices"] = flatten_indices_tensor

    @classmethod
    def dispatch_update_selected_indices(
        cls,
        wrappers,
        selected_indices_metadatas,
        spec_cache_args,  # current stage
        input_metadata,  # last stage
        device,
    ):
        if len(selected_indices_metadatas) == 0:
            return

        # step 1. apply topk on `dump_logits`
        assert len(wrappers) == 2, "[attn_core_wrapper, topk_wrapper]"
        topk_wrapper = wrappers[1]

        if input_metadata is None:
            # This only happens when all requests are STALL in last stage
            for metadata in selected_indices_metadatas:
                assert metadata["num_selected_tokens"] == 0
            return

        # these are 1 stage prev
        request_type = input_metadata["request_type"]
        verify_kv_indptr = input_metadata["verify_kv_indptr"]
        dump_logits = input_metadata["dump_logits"]

        # these are current stage
        all_kv_indptr_cpu = spec_cache_args["all_kv_indptr_cpu"]
        all_kv_indices_gpu = spec_cache_args["all_kv_indices_gpu"]

        num_verify_requests = request_type.count(Request.ReqExecType.VERIFY.value)
        if num_verify_requests == 0:
            # all selected_indices are from normal requests
            for metadata in selected_indices_metadatas:
                assert metadata["num_selected_tokens"] == 0
            return

        assert input_metadata["dump_logits_len"] > 0
        in_val_start_pos = np.empty(num_verify_requests, dtype=np.int32)
        in_idx_start_pos = np.empty(num_verify_requests, dtype=np.int32)
        in_lens = np.empty(num_verify_requests, dtype=np.int32)
        ks = np.empty(num_verify_requests, dtype=np.int32)
        selected_indices_ptrs = np.empty(num_verify_requests, dtype=np.int64)

        verify_idx = 0
        for metadata in selected_indices_metadatas:
            idx = metadata["idx"]  # cur iter
            prev_idx = metadata["prev_idx"]  # prev iter
            if prev_idx == -1:
                # used to skip async normal requests
                continue
            if request_type[prev_idx] == Request.ReqExecType.VERIFY.value:
                in_val_start_pos[verify_idx] = verify_kv_indptr[prev_idx]
                in_idx_start_pos[verify_idx] = all_kv_indptr_cpu[idx]
                in_lens[verify_idx] = metadata["num_effective_kv_len"]
                ks[verify_idx] = metadata["num_selected_tokens"]
                selected_indices_ptrs[verify_idx] = metadata["ptr"]
                verify_idx += 1
            else:
                assert metadata["num_selected_tokens"] == 0
        del verify_idx

        in_val_start_pos = torch.from_numpy(in_val_start_pos).to(device=device)
        in_idx_start_pos = torch.from_numpy(in_idx_start_pos).to(device=device)
        in_lens = torch.from_numpy(in_lens).to(device=device)
        ks = torch.from_numpy(ks).to(device=device)
        selected_indices_ptrs = torch.from_numpy(selected_indices_ptrs).to(
            device=device
        )

        # call kernel
        _, out_idx = topk_wrapper.run(
            in_val=dump_logits,
            in_idx=all_kv_indices_gpu,
            in_val_start_pos=in_val_start_pos,
            in_idx_start_pos=in_idx_start_pos,
            in_lens=in_lens,
            ks=ks,
            num_requests=num_verify_requests,
        )

        # step 2. update selected indices
        assert out_idx.shape[-1] == cls.MAX_LEN_SELECTED_KV_CACHE
        grid = (
            cls.NUM_LAYERS * cls.NUM_KV_HEADS,
            num_verify_requests,
            1,
        )
        cls.update_selected_indices_kernel[grid](
            out_idx,
            ks,
            selected_indices_ptrs,
            cls.MAX_LEN_SELECTED_KV_CACHE,
        )

    @staticmethod
    def concat_selected_indices(
        input_metadata_host,
        input_metadata_device,
        requests,
        o_kv_indices_buf,
        num_layers,
        num_kv_heads,
    ):
        """
        This function is used to concatenate `draft indices`, by [[selected indices], [recent indices]]
        Only used for draft requests. Other requests use existing `kv-indices`.
        """
        assert o_kv_indices_buf.ndim == 3, "[num_layers, num_kv_heads, n_len]"
        num_draft_requests = sum(
            request.status == Request.ReqExecType.DRAFT for request in requests
        )
        if num_draft_requests == 0:
            return o_kv_indices_buf

        selected_kv_indices_ptrs = np.empty(num_draft_requests, dtype=np.int64)
        len_selected_kv_indices = np.empty(num_draft_requests, dtype=np.int32)
        all_kv_indices_offsets = np.empty(num_draft_requests, dtype=np.int32)
        len_all_kv_indices = np.empty(num_draft_requests, dtype=np.int32)
        draft_kv_indptr = np.empty(num_draft_requests, dtype=np.int32)

        kv_indptr = input_metadata_host["kv_indptr"]
        all_kv_indptr = input_metadata_host["all_kv_indptr"]
        all_kv_indices_tensor = input_metadata_device["all_kv_indices"]

        draft_idx = 0
        for idx, request in enumerate(requests):
            if request.status == Request.ReqExecType.DRAFT:
                # no offloaded indices
                assert request.kv_cache_ptr.len_selected_indices_cpu == 0
                selected_kv_indices_ptrs[draft_idx] = (
                    request.kv_cache_ptr.selected_indices_ptr
                )
                len_selected_kv_indices[draft_idx] = (
                    request.kv_cache_ptr.len_selected_indices_gpu
                )
                # Recent tokens appended in draft stage
                len_recent_indices = (
                    request.kv_cache_ptr.len_loading_indices
                    - request.kv_cache_ptr.len_selected_indices_gpu
                )
                assert len_recent_indices >= 0
                all_kv_indices_offsets[draft_idx] = (
                    all_kv_indptr[idx + 1] - len_recent_indices
                )
                len_all_kv_indices[draft_idx] = len_recent_indices
                draft_kv_indptr[draft_idx] = kv_indptr[idx]
                draft_idx += 1
        del draft_idx

        # sanity check
        len_total_kv_indices = np.sum(len_all_kv_indices) + np.sum(
            len_selected_kv_indices
        )
        assert len_total_kv_indices < o_kv_indices_buf.shape[-1]
        assert o_kv_indices_buf.shape[-1] == PillarCachePtr.MAX_TOTAL_DRAFT_KV_LEN

        # call kernel
        selected_kv_indices_ptrs = torch.from_numpy(selected_kv_indices_ptrs).to(
            device=o_kv_indices_buf.device
        )
        all_kv_indices_offsets = torch.from_numpy(all_kv_indices_offsets).to(
            device=o_kv_indices_buf.device
        )
        len_selected_kv_indices = torch.from_numpy(len_selected_kv_indices).to(
            device=o_kv_indices_buf.device
        )
        len_all_kv_indices = torch.from_numpy(len_all_kv_indices).to(
            device=o_kv_indices_buf.device
        )
        draft_kv_indptr = torch.from_numpy(draft_kv_indptr).to(
            device=o_kv_indices_buf.device
        )

        grid = (
            num_draft_requests,
            num_layers * num_kv_heads,
        )
        StreamingCachePtr.concat_kv_indices_contiguous_kernel[grid](
            selected_kv_indices_ptrs,
            PillarCachePtr.MAX_LEN_SELECTED_KV_CACHE,
            len_selected_kv_indices,
            all_kv_indices_tensor,
            all_kv_indices_offsets,
            len_all_kv_indices,
            o_kv_indices_buf,
            o_kv_indices_buf.shape[-1],
            draft_kv_indptr,
            num_requests=num_draft_requests,
            BLOCK_SIZE=128,
        )

        return o_kv_indices_buf

    @staticmethod
    @triton.jit
    def update_selected_indices_kernel(
        selected_kv_indices,
        len_selected_indices,
        selected_indices_ptrs,
        stride,
    ):
        head_id = tl.program_id(0)
        req_id = tl.program_id(1)
        num_heads = tl.num_programs(0)

        l_selected = tl.load(len_selected_indices + req_id)
        out_addr = tl.load(selected_indices_ptrs + req_id)
        input_ptr = selected_kv_indices + (req_id * num_heads + head_id) * stride
        out_ptr = out_addr.to(tl.pointer_type(tl.int32)) + head_id * stride

        pid_inner = tl.program_id(2)
        inner_stride = tl.num_programs(2)
        for i in range(pid_inner, l_selected, inner_stride):
            val = tl.load(input_ptr + i)
            tl.store(out_ptr + i, val)
