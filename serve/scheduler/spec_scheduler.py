import array
import logging
import random
from typing import List, Tuple, Type

import numpy as np
import torch
import torch.cuda.nvtx as nvtx

from serve.model.model import KVPool, Transformer
from serve.request.kv_cache_ptr.base import FullCachePtr
from serve.request.request import Request
from serve.sampling.rejection_sampler import RejectionSampler
from serve.scheduler.scheduler import BaseScheduler
from serve.utils import wandb_logger

logger = logging.getLogger(__name__)


class SpecScheduler(BaseScheduler):
    """
    Speculative scheduler that extends BaseScheduler with speculative decoding capabilities.

    This scheduler implements three execution modes:
    - NORMAL: Regular token generation
    - DRAFT: Speculative token generation
    - VERIFY: Verification of speculative tokens
    """

    def __init__(
        self,
        kv_pool: KVPool,
        model: Transformer,
        **kwargs,
    ):
        super().__init__(
            kv_pool=kv_pool,
            model=model,
            **kwargs,
        )

        assert hasattr(self.kv_pool, "kv_cache_class")
        self.spec_cache_class = self.kv_pool.kv_cache_class
        self.spec_stride = kwargs["spec_stride"]
        self.spec_cache_args = kwargs

        # init RejectionSampler
        assert self.spec_stride + 1 <= RejectionSampler.MAX_SPEC_LEN
        self.rejection_sampler = RejectionSampler()

    def add_request(self, request: Request, is_restore: bool = False) -> None:
        """
        This function puts an idle request into the active request queue,
        whether a new one or a restored one.
        Also, this function assigns an initial (speculative) index to the request.
        """
        self.active_requests.append(request)

        # Trying to permute the speculative steps
        # for unified batch scheudler
        if is_restore:
            assert request.status != Request.ReqExecType.NORMAL
            assert request.initial_idx is not None
        else:
            assert request.initial_idx is None
            if self._args["batch_policy"] == "random":
                request.cur_spec_idx = random.randint(0, self.spec_stride - 1)
            elif self._args["batch_policy"] == "greedy":
                _bucket_hist = np.zeros(self.spec_stride + 1, dtype=np.int32)
                for _req in self.active_requests:
                    if _req.status is not Request.ReqExecType.NORMAL:
                        _bucket_hist[_req.cur_spec_idx] += 1
                # k-1 bucket will become verify next iteration
                # which cannot be selected
                _bucket_hist[self.spec_stride - 1] = 2 * len(self.active_requests)
                _min_idx = (np.argmin(_bucket_hist) + 1) % (self.spec_stride + 1)
                assert _min_idx != self.spec_stride
                request.cur_spec_idx = _min_idx
            elif self._args["batch_policy"] == "naive":
                _bucket_hist = np.zeros(self.spec_stride + 1, dtype=np.int32)
                for _req in self.active_requests:
                    if _req.status is not Request.ReqExecType.NORMAL:
                        _bucket_hist[_req.cur_spec_idx] += 1
                # k-1 bucket will become verify next iteration
                # which cannot be selected
                _bucket_hist[self.spec_stride - 1] = 0
                _max_idx = (np.argmax(_bucket_hist) + 1) % (self.spec_stride + 1)
                assert _max_idx != self.spec_stride
                request.cur_spec_idx = _max_idx
            else:
                raise ValueError(f"Unknown batch policy: {self._args['batch_policy']}")
            request.initial_idx = request.cur_spec_idx

    def is_request_complete(self, request: Request, check_idx=-1) -> bool:
        # sanity-check
        assert check_idx == -1 or self.async_on == True
        if len(request.token_ids) + check_idx < 0:
            # too short
            return False

        if request.status == Request.ReqExecType.NORMAL:
            # consider the dummy token_id by `check_idx`
            return (
                request.len_oracle_ctx is not None
                and request.len_kv_cache + check_idx >= request.len_oracle_ctx - 1
            ) or request.token_ids[check_idx] in self.eos_token_id

        elif request.status == Request.ReqExecType.DRAFT:
            # will not return until verified
            return False

        elif self.async_on and request.status == Request.ReqExecType.VERIFY:
            # `verify_output` will be called in STALL
            return False

        elif request.status in (Request.ReqExecType.VERIFY, Request.ReqExecType.STALL):
            if (
                request.len_oracle_ctx is not None
                and request.len_kv_cache >= request.len_oracle_ctx
            ):
                return True
            # Check for EOS token in last spec_stride+1 tokens
            for i in range(self.spec_stride + 1, 0, -1):
                if len(request.token_ids) - request.len_input < i:
                    break
                if request.token_ids[-i] in self.eos_token_id:
                    # clear tokens after the EOS token
                    request.token_ids = request.token_ids[:-i]
                    return True
            return False
        else:
            assert False, f"Unknown request status: {request.status}"

    def prepare_forward_inputs(
        self,
    ):
        """
        Collect all needed metadata for model forward.
        Collect common variable first, then call specialized function.
        """
        token_ids = array.array("i")
        token_indptr = array.array("i", [0])
        pos_offsets = array.array("i")

        all_kv_indices = array.array("i")
        all_kv_indptr = array.array("i", [0])
        kv_len_arr = array.array("i")
        all_kv_len_arr = array.array("i")

        # for token-probs scatter
        probs_src_start_pos = array.array("i")
        probs_token_num_list = array.array("i")
        probs_tgt_tensor_ptrs = array.array("q")

        nvtx.range_push("Metadata collect")
        for request in self.active_requests:
            # Handle verification mode specially
            if request.status == Request.ReqExecType.VERIFY:
                # include bonus token
                num_tokens = self.spec_stride + 1 - request.initial_idx
                request.kv_cache_ptr.allocate_tokens(1)
            elif request.status in (
                Request.ReqExecType.DRAFT,
                Request.ReqExecType.NORMAL,
            ):
                # chunked prefill
                num_tokens = min(
                    request.len_token_ids - request.len_kv_cache,
                    self._args["chunk_size"],
                )
                request.kv_cache_ptr.allocate_tokens(num_tokens)
                if request.status == Request.ReqExecType.DRAFT:
                    assert request.draft_probs is not None
                    assert request.len_draft_probs < request.draft_probs.shape[0]
                    assert num_tokens == 1

                    probs_src_start_pos.append(token_indptr[-1])
                    probs_token_num_list.append(num_tokens)
                    probs_tgt_tensor_ptrs.append(
                        request.draft_probs[request.len_draft_probs].data_ptr()
                    )
                    request.len_draft_probs += num_tokens  # clear after verify
            elif request.status == Request.ReqExecType.STALL:
                assert self.async_on == True
                # NOTE(Yilong): still append indices for `update_selected_indices`
                num_tokens = 0
            else:
                raise ValueError(f"Unknown request status: {request.status}")

            # Update token pointers and offsets
            if num_tokens > 0:
                token_ids.extend(
                    request.token_ids[
                        request.len_kv_cache - num_tokens : request.len_kv_cache
                    ]
                )
            token_indptr.append(token_indptr[-1] + num_tokens)
            pos_offsets.append(request.len_kv_cache - num_tokens)

            # Update KV cache metadata
            all_kv_indices.extend(request.kv_cache_ptr.indices)
            all_kv_indptr.append(all_kv_indptr[-1] + len(request.kv_cache_ptr.indices))
            kv_len_arr.append(request.len_loading_kv_cache)
            all_kv_len_arr.append(request.len_kv_cache)
        nvtx.range_pop()

        # zero-forward caused by all STALL requests
        if len(token_ids) == 0:
            return None, None

        input_metadata = {
            "token_ids": token_ids,
            "token_indptr": token_indptr,
            "pos_offsets": pos_offsets,
            "all_kv_indices": all_kv_indices,
            "all_kv_indptr": all_kv_indptr,
            "kv_len_arr": kv_len_arr,
            "all_kv_len_arr": all_kv_len_arr,
        }
        input_metadata_tensors = dict()

        nvtx.range_push("Metadata to GPU")
        for key, arr in input_metadata.items():
            tensor_data_pinned = torch.frombuffer(arr, dtype=torch.int32)
            tensor_data = tensor_data_pinned.to(self.model.device, non_blocking=True)
            input_metadata_tensors[key] = tensor_data
        nvtx.range_pop()

        # convert back to numpy
        probs_src_start_pos = np.array(probs_src_start_pos, dtype=np.int32)
        probs_token_num_list = np.array(probs_token_num_list, dtype=np.int32)
        probs_tgt_tensor_ptrs = np.array(probs_tgt_tensor_ptrs, dtype=np.int64)

        input_metadata.update(
            {
                "probs_src_start_pos": probs_src_start_pos,
                "probs_token_num_list": probs_token_num_list,
                "probs_tgt_tensor_ptrs": probs_tgt_tensor_ptrs,
            }
        )

        # collect spec kv metadata w/ in-place update
        self.spec_cache_class.dispatch_prepare_fwd_metadata(
            input_metadata=input_metadata,
            input_metadata_tensors=input_metadata_tensors,
            active_requests=self.active_requests,
            device=self.model.device,
        )

        # NOTE (Yilong): save in `spec-cache-args` rather than `input_metadata`
        # as `normal->draft` and `verify/stall->draft` has different cycles
        # `normal->draft` uses current stage; while `verify/stall->draft` uses previous stage
        # we save STALL metadata twice to solve this
        self.spec_cache_args.update(
            {
                "all_kv_indices_gpu": input_metadata_tensors["all_kv_indices"],
                "all_kv_indptr_cpu": input_metadata["all_kv_indptr"],
            }
        )

        # pop useless metadata
        input_metadata_tensors.pop("all_kv_indices")
        input_metadata.pop("all_kv_indices")

        return input_metadata, input_metadata_tensors

    def verify_outputs(
        self,
        request: Request,
        output_token_ids: list[int],
        **kwargs,
    ):
        """
        This function takes the output token-ids from the rejection sampler,
        and update the requests metadata accordingly by retracting rejected token-ids,
        rejecting kv-caches, and appending the recovered/bonus token-ids.
        """
        spec_tokens_count = request.len_draft_probs
        accepted_tokens = len(output_token_ids)
        num_retracted_kv_cache_tokens = spec_tokens_count + 1 - accepted_tokens

        assert spec_tokens_count > 0  # at least from draft to verify
        assert accepted_tokens > 0  # at least one bonus token
        assert num_retracted_kv_cache_tokens >= 0

        # maintain token-ids
        del request.token_ids[-spec_tokens_count:]
        request.token_ids.extend(output_token_ids)
        # retract rejected kv-cache tokens
        request.kv_cache_ptr.retract_tokens(num_retracted_kv_cache_tokens)
        # Metadata for selected-indices update when everything is accepted.
        kwargs["num_accepted_tokens"] = accepted_tokens
        selected_indices_metadata = request.kv_cache_ptr.get_select_metadata(**kwargs)
        selected_indices_metadata["len_kv_indices"] = request.kv_cache_ptr.len_kv_cache

        # log Specgen metrics
        self.log_throughput_metrics(
            spec_tokens_count, accepted_tokens - 1, accepted_tokens
        )
        return selected_indices_metadata

    def update_requests(
        self,
        token_indptr: array.array,  # current iter
        gen_token_ids_gpu: torch.Tensor,  # following three args are last iter w/ async on
        gen_token_probs_gpu: torch.Tensor,
        input_metadata: dict,
    ) -> List[Request]:
        """
        This function updates the requests' metadata immediately after calling model forward,
        to provide correct kernel configurations for next forward pass.
        1) A dummy token_id will appended if async is on.
        2) If async, nothing happens on verify requests as next cycle they are idle.
            Then `verify_outputs` will be called in STALL.
        3) If sync, execute verify for verification requests.
        """
        completed_requests: List[Request] = []
        remaining_requests: List[Request] = []

        gen_token_ids_cpu = (
            gen_token_ids_gpu.cpu().numpy() if gen_token_ids_gpu is not None else None
        )

        # 1. Scatter the generated token probs from draft requests
        # as there bufs are never offloaded, we just use the collected metadata from last cycle
        if input_metadata is not None:
            probs_src_start_pos = input_metadata["probs_src_start_pos"]
            probs_token_num_list = input_metadata["probs_token_num_list"]
            probs_tgt_tensor_ptrs = input_metadata["probs_tgt_tensor_ptrs"]
            # Consume the generated probs into local buffer by scattering
            if len(probs_src_start_pos) > 0:
                nvtx.range_push("Scatter draft token probs")
                FullCachePtr.scatter_token_probs(
                    src_tensor=gen_token_probs_gpu,
                    src_start_pos=probs_src_start_pos,
                    tgt_tensor_ptrs_host=probs_tgt_tensor_ptrs,
                    token_num_list=probs_token_num_list,
                )
                nvtx.range_pop()

        # 2. Gathering the necessary data for batched verification,
        # including draft_token_ids, bonus_token_id, draft_probs, and target_probs
        nvtx.range_push("Gathering metadata for verification")
        draft_token_ids_host = array.array("i")
        num_draft_tokens_host = array.array("i", [0])  # for cumsum
        bonus_token_ids_host = array.array("i")

        # for draft_probs and target_probs tensor
        draft_probs_src_tensor_ptrs_host = array.array("q")
        target_probs_src_start_pos_host = array.array("i")

        selected_status = (
            Request.ReqExecType.STALL if self.async_on else Request.ReqExecType.VERIFY
        )
        for idx, request in enumerate(self.active_requests):
            if request.status == selected_status:
                if self.async_on:
                    assert request.status == Request.ReqExecType.STALL
                    assert hasattr(request, "token_indptr")
                    start_idx, end_idx = request.token_indptr
                    del request.token_indptr
                else:
                    # Only sync mode can directly access metadata w/ current idx
                    assert request.status == Request.ReqExecType.VERIFY
                    start_idx, end_idx = token_indptr[idx], token_indptr[idx + 1]

                num_draft_tokens = end_idx - start_idx - 1  # excluding bonus token
                assert num_draft_tokens > 0
                assert num_draft_tokens == request.len_draft_probs

                # Note that bonus token-id is not maintained in token_ids before verify
                num_draft_tokens_host.append(num_draft_tokens)
                draft_token_ids_host.extend(request.token_ids[-num_draft_tokens:])
                bonus_token_ids_host.append(gen_token_ids_cpu[end_idx - 1])
                draft_probs_src_tensor_ptrs_host.append(request.draft_probs.data_ptr())
                target_probs_src_start_pos_host.append(start_idx)

        num_draft_tokens_host = np.array(num_draft_tokens_host, dtype=np.int32)
        num_total_draft_tokens = np.sum(num_draft_tokens_host)
        nvtx.range_pop()

        # 3. perform verification if needed
        output_token_ids_host = None
        if num_total_draft_tokens > 0:
            nvtx.range_push("Gather draft and target token probs")
            draft_probs_tensor = torch.empty(
                (num_total_draft_tokens, self.kv_pool.vocab_size),
                dtype=torch.float32,
                device=self.kv_pool.device,
            )
            target_probs_tensor = torch.empty(
                (num_total_draft_tokens, self.kv_pool.vocab_size),
                dtype=torch.float32,
                device=self.kv_pool.device,
            )
            cu_num_draft_tokens_host = np.cumsum(num_draft_tokens_host, dtype=np.int32)
            cu_num_draft_tokens_device = torch.from_numpy(cu_num_draft_tokens_host).to(
                self.kv_pool.device, non_blocking=True
            )
            # 1. gather the draft probs
            FullCachePtr.gather_token_probs(
                src_tensor_ptrs_host=np.array(
                    draft_probs_src_tensor_ptrs_host, dtype=np.int64
                ),
                tgt_tensor=draft_probs_tensor,
                tgt_start_pos=cu_num_draft_tokens_host[:-1],
                token_num_list=num_draft_tokens_host[1:],
            )
            # 2. Copy the target probs from gen_token_probs_gpu
            FullCachePtr.copy_token_probs(
                src_tensor=gen_token_probs_gpu,
                src_start_pos=np.array(target_probs_src_start_pos_host, dtype=np.int32),
                tgt_tensor=target_probs_tensor,
                tgt_start_pos=cu_num_draft_tokens_host[:-1],
                token_num_list=num_draft_tokens_host[1:],
            )
            nvtx.range_pop()

            # 3. perform verification
            nvtx.range_push("Perform verification")
            draft_token_ids_device = torch.frombuffer(
                draft_token_ids_host, dtype=torch.int32
            ).to(self.kv_pool.device, non_blocking=True)
            bonus_token_ids_device = torch.frombuffer(
                bonus_token_ids_host, dtype=torch.int32
            ).to(self.kv_pool.device, non_blocking=True)

            output_token_ids_device = self.rejection_sampler(
                draft_token_ids=draft_token_ids_device,
                num_draft_tokens=num_draft_tokens_host[1:],
                cu_num_draft_tokens=cu_num_draft_tokens_device[1:],
                max_spec_len=RejectionSampler.MAX_SPEC_LEN,
                draft_probs=draft_probs_tensor,
                target_probs=target_probs_tensor,
                bonus_token_ids=bonus_token_ids_device,
            )
            nvtx.range_pop()

            # 4. parse the output into per-request format
            nvtx.range_push("Parse output into per-request format")
            output_token_ids_host = RejectionSampler.parse_output(
                output_token_ids=output_token_ids_device,
                vocab_size=self.kv_pool.vocab_size,
            )
            nvtx.range_pop()

        # Return the metadata for updating selected indices of all requests at once later
        # to avoid frequent HtoD overhead
        nvtx.range_push("Status transition and metadata collect")
        selected_indices_metadatas = []
        # counter for checked verifiable requests
        cnt_verify_idx = 0
        for idx, request in enumerate(self.active_requests):
            # Step 1. Process the generated tokens for this request
            if request.status in (
                Request.ReqExecType.NORMAL,
                Request.ReqExecType.DRAFT,
            ):
                # Do nothing if chunked prefill
                if request.len_kv_cache == request.len_token_ids:
                    # Normal/draft mode: just append the new generated (or **dummy**) token_id
                    if not self.async_on:
                        # only sync mode can directly access metadata w/ current idx
                        request.token_ids.append(
                            gen_token_ids_cpu[token_indptr[idx + 1] - 1]
                        )
                    else:
                        request.token_ids.append(-233)

                    # check whether this is the prefilling request
                    # approve for next chunked requests
                    if (
                        self.prefilling_request is not None
                        and self.prefilling_request == request
                    ):
                        assert request.prefilled
                        self.prefilling_request = None
                else:
                    assert not request.prefilled
            elif request.status == Request.ReqExecType.VERIFY:
                if not self.async_on:
                    # immediately verify if synchronous
                    # will append one more token-id in `verify_outputs`
                    selected_indices_metadata = self.verify_outputs(
                        request,
                        output_token_ids_host[cnt_verify_idx],
                        **self.spec_cache_args,
                    )
                    cnt_verify_idx += 1

                    selected_indices_metadata["idx"] = idx
                    selected_indices_metadata["prev_idx"] = idx  # must be sync
                    selected_indices_metadatas.append(selected_indices_metadata)
                else:
                    # do nothing as the successor STALL will not involve in next cycle
                    # token-id will be maintained in `verify_outputs`
                    pass
            elif request.status == Request.ReqExecType.STALL:
                assert self.async_on == True

                # update token_id in `verify_outputs`
                # we do not use dummy token_id as this is quite dynamic in verification
                selected_indices_metadata = self.verify_outputs(
                    request,
                    output_token_ids_host[cnt_verify_idx],
                    **self.spec_cache_args,
                )
                cnt_verify_idx += 1

                selected_indices_metadata["idx"] = idx
                # must be async, set to prev_idx
                selected_indices_metadata["prev_idx"] = request.prev_idx
                selected_indices_metadatas.append(selected_indices_metadata)
            else:
                raise ValueError(f"Unknown request status: {request.status}")

            # Step 2. Check if the request is complete
            # record token_indptr for `update_metadata`
            if self.is_request_complete(request, -2 if self.async_on else -1):
                request.retire()
                completed_requests.append(request)
                continue
            else:
                # NOTE: first draft after stall will not have update
                # This is to avoid None Type reference error
                if (
                    self.async_on
                    and request.status != Request.ReqExecType.STALL
                    and request.prefilled
                ):
                    # token_indptr is current iter, which is correct
                    request.token_indptr = (
                        token_indptr[idx],
                        token_indptr[idx + 1],
                    )
                remaining_requests.append(request)

            # Step 3. Transition request state for next iteration
            if request.status == Request.ReqExecType.NORMAL:
                # NOTE (Yilong): This potentially can be adaptive
                # Only draft after fully prefilled
                if request.prefilled:
                    assert request.len_draft_probs == 0
                    request.status = Request.ReqExecType.DRAFT

                    # Create speculative cache from current cache
                    request.kv_cache_ptr, selected_indices_metadata = (
                        self.spec_cache_class.from_kv_cache(
                            request.kv_cache_ptr, **self.spec_cache_args
                        )
                    )
                    selected_indices_metadata["idx"] = idx
                    # assume normal directly go to draft. no previous stage
                    selected_indices_metadata["prev_idx"] = -1 if self.async_on else idx
                    selected_indices_metadata["len_kv_indices"] = (
                        request.kv_cache_ptr.len_kv_cache
                    )
                    selected_indices_metadatas.append(selected_indices_metadata)
            elif request.status == Request.ReqExecType.DRAFT:
                request.cur_spec_idx += 1
                assert request.cur_spec_idx <= self.spec_stride
                assert (
                    request.cur_spec_idx - request.initial_idx
                    == request.len_draft_probs
                )
                # Draft → Verify transition (when we've collected enough spec tokens)
                if request.cur_spec_idx == self.spec_stride:
                    request.status = Request.ReqExecType.VERIFY
            elif request.status == Request.ReqExecType.VERIFY:
                assert (
                    request.cur_spec_idx - request.initial_idx
                    == request.len_draft_probs
                )
                request.cur_spec_idx = 0  # clean
                request.initial_idx = 0
                if self.async_on:
                    # avoid blocking CPU
                    request.status = Request.ReqExecType.STALL
                else:
                    # directly verify as synchronous CPU
                    request.len_draft_probs = 0  # update the length
                    request.status = Request.ReqExecType.DRAFT
            elif request.status == Request.ReqExecType.STALL:
                assert self.async_on == True
                request.len_draft_probs = 0  # update the length
                request.status = Request.ReqExecType.DRAFT
            else:
                raise ValueError(f"Unknown request status: {request.status}")
        assert output_token_ids_host is None or cnt_verify_idx == len(
            output_token_ids_host
        )
        assert cnt_verify_idx == len(num_draft_tokens_host) - 1
        nvtx.range_pop()

        # Final step: update the selected indices for all requests at once
        nvtx.range_push("Update selected indices")
        self.spec_cache_class.dispatch_update_selected_indices(
            wrappers=self.kv_pool.wrappers,
            selected_indices_metadatas=selected_indices_metadatas,
            spec_cache_args=self.spec_cache_args,  # current stage
            input_metadata=input_metadata,  # previous stage
            device=self.kv_pool.device,
        )
        nvtx.range_pop()

        # NOTE (Yilong): this is to log prev-step's idx
        # for async correctness of PillarCachePtr
        # as `verify kv indptr` need pre-idx to index
        for idx, request in enumerate(self.active_requests):
            request.prev_idx = idx

        # update active requests
        self.active_requests = remaining_requests
        return completed_requests

    def log_throughput_metrics(
        self, num_total_spec_tokens: int, num_acc_spec_tokens: int, num_acc_tokens: int
    ) -> None:
        wandb_logger.log(
            metrics={
                "generated_tokens": num_acc_tokens,
            },
            categories="engine",
            op="sum",
        )

        if num_total_spec_tokens > 0:
            wandb_logger.log(
                metrics={
                    "total_speculative_tokens": num_total_spec_tokens,
                    "accepted_speculative_tokens": num_acc_spec_tokens,
                },
                categories="spec",
                op="sum",
            )
            accepted_speculative_tokens = wandb_logger.log_data.get(
                "spec/accepted_speculative_tokens", 1
            )
            total_speculative_tokens = wandb_logger.log_data.get(
                "spec/total_speculative_tokens", 1
            )
            acceptance_rate = accepted_speculative_tokens / total_speculative_tokens
            wandb_logger.log(
                metrics={
                    "acceptance_rate": acceptance_rate,
                },
                categories="spec",
                op="replace",
            )
