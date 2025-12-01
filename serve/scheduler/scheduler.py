import array
import logging
import time
from typing import Dict, List, Tuple

import torch
import torch.cuda.nvtx as nvtx

from serve.model.model import KVPool
from serve.profiler import Profiler
from serve.request.request import Request
from serve.sampling.sampler import Sampler, SamplingMetadata
from serve.utils import (
    get_local_rank,
    index_copy,
    interval_trigger,
    is_first_rank,
    wandb_logger,
)

logger = logging.getLogger(__name__)

# For GPU and CPU overlaping
stream_forward = torch.cuda.Stream(device=f"cuda:{get_local_rank()}")


class BaseScheduler:
    def __init__(
        self,
        kv_pool: KVPool,
        model,
        **kwargs,
    ):
        self.active_requests: List[Request] = []
        self.offload_requests: List[Request] = []

        # stack head offloading request
        self.offloading_request: Request = None
        # stack head chunked prefill request
        self.prefilling_request: Request = None

        self.kv_pool = kv_pool
        self.model = model
        self.max_batch_size = kwargs["max_batch_size"]
        self.eos_token_id = kwargs["eos_token_id"]
        self.admit_policy = kwargs["admit_policy"]
        self.async_on = kwargs["async_cpu"]
        self._args = kwargs

        # construct sampler
        self.sampler = Sampler(tp_ranks=kwargs["tp_ranks"])
        self.temperature = kwargs["temperature"]
        self.sampling_metadata = SamplingMetadata(
            temperature=self.temperature,
            is_greedy_sampling=(self.temperature == 0.0),
        )

        # init profiler
        self.prof = Profiler(
            tag="schedule",
            enable=kwargs["enable_torch_profiler"] and is_first_rank(),
            wait=4480,
            warmup=10,
            active=40,
            repeat=1,
            result_dir=kwargs["profiler_result_dir"],
        )

    @property
    def total_kv_cache(self) -> int:
        """Count the total number of KV tokens in the active requests."""
        return sum(req.len_kv_cache for req in self.active_requests)

    @property
    def total_loading_kv_cache(self) -> int:
        """Count the total number of loading KV tokens in the active requests."""
        return sum(req.len_loading_kv_cache for req in self.active_requests)

    @property
    def kv_cache_usage_ratio(self) -> float:
        """Calculate the KV capacity usage as a ratio."""
        total_capacity = self.kv_pool.capacity * self.kv_pool.page_size
        return self.total_kv_cache / total_capacity if total_capacity > 0 else 0.0

    @property
    def loading_kv_cache_usage_ratio(self) -> float:
        """Calculate the loading KV capacity usage as a ratio."""
        total_capacity = self.kv_pool.capacity * self.kv_pool.page_size
        return (
            self.total_loading_kv_cache / total_capacity if total_capacity > 0 else 0.0
        )

    @property
    def host_kv_cache_usage_ratio(self) -> float:
        """Calculate the host KV capacity usage as a ratio."""
        total_capacity = self.kv_pool._MAX_CPU_CAPACITY * self.kv_pool.page_size
        return (
            (1 - len(self.kv_pool._free_cpu) / total_capacity)
            if total_capacity > 0
            else 0.0
        )

    def add_request(self, request: Request, is_restore: bool = False) -> None:
        self.active_requests.append(request)

    def admit_requests(self, pending_requests: List[Request]) -> None:
        if self.admit_policy == "naive":
            assert False, "Naive policy is deprecated"
            token_capcaity = self.kv_pool.capacity * self.kv_pool.page_size
            tokens_needed_sum = sum(
                min(req.len_token_ids - req.len_kv_cache, self._args["chunk_size"])
                for req in self.active_requests
            )
            available_capacity = (
                token_capcaity - self.total_kv_cache - tokens_needed_sum
            )
            # Retract reqeusts if needed
            if available_capacity < 0:
                self.active_requests.sort(key=lambda x: x.len_kv_cache, reverse=True)
                nvtx.range_push("Retracting requests")
                while available_capacity < 0:
                    request = None
                    for req in self.active_requests[::-1]:
                        if req.status == Request.ReqExecType.STALL:
                            logger.warning(
                                f"Skip retracting STALL request, tweaking the strict monotonous length."
                            )
                            continue
                        request = req
                        break
                    assert request is not None

                    # directly release all the kv cache
                    request.retire()
                    request.attach_kv_pool(self.kv_pool)
                    # retract draft token ids
                    if (
                        request.status == Request.ReqExecType.DRAFT
                        or request.status == Request.ReqExecType.VERIFY
                    ):
                        del request.token_ids[
                            request.initial_idx - request.cur_spec_idx :
                        ]
                    elif self.async_on:
                        assert request.status == Request.ReqExecType.NORMAL
                        # retract the last token, which is dummy token
                        if request.token_ids[-1] == -233:
                            del request.token_ids[-1:]
                    # clear status
                    request.status = Request.ReqExecType.NORMAL
                    request.cur_spec_idx = 0
                    request.initial_idx = None
                    if hasattr(request, "token_indptr"):
                        del request.token_indptr
                    # retract kv cache
                    self.active_requests.remove(request)
                    pending_requests.append(request)

                    available_capacity = (
                        self.kv_pool.num_free_pages * self.kv_pool.page_size
                        - tokens_needed_sum
                    )
                    logger.info(
                        f"Retracted request with generated length {request.len_token_ids-request.len_input}, "
                        f"available capacity: {available_capacity}"
                    )
                nvtx.range_pop()
            elif (
                pending_requests
                and len(self.active_requests) < self.max_batch_size
                and available_capacity > self._args["chunk_size"] * 2  # headroom
            ):
                request = pending_requests.pop()
                self.add_request(request)

        elif self.admit_policy == "oracle":
            current_oracle_ctx_sum = sum(
                req.len_oracle_ctx for req in self.active_requests
            )
            for req in pending_requests[:]:
                if (
                    req.len_oracle_ctx + current_oracle_ctx_sum
                    < self.kv_pool.capacity * self.kv_pool.page_size
                    and len(self.active_requests) < self.max_batch_size
                ):
                    req.attach_kv_pool(self.kv_pool)  # late attach
                    self.add_request(req)
                    current_oracle_ctx_sum += req.len_oracle_ctx
                    pending_requests.remove(req)

                    # lazy admit (coarse grained chunked prefill)
                    break

        elif self.admit_policy == "offloading":
            # Do not include chunk size here to leave headroom
            tokens_needed_sum = sum(
                req.len_token_ids - req.len_kv_cache for req in self.active_requests
            )
            available_capacity = (
                self.kv_pool.num_free_pages * self.kv_pool.page_size - tokens_needed_sum
            )

            # Offload requests if needed
            if available_capacity < 0:
                # Sort active requests by len_kv_cache
                self.active_requests.sort(key=lambda x: x.len_kv_cache, reverse=True)
                nvtx.range_push("Offload requests")
                while available_capacity < 0:
                    if self.offloading_request:
                        self.offloading_request = self.offloading_request.offload()

                        available_capacity = (
                            self.kv_pool.num_free_pages * self.kv_pool.page_size
                            - tokens_needed_sum
                        )
                        logger.info(
                            f"Offloaded request, available capacity: {available_capacity}"
                        )
                    else:
                        # NOTE: Temporary fix for the interupted matching issue
                        # this destory the strict monotonous
                        request = None
                        for req in self.active_requests[::-1]:
                            if req.status == Request.ReqExecType.STALL:
                                logger.warning(
                                    f"Skip offloading STALL request, tweaking the strict monotonous length."
                                )
                                continue
                            request = req
                            break
                        assert request is not None

                        self.offloading_request = request.offload()
                        self.offload_requests.append(request)
                        self.active_requests.remove(request)

                        available_capacity = (
                            self.kv_pool.num_free_pages * self.kv_pool.page_size
                            - tokens_needed_sum
                        )
                        logger.info(
                            f"Offloaded request, available capacity: {available_capacity}"
                        )
                nvtx.range_pop()
            else:
                # If there are offloaded requests, try to restore them
                nvtx.range_push("Restore offloaded requests")
                if (
                    len(self.offload_requests) > 0
                    and len(self.active_requests) < self.max_batch_size
                    and self.offload_requests[-1].MAX_OFFLOADING_LEN
                    < available_capacity
                ):
                    request = self.offload_requests[-1]
                    self.offloading_request = request.restore()
                    if self.offloading_request is None:
                        self.add_request(request, is_restore=True)
                        self.offload_requests.pop()

                    available_capacity = (
                        self.kv_pool.num_free_pages * self.kv_pool.page_size
                        - tokens_needed_sum
                    )
                    logger.info(
                        f"Restored offloaded request, available capacity: {available_capacity}"
                    )
                nvtx.range_pop()

                if len(self.offload_requests) == 0:
                    # If no offloaded requests, try to add pending requests
                    # Only try first one to assure one per-iteration
                    if (
                        len(pending_requests) > 0
                        and len(self.active_requests) < self.max_batch_size
                        and pending_requests[0].len_token_ids < available_capacity
                        and self.prefilling_request is None
                    ):
                        # still exaggerate needed space by ignoring chunk size
                        # leaving headroom
                        request = pending_requests[0]
                        available_capacity -= request.len_token_ids

                        # late attach kv_pool to reduce peak memory
                        request.attach_kv_pool(self.kv_pool)
                        self.prefilling_request = request
                        self.add_request(request)
                        pending_requests.remove(request)
        else:
            raise ValueError(f"Unknown admit policy: {self.admit_policy}")

    def is_request_complete(self, request: Request, check_idx=-1) -> bool:
        assert request.status == Request.ReqExecType.NORMAL

        if len(request.token_ids) + check_idx < 0:
            return False

        return (
            request.len_oracle_ctx is not None
            and request.len_kv_cache + check_idx >= request.len_oracle_ctx - 1
        ) or request.token_ids[check_idx] in self.eos_token_id

    def prepare_forward_inputs(
        self,
    ):
        """
        This function iterates on all active requests, obtains corresponding metadata,
        such as qo_indtpr, kv_indtpr, kv_indices, etc.
        """
        token_ids = array.array("i")
        token_indptr = array.array("i", [0])
        pos_offsets = array.array("i")

        kv_indices = array.array("i")
        kv_indptr = array.array("i", [0])
        kv_len_arr = array.array("i")

        # Collect metadata on CPU
        for request in self.active_requests:
            # sanity-check
            assert not self.is_request_complete(request, -2 if self.async_on else -1)
            assert request.status == Request.ReqExecType.NORMAL

            # chunked prefill
            num_tokens = min(
                request.len_token_ids - request.len_kv_cache, self._args["chunk_size"]
            )
            request.kv_cache_ptr.allocate_tokens(num_tokens)

            token_ids.extend(
                request.token_ids[
                    request.len_kv_cache - num_tokens : request.len_kv_cache
                ]
            )
            token_indptr.append(token_indptr[-1] + num_tokens)
            pos_offsets.append(request.len_kv_cache - num_tokens)

            kv_indices.extend(request.kv_cache_ptr.indices)
            kv_indptr.append(kv_indptr[-1] + len(request.kv_cache_ptr.indices))
            kv_len_arr.append(request.len_kv_cache)

        input_metadata = {
            "token_ids": token_ids,
            "token_indptr": token_indptr,
            "pos_offsets": pos_offsets,
            "kv_indices": kv_indices,
            "kv_indptr": kv_indptr,
            "kv_len_arr": kv_len_arr,
        }

        # Send metadata to GPU
        input_metadata_tensors = dict()
        nvtx.range_push("Metadata to GPU")
        for key, arr in input_metadata.items():
            nvtx.range_push(f"Array to tensor")
            tensor_data_pinned = torch.frombuffer(arr, dtype=torch.int32)
            nvtx.range_pop()
            tensor_data = tensor_data_pinned.to(self.model.device, non_blocking=True)
            input_metadata_tensors[key] = tensor_data
        nvtx.range_pop()

        return input_metadata, input_metadata_tensors

    def update_requests(
        self,
        token_indptr: array.array,
        gen_token_ids_gpu: torch.Tensor,
        gen_token_probs_gpu: torch.Tensor,
        input_metadata: dict,
    ) -> List[Request]:
        """
        This function updates the requests' metadata immediately after calling forward,
        which is necessary for `preparing_forward_inputs` that needs length information.
        Put dummy token_id if async is on.
        """
        completed_requests: List[Request] = []
        remaining_requests: List[Request] = []

        # Get all gen_token_ids from GPU at once
        gen_token_ids_cpu = (
            gen_token_ids_gpu.cpu().numpy() if not self.async_on else None
        )

        for idx, request in enumerate(self.active_requests):
            assert request.status == Request.ReqExecType.NORMAL
            # Step 1: update requests token-id
            # Only update for new token ids
            # keep same if **chunked prefill** enabled
            if request.len_kv_cache == request.len_token_ids:
                # If async is on, preallocate space for the last token
                # After the forward pass, the last token will be filled with correct value
                # Just set it to a **dummy negative value** to raise an error
                # if performing the next forward without updating the correct value
                request.token_ids.append(
                    gen_token_ids_cpu[token_indptr[idx + 1] - 1]
                    if not self.async_on
                    else -233
                )

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

            # Step 2: check if request is complete
            if self.is_request_complete(request, -2 if self.async_on else -1):
                request.retire()
                completed_requests.append(request)
            else:
                # Step 3: store `token_indptr` pair for update token-ids
                # store the token_indptr for updating the token_ids after the forward pass
                # Only update requests if fully prefilled
                if self.async_on and request.prefilled:
                    request.token_indptr = (token_indptr[idx], token_indptr[idx + 1])
                remaining_requests.append(request)

        self.active_requests = remaining_requests
        return completed_requests

    def update_metadata_device(
        self,
        input_metadata_cpu: Dict[str, torch.Tensor],
        input_metadata_gpu: Dict[str, torch.Tensor],
        gen_token_ids_gpu: torch.Tensor,
        execute_stream: torch.cuda.Stream,
        control_stream: torch.cuda.Stream,
    ) -> None:
        """
        Update the dummy token-ids on GPU w/ real token-ids.
        Guarantee dependency by stream barriers.
        NOTE(Yilong): separate device / host update to avoid D2H on critical path.
        """
        token_ids_gpu = input_metadata_gpu["token_ids"]
        token_indptr_cpu = input_metadata_cpu["token_indptr"]

        # Collect indices for updating metadata token_ids on GPU at once
        nvtx.range_push("Collect idx")
        src_idx = array.array("i")
        des_idx = array.array("i")
        for idx, request in enumerate(self.active_requests + self.offload_requests):
            # only iterate without modification
            if (
                hasattr(request, "token_indptr")
                and request.status != Request.ReqExecType.STALL
            ):
                assert request.status in (
                    Request.ReqExecType.NORMAL,
                    Request.ReqExecType.DRAFT,
                    Request.ReqExecType.VERIFY,
                )

                # Update token_ids with generated tokens
                _, end_idx = request.token_indptr
                assert request.token_ids[-1] == -233, "dummy token_id should be set"

                if idx < len(self.active_requests):
                    # token_indptr_cpu is the position of the new metadata
                    # request.token_indptr is the position of generated tokens in the previous forward pass
                    des_idx.append(token_indptr_cpu[idx + 1] - 1)
                    src_idx.append(end_idx - 1)
        nvtx.range_pop()

        # Update token_ids metadata on GPU at once
        if len(src_idx) > 0:
            # Step 1. launch H2D on control stream instead of critical path
            with torch.cuda.stream(control_stream):
                src_idx_device = torch.frombuffer(src_idx, dtype=torch.int32).to(
                    token_ids_gpu.device, non_blocking=True
                )
                des_idx_device = torch.frombuffer(des_idx, dtype=torch.int32).to(
                    token_ids_gpu.device, non_blocking=True
                )

            # Step 2. launch barriers
            execute_stream.wait_stream(control_stream)

            # Step 3. launch D2D copy on critical path
            with torch.cuda.stream(execute_stream):
                index_copy(
                    dst=token_ids_gpu,
                    src=gen_token_ids_gpu,
                    src_idx=src_idx_device,
                    des_idx=des_idx_device,
                )

    def update_metadata_host(
        self,
        input_metadata_cpu: Dict[str, torch.Tensor],
        input_metadata_gpu: Dict[str, torch.Tensor],
        gen_token_ids_gpu: torch.Tensor,
    ) -> None:
        """
        Update the dummy token_ids on CPU w/ real token_ids.
        Called right after forward pass, before `update_requests`
        """
        gen_token_ids_cpu = gen_token_ids_gpu.cpu().numpy()

        for idx, request in enumerate(self.active_requests + self.offload_requests):
            # if has attribute token_indptr, its token_ids and metadata needs to be updated
            # even if offloaded. Specifically, STALL requests are not executed in this iter,
            # which is properly handled in **update_requests**.
            if (
                hasattr(request, "token_indptr")
                and request.status != Request.ReqExecType.STALL
            ):
                assert request.status in (
                    Request.ReqExecType.NORMAL,
                    Request.ReqExecType.DRAFT,
                    Request.ReqExecType.VERIFY,
                )

                # Update token_ids with generated tokens
                _, end_idx = request.token_indptr
                del request.token_indptr  # clean states

                assert request.token_ids[-1] == -233, "dummy token_id should be set"
                request.token_ids[-1] = gen_token_ids_cpu[end_idx - 1]

    def run(self, pending_requests: List[Request]) -> List[Request]:
        from tqdm import tqdm

        completed_requests: List[Request] = []
        total_requests = len(pending_requests)

        # add request id for every request
        for idx, request in enumerate(pending_requests):
            request.request_id = idx

        _timestamp = time.perf_counter()

        # For async, the generated token ids tensor from the previous forward pass
        last_gen_token_ids_tensor = None
        last_gen_token_probs_tensor = None
        last_gen_input_metadata = None
        output_snapshot_barrier = None
        pbar = tqdm(total=total_requests, desc="Specgen inference")
        while pending_requests or self.active_requests or self.offload_requests:
            # Admit new requests
            nvtx.range_push("Admit requests")
            self.admit_requests(pending_requests)
            nvtx.range_pop()

            assert (
                len(self.active_requests)
                + len(pending_requests)
                + len(completed_requests)
                + len(self.offload_requests)
                == total_requests
            )

            nvtx.range_push("Prepare forward inputs")
            input_metadata, input_metadata_tensors = self.prepare_forward_inputs()
            nvtx.range_pop()

            if input_metadata is not None:
                # Feed in input metadata if input_metadata is not None
                self.model.register_input(**input_metadata_tensors)

                nvtx.range_push("Log scheduler metrics")
                with interval_trigger(tag="scheduler_heartbead", ntimes=100) as fire:
                    if fire:
                        logger.info(
                            f"[Heartbeat] step: {self.prof._step}, batch size {len(self.active_requests)}, GEMM bsz {len(input_metadata['token_ids'])}, "
                            f"KV-Cache Utils: {self.kv_cache_usage_ratio*100:.2f}%, Host KV-Cache Utils: {self.host_kv_cache_usage_ratio*100:.2f}%, "
                            f"remaining: {len(pending_requests)}, completed: {len(completed_requests)}, offloaded: {len(self.offload_requests)}"
                        )
                self.log_scheduler_metrics(
                    pending_requests,
                    completed_requests,
                    len(input_metadata_tensors["token_ids"]),
                )
                nvtx.range_pop()

                # If async is on, update the metadata on GPU after the previous forward pass finished for next forward pass
                if self.async_on:
                    if last_gen_token_ids_tensor is not None:
                        nvtx.range_push("Update metadata device")
                        self.update_metadata_device(
                            input_metadata,
                            input_metadata_tensors,
                            last_gen_token_ids_tensor,
                            execute_stream=stream_forward,
                            control_stream=torch.cuda.current_stream(),
                        )
                        nvtx.range_pop()
                    # Make sure the next forward pass that we are about to
                    # enqueue on `stream_forward` happens **after** the
                    # metadata update we just scheduled on the default stream.
                    # Otherwise the model may read stale token ids.
                    stream_forward.synchronize()
                    stream_forward.wait_stream(torch.cuda.current_stream())

                nvtx.range_push("Forward pass")
                # New cuda stream for async on
                with (
                    torch.cuda.stream(stream_forward)
                    if self.async_on
                    else torch.cuda.stream(torch.cuda.current_stream())
                ):
                    # Actual model forward pass
                    gen_padded_token_logits_tensor = self.model(
                        **input_metadata_tensors
                    )
                    # CUDA Graph could potentially return padded logits
                    # use sampler to clip out padded tokens
                    # Implement in a hacky way to leverage the TP-friendly layout
                    gen_token_ids_tensor, gen_token_probs_tensor = self.sampler(
                        logits=gen_padded_token_logits_tensor,
                        sampling_metadata=self.sampling_metadata,
                        clip_length=input_metadata_tensors["token_ids"].shape[0],
                    )
                    # unbind graph output to avoid race condition
                    del gen_padded_token_logits_tensor

                    gen_token_ids_tensor = gen_token_ids_tensor.to(torch.int32)
                    # NOTE (Yilong): snapshot to avoid race condition
                    # only used for dump-logits in Top-K cache rightnow
                    if output_snapshot_barrier is not None:
                        # sync to avoid race condition
                        output_snapshot_barrier.wait()
                    input_metadata.update(self.kv_pool.output_snapshot(input_metadata))
                nvtx.range_pop()

                # If async is on, update the metadata on CPU on control stream
                # Separate from device update to avoid critical path bottleneck
                if self.async_on:
                    if last_gen_token_ids_tensor is not None:
                        nvtx.range_push("Update metadata host")
                        self.update_metadata_host(
                            input_metadata,
                            input_metadata_tensors,
                            last_gen_token_ids_tensor,
                        )
                        nvtx.range_pop()
            else:
                # syncronize previous run
                # as input is None, no need for another kernel launch
                stream_forward.synchronize()

            nvtx.range_push("Update requests")
            # NOTE (Yilong): Here we got the output from the last iter
            # which has been already used by `update_metadata`
            # 1) add a dummy token_id to normal/draft requests
            # 2) do nothing to verify requests
            # 3) execute verify for verification requests (Specgen)
            gen_token_indptr = (
                input_metadata["token_indptr"] if input_metadata is not None else None
            )
            newly_completed_requests = self.update_requests(
                gen_token_indptr,  # current iter
                last_gen_token_ids_tensor if self.async_on else gen_token_ids_tensor,
                (
                    last_gen_token_probs_tensor
                    if self.async_on
                    else gen_token_probs_tensor
                ),
                last_gen_input_metadata if self.async_on else input_metadata,
            )
            completed_requests.extend(newly_completed_requests)

            last_gen_token_ids_tensor = gen_token_ids_tensor
            last_gen_token_probs_tensor = gen_token_probs_tensor
            last_gen_input_metadata = input_metadata
            output_snapshot_barrier = torch.cuda.current_stream().record_event()
            self.model.kv_cache.step()  # flip phase
            nvtx.range_pop()

            nvtx.range_push("Log throughput metrics")
            # torch-profiler
            self.prof.step()

            # pbar
            if newly_completed_requests:
                pbar.update(len(newly_completed_requests))

            # wangdb logging
            if self._args["kv_cache"] == "full":
                # Specgen TPS is logged in Specgen's log_throughput_metrics
                self.log_throughput_metrics(len(input_metadata["token_ids"]))

            _end_timestamp = time.perf_counter()
            self.log_performance_metrics(_end_timestamp - _timestamp)
            _timestamp = _end_timestamp

            wandb_logger.step()
            self.log_refresh_metrics()  # refresh step-specific metrics
            nvtx.range_pop()

        pbar.close()
        # Sort completed requests by request_id
        completed_requests.sort(key=lambda x: x.request_id)

        return completed_requests

    def log_scheduler_metrics(
        self,
        pending_requests: List[Request],
        completed_requests: List[Request],
        num_gemm_tokens: int,
    ) -> None:
        # Count requests in different states
        draft_requests_count = sum(
            1 for req in self.active_requests if req.status == Request.ReqExecType.DRAFT
        )
        verify_requests_count = sum(
            1
            for req in self.active_requests
            if req.status == Request.ReqExecType.VERIFY
        )
        stall_requests_count = sum(
            1 for req in self.active_requests if req.status == Request.ReqExecType.STALL
        )
        normal_requests_count = sum(
            1
            for req in self.active_requests
            if req.status == Request.ReqExecType.NORMAL
        )

        # Log scheduler metrics
        wandb_logger.log(
            metrics={
                "batch_size": len(self.active_requests),
                "pending_requests": len(pending_requests),
                "completed_requests": len(completed_requests),
                "offloaded_requests": len(self.offload_requests),
                "draft_requests": draft_requests_count,
                "verify_requests": verify_requests_count,
                "normal_requests": normal_requests_count,
                "stall_requests": stall_requests_count,
            },
            categories="scheduler",
            op="replace",
        )

        wandb_logger.log(
            metrics={
                "kv_cache_usage": self.kv_cache_usage_ratio,
                "loading_kv_cache_usage": self.loading_kv_cache_usage_ratio,
                "num_gemm_tokens": num_gemm_tokens,
            },
            categories="engine",
            op="replace",
        )

    def log_throughput_metrics(self, num_acc_tokens: int) -> None:
        wandb_logger.log(
            metrics={
                "generated_tokens": num_acc_tokens,
            },
            categories="engine",
            op="sum",
        )

    def log_performance_metrics(self, step_time) -> None:
        wandb_logger.log(
            metrics={
                "step_time": step_time,
            },
            categories="engine",
            op="replace",
        )

        total_tokens = wandb_logger.log_data.get("engine/generated_tokens", 1)
        wandb_logger.log(
            metrics={
                "tokens_per_second": total_tokens / step_time,
            },
            categories="engine",
            op="replace",
        )

    def log_refresh_metrics(self) -> None:
        # set metric to 0
        # these are step-specific
        wandb_logger.log(
            metrics={
                "generated_tokens": 0,
            },
            categories="engine",
            op="replace",
        )
