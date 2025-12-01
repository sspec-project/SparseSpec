# Copyright (c) vLLM contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified from: https://github.com/vllm-project/vllm
# By happierpig on 03-23-2025

import contextlib
import logging
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from serve.model.model import Transformer
from serve.request.kv_cache_ptr.pillar import PillarCachePtr
from serve.request.request import Request

logger = logging.getLogger(__name__)


class CUDAGraphModelWrapper:
    """
    A wrapper for a Transformer model that uses CUDA Graphs to optimize inference.
    This class provides entry point for inference, which is dispatched into many discrete batch sizes.
    """

    def __init__(self, model: Transformer):
        self.model: Transformer = model
        assert self.model.kv_cache is not None

        self.graph_runners = defaultdict(list)
        self.eager_runner = _EagerModelRunner(model)

        self.captured_bsz: np.ndarray = None
        self.max_padded_ratio: float = 0.0

        # statistic counter
        self.num_total_tokens: int = 0
        self.num_captured_tokens: int = 0
        self.num_padded_tokens: int = 0

    def eager_cuda_graph_mode(
        self,
        max_batch_size: int = 256,
        num_cuda_graphs: int = 16,
        max_padded_ratio: float = 0.1,
    ):
        """
        Capture as many CUDA graphs as possible for the given max batch size.
        """
        assert max_batch_size > 0
        self.max_padded_ratio = max_padded_ratio

        min_batch_size: int = 1

        def cosine_buckets_int(min_x, max_x, N):
            """
            Generate N unique integer values in the range [min_x, max_x] using a cosine distribution.
            Example:
                print(cosine_buckets_int(1, 256, 16))
                output: [1 3 11 22 38 58 80 104 128 153 177 199 219 235 246 254 256]
            """
            i = np.arange(N + 1)
            x_float = min_x + (max_x - min_x) * (1 - np.cos(np.pi * i / N)) / 2
            x_int = np.round(x_float).astype(int)

            unique_vals, indices = np.unique(x_int, return_index=True)
            sorted_order = np.argsort(indices)
            x_int_no_dup = unique_vals[sorted_order]

            return x_int_no_dup

        def heuristic_buckets_int(min_x, max_x, N):
            """
            Generate N unique integer values in the range [min_x, max_x] using a heuristic distribution.
            """
            x_int = [1, 2, 4, 8, 16, 32]
            x_int.extend(list(range(64, max_x, 64)))
            x_int.append(max_x)
            x_int = np.array(x_int)

            if len(x_int) > N:
                x_int = x_int[:N]
                x_int[-1] = max_x
            # unique
            unique_vals, indices = np.unique(x_int, return_index=True)
            sorted_order = np.argsort(indices)
            x_int_no_dup = unique_vals[sorted_order]

            return x_int_no_dup

        captured_args = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640]
        self.capture(captured_args)

    @torch.inference_mode()
    def capture(self, captured_args):
        logger.info(f"Capturing CUDA graphs for batch sizes: {captured_args}")

        torch.cuda.empty_cache()
        mem_size_start_capture: float = torch.cuda.memory_reserved() / 1e9
        mem_size_end_capture = mem_size_start_capture

        for _ in range(2):
            pbar = tqdm(captured_args, desc="Capturing graph")
            # batch_size -> GEMM input batch size
            for batch_size in pbar:
                # Step 1, construct random input
                # simply assume all requests are decode requests
                # NOTE(Yilong): specialized for different kv-cache-ptr backend
                seq_len = 128  # arbitrary dummy length
                num_kv_pages = seq_len // self.kv_cache.page_size * batch_size + 1024
                if num_kv_pages >= self.kv_cache.capacity:
                    # skip this batch, as basic capture already exceeds the capacity
                    logger.warning(
                        f"Needed {num_kv_pages} exceeds the capacity {self.kv_cache.capacity}, "
                        f"skipping batch size {batch_size}..."
                    )
                    continue

                idx = torch.randint(
                    0,
                    self.model.config.vocab_size,
                    (batch_size,),
                    device=self.model.device,
                    dtype=torch.int32,
                )

                qo_indptr_host = [0]
                kv_page_indices_host = list(range(0, num_kv_pages))
                kv_page_indptr_host = [0]
                kv_len_arr_host = []
                pos_offsets_host = []

                for _ in range(batch_size):
                    # assume all requests are decode requests
                    qo_len = 1
                    kv_len = seq_len
                    page_per_seq = (
                        kv_len + self.kv_cache.page_size - 1
                    ) // self.kv_cache.page_size

                    qo_indptr_host.append(qo_indptr_host[-1] + qo_len)
                    kv_page_indptr_host.append(kv_page_indptr_host[-1] + page_per_seq)
                    kv_len_arr_host.append(kv_len)
                    pos_offsets_host.append(kv_len)

                qo_indptr = torch.tensor(
                    qo_indptr_host, device=self.model.device, dtype=torch.int32
                )
                kv_page_indices = torch.tensor(
                    kv_page_indices_host, device=self.model.device, dtype=torch.int32
                )
                kv_page_indptr = torch.tensor(
                    kv_page_indptr_host, device=self.model.device, dtype=torch.int32
                )
                kv_len_arr = torch.tensor(
                    kv_len_arr_host, device=self.model.device, dtype=torch.int32
                )
                pos_offsets = torch.tensor(
                    pos_offsets_host, device=self.model.device, dtype=torch.int32
                )

                kwargs = {
                    "token_ids": idx,
                    "pos_offsets": pos_offsets,
                    "token_indptr": qo_indptr,
                    "kv_indices": kv_page_indices,
                    "kv_indptr": kv_page_indptr,
                    "kv_len_arr": kv_len_arr,
                }

                # Add more metadata if pillar
                # [request_type, verify_kv_indptr, flatten_indices] + [flatten_indices_len]
                if self.kv_cache.kv_cache_class == PillarCachePtr:
                    # just assume all requests are normal decode requests
                    request_type = torch.full(
                        (batch_size,),
                        Request.ReqExecType.NORMAL.value,
                        device=self.model.device,
                        dtype=torch.int32,
                    )
                    verify_kv_indptr = torch.zeros(
                        (batch_size,),
                        device=self.model.device,
                        dtype=torch.int32,
                    )
                    flatten_indices = PillarCachePtr.draft_kv_indices_tensor_buf
                    flatten_indices_len = 0  # no draft requests

                    kwargs.update(
                        {
                            "request_type": request_type,
                            "verify_kv_indptr": verify_kv_indptr,
                            "flatten_indices": flatten_indices,
                            "flatten_indices_len": flatten_indices_len,
                            "all_kv_len_arr": kv_len_arr,
                            "all_kv_indptr": kv_page_indptr,
                        }
                    )

                # Register input first
                self.register_input(**kwargs)

                # Step 2, Warmup
                for _ in range(8):
                    self.eager_runner(**kwargs)

                # Step 3, capture CUDA graph
                graph_runner = _CUDAGraphModelRunner(self.model)
                graph_runner.capture(**kwargs)
                self.graph_runners[batch_size].append(graph_runner)

                mem_size_end_capture = torch.cuda.memory_reserved() / 1e9
                mem_diff = mem_size_end_capture - mem_size_start_capture
                pbar.set_postfix_str(f"Used Mem: {mem_diff:.3f} GB")

            # step for next phase
            self.kv_cache.step()

        self.captured_bsz = np.array(list(self.graph_runners.keys()))
        self.captured_bsz.sort()  # sort, ascending

        logger.info(f"CUDA graph construction finished.")

    @torch.inference_mode()
    def forward(
        self,
        **kwargs,
    ) -> torch.Tensor:
        assert kwargs["pos_offsets"].size(0) == kwargs["kv_len_arr"].size(0)
        assert kwargs["token_indptr"].size(0) == kwargs["kv_indptr"].size(0)

        batch_size = kwargs["token_ids"].size(0)  # GEMM input batch size
        self.num_total_tokens += batch_size

        # default to eager runner
        model_executor = self.eager_runner
        if self.captured_bsz is not None:
            idx = np.searchsorted(self.captured_bsz, batch_size, side="left")
            if (
                idx < len(self.captured_bsz)
                and abs((batch_size - self.captured_bsz[idx]) / batch_size)
                < self.max_padded_ratio
            ):
                # use the CUDA graph
                padded_batch_size = self.captured_bsz[idx]
                assert padded_batch_size >= batch_size
                self.num_padded_tokens += padded_batch_size - batch_size
                self.num_captured_tokens += batch_size

                model_executor = self.graph_runners[padded_batch_size][
                    self.kv_cache.phase
                ]

        return model_executor(**kwargs)

    @torch.inference_mode()
    def register_input(self, **kwargs):
        """
        This function is used to fillin all the corresponding fwd buffers,
        before the actual forward pass happens, to overlap CPU/GPU.
        """
        self.kv_cache.pre_attn_fwd(**kwargs)

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    @property
    def device(self):
        return self.model.device

    @property
    def kv_cache(self):
        return self.model.kv_cache


class _CUDAGraphModelRunner:
    """
    A class that captures and replays CUDA graphs for a Transformer model under certain batch sizes.
    """

    def __init__(self, model: Transformer):
        self.model: Transformer = model
        assert self.model.kv_cache is not None

        self.cuda_graph: torch.cuda.CUDAGraph = None
        self.graph_input: torch.Tensor = None
        self.graph_output: torch.Tensor = None

    def capture(
        self,
        **kwargs,
    ) -> torch.Tensor:
        assert self.cuda_graph is None

        idx = kwargs["token_ids"]
        self.cuda_graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.graph(self.cuda_graph),
            (
                self.model.ccl.capture_context()
                if self.model.ccl is not None
                else contextlib.nullcontext()
            ),
        ):
            out = self.model(idx)

        torch.cuda.synchronize()
        self.graph_input = idx
        self.graph_output = out

    def forward(
        self,
        **kwargs,
    ) -> torch.Tensor:
        idx = kwargs["token_ids"]
        assert self.cuda_graph is not None
        assert self.graph_input.size(0) >= idx.size(0)

        self.graph_input[0 : idx.size(0)] = idx
        self.cuda_graph.replay()

        # NOTE(Yilong): we neither clip to input length, nor clone the buffer
        # 1. use sampler to clip, for efficient lm-head TP layout
        #   ref to `serve/sampling/sampler.py: Sampler.forward`
        # 2. the output will be consumed immediately on the same stream
        #   therefore no race condition
        return self.graph_output

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


class _EagerModelRunner:
    def __init__(self, model: Transformer):
        self.model: Transformer = model
        assert self.model.kv_cache is not None

    def capture(self, x):
        raise NotImplementedError("Does not support capture.")

    @torch.inference_mode()
    def forward(
        self,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(kwargs["token_ids"])

    def __call__(self, **kwargs) -> torch.Tensor:
        return self.forward(**kwargs)

    @property
    def device(self):
        return self.model.device
