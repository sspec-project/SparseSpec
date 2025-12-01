# Copyright (c) 2025
#
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
#
# -----------------------------------------------------------------------------
# This file includes portions of code adapted from the vLLM project:
#   "A high-throughput and memory-efficient inference and serving engine
#    for LLMs" (https://docs.vllm.ai)
# Licensed under the Apache License, Version 2.0
# -----------------------------------------------------------------------------


import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from serve.sampling.fast_fused_op import triton_fused_sampling, triton_strided_argmax

logger = logging.getLogger(__name__)

_SAMPLING_EPS = 1e-5


@dataclass
class SamplingMetadata:
    """
    Current implementation: entire batch shares the same.
    """

    temperature: float
    is_greedy_sampling: bool


class TopKTopPSampler:
    """
    Module that performs optional top-k and top-p filtering followed by
    weighted random sampling of logits.

    Implementations may update the logits tensor in-place.
    """

    def __init__(self) -> None:
        self.forward = self.forward_native

    def forward_native(
        self,
        logits: torch.Tensor,
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        PyTorch-native implementation of top-k and top-p sampling.

        Return next-tokens and probs.
        """
        assert k is None and p is None  # Consider probs w/ k and p
        logits = TopKTopPSampler.apply_top_k_top_p(logits, k, p)
        probs = logits.softmax(dim=-1, dtype=torch.float32)

        return TopKTopPSampler.random_sample(probs), probs

    @staticmethod
    def apply_top_k_top_p(
        logits: torch.Tensor,
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if p is None:
            if k is None:
                return logits

            # Avoid sorting vocab for top-k only case.
            return apply_top_k_only(logits, k)

        logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

        if k is not None:
            # Apply top-k.
            top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
            # Get all the top_k values.
            top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
            top_k_mask = logits_sort < top_k_mask
            logits_sort.masked_fill_(top_k_mask, -float("inf"))

        if p is not None:
            # Apply top-p.
            probs_sort = logits_sort.softmax(dim=-1)
            probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
            top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
            # at least one
            top_p_mask[:, -1] = False
            logits_sort.masked_fill_(top_p_mask, -float("inf"))

        # Re-sort the probabilities.
        logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
        return logits

    @staticmethod
    def random_sample(
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """Randomly sample from the probabilities.

        We use this function instead of torch.multinomial because torch.multinomial
        causes CPU-GPU synchronization.
        """
        q = torch.empty_like(probs)
        q.exponential_()

        scores = probs / q
        return scores.argmax(dim=-1).view(-1)


class Sampler(nn.Module):
    """
    A layer that samples the next tokens from the model's outputs
    with the following steps in order:
    1. Apply temperature
    2. Sampling the next tokens
    3. Return the sampled tokens and probs
    """

    def __init__(self, tp_ranks: int = 1):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler()
        # NOTE: `tp_ranks` is used to do fused sampling w/ TP lm-head
        self.tp_ranks = tp_ranks

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        clip_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # NOTE: clip padded cuda graph tokens here
        # ref to `serve/model/graph_model.py: _CUDAGraphModelRunner`
        assert logits.shape[0] >= clip_length

        # we pass vllm implementation of sample w/ customized TP strided kernels
        if sampling_metadata.is_greedy_sampling:
            greedy_sampled = triton_strided_argmax(logits, self.tp_ranks).long()
            probs = torch.zeros_like(logits, dtype=torch.float32)
            probs.scatter_(dim=-1, index=greedy_sampled.unsqueeze(-1), value=1.0)
            return greedy_sampled[:clip_length], probs[:clip_length]

        assert sampling_metadata.temperature > 0.0
        q = torch.empty(
            logits.shape,
            dtype=torch.float32,
            device=logits.device,
        )
        q.exponential_()
        random_sampled, probs = triton_fused_sampling(
            logits=logits,
            q=q,
            temperature=sampling_metadata.temperature,
            tp_ranks=self.tp_ranks,
        )

        return random_sampled[:clip_length], probs[:clip_length]

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """

        if sampling_metadata.is_greedy_sampling:
            greedy_sampled = self.greedy_sample(logits)
            probs = torch.zeros_like(logits)
            probs.scatter_(dim=-1, index=greedy_sampled.unsqueeze(-1), value=1.0)
            return greedy_sampled, probs

        assert sampling_metadata.temperature >= 0.0

        # Apply temperature.
        logits = self.apply_temperature(logits, sampling_metadata.temperature)
        # Apply top_k and/or top_p.
        random_sampled, probs = self.topk_topp_sampler.forward(
            logits,
            None,
            None,
        )

        return random_sampled, probs

    def apply_temperature(
        self,
        logits: torch.Tensor,
        temp: float,
    ) -> torch.Tensor:
        # Use in-place division to avoid creating a new tensor.
        return logits.div_(temp)

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)
