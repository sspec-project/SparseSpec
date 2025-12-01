import array
from enum import Enum
from typing import Optional, Sequence

import torch

from serve.model.model import KVPool
from serve.request.kv_cache_ptr import FullCachePtr
from serve.sampling.rejection_sampler import RejectionSampler


class Request:
    MAX_OFFLOADING_LEN = 1024

    class ReqExecType(Enum):
        NORMAL = 0
        DRAFT = 1
        VERIFY = 2
        STALL = 3

    def __init__(
        self,
        token_ids: Optional[Sequence[int]],
        desired_output_length: int,
    ):
        # convert list to array
        self.token_ids = array.array("i", token_ids)
        self.len_input = len(token_ids)
        self.len_output = desired_output_length

        self.status: Request.ReqExecType = self.ReqExecType.NORMAL
        self.kv_cache_ptr = None

        self.cur_spec_idx = 0
        self.initial_idx: int = None

        # local buffer for draft token probs
        # maintain on-device to avoid expensive D2H
        self.draft_probs = None
        self.len_draft_probs = 0  # 0-base

    def attach_kv_pool(self, kv_pool: KVPool):
        """
        Init necessary metadata for this request.
        Must be called before execution.
        """
        self.kv_cache_ptr = FullCachePtr(kv_pool)
        assert self.draft_probs is None

        # NOTE(Yilong): late init to avoid memory leak.
        # this tensor is huge and never offloaded
        self.len_draft_probs = 0
        self.draft_probs = torch.empty(
            (RejectionSampler.MAX_SPEC_LEN, kv_pool.vocab_size),
            dtype=torch.float32,
            device=kv_pool.device,
        )

    def retire(self):
        """
        Release kv_cache after completion.
        Called to avoid memory leak.
        """
        del self.kv_cache_ptr
        self.kv_cache_ptr = None
        del self.draft_probs
        self.draft_probs = None
        self.len_draft_probs = 0

    def offload(self):
        rest_len = self.kv_cache_ptr.offload(Request.MAX_OFFLOADING_LEN)
        return self if rest_len != 0 else None

    def restore(self):
        rest_len = self.kv_cache_ptr.restore(Request.MAX_OFFLOADING_LEN)
        return self if rest_len != 0 else None

    @property
    def prefilled(self) -> bool:
        """
        Check whether the request is fully prefilled.
        """
        return self.len_token_ids > self.len_input

    @property
    def len_oracle_ctx(self):
        return self.len_input + self.len_output if self.len_output != None else None

    @property
    def len_kv_cache(self):
        return self.kv_cache_ptr.len_kv_cache

    @property
    def len_token_ids(self):
        return len(self.token_ids)

    @property
    def len_loading_kv_cache(self):
        if self.status in (self.ReqExecType.NORMAL, self.ReqExecType.VERIFY):
            return len(self.kv_cache_ptr.indices)
        if self.status == self.ReqExecType.DRAFT:
            # Return the length of the loading indices instead of the length of selected indices
            return self.kv_cache_ptr.len_loading_indices
        if self.status == self.ReqExecType.STALL:
            return 0
        assert False, f"Unknown status: {self.status}"
