"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import array
import math
from dataclasses import dataclass

import flashinfer
import numpy as np
import torch
from flashinfer.testing.utils import bench_gpu_time

from serve.attention.backend import BatchTopKAttention


@dataclass
class Req:
    kv_len: int
    qo_len: int
    request_type: int
    kv_indices: np.ndarray

    def sanity_check(self, page_size, spec_stride):
        if self.request_type == 1:
            # draft
            assert self.qo_len == 1

        if self.request_type == 2:
            # verify
            assert self.qo_len <= spec_stride

        assert len(self.kv_indices) == math.ceil(self.kv_len / page_size)


def _build_reqs(
    batch_size=128,
    spec_stride=16,
    page_block_size=1,
    min_seq_len=1024,
    max_seq_len=8 * 1024,
    sparsity=0.10,
):
    reqs = []
    page_cnt = 0
    for i in range(batch_size):
        kv_len = np.random.randint(min_seq_len, max_seq_len)
        if i % spec_stride == 0:
            qo_len = spec_stride
            request_type = 2
        else:
            qo_len = 1
            request_type = 1
            kv_len = math.ceil(kv_len * sparsity)

        assert kv_len > 0
        num_pages = math.ceil(kv_len / page_block_size)
        kv_indices = np.arange(page_cnt, page_cnt + num_pages, dtype=np.int32)
        page_cnt += num_pages

        reqs.append(Req(kv_len, qo_len, request_type, kv_indices))
        reqs[-1].sanity_check(page_block_size, spec_stride)

    return reqs, page_cnt


def ref_attention(
    reqs: list[Req],
    q: torch.Tensor,
    kv_data: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_block_size: int,
    causal: bool,
    layout: str,
    dtype: torch.dtype,
    device: torch.device,
):
    wrapper_old = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        kv_layout=layout,
        backend="fa2",
    )

    seq_lens = np.array([req.kv_len for req in reqs], dtype=np.int32)
    seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    last_page_len = (seq_lens - 1) % page_block_size + 1

    qo_lens = np.array([req.qo_len for req in reqs], dtype=np.int32)
    qo_lens = torch.tensor(qo_lens, dtype=torch.int32, device=device)
    qo_indptr = torch.cat(
        [torch.tensor([0], device=device), torch.cumsum(qo_lens, 0)], dim=0
    ).int()

    kv_indptr_host = array.array("i", [0])
    kv_indices_host = array.array("i", [])
    for req in reqs:
        kv_indices_host.extend(req.kv_indices)
        kv_indptr_host.append(len(kv_indices_host))
    kv_indptr = torch.tensor(kv_indptr_host, dtype=torch.int32, device=device)
    kv_indices = torch.tensor(kv_indices_host, dtype=torch.int32, device=device)

    wrapper_old.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_block_size,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    times_list = bench_gpu_time(lambda: wrapper_old.run(q, kv_data, return_lse=True))
    ms_old = np.mean(times_list)

    return ms_old


def customized_attention(
    reqs: list[Req],
    q: torch.Tensor,
    kv_data: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_block_size: int,
    causal: bool,
    layout: str,
    dtype: torch.dtype,
    device: torch.device,
    spec_stride: int,
):
    wrapper = BatchTopKAttention(
        kv_layout=layout,
        device=device,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_qo_heads,
        head_dim=head_dim,
        q_data_type=dtype,
        kv_data_type=dtype,
        MAX_SPEC_STRIDE=spec_stride,
    )

    gqa_group_size = num_qo_heads // num_kv_heads
    assert num_qo_heads % num_kv_heads == 0

    seq_lens = np.array([req.kv_len for req in reqs], dtype=np.int32)
    kv_len_arr = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    qo_lens = np.array([req.qo_len for req in reqs], dtype=np.int32)
    qo_lens = torch.tensor(qo_lens, dtype=torch.int32, device=device)
    qo_indptr = torch.cat(
        [torch.tensor([0], device=device), torch.cumsum(qo_lens, 0)], dim=0
    ).int()

    kv_indptr_host = array.array("i", [])
    kv_indices_host = array.array("i", [])
    draft_kv_indices_host = array.array("i", [])
    request_type_host = array.array("i", [])
    verify_kv_indptr_host = array.array("i", [0])
    for req in reqs:
        request_type_host.append(req.request_type)
        if req.request_type == 1:
            # draft requests
            kv_indptr_host.append(len(draft_kv_indices_host))
            draft_kv_indices_host.extend(req.kv_indices)
        else:
            # other requests
            kv_indptr_host.append(len(kv_indices_host))
            kv_indices_host.extend(req.kv_indices)

        if req.request_type == 2:
            # verify requests
            assert req.qo_len * gqa_group_size <= 128  # cuda-level limit
            verify_kv_indptr_host.append(req.kv_len + verify_kv_indptr_host[-1])
        else:
            verify_kv_indptr_host.append(verify_kv_indptr_host[-1])

    kv_indptr = torch.tensor(kv_indptr_host, dtype=torch.int32, device=device)
    kv_indices = torch.tensor(kv_indices_host, dtype=torch.int32, device=device)
    request_type = torch.tensor(request_type_host, dtype=torch.int32, device=device)
    verify_kv_indptr = torch.tensor(
        verify_kv_indptr_host, dtype=torch.int32, device=device
    )

    if len(draft_kv_indices_host) > wrapper.MAX_TOTAL_DRAFT_KV_LEN:
        return 100000
    if verify_kv_indptr_host[-1] > wrapper.MAX_TOTAL_VERIFY_KV_LEN:
        return 100000

    # padded draft_kv_indices_host to MAX_TOTAL_DRAFT_KV_LEN
    draft_kv_indices_host = draft_kv_indices_host + array.array(
        "i", [0] * (wrapper.MAX_TOTAL_DRAFT_KV_LEN - len(draft_kv_indices_host))
    )
    draft_kv_indices = torch.tensor(
        draft_kv_indices_host, dtype=torch.int32, device=device
    )
    # repeat draft_kv_indices `num_kv_heads` times
    draft_kv_indices = draft_kv_indices.repeat(num_kv_heads, 1)
    assert draft_kv_indices.shape == (num_kv_heads, wrapper.MAX_TOTAL_DRAFT_KV_LEN)

    # init dump_logits
    dump_logits = torch.zeros(
        num_kv_heads,
        wrapper.MAX_TOTAL_VERIFY_KV_LEN,
        dtype=torch.float16,
        device=device,
    )

    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_len_arr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        page_block_size,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    # warmup and get the out, lse
    out, lse = wrapper.run_attention(
        q,
        kv_data,
        kv_indices,
        draft_kv_indices,
        request_type,
    )

    # test core attention
    times_list = bench_gpu_time(
        lambda: wrapper.run_attention(
            q,
            kv_data,
            kv_indices,
            draft_kv_indices,
            request_type,
        )
    )
    ms_new = np.mean(times_list)

    # test score rematerialize
    times_list = bench_gpu_time(
        lambda: wrapper.rematerialize_scores(
            q,
            out,
            lse,
            kv_data,
            kv_indices,
            request_type,
            verify_kv_indptr,
            dump_logits,
        )
    )
    ms_new_score_rematerialize = np.mean(times_list)

    return ms_new, ms_new_score_rematerialize


def _run_check(
    batch_size=128,
    spec_stride=16,
    page_block_size=1,
    num_kv_heads=1,
    num_qo_heads=1,
    head_dim=128,
    test_dtype=torch.bfloat16,
    device="cuda",
):
    """
    Run both implementations and return (output_old, lse_old, output_new, lse_new)
    """
    layout = "NHD"
    causal = True
    dev = torch.device(device)

    # set random seed
    np.random.seed(30)
    torch.manual_seed(30)
    torch.cuda.manual_seed(30)

    # get request length
    reqs, num_pages = _build_reqs(batch_size, spec_stride, page_block_size)
    num_total_qo_len = sum(req.qo_len for req in reqs)

    # get data
    q = torch.rand(
        num_total_qo_len, num_qo_heads, head_dim, dtype=test_dtype, device=dev
    )
    kv_data = torch.randn(
        num_pages,
        2,
        page_block_size,
        num_kv_heads,
        head_dim,
        dtype=test_dtype,
        device=dev,
    )

    # run ref
    ms_old = ref_attention(
        reqs,
        q,
        kv_data,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_block_size,
        causal,
        layout,
        test_dtype,
        dev,
    )

    # run customized
    ms_new, ms_new_score_rematerialize = customized_attention(
        reqs,
        q,
        kv_data,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_block_size,
        causal,
        layout,
        test_dtype,
        dev,
        spec_stride,
    )

    total_bytes = (
        q.numel() * q.element_size() + kv_data.numel() * kv_data.element_size()
    )
    mem_MB = total_bytes / 1024**2
    bw_old = total_bytes / (ms_old * 1e-3) / 1024**3
    bw_new = total_bytes / (ms_new * 1e-3) / 1024**3

    # calculate total loading bytes for score rematerialize
    total_bytes = 0
    for req in reqs:
        if req.request_type == 2:
            # only load k without v
            total_bytes += kv_data.element_size() * req.kv_len * num_kv_heads * head_dim

    mem_score_rematerialize = total_bytes / 1024**2
    bw_score_rematerialize = total_bytes / (ms_new_score_rematerialize * 1e-3) / 1024**3

    print(
        f"batch_size: {batch_size}, spec_stride: {spec_stride}, page_block_size: {page_block_size}, "
        f"num_kv_heads: {num_kv_heads}, gqa_group_size: {gqa_group_size}, head_dim: {head_dim}, test_dtype: {test_dtype}, "
        f"ms_ref_core: {ms_old}, ms_new_core: {ms_new}, ms_score_rematerialize: {ms_new_score_rematerialize}, "
        f"mem_core: {mem_MB}, bw_ref_core: {bw_old}, bw_new_core: {bw_new}, "
        f"mem_score_rematerialize: {mem_score_rematerialize}, bw_score_rematerialize: {bw_score_rematerialize}"
    )


def bench_batch_attention_correctness(
    batch_size,
    spec_stride,
    page_block_size,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    test_dtype,
):
    num_qo_heads = num_kv_heads * gqa_group_size
    _run_check(
        batch_size=batch_size,
        spec_stride=spec_stride,
        page_block_size=page_block_size,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_qo_heads,
        head_dim=head_dim,
        test_dtype=test_dtype,
        device="cuda",
    )


if __name__ == "__main__":
    for batch_size in [64, 128]:
        for spec_stride in [16]:
            for page_block_size in [1]:
                for num_kv_heads in [4]:
                    for gqa_group_size in [1, 2, 4, 8]:
                        for head_dim in [128]:
                            for test_dtype in [torch.float16]:
                                bench_batch_attention_correctness(
                                    batch_size=batch_size,
                                    spec_stride=spec_stride,
                                    page_block_size=page_block_size,
                                    num_kv_heads=num_kv_heads,
                                    gqa_group_size=gqa_group_size,
                                    head_dim=head_dim,
                                    test_dtype=test_dtype,
                                )
