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
import pytest
import torch
import torch.nn.functional as F

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


def ref_core_attention(
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
    out_old, lse_old = wrapper_old.run(q, kv_data, return_lse=True)

    return out_old, lse_old


def ref_score_rematerialize(
    q: torch.Tensor,
    k: torch.Tensor,
    causal: bool = True,
):
    assert q.shape[-1] == k.shape[-1]
    qo_len, num_qo_heads, head_dim = q.shape
    kv_len, num_kv_heads, head_dim = k.shape

    gqa_group_size = num_qo_heads // num_kv_heads
    assert num_qo_heads % num_kv_heads == 0

    # calculate logits
    q_rg = q.reshape(qo_len, num_kv_heads, gqa_group_size, head_dim).contiguous()
    logits = torch.einsum("thgd,khd->thgk", q_rg, k)
    logits = logits.reshape(qo_len, num_qo_heads, kv_len).to(torch.float32)
    logits = logits / math.sqrt(head_dim)

    # apply causal mask
    if causal:
        mask = torch.triu(
            torch.ones(qo_len, kv_len, dtype=torch.bool, device=logits.device),
            diagonal=kv_len - qo_len + 1,
        )
        logits.masked_fill_(mask.unsqueeze(1), float("-inf"))

    scores = F.softmax(logits, dim=-1)

    # average across gqa-group-size and qo-len
    scores = (
        scores.reshape(qo_len, num_kv_heads, gqa_group_size, kv_len)
        .transpose(1, 2)
        .reshape(-1, num_kv_heads, kv_len)
    )
    scores = scores.mean(dim=0)

    return scores


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
        pytest.skip(
            f"Draft KV length ({len(draft_kv_indices_host)}) larger than MAX_TOTAL_DRAFT_KV_LEN ({wrapper.MAX_TOTAL_DRAFT_KV_LEN}), skip test"
        )
    if verify_kv_indptr_host[-1] > wrapper.MAX_TOTAL_VERIFY_KV_LEN:
        pytest.skip("Verify KV length is too long, skip test")

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

    out, lse = wrapper.run_attention(
        q,
        kv_data,
        kv_indices,
        draft_kv_indices,
        request_type,
    )

    wrapper.rematerialize_scores(
        q,
        out,
        lse,
        kv_data,
        kv_indices,
        request_type,
        verify_kv_indptr,
        dump_logits,
    )

    return out, lse, dump_logits


def run_attention_attention_check(
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
    o_ref, lse_ref = ref_core_attention(
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
    o_new, lse_new, _ = customized_attention(
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

    torch.cuda.synchronize()

    if test_dtype == torch.float16:
        torch.testing.assert_close(o_ref, o_new, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(lse_ref, lse_new, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(o_ref, o_new, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(lse_ref, lse_new, rtol=1e-2, atol=1e-2)


def run_score_rematerialize_check(
    batch_size=128,
    spec_stride=16,
    page_block_size=1,
    num_kv_heads=1,
    num_qo_heads=1,
    head_dim=128,
    test_dtype=torch.bfloat16,
    device="cuda",
):
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

    gqa_group_size = num_qo_heads // num_kv_heads
    assert num_qo_heads % num_kv_heads == 0

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

    # run customized
    _, _, scores_new = customized_attention(
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

    torch.cuda.synchronize()

    # run ref
    verify_kv_indptr_host = array.array("i", [0])
    qo_indptr_host = array.array("i", [0])
    for idx, req in enumerate(reqs):
        qo_indptr_host.append(req.qo_len + qo_indptr_host[-1])
        if req.request_type == 2:
            # verify requests
            assert req.qo_len * gqa_group_size <= 128  # cuda-level limit
            verify_kv_indptr_host.append(req.kv_len + verify_kv_indptr_host[-1])
        else:
            verify_kv_indptr_host.append(verify_kv_indptr_host[-1])

        if req.request_type != 2:
            continue

        # gather q and k
        q_req = q[qo_indptr_host[idx] : qo_indptr_host[idx + 1]]
        k_req = (
            kv_data[
                req.kv_indices[0] : req.kv_indices[-1] + 1,
                0,
            ]
            .reshape(-1, num_kv_heads, head_dim)
            .contiguous()
        )
        k_req = k_req[: req.kv_len]

        # calculate score
        scores_ref = ref_score_rematerialize(q_req, k_req, causal).to(scores_new.dtype)

        torch.testing.assert_close(
            scores_new[
                :,
                verify_kv_indptr_host[idx] : verify_kv_indptr_host[idx + 1]
                - req.qo_len,
            ],
            scores_ref[:, : -req.qo_len],
            rtol=1e-2,
            atol=1e-2,
        )


@pytest.mark.parametrize("batch_size", [64, 128])
@pytest.mark.parametrize("spec_stride", [8, 16])
@pytest.mark.parametrize("page_block_size", [1, 8])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("gqa_group_size", [1, 4])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("test_dtype", [torch.bfloat16, torch.float16])
def test_batch_attention_correctness(
    batch_size,
    spec_stride,
    page_block_size,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    test_dtype,
):
    num_qo_heads = num_kv_heads * gqa_group_size
    run_attention_attention_check(
        batch_size=batch_size,
        spec_stride=spec_stride,
        page_block_size=page_block_size,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_qo_heads,
        head_dim=head_dim,
        test_dtype=test_dtype,
        device="cuda",
    )


@pytest.mark.parametrize("batch_size", [32, 64, 128])
@pytest.mark.parametrize("spec_stride", [4, 8, 16])
@pytest.mark.parametrize("page_block_size", [1, 8])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("gqa_group_size", [1, 4, 8])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("test_dtype", [torch.bfloat16, torch.float16])
def test_batch_score_rematerialize_correctness(
    batch_size,
    spec_stride,
    page_block_size,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    test_dtype,
):
    if spec_stride * gqa_group_size > 64:
        # NOTE: hardcode in score kernel
        # will discard qo_idx > 64
        pytest.skip("spec_stride * gqa_group_size > 64, skip test")

    num_qo_heads = num_kv_heads * gqa_group_size
    run_score_rematerialize_check(
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
    test_batch_score_rematerialize_correctness(
        batch_size=64,
        spec_stride=16,
        page_block_size=1,
        num_kv_heads=1,
        gqa_group_size=4,
        head_dim=128,
        test_dtype=torch.float16,
    )
