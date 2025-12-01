import array
import math
from dataclasses import dataclass

import flashinfer
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from serve.attention.backend import BatchTopKWrapper

MAX_TOTAL_VERIFY_KV_LEN = 512 * 1024
MAX_DRAFT_KV_LEN = 8192


def ref_topk(
    in_val: torch.Tensor,
    in_val_start_pos_host: np.ndarray,
    in_lens_host: np.ndarray,
    ks_host: np.ndarray,
    num_requests: int,
):
    # init output buffer first
    batch_size = in_val.shape[0]
    out_val = torch.empty(
        num_requests,
        batch_size,
        MAX_DRAFT_KV_LEN,
        device=in_val.device,
        dtype=in_val.dtype,
    )
    out_idx = torch.empty(
        num_requests,
        batch_size,
        MAX_DRAFT_KV_LEN,
        device=in_val.device,
        dtype=torch.int32,
    )

    for req_idx in range(num_requests):
        for batch_idx in range(batch_size):
            k = ks_host[req_idx]
            in_len = in_lens_host[req_idx]

            values, indices = torch.topk(
                in_val[
                    batch_idx,
                    in_val_start_pos_host[req_idx] : in_val_start_pos_host[req_idx]
                    + in_len,
                ],
                k,
                dim=-1,
                largest=True,
                sorted=False,
            )

            out_val[req_idx, batch_idx, :k] = values
            out_idx[req_idx, batch_idx, :k] = indices

    return out_val, out_idx


def customized_topk(
    in_val: torch.Tensor,
    in_val_start_pos_host: np.ndarray,
    in_lens_host: np.ndarray,
    ks_host: np.ndarray,
    num_requests: int,
):
    wrapper = BatchTopKWrapper(
        MAX_TOTAL_VERIFY_KV_LEN=MAX_TOTAL_VERIFY_KV_LEN,
        MAX_DRAFT_KV_LEN=MAX_DRAFT_KV_LEN,
    )
    # check k < wrapper.MAX_DRAFT_KV_LEN
    assert ks_host.max() < wrapper.MAX_DRAFT_KV_LEN

    # construct 0-base indices array
    in_idx_host = np.arange(0, np.sum(in_lens_host), dtype=np.int32)
    in_idx_start_pos_host = np.zeros_like(in_val_start_pos_host)

    for i in range(num_requests):
        if i + 1 < num_requests:
            in_idx_start_pos_host[i + 1] = in_idx_start_pos_host[i] + in_lens_host[i]
        in_idx_host[
            in_idx_start_pos_host[i] : in_idx_start_pos_host[i] + in_lens_host[i]
        ] = np.arange(0, in_lens_host[i], dtype=np.int32)

    # get tensor metadata
    in_idx = torch.from_numpy(in_idx_host).to(in_val.device)
    in_val_start_pos = torch.from_numpy(in_val_start_pos_host).to(in_val.device)
    in_idx_start_pos = torch.from_numpy(in_idx_start_pos_host).to(in_val.device)
    in_lens = torch.from_numpy(in_lens_host).to(in_val.device)
    ks = torch.from_numpy(ks_host).to(in_val.device)

    # run topk
    out_val, out_idx = wrapper.run(
        in_val,
        in_idx.to(torch.int32),
        in_val_start_pos.to(torch.int32),
        in_idx_start_pos.to(torch.int32),
        in_lens.to(torch.int32),
        ks.to(torch.int32),
        num_requests,
    )

    return out_val, out_idx


def check_results(
    val: torch.Tensor,
    idx: torch.Tensor,
    val_ref: torch.Tensor,
    idx_ref: torch.Tensor,
    ks_host: np.ndarray,
    batch_size: int,
    num_requests: int,
):
    for req_idx in range(num_requests):
        for batch_idx in range(batch_size):
            k = ks_host[req_idx]
            torch.testing.assert_close(
                torch.sum(val[req_idx, batch_idx, :k]),
                torch.sum(val_ref[req_idx, batch_idx, :k]),
                atol=1e-3,
                rtol=1e-3,
            )


@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("num_requests", [8, 16])
@pytest.mark.parametrize("max_len", [16384, 32768])
@pytest.mark.parametrize("k_ratio", [0.05, 0.10, 0.15])
def test_batch_topk_sort(
    batch_size,
    num_requests,
    max_len,
    k_ratio,
):
    in_lens_host = np.random.randint(
        max_len // 2, max_len, size=(num_requests), dtype=np.int32
    )
    if np.sum(in_lens_host) >= MAX_TOTAL_VERIFY_KV_LEN:
        pytest.skip("Skip test due to exceeding MAX_TOTAL_VERIFY_KV_LEN")

    in_val_start_pos_host = np.zeros(num_requests, dtype=np.int32)
    in_val_start_pos_host[1:] = np.cumsum(in_lens_host)[:-1]

    ks_host = in_lens_host.copy()
    for i in range(num_requests):
        ks_host[i] = max(16, int(in_lens_host[i] * k_ratio))
        ks_host[i] = min(ks_host[i], MAX_DRAFT_KV_LEN, in_lens_host[i])

    # init data
    in_val = (
        torch.randn(
            batch_size,
            MAX_TOTAL_VERIFY_KV_LEN,
            device="cuda",
            dtype=torch.float16,
        )
        * 10
    )
    in_val += torch.randn_like(in_val) * 5

    # run customized topk
    val_new, idx_new = customized_topk(
        in_val,
        in_val_start_pos_host,
        in_lens_host,
        ks_host,
        num_requests,
    )

    # run ref topk
    val_ref, idx_ref = ref_topk(
        in_val,
        in_val_start_pos_host,
        in_lens_host,
        ks_host,
        num_requests,
    )

    # run check
    check_results(
        val_new,
        idx_new,
        val_ref,
        idx_ref,
        ks_host,
        batch_size,
        num_requests,
    )


if __name__ == "__main__":
    test_batch_topk_sort(
        batch_size=16,
        num_requests=8,
        max_len=16 * 1024,
        k_ratio=0.10,
    )
