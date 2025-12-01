import array
import math
from dataclasses import dataclass

import flashinfer
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from flashinfer.testing.utils import bench_gpu_time

from serve.attention.backend import BatchTopKWrapper

MAX_TOTAL_VERIFY_KV_LEN = 512 * 1024
MAX_DRAFT_KV_LEN = 8192


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
    times_list = bench_gpu_time(
        lambda: wrapper.run(
            in_val,
            in_idx.to(torch.int32),
            in_val_start_pos.to(torch.int32),
            in_idx_start_pos.to(torch.int32),
            in_lens.to(torch.int32),
            ks.to(torch.int32),
            num_requests,
        )
    )

    return np.mean(times_list)


def bench_batch_topk_sort(
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
    ms_time = customized_topk(
        in_val,
        in_val_start_pos_host,
        in_lens_host,
        ks_host,
        num_requests,
    )

    # add read bytes
    loading_bytes = np.sum(in_lens_host) * batch_size * (in_val.element_size() + 4)
    # add write bytes
    loading_bytes += np.sum(ks_host) * batch_size * (in_val.element_size() + 4)
    mem_MB = loading_bytes / 1024**2
    bw_MB = loading_bytes / (ms_time * 1e-3) / 1024**3

    print(
        f"batch_size: {batch_size}, num_requests: {num_requests}, max_len: {max_len}, k_ratio: {k_ratio}, "
        f"time (ms): {ms_time:.2f}, mem (MB): {mem_MB:.2f}, bw (GB/s): {bw_MB:.2f}"
    )


if __name__ == "__main__":
    for batch_size in [64, 128]:
        for num_requests in [8, 16]:
            for max_len in [16 * 1024, 32 * 1024]:
                for k_ratio in [0.05, 0.10, 0.15]:
                    bench_batch_topk_sort(
                        batch_size=batch_size,
                        num_requests=num_requests,
                        max_len=max_len,
                        k_ratio=k_ratio,
                    )
