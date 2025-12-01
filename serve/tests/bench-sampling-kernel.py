import array
import math

import flashinfer
import numpy as np
import torch
import triton
from flashinfer.testing.utils import bench_gpu_time

from serve.sampling.rejection_sampler import (
    RejectionSampler,
    sample_recovered_tokens_kernel,
)


def bench_sample_recovered_tokens_kernel(
    batch_size,
    max_spec_len,
    vocab_size,
):
    # synthesize data
    num_draft_tokens_host = np.random.randint(
        1,
        max_spec_len,
        size=(batch_size,),
        dtype=np.int32,
    )
    cu_num_draft_tokens_host = np.cumsum(num_draft_tokens_host)

    num_tokens = np.sum(num_draft_tokens_host)
    draft_token_ids_host = np.random.randint(
        0,
        vocab_size,
        size=(num_tokens,),
        dtype=np.int32,
    )
    draft_probs_device = torch.randn(
        num_tokens,
        vocab_size,
        dtype=torch.float32,
        device="cuda",
    )
    target_probs_device = torch.randn(
        num_tokens,
        vocab_size,
        dtype=torch.float32,
        device="cuda",
    )
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device="cuda",
    )
    q.exponential_()

    draft_token_ids_device = (
        torch.from_numpy(draft_token_ids_host).to(torch.int32).to("cuda")
    )
    cu_num_draft_tokens_device = (
        torch.from_numpy(cu_num_draft_tokens_host).to(torch.int32).to("cuda")
    )
    recovered_token_ids_device = torch.empty_like(draft_token_ids_device)

    ms_list = bench_gpu_time(
        lambda: sample_recovered_tokens_kernel[
            (batch_size, RejectionSampler.MAX_SPEC_LEN)
        ](
            recovered_token_ids_device,
            cu_num_draft_tokens_device,
            draft_token_ids_device,
            draft_probs_device,
            target_probs_device,
            q,
            vocab_size,
            NO_DRAFT_PROBS=False,
            BLOCK_SIZE=16384,
            num_warps=8,
        )
    )

    ms_avg = np.mean(ms_list)
    loading_bytes = (
        recovered_token_ids_device.numel() * recovered_token_ids_device.element_size()
    )
    loading_bytes += (
        cu_num_draft_tokens_device.numel() * cu_num_draft_tokens_device.element_size()
    )
    loading_bytes += (
        draft_token_ids_device.numel() * draft_token_ids_device.element_size()
    )
    loading_bytes += draft_probs_device.numel() * draft_probs_device.element_size()
    loading_bytes += target_probs_device.numel() * target_probs_device.element_size()
    loading_bytes += q.numel() * q.element_size()
    mem_MB = loading_bytes / 1024**2
    bw_MB = loading_bytes / (ms_avg * 1e-3) / 1024**3

    print(
        f"batch_size: {batch_size}, max_spec_len: {max_spec_len}, vocab_size: {vocab_size}, "
        f"time (ms): {ms_avg:.2f}, mem (MB): {mem_MB:.2f}, bw (GB/s): {bw_MB:.2f}"
    )


if __name__ == "__main__":
    for batch_size in [8, 16, 32]:
        for max_spec_len in [16]:
            for vocab_size in [128256, 152064]:
                bench_sample_recovered_tokens_kernel(
                    batch_size=batch_size,
                    max_spec_len=max_spec_len,
                    vocab_size=vocab_size,
                )
