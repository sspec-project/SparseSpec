"""
Ref:
1. https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
2. https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/softmax.py
10-14-2025 by happierpig
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


@torch.compile(fullgraph=True)
def torch_native_fused_sampling(
    logits: torch.Tensor, q: torch.Tensor, temperature: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = logits.to(torch.float32)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    scores = probs / q
    return scores.argmax(dim=-1).view(-1).to(torch.int32), probs


def triton_fused_sampling(
    logits: torch.Tensor, q: torch.Tensor, temperature: float, tp_ranks: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert q.dtype == torch.float32
    assert logits.is_contiguous()

    num_tokens, vocab_size = logits.shape
    assert vocab_size % tp_ranks == 0

    probs = torch.empty(
        (num_tokens, vocab_size),
        dtype=torch.float32,
        device=logits.device,
    )
    token_ids = torch.empty(
        (num_tokens,),
        dtype=torch.int32,
        device=logits.device,
    )

    BLOCK_SIZE = 8192
    num_warps = 16

    _triton_fused_sampling_kernel[(num_tokens,)](
        logits,
        q,
        q.stride(0),
        1 / temperature,
        probs,
        probs.stride(0),
        token_ids,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        TP_RANKS=tp_ranks,
        num_warps=num_warps,
    )

    return token_ids, probs


@triton.jit
def _triton_fused_sampling_kernel(
    logits,
    q,
    q_row_stride,
    inv_temperature,
    probs,
    probs_row_stride,
    token_ids,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    TP_RANKS: tl.constexpr,
):
    """
    Fused temperature sampling with online softmax.
    Specialized for tp logits, which is strided by all-gather.
    logits layout: shape (tp_ranks, num_tokens, vocab_size // tp_ranks)
        stride (num_tokens * vocab_size // tp_ranks, vocab_size // tp_ranks, 1)
    """
    row_id = tl.program_id(0)
    num_rows = tl.num_programs(0)
    offs = tl.arange(0, BLOCK_SIZE)

    m = -float("inf")
    d = float(0.0)

    # First pass of the online softmax
    for tp_idx in tl.range(0, TP_RANKS):
        vocab_start_idx = vocab_size // TP_RANKS * tp_idx
        vocab_end_idx = vocab_start_idx + vocab_size // TP_RANKS
        for start in tl.range(vocab_start_idx, vocab_end_idx, BLOCK_SIZE):
            idx = start + offs
            mask = idx < vocab_end_idx
            xblk = tl.load(
                logits
                + tp_idx * num_rows * vocab_size // TP_RANKS
                + row_id * vocab_size // TP_RANKS
                + (idx - vocab_start_idx),
                mask=mask,
                other=-float("inf"),
                cache_modifier=".ca",
            )

            # cast xblk to float32
            xblk = tl.cast(xblk, tl.float32)
            xblk = xblk * inv_temperature

            blk_max = tl.max(xblk, axis=0)
            new_m = tl.maximum(m, blk_max)
            d = d * tl.exp(m - new_m) + tl.sum(tl.exp(xblk - new_m), axis=0)
            m = new_m

    # Second pass of online softmax
    # calculate probs, get token-id by online argmax
    max_score = -float("inf")
    max_idx = 0
    for tp_idx in tl.range(0, TP_RANKS):
        vocab_start_idx = vocab_size // TP_RANKS * tp_idx
        vocab_end_idx = vocab_start_idx + vocab_size // TP_RANKS
        for start in tl.range(vocab_start_idx, vocab_end_idx, BLOCK_SIZE):
            idx = start + offs
            mask = idx < vocab_end_idx
            xblk = tl.load(
                logits
                + tp_idx * num_rows * vocab_size // TP_RANKS
                + row_id * vocab_size // TP_RANKS
                + (idx - vocab_start_idx),
                mask=mask,
                other=-float("inf"),
                cache_modifier=".ca",
            )

            # cast xblk to float32
            xblk = tl.cast(xblk, tl.float32)
            xblk = xblk * inv_temperature
            yblk = tl.exp(xblk - m) / d  # probs

            # store probs
            tl.store(
                probs + row_id * probs_row_stride + idx,
                yblk,
                mask=mask,
                cache_modifier=".cs",
            )

            # random sampling
            qblk = tl.load(
                q + row_id * q_row_stride + idx,
                mask=mask,
                other=-float("inf"),
                cache_modifier=".cv",
            )
            scores = yblk / qblk

            # online argmax for token-ids
            local_val = tl.max(scores, axis=-1)
            local_idx = tl.argmax(scores, axis=-1) + start

            better = local_val > max_score
            max_score = tl.where(better, local_val, max_score)
            max_idx = tl.where(better, local_idx, max_idx)

    tl.store(token_ids + row_id, max_idx)


def triton_strided_argmax(logits: torch.Tensor, tp_ranks: int = 1) -> torch.Tensor:
    assert logits.is_contiguous()

    num_tokens, vocab_size = logits.shape
    assert vocab_size % tp_ranks == 0

    token_ids = torch.empty(
        (num_tokens,),
        dtype=torch.int32,
        device=logits.device,
    )

    BLOCK_SIZE = 8192
    num_warps = 16

    _triton_strided_argmax_kernel[(num_tokens,)](
        logits,
        token_ids,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        TP_RANKS=tp_ranks,
        num_warps=num_warps,
    )

    return token_ids


@triton.jit
def _triton_strided_argmax_kernel(
    logits,
    token_ids,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    TP_RANKS: tl.constexpr,
):
    row_id = tl.program_id(0)
    num_rows = tl.num_programs(0)
    offs = tl.arange(0, BLOCK_SIZE)

    max_score = -float("inf")
    max_idx = 0
    for tp_idx in tl.range(0, TP_RANKS):
        vocab_start_idx = vocab_size // TP_RANKS * tp_idx
        vocab_end_idx = vocab_start_idx + vocab_size // TP_RANKS
        for start in tl.range(vocab_start_idx, vocab_end_idx, BLOCK_SIZE):
            idx = start + offs
            mask = idx < vocab_end_idx
            xblk = tl.load(
                logits
                + tp_idx * num_rows * vocab_size // TP_RANKS
                + row_id * vocab_size // TP_RANKS
                + (idx - vocab_start_idx),
                mask=mask,
                other=-float("inf"),
                cache_modifier=".ca",
            )
            xblk = tl.cast(xblk, tl.float32)

            # online argmax for token-ids
            local_val = tl.max(xblk, axis=-1)
            local_idx = tl.argmax(xblk, axis=-1) + start

            better = local_val > max_score
            max_score = tl.where(better, local_val, max_score)
            max_idx = tl.where(better, local_idx, max_idx)

    tl.store(token_ids + row_id, max_idx)
