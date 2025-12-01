from typing import Tuple

import pytest
import torch
from triton.testing import do_bench

from serve.sampling.fast_fused_op import (
    torch_native_fused_sampling,
    triton_fused_sampling,
    triton_strided_argmax,
)


def ref_sampling(
    logits: torch.Tensor, q: torch.Tensor, temperature: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = logits.to(torch.float32)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    scores = probs / q
    return scores.argmax(dim=-1).view(-1), probs


def ref_unpack_logits(logits: torch.Tensor, tp_ranks: int) -> torch.Tensor:
    if tp_ranks == 1:
        return logits

    num_tokens, vocab_size = logits.shape
    assert vocab_size % tp_ranks == 0
    assert logits.is_contiguous()

    chunks = torch.chunk(logits.view(-1), tp_ranks, dim=0)
    chunks = [chunk.view(num_tokens, vocab_size // tp_ranks) for chunk in chunks]

    return torch.cat(chunks, dim=1)


@pytest.mark.parametrize("num_tokens", [16, 32, 64, 80, 128, 256, 280, 384, 448, 512])
@pytest.mark.parametrize("vocab_size", [128 * 1024, 151936])
@pytest.mark.parametrize("temperature", [0.6, 1.0])
@pytest.mark.parametrize("tp_ranks", [1, 2, 4])
def test_fast_fused_sampling(num_tokens, vocab_size, temperature, tp_ranks):
    x = torch.randn(num_tokens, vocab_size, device="cuda", dtype=torch.float16)
    q = torch.empty(num_tokens, vocab_size, device="cuda", dtype=torch.float32)
    q.exponential_()

    triton_token_ids, triton_probs = triton_fused_sampling(x, q, temperature, tp_ranks)

    unpack_x = ref_unpack_logits(x, tp_ranks)
    ref_token_ids, ref_probs = ref_sampling(unpack_x, q, temperature)
    torch_native_token_ids, torch_native_probs = torch_native_fused_sampling(
        unpack_x, q, temperature
    )
    ref_token_ids = ref_token_ids.to(torch.int32)

    assert torch.allclose(torch_native_token_ids, ref_token_ids)
    assert torch.allclose(torch_native_probs, ref_probs)
    assert torch.allclose(triton_token_ids, ref_token_ids)
    assert torch.allclose(triton_probs, ref_probs)


@pytest.mark.parametrize("num_tokens", [16, 32, 64, 80, 128, 256, 280, 384, 448, 512])
@pytest.mark.parametrize("vocab_size", [128 * 1024, 151936])
@pytest.mark.parametrize("tp_ranks", [1, 2, 4])
def test_fast_fused_strided_argmax(num_tokens, vocab_size, tp_ranks):
    x = torch.randn(num_tokens, vocab_size, device="cuda", dtype=torch.float16)
    triton_token_ids = triton_strided_argmax(x, tp_ranks)
    unpack_x = ref_unpack_logits(x, tp_ranks)
    ref_token_ids = unpack_x.argmax(dim=-1).view(-1).to(torch.int32)

    assert torch.allclose(triton_token_ids, ref_token_ids)


if __name__ == "__main__":
    test_fast_fused_sampling(16, 128 * 1024, 0.6)
