from typing import Tuple

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


if __name__ == "__main__":
    for num_tokens in [16, 32, 64, 80, 128, 256, 280, 384, 448, 512]:
        for vocab_size in [128 * 1024, 151936]:
            for tp_ranks in [1, 2, 4]:
                x = torch.randn(
                    num_tokens, vocab_size, device="cuda", dtype=torch.float16
                )
                q = torch.empty(
                    num_tokens, vocab_size, device="cuda", dtype=torch.float32
                )
                q.exponential_()

                unpack_x = ref_unpack_logits(x, tp_ranks)

                native_time = do_bench(
                    lambda: ref_sampling(unpack_x, q, 0.6), warmup=10, rep=100
                )
                torch_native_time = do_bench(
                    lambda: torch_native_fused_sampling(unpack_x, q, 0.6),
                    warmup=10,
                    rep=100,
                )
                triton_time = do_bench(
                    lambda: triton_fused_sampling(x, q, 0.6, tp_ranks),
                    warmup=10,
                    rep=100,
                )

                print(
                    f"[fused sampling benchmark] "
                    f"num_tokens: {num_tokens}, vocab_size: {vocab_size}, tp_ranks: {tp_ranks}, "
                    f"native time: {native_time:.4f}ms, torch native time: {torch_native_time:.4f}ms, triton time: {triton_time:.4f}ms"
                )

    for num_tokens in [16, 32, 64, 80, 128, 256, 280, 384, 448, 512]:
        for vocab_size in [128 * 1024, 151936]:
            for tp_ranks in [1, 2, 4]:
                x = torch.randn(
                    num_tokens, vocab_size, device="cuda", dtype=torch.float16
                )
                q = torch.empty(
                    num_tokens, vocab_size, device="cuda", dtype=torch.float32
                )
                q.exponential_()

                unpack_x = ref_unpack_logits(x, tp_ranks)

                native_time = do_bench(
                    lambda: torch.argmax(unpack_x, dim=-1).view(-1).to(torch.int32),
                    warmup=10,
                    rep=100,
                )
                triton_time = do_bench(
                    lambda: triton_strided_argmax(x, tp_ranks), warmup=10, rep=100
                )

                print(
                    f"[strided argmax benchmark] "
                    f"num_tokens: {num_tokens}, vocab_size: {vocab_size}, tp_ranks: {tp_ranks}, "
                    f"native time: {native_time:.4f}ms, triton time: {triton_time:.4f}ms"
                )
