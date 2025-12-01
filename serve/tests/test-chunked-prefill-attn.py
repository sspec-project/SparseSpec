import pytest
import torch
from flashinfer.attention import BatchAttention


def ref_chunked_prefill_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    qo_len, num_qo_heads, head_dim = q.shape
    kv_len, _, block_size, num_kv_heads, _ = kv.shape

    assert block_size == 1
    assert kv_len == qo_len

    o = torch.empty_like(q)

    wrapper = BatchAttention(kv_layout="NHD")
    for chunk_start in range(0, qo_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, qo_len)
        effective_chunk_size = chunk_end - chunk_start

        q_indptr = torch.tensor(
            [0, effective_chunk_size], dtype=torch.int32, device=q.device
        )
        kv_indptr = torch.tensor([0, chunk_end], dtype=torch.int32, device=kv.device)
        seq_lens = torch.tensor([chunk_end], dtype=torch.int32, device=q.device)
        kv_indices = torch.arange(chunk_end, dtype=torch.int32, device=q.device)

        wrapper.plan(
            q_indptr,
            kv_indptr,
            seq_lens,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            1,
            causal=True,
            q_data_type=q.dtype,
            kv_data_type=kv.dtype,
            logits_soft_cap=0.0,
        )

        chunk_o, _ = wrapper.run(
            q=q[chunk_start:chunk_end],
            kv_cache=kv,
            kv_indices=kv_indices,
        )

        o[chunk_start:chunk_end] = chunk_o

    return o


@pytest.mark.parametrize("num_qo_heads", [16, 32])
@pytest.mark.parametrize("num_kv_heads", [1, 8])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("seqlen", [80, 117, 128, 200])
@pytest.mark.parametrize("chunk_size", [16, 32, 64])
def test_chunked_prefill_attn(num_qo_heads, num_kv_heads, head_dim, seqlen, chunk_size):
    q = torch.randn(seqlen, num_qo_heads, head_dim, device="cuda", dtype=torch.float16)
    kv = torch.randn(
        seqlen, 2, 1, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
    )

    o_chunked = ref_chunked_prefill_attn(q, kv, chunk_size=chunk_size)
    o_full = ref_chunked_prefill_attn(q, kv, chunk_size=16384)

    torch.testing.assert_close(o_chunked, o_full, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    num_qo_heads = 16
    num_kv_heads = 8
    head_dim = 128

    seqlen = 80

    q = torch.randn(seqlen, num_qo_heads, head_dim, device="cuda", dtype=torch.float16)
    kv = torch.randn(
        seqlen, 2, 1, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
    )

    o_chunked = ref_chunked_prefill_attn(q, kv, chunk_size=32)
    o_full = ref_chunked_prefill_attn(q, kv, chunk_size=1024)

    torch.testing.assert_close(o_chunked, o_full, rtol=1e-3, atol=1e-3)
