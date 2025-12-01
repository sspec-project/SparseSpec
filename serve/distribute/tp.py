# Copyright 2024 Infini-AI-Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifed by happierpig on 03-04-2025
# Adopted from: https://github.com/Infini-AI-Lab/MagicDec/

import logging
import os
from itertools import accumulate
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from serve.distribute.communicator import CudaCommunicator
from serve.model.config import ModelArgs
from serve.model.model import Attention, FeedForward, Transformer
from serve.utils import cleanup, get_local_rank, get_world_size, is_first_rank

logger = logging.getLogger(__name__)


def local_break():
    if is_first_rank():
        breakpoint()
    dist.barrier()


def init_dist():
    """Initialize distributed torch lib. Called at entry in each process."""
    local_rank = get_local_rank()
    world_size = get_world_size()
    # Called after setup `CUDA_VISIBLE_DEVICES`
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        rank=local_rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    global_group = dist.group.WORLD

    return local_rank, global_group


def select_kv_heads(num_kv_heads):
    """Select the local kv heads for the current process."""
    local_rank = get_local_rank()
    world_size = get_world_size()

    base_heads = num_kv_heads // world_size
    remainder = num_kv_heads % world_size
    if remainder != 0:
        logger.warning(
            f"num_kv_heads={num_kv_heads} is not divisible by world_size={world_size}."
        )

    distribution = [base_heads] * world_size
    for i in range(remainder):
        distribution[i] += 1
    cumulative_distribution = list(accumulate(distribution))
    if local_rank == 0:
        start = 0
        end = cumulative_distribution[0]
    else:
        start = cumulative_distribution[local_rank - 1]
        end = cumulative_distribution[local_rank]

    return start, end


@cleanup
def apply_tp_linear_attn(
    linear: nn.Linear,
    style: str,
    num_kv_heads: int,
    num_qo_heads: int,
    head_dim: int,
    weight_splits: Optional[list[int]] = None,
):
    """Shard linear weight in attention module."""
    gqa_group_size = num_qo_heads // num_kv_heads
    kv_start, kv_end = select_kv_heads(num_kv_heads)

    q_start = kv_start * gqa_group_size * head_dim
    q_end = kv_end * gqa_group_size * head_dim
    kv_start = kv_start * head_dim
    kv_end = kv_end * head_dim

    # Linear's weight matrix is transposed, and is of shape
    # (linear.out_features, linear.in_features)
    dim_lookup = {"colwise": (0, "out_features"), "rowwise": (1, "in_features")}
    assert style in dim_lookup
    shard_dim, size_attr = dim_lookup[style]

    def shard(x, dim, start, end):
        """Return cloned/continuous shard of the tensor x."""
        if dim == 0:
            return x[start:end, :].clone().contiguous()
        elif dim == 1:
            return x[:, start:end].clone().contiguous()

    def shard_qkv(qkv, dim, weight_splits):
        """Shard the qkv tensor along the given dimension."""
        q, k, v = qkv.split(weight_splits, dim=dim)
        q = shard(q, dim, q_start, q_end)
        k = shard(k, dim, kv_start, kv_end)
        v = shard(v, dim, kv_start, kv_end)
        return torch.cat((q, k, v), dim=dim).clone().contiguous()

    if weight_splits is not None:
        # attention
        assert len(weight_splits) == 3
        sharded_weight = shard_qkv(linear.weight, shard_dim, weight_splits)
        if hasattr(linear, "scales") and style == "colwise":
            linear.scales = shard_qkv(linear.scales, 0, weight_splits)
    else:
        sharded_weight = shard(linear.weight, shard_dim, q_start, q_end)
        if hasattr(linear, "scales") and style == "colwise":
            linear.scales = shard(linear.scales, 0, q_start, q_end)

    linear.weight = nn.Parameter(sharded_weight, requires_grad=False)
    linear.out_features, linear.in_features = linear.weight.shape
    setattr(linear, size_attr, linear.weight.shape[shard_dim])

    # Handle bias if it exists
    if linear.bias is not None and style == "colwise":
        if weight_splits is not None:
            # For attention QKV bias, apply same partitioning as weights
            q_bias, k_bias, v_bias = linear.bias.split(weight_splits, dim=0)
            q_bias = q_bias[q_start:q_end]
            k_bias = k_bias[kv_start:kv_end]
            v_bias = v_bias[kv_start:kv_end]
            sharded_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
        else:
            # For other biases in attention (e.g., wo)
            sharded_bias = linear.bias[q_start:q_end]

        linear.bias = nn.Parameter(sharded_bias, requires_grad=False)


@cleanup
def apply_tp_linear_mlp(
    linear: nn.Linear,
    style: str,
):
    """Shard linear weight (i.e., GEMM)."""
    local_rank = get_local_rank()
    world_size = get_world_size()

    # Linear's weight matrix is transposed, and is of shape
    # (linear.out_features, linear.in_features)
    dim_lookup = {"colwise": (0, "out_features"), "rowwise": (1, "in_features")}
    assert style in dim_lookup
    shard_dim, size_attr = dim_lookup[style]

    def shard(x, dim):
        """Shard the tensor x along the given dimension. Return a new cloned tensor."""
        if dim == 1:
            # row-wise: split on intermediate dimension (w_down)
            _view = torch.chunk(x, world_size, dim=dim)[local_rank]
        else:
            # NOTE: This is coupled with the w13 fused weight
            _chunks = torch.chunk(x, 2 * world_size, dim=dim)
            _view = torch.cat(
                [_chunks[local_rank], _chunks[local_rank + world_size]], dim=dim
            )
        return _view.clone().contiguous()

    sharded_weight = shard(linear.weight, shard_dim)
    if hasattr(linear, "scales") and style == "colwise":
        linear.scales = shard(linear.scales, 0)

    linear.weight = nn.Parameter(sharded_weight, requires_grad=False)
    linear.out_features, linear.in_features = linear.weight.shape
    setattr(linear, size_attr, linear.weight.shape[shard_dim])

    # Handle bias if it exists
    if linear.bias is not None and style == "colwise":
        # For column-wise sharding, the bias needs to be partitioned
        # as it's aligned with the output features
        sharded_bias = shard(linear.bias, 0)
        linear.bias = nn.Parameter(sharded_bias, requires_grad=False)


@cleanup
def apply_tp_linear_lm_head(
    linear: nn.Linear,
):
    """Shard lm_head weight."""
    local_rank = get_local_rank()
    world_size = get_world_size()

    def shard(x):
        _chunks = torch.chunk(x, world_size, dim=0)
        return _chunks[local_rank].clone().contiguous()

    sharded_weight = shard(linear.weight)

    linear.weight = nn.Parameter(sharded_weight, requires_grad=False)
    linear.out_features, linear.in_features = linear.weight.shape


@cleanup
def apply_tp_embedding(
    linear: nn.Embedding,
):
    """Shard embedding weight."""
    local_rank = get_local_rank()
    world_size = get_world_size()

    def shard(x):
        _chunks = torch.chunk(x, world_size, dim=0)
        return _chunks[local_rank].clone().contiguous()

    sharded_weight = shard(linear.weight)

    linear.weight = nn.Parameter(sharded_weight, requires_grad=False)
    linear.num_embeddings, linear.embedding_dim = linear.weight.shape


def apply_tp_ffn(mlp: FeedForward, config: ModelArgs, group, ccl) -> None:
    # must be merged first for perf
    assert hasattr(mlp, "w12")
    assert hasattr(mlp, "down_proj")

    # Megatron-style split
    apply_tp_linear_mlp(mlp.w12, "colwise")
    apply_tp_linear_mlp(mlp.down_proj, "rowwise")

    mlp.intermediate_size = config.intermediate_size
    mlp.process_group = group
    mlp.ccl = ccl


def apply_tp_attn(attn: Attention, config: ModelArgs, group, ccl) -> None:
    # must be merged first for perf
    assert hasattr(attn, "wqkv")
    assert hasattr(attn, "o_proj")

    # Megatron-style num_heads partition
    kv_size = attn.num_kv_heads * attn.head_dim
    apply_tp_linear_attn(
        linear=attn.wqkv,
        style="colwise",
        weight_splits=[attn.dim, kv_size, kv_size],
        num_kv_heads=attn.num_kv_heads,
        num_qo_heads=attn.num_qo_heads,
        head_dim=attn.head_dim,
    )
    apply_tp_linear_attn(
        linear=attn.o_proj,
        style="rowwise",
        num_kv_heads=attn.num_kv_heads,
        num_qo_heads=attn.num_qo_heads,
        head_dim=attn.head_dim,
    )

    # overwrite according to the updated config
    attn.num_qo_heads = config.num_qo_heads
    attn.dim = config.dim
    attn.head_dim = attn.dim // attn.num_qo_heads
    attn.num_kv_heads = config.num_kv_heads

    attn.process_group = group
    attn.ccl = ccl


def apply_tp_transformer(transformer: Transformer, process_group, ccl) -> None:
    # overwrite config before transformer.setup_cache is called
    num_qo_heads = transformer.config.num_qo_heads
    num_kv_heads = transformer.config.num_kv_heads
    gqa_group_size = num_qo_heads // num_kv_heads

    start, end = select_kv_heads(num_kv_heads)
    local_num_kv_heads = end - start
    local_num_qo_heads = local_num_kv_heads * gqa_group_size
    local_dim = transformer.config.hidden_size * local_num_kv_heads // num_kv_heads

    # Partition on `num_heads` will not change `head_dim` per head
    transformer.config.num_attention_heads = local_num_qo_heads
    transformer.config.hidden_size = local_dim
    transformer.config.num_key_value_heads = local_num_kv_heads

    transformer.process_group = process_group
    transformer.ccl = ccl
    transformer.world_size = dist.get_world_size(process_group)
    transformer.rank = dist.get_rank(process_group)

    # split fused w13 w/ TP
    assert transformer.config.intermediate_size % transformer.world_size == 0
    transformer.config.intermediate_size //= transformer.world_size

    # split lm-head
    assert transformer.config.vocab_size % transformer.world_size == 0
    apply_tp_linear_lm_head(transformer.lm_head)

    # split embedding
    apply_tp_embedding(transformer.embed_tokens)
    transformer.config.vocab_start_index = (
        transformer.config.vocab_size // transformer.world_size
    ) * transformer.rank
    transformer.config.vocab_end_index = (
        transformer.config.vocab_size // transformer.world_size
    ) * (transformer.rank + 1)


def apply_tp(model: Transformer, group, ccl) -> None:
    """
    Apply tensor parallelism to the model.
    Following Megatron's partitioning strategy.
    """
    apply_tp_transformer(model, group, ccl)
    for block in model.layers:
        apply_tp_ffn(block.mlp, model.config, group, ccl)
        apply_tp_attn(block.self_attn, model.config, group, ccl)


def setup_tp_worker(
    model: Transformer,
    tp_ranks: int,
    device: str,
    dtype: torch.dtype = torch.float16,
    reduce_method: str = "customized",
):
    assert 1 <= tp_ranks <= torch.cuda.device_count()
    if tp_ranks == 1:
        """Single GPU mode"""
        return model.to(device=device, dtype=dtype).eval()

    # Must called by torchrun
    _, process_group = init_dist()

    # ref: https://github.com/pytorch/rfcs/pull/17
    with torch.inference_mode(False):
        ccl: CudaCommunicator = CudaCommunicator(
            device_group=process_group, method=reduce_method
        )

    apply_tp(model, process_group, ccl)
    model = model.to(device=device, dtype=dtype)

    torch.cuda.empty_cache()
    return model.eval()


def end_tp_worker(tp_ranks: int):
    if tp_ranks == 1:
        return
    dist.barrier()
    dist.destroy_process_group()
    torch.cuda.empty_cache()
