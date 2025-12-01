import array
import json
import logging
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Tuple

import flashinfer
import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load
from tqdm import tqdm

from serve.distribute.communicator import CudaCommunicator
from serve.model.config import ModelArgs
from serve.model.model_perf import *
from serve.utils import ctx_torch_default, get_masked_input_and_mask

logger = logging.getLogger(__name__)


class KVPool:
    """
    Centralized key-value memory pool for all requests.
    Since use num_ranks tp on num_kv_heads, all GPUs are identical.
    Therefore, a single replica is enough.
    NOTE: physical memory is separated from the logical memory pool.
    """

    def __init__(
        self,
        config: ModelArgs,
        capacity: int,
        device: str,
        kv_cache_class,
        dtype: torch.dtype = torch.float16,
        page_size: int = 1,
        vanilla_backend: bool = False,
    ):
        self._free_gpu = set(range(capacity))
        self.num_layers = config.num_layers
        self.num_kv_heads = config.num_kv_heads
        self.num_qo_heads = config.num_qo_heads
        self.head_dim = config.head_dim
        self.capacity = capacity
        self.page_size = page_size
        self.device = device
        self.vocab_size = config.vocab_size  # for draft probs

        self._physical_mem_gpu = torch.empty(
            (
                self.num_layers,
                self.capacity,
                2,
                self.page_size,
                self.num_kv_heads,
                self.head_dim,
            ),
            dtype=dtype,
            device=device,
        )

        # NOTE(Yilong): This could be super slow on certain machines.
        self._MAX_CPU_CAPACITY = math.ceil(self.capacity * 1.5)
        self._free_cpu = array.array("i", range(self._MAX_CPU_CAPACITY))
        self._physical_mem_cpu = torch.empty(
            (
                self._MAX_CPU_CAPACITY,
                self.num_layers,
                2,
                self.page_size,
                self.num_kv_heads,
                self.head_dim,
            ),
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )

        assert self.page_size == 1, "kv_last_page_len"
        self._kv_layout: str = "NHD"
        self.vanilla_backend = vanilla_backend

        # Move init to kv-cache-ptr dispatch
        self.init_attn_backend(kv_cache_class)

        # used for cuda graph double-buffering
        self.phase_idx = 0

    def init_attn_backend(self, kv_cache_class):
        """
        Init attention backend corresponding to the kv_cache_class.
        Wrappers is a list of wrapper.
        [AttentionWrapper, BatchTopKWrapper if exist]
        """
        assert not hasattr(self, "kv_cache_class"), "kv_cache_class set"
        assert kv_cache_class._INITIALIZED == 1, "kv_cache_class not initialized"

        self.kv_cache_class = kv_cache_class
        self._wrappers, self._attn_fwd_metadata = [], []

        # double buffering cuda graph
        for _ in range(2):
            wrappers, attn_fwd_metadata = kv_cache_class.dispatch_wrappers_and_metadata(
                kv_layout=self._kv_layout,
                page_size=self.page_size,
                enable_vanilla_backend=self.vanilla_backend,
                device=self.device,
            )
            self._wrappers.append(wrappers)
            self._attn_fwd_metadata.append(attn_fwd_metadata)

        # reassign float buffer to reduce memory usage
        float_workspace_buffer = torch.empty(
            256 * 1024 * 1024,  # 256MB
            dtype=torch.uint8,
            device=self.device,
        )
        for wrappers in self._wrappers:
            # replace float buffer
            wrappers[0].float_workspace_buffer = None
            wrappers[0].float_workspace_buffer = float_workspace_buffer

        torch.cuda.empty_cache()

    @property
    def phase(self) -> int:
        return self.phase_idx

    @property
    def attn_metadata(self):
        return self._attn_fwd_metadata[self.phase]

    @property
    def wrappers(self):
        return self._wrappers[self.phase]

    @property
    def num_free_pages(self) -> int:
        return len(self._free_gpu)

    def step(self):
        self.phase_idx ^= 1

    def alloc_page(self) -> int:
        assert len(self._free_gpu) > 0, "Out of memory"
        idx = self._free_gpu.pop()
        return idx

    def free_page(self, idx: int):
        assert idx not in self._free_gpu, f"Page {idx} already free"
        assert 0 <= idx < self.capacity
        self._free_gpu.add(idx)

    def alloc_page_cpu(self):
        assert len(self._free_cpu) > 0, "Out of memory"
        idx = self._free_cpu.pop()
        return idx

    def free_page_cpu(self, idx: int):
        assert idx not in self._free_cpu, f"Page {idx} already restored"
        assert 0 <= idx < self._physical_mem_cpu.shape[0]
        self._free_cpu.append(idx)

    def offload(self, idx_gpu, idx_cpu):
        self._physical_mem_cpu[idx_cpu[0] : idx_cpu[-1] + 1, ...].copy_(
            self._physical_mem_gpu[:, idx_gpu, ...].transpose(0, 1), non_blocking=True
        )

    def restore(self, idx_cpu, idx_gpu):
        self._physical_mem_gpu[:, idx_gpu, ...] = (
            self._physical_mem_cpu[idx_cpu[0] : idx_cpu[-1] + 1, ...]
            .to(self.device, non_blocking=True)
            .transpose(0, 1)
        )

    def apply_batch_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        config: ModelArgs,
    ):
        assert hasattr(self, "kv_cache_class")
        assert config.high_freq_factor is None and config.low_freq_factor is None

        batch_size = self.attn_metadata["batch_size_host"]
        if self.vanilla_backend:
            # not cuda graph compatible
            return flashinfer.apply_rope(
                q=q,
                k=k,
                indptr=self.attn_metadata["qo_indptr"][: batch_size + 1],
                offsets=self.attn_metadata["pos_offsets"][:batch_size],
                interleave=False,
                rope_scale=config.scaling_factor,
                rope_theta=config.rope_base,
            )
        else:
            # this is customized cuda-graph compatiable version
            flashinfer.apply_rope_persistent_in_place(
                q=q,
                k=k,
                indptr=self.attn_metadata["qo_indptr"][: batch_size + 1],
                offsets=self.attn_metadata["pos_offsets"][:batch_size],
                batch_size=self.attn_metadata["batch_size_tensor"],
                interleave=False,
                rope_scale=config.scaling_factor,
                rope_theta=config.rope_base,
            )
            return q, k

    def append_batch_kv_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
    ):
        assert hasattr(self, "kv_cache_class")
        self.kv_cache_class.dispatch_append_kv_cache(
            k=k,
            v=v,
            layer_idx=layer_idx,
            kv_pool=self,
        )

    def pre_attn_fwd(
        self,
        **kwargs,
    ):
        """
        Prepare metadata for the forward pass. Called once per iter.
        Use kwargs for variable arguments (modular).
        """
        assert hasattr(self, "kv_cache_class")
        assert self.page_size == 1

        nvtx.range_push("pre_attn_fwd")
        self.kv_cache_class.dispatch_attn_plan(
            wrappers=self.wrappers,
            attn_fwd_metadata=self.attn_metadata,
            kv_pool=self,
            **kwargs,
        )
        nvtx.range_pop()

    def batch_attn_fwd(
        self,
        q: torch.Tensor,
        layer_idx: int,
    ):
        assert hasattr(self, "kv_cache_class")
        assert q.dtype == self._physical_mem_gpu.dtype

        out = self.kv_cache_class.dispatch_attn_fwd(
            wrappers=self.wrappers,
            attn_fwd_metadata=self.attn_metadata,
            kv_pool=self,
            q=q,
            layer_idx=layer_idx,
        )
        return out

    def output_snapshot(self, input_metadata: dict) -> dict:
        assert hasattr(self, "kv_cache_class")
        return self.kv_cache_class.dispatch_output_snapshot(
            wrappers=self.wrappers,
            attn_fwd_metadata=self.attn_metadata,
            input_metadata=input_metadata,
        )


class Transformer(nn.Module):

    def __init__(
        self, config: ModelArgs, device: str = "cpu", dtype=torch.float16
    ) -> None:
        super().__init__()

        with ctx_torch_default(device=device, dtype=dtype):
            self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
            self.layers = nn.ModuleList(
                TransformerBlock(config) for _ in range(config.num_layers)
            )
            self.norm = RMSNorm(config.dim, eps=config.norm_eps)
            self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

            # Tie embeddings if requested by config (common for Qwen3 checkpoints)
            if getattr(config, "tie_word_embeddings", False):
                self.lm_head.weight = self.embed_tokens.weight

        self.config = config
        self.world_size = None
        self.rank = None
        self.process_group = None
        self.ccl: CudaCommunicator | None = None

    def attach_kv_pool(self, kv_pool: KVPool):
        self.kv_cache = kv_pool
        for layer_idx, b in enumerate(self.layers):
            b.self_attn.kv_cache = kv_pool
            b.self_attn.layer_idx = layer_idx

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        assert hasattr(self, "kv_cache"), "KVPool not attached"
        assert idx.dim() == 1, "input_ids are flattened with indptr"

        # shard embed_tokens in TP way
        # such vocab metadata is assigned when sharding
        if self.process_group is not None:
            idx, mask = get_masked_input_and_mask(
                idx, self.config.vocab_start_index, self.config.vocab_end_index
            )
        x = self.embed_tokens(idx)
        if self.process_group is not None:
            x.masked_fill_(mask.unsqueeze(-1), 0)
            x = self.ccl.all_reduce(x)

        for block in self.layers:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        # below shards lm-head in TP way
        if self.process_group is not None:
            assert logits.dim() == 2, "logits are sharded"
            num_tokens, sharded_vocab_size = logits.size()
            assert sharded_vocab_size * self.world_size == self.config.vocab_size

            # NOTE(Yilong): this layout must be handled by customized sampler!!
            recv_buf = torch.empty(
                self.config.vocab_size * num_tokens,
                dtype=logits.dtype,
                device=logits.device,
            )
            dist.all_gather_into_tensor(
                recv_buf, logits.view(-1), group=self.process_group
            )
            logits = recv_buf.view(num_tokens, self.config.vocab_size)

        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def _strip_prefix_if_present(
        state_dict: dict[str, torch.Tensor], prefix: str = "model."
    ) -> dict[str, torch.Tensor]:
        """
        Remove `prefix` from every key that starts with it and return a *dict*.
        """
        if not any(k.startswith(prefix) for k in state_dict):
            return state_dict  # nothing to change

        return {
            (
                k[len(prefix) :] if k.startswith(prefix) else k
            ): v  # keep key→value pairs!
            for k, v in state_dict.items()
        }

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str | os.PathLike,
        *,
        revision: str = "main",
        subfolder: str = "",
        device="cpu",
        dtype=torch.float32,
        strict: bool = False,
        download_kwargs: dict | None = None,
        **model_kwargs,
    ):
        download_kwargs = download_kwargs or {}

        repo_path = Path(repo_id)
        use_local = repo_path.exists() and repo_path.is_dir()

        cfg_path = (
            repo_path / subfolder / "config.json"
            if use_local
            else hf_hub_download(
                repo_id,
                "config.json",
                revision=revision,
                subfolder=subfolder,
                **download_kwargs,
            )
        )

        def _load_file(path: os.PathLike) -> dict[str, torch.Tensor]:
            if str(path).endswith(".safetensors"):
                return safe_load(str(path), device="cpu")
            return torch.load(str(path), map_location="cpu", weights_only=True)

        def _fetch(path_or_name: str) -> Path:
            if use_local:
                return repo_path / subfolder / path_or_name
            return Path(
                hf_hub_download(
                    repo_id,
                    path_or_name,
                    revision=revision,
                    subfolder=subfolder,
                    **download_kwargs,
                )
            )

        try:
            index_file = _fetch("model.safetensors.index.json")
            with open(index_file, "r") as f:
                index_data = json.load(f)
            shard_names = sorted(set(index_data["weight_map"].values()))
            weight_files = [_fetch(name) for name in shard_names]
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            # fall back to single‑file checkpoint logic
            cand = ["model.safetensors", "pytorch_model.bin"]
            for name in cand:
                try:
                    weight_files = [_fetch(name)]
                    break
                except FileNotFoundError:
                    continue
            else:  # still nothing? look for *-of-*.safetensors locally
                safes = sorted((repo_path / subfolder).glob("*-of-*.safetensors"))
                if not safes:
                    raise FileNotFoundError("No checkpoint shards found.")
                weight_files = safes

        state_dict: dict[str, torch.Tensor] = OrderedDict()
        for wf in weight_files:
            part = _load_file(wf)
            overlap = state_dict.keys() & part.keys()
            if overlap:
                raise ValueError(
                    f"Duplicate keys across shards: {sorted(overlap)[:5]}…"
                )
            state_dict.update(part)

        def _strip_prefix(
            d: dict[str, torch.Tensor], prefix: str
        ) -> dict[str, torch.Tensor]:
            if not any(k.startswith(prefix) for k in d):
                return d
            return {
                (k[len(prefix) :] if k.startswith(prefix) else k): v
                for k, v in d.items()
            }

        for p in ("model.", "module."):
            state_dict = _strip_prefix(state_dict, p)

        # If checkpoint is tied and doesn't store lm_head.weight explicitly,
        # alias it from embed_tokens.weight to avoid missing-weights warnings.
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"]

        ckpt_dtype = next(iter(state_dict.values())).dtype
        # Infer presence of q/k/v biases from checkpoint and align config
        has_qkv_bias = any(
            key.endswith("self_attn.q_proj.bias")
            or key.endswith("self_attn.k_proj.bias")
            or key.endswith("self_attn.v_proj.bias")
            for key in state_dict.keys()
        )

        config = ModelArgs.from_name(str(cfg_path))
        if has_qkv_bias:
            config.attention_bias = True

        model = cls(config, device="cpu", dtype=ckpt_dtype)  # avoid frequent H2D

        missing, unexpected = model.load_state_dict(
            state_dict, strict=strict, assign=True
        )
        if missing:
            # Suppress missing lm_head when embeddings are tied or aliased
            if not (
                len(missing) == 1
                and missing[0] == "lm_head.weight"
                and (
                    getattr(model.config, "tie_word_embeddings", False)
                    or "lm_head.weight" in state_dict
                )
            ):
                logger.warning(f"Missing weights: {missing}")
        if unexpected:
            logger.warning(f"Unexpected weights: {unexpected}")

        model.eval().to(device=device, dtype=dtype)

        # start merging continous weights
        for block in tqdm(model.layers, desc="Merging weights"):
            block.self_attn.merge()
            block.mlp.merge()
        torch.cuda.empty_cache()

        return model


class TransformerBlock(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.input_layernorm = RMSNorm(config.dim, config.norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.dim, config.norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x) + residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x) + residual
        return x


class Attention(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        assert config.dim % config.num_qo_heads == 0

        # Projections: respect attention_bias from config to match checkpoints
        self.q_proj = nn.Linear(
            config.dim,
            config.num_qo_heads * config.head_dim,
            bias=getattr(config, "attention_bias", getattr(config, "qkv_bias", False)),
        )
        self.k_proj = nn.Linear(
            config.dim,
            config.num_kv_heads * config.head_dim,
            bias=getattr(config, "attention_bias", getattr(config, "qkv_bias", False)),
        )
        self.v_proj = nn.Linear(
            config.dim,
            config.num_kv_heads * config.head_dim,
            bias=getattr(config, "attention_bias", getattr(config, "qkv_bias", False)),
        )
        # For some models (e.g., Qwen3-4B), num_qo_heads * head_dim != dim.
        # The attention output is of size [seqlen, num_qo_heads * head_dim],
        # so the projection should map from that to hidden dim.
        self.o_proj = nn.Linear(
            config.num_qo_heads * config.head_dim, config.dim, bias=False
        )

        # distributed metadata
        self.process_group = None
        self.ccl: CudaCommunicator | None = None

        # kv cache hooks
        self.kv_cache: KVPool | None = None
        self.layer_idx: int | None = None

        # quick aliases
        self.num_qo_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.dim = config.dim

        # Only Qwen3 uses per-head q/k RMSNorm (q_norm/k_norm)
        name = str(getattr(config, "model_name", ""))
        if "qwen3" in name.lower():
            self.q_norm = RMSNorm(config.head_dim, eps=config.norm_eps)
            self.k_norm = RMSNorm(config.head_dim, eps=config.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def merge(self):
        """
        This function is called after loading the model weights to
        merge wq,wk,wv into one tensor for a single GEMM call.
        """
        assert not hasattr(self, "wqkv")
        if_qkv_bias = self.q_proj.bias is not None
        dtype = self.q_proj.weight.dtype
        device = self.q_proj.weight.device

        wqkv_weight = torch.cat(
            [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0
        )
        if if_qkv_bias:
            wqkv_bias = torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], dim=0
            )
        else:
            wqkv_bias = None

        self.wqkv = nn.Linear(
            self.q_proj.in_features,
            self.q_proj.out_features
            + self.k_proj.out_features
            + self.v_proj.out_features,
            bias=if_qkv_bias,
            dtype=dtype,
            device=device,
        )

        with torch.no_grad():
            self.wqkv.weight.copy_(wqkv_weight)
            if if_qkv_bias:
                self.wqkv.bias.copy_(wqkv_bias)

        # clean memory
        del self.q_proj
        del self.k_proj
        del self.v_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seqlen, _ = x.shape

        if hasattr(self, "wqkv"):
            q_size = self.num_qo_heads * self.head_dim
            kv_size = self.num_kv_heads * self.head_dim
            q, k, v = self.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)
            q = q.view(seqlen, self.num_qo_heads, self.head_dim)
            k = k.view(seqlen, self.num_kv_heads, self.head_dim)
            v = v.view(seqlen, self.num_kv_heads, self.head_dim)
        else:
            q = self.q_proj(x).view(seqlen, self.num_qo_heads, self.head_dim)
            k = self.k_proj(x).view(seqlen, self.num_kv_heads, self.head_dim)
            v = self.v_proj(x).view(seqlen, self.num_kv_heads, self.head_dim)

        # Optional per-head RMSNorm on q/k (e.g., Qwen3). Apply before RoPE.
        if getattr(self, "q_norm", None) is not None:
            q = self.q_norm.forward_qk(q)
        if getattr(self, "k_norm", None) is not None:
            k = self.k_norm.forward_qk(k)

        # Rope + append KV cache
        q, k = self.kv_cache.apply_batch_rope(q, k, self.config)
        self.kv_cache.append_batch_kv_cache(k, v, self.layer_idx)

        # perform computation
        y = self.kv_cache.batch_attn_fwd(q, self.layer_idx)

        # Flatten heads before output projection. Do NOT assume heads*head_dim == dim.
        y = y.contiguous().view(seqlen, self.num_qo_heads * self.head_dim)
        y = self.o_proj(y)

        if self.process_group is not None:
            y = self.ccl.all_reduce(y)
        return y


class FeedForward(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden = config.intermediate_size
        self.gate_proj = nn.Linear(config.dim, hidden, bias=False)
        self.up_proj = nn.Linear(config.dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.dim, bias=False)

        self.process_group = None
        self.ccl: CudaCommunicator | None = None

    def merge(self):
        """
        This function is called after loading the model weights to
        merge [up, gate] into one tensor for a single GEMM call.
        """
        assert not hasattr(self, "w12")
        dtype = self.gate_proj.weight.dtype
        device = self.gate_proj.weight.device
        if_bias = self.gate_proj.bias is not None

        w12_weight = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
        if if_bias:
            w12_bias = torch.cat([self.gate_proj.bias, self.up_proj.bias], dim=0)
        else:
            w12_bias = None

        self.w12 = nn.Linear(
            self.gate_proj.in_features,
            self.gate_proj.out_features + self.up_proj.out_features,
            bias=if_bias,
            dtype=dtype,
            device=device,
        )

        with torch.no_grad():
            self.w12.weight.copy_(w12_weight)
            if if_bias:
                self.w12.bias.copy_(w12_bias)

        # clean memory
        del self.gate_proj
        del self.up_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "w12"):
            y = self.down_proj(flashinfer.activation.silu_and_mul(self.w12(x)))
        else:
            a, b = self.gate_proj(x), self.up_proj(x)
            y = flashinfer.activation.silu_and_mul(torch.cat([a, b], dim=-1))
            y = self.down_proj(y)

        if self.process_group is not None:
            y = self.ccl.all_reduce(y)
        return y


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return flashinfer.norm.rmsnorm(x, self.weight, self.eps)

    def forward_qk(self, x: torch.Tensor) -> torch.Tensor:
        return flashinfer.norm.qk_rmsnorm(x, self.weight, self.eps)

    def forward_fused_residual(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        return flashinfer.norm.fused_add_rmsnorm(x, residual, self.weight, self.eps)
