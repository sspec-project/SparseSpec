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

# Modifed by happierpig on 03-02-2025
# Adopted from: https://github.com/Infini-AI-Lab/MagicDec/

import json
import logging
import os
from dataclasses import dataclass

from transformers import AutoConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelArgs:
    # HF-aligned names
    max_position_embeddings: int = 2048
    scaling_factor: float = 1.0
    low_freq_factor: int = None
    high_freq_factor: int = None
    vocab_size: int = 32000
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = -1
    hidden_size: int = 4096
    intermediate_size: int = None
    head_dim: int = 128
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    eos_token_id: int = -1

    # Optional extras for reference; not required by HF parity
    original_max_position_embeddings: int = None
    model_name: str = None

    def __post_init__(self):
        if self.num_key_value_heads == -1:
            self.num_key_value_heads = self.num_attention_heads
        assert self.intermediate_size is not None
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    # Backward-compat properties (temporary) to avoid breaking existing code paths
    @property
    def block_size(self):
        return self.max_position_embeddings

    @property
    def num_layers(self):
        return self.num_hidden_layers

    @property
    def num_qo_heads(self):
        return self.num_attention_heads

    @property
    def dim(self):
        return self.hidden_size

    @property
    def num_kv_heads(self):
        return self.num_key_value_heads

    @property
    def norm_eps(self):
        return self.rms_norm_eps

    @property
    def rope_base(self):
        return self.rope_theta

    @property
    def qkv_bias(self):
        return self.attention_bias

    @classmethod
    def from_name(cls, name: str):
        # Always resolve via HF AutoConfig (supports repo ids and local directories)
        ac = AutoConfig.from_pretrained(name, trust_remote_code=True)
        ret = cls._from_hf_config_dict(ac.to_dict())
        ret.model_name = name
        return ret

    @classmethod
    def _from_hf_config_dict(cls, cfg: dict):
        """Directly copy HF fields into ModelArgs (filter to dataclass fields), strict required-check."""
        allowed = set(cls.__dataclass_fields__.keys())
        params = {k: cfg[k] for k in allowed if k in cfg}
        # Backward-compat: map common HF aliases for attention bias
        if "attention_bias" not in params:
            if "qkv_bias" in cfg:
                params["attention_bias"] = cfg["qkv_bias"]
            elif "qkv_proj_bias" in cfg:
                params["attention_bias"] = cfg["qkv_proj_bias"]
        required = [
            "max_position_embeddings",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "vocab_size",
            "rope_theta",
            "rms_norm_eps",
            "eos_token_id",
        ]
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(f"Missing required HF config fields: {missing}")
        params["model_name"] = None
        return cls(**params)

    def element_size(self):
        model_size = self.model_name.split("-")[-1].lower()
        if "8b" == model_size:
            return 8 * 1e9
        elif "7b" == model_size:
            return 7 * 1e9
        elif "4b" == model_size:
            return 4 * 1e9
        elif "32b" == model_size:
            return 32 * 1e9
        elif "1.7b" == model_size:
            return 1.7 * 1e9
        elif "14b" == model_size:
            return 14 * 1e9
        else:
            raise ValueError(f"Unsupported {self.model_name}")
