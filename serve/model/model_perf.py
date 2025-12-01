import bisect
import json
import logging
from typing import List, Tuple

import torch

from serve.model.config import ModelArgs

logger = logging.getLogger(__name__)


class ModelPerf:

    def __init__(self):
        self._init_flag: bool = False

    @property
    def init(self) -> bool:
        return self._init_flag

    @classmethod
    def get_device_name(cls) -> str:
        assert torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        return gpu_name.replace(" ", "_")

    @classmethod
    def get_profile_name(cls, config: ModelArgs, tp_ranks: int) -> str:
        return f"{cls.get_device_name()}_{config.model_name}_tp{tp_ranks}.jsonl"

    @classmethod
    def get_device_property(cls) -> dict:
        assert torch.cuda.is_available()
        gpu_name = cls.get_device_name()

        match gpu_name:
            case "NVIDIA_H200":
                prop = {
                    "mem_size": 141 * 1e9,
                    "mem_bandwidth": 4.8 * 1e12,
                    "peak_tensor_fp16": 989.4 * 1e12,
                    "net_bandwidth": 900 * 1e9,
                }
            case "NVIDIA_H100_80GB_HBM3":
                prop = {
                    "mem_size": 80 * 1e9,
                    "mem_bandwidth": 3.352 * 1e12,
                    "peak_tensor_fp16": 989.4 * 1e12,
                    "net_bandwidth": 900 * 1e9,
                }
            case "NVIDIA_A100":
                # SXM-80GB
                prop = {
                    "mem_size": 80 * 1e9,
                    "mem_bandwidth": 2.039 * 1e12,
                    "peak_tensor_fp16": 312 * 1e12,
                    "net_bandwidth": 600 * 1e9,
                }
            case _:
                assert False, f"Unsupported GPU: {gpu_name}"
        return prop

    @classmethod
    def get_all_gemm_shapes(
        cls, tp_ranks: int, config: ModelArgs
    ) -> List[Tuple[int, int]]:
        shapes = {}  # (N,K)

        assert tp_ranks > 0
        assert config.num_kv_heads % tp_ranks == 0

        # Megatron-LM
        w_qkv = (
            config.head_dim
            * (config.num_qo_heads + config.num_kv_heads * 2)
            // tp_ranks,
            config.dim,
        )
        w_o = (config.dim, config.head_dim * config.num_qo_heads // tp_ranks)
        w_ffn_up = (config.intermediate_size // tp_ranks, config.dim)
        w_ffn_down = (config.dim, config.intermediate_size // tp_ranks)

        shapes[w_qkv] = True
        shapes[w_o] = True
        shapes[w_ffn_up] = True
        shapes[w_ffn_down] = True

        return list(shapes.keys())

    def initialize(self, tp_ranks: int, config: ModelArgs, path: str = "."):
        assert not self.init
        self._init_flag = True

        self.gpu_prop = self.get_device_property()
        profile_perf_path = f"{path}/{self.get_profile_name(config, tp_ranks)}"

        def load_gemm_profile(path: str) -> dict:
            # Tuple [int, int] -> List[Tuple[int, int]]
            gemm_profile: dict = {}
            with open(path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    key = (data["n"], data["k"])
                    if key not in gemm_profile:
                        gemm_profile[key] = []
                    gemm_profile[key].append((data["m"], data["latency"]))
            # sort each list by m
            for key in gemm_profile:
                gemm_profile[key].sort(key=lambda x: x[0])
            assert len(gemm_profile) > 0

            return gemm_profile

        self.gemm_profile = load_gemm_profile(profile_perf_path)

        logger.info(f"ModelPerf initialized with {profile_perf_path}")
        logger.info(f"GPU Property: {self.gpu_prop}")
        logger.info(f"GEMM NxK: {list(self.gemm_profile.keys())}")

    def eval_gemm(self, m: int, n: int, k: int) -> float:
        assert self.init

        key = (n, k)
        if key not in self.gemm_profile:
            assert False, f"(N,K) {key} not found in GEMM profile"

        def get_latency_by_bisect(srcData: List[Tuple[int, float]], trtM) -> float:
            srcMs = [x[0] for x in srcData]
            idx = bisect.bisect_left(srcMs, trtM)

            if idx == 0:
                return srcData[0][1]
            if idx == len(srcData):
                return srcData[-1][1]

            m_low, latency_low = srcData[idx - 1]
            m_high, latency_high = srcData[idx]

            # weighted average
            weight_high = (trtM - m_low) / (m_high - m_low)
            weight_low = 1 - weight_high

            # return in seconds, while the profile is in ms
            return weight_low * latency_low + weight_high * latency_high

        return get_latency_by_bisect(self.gemm_profile[key], m) * 1e-3

    def eval_mem(self, size: int, utils: float = 0.75) -> float:
        assert self.init
        return size / (self.gpu_prop["mem_bandwidth"] * utils)

    def eval_net(self, size: int, utils: float = 0.75) -> float:
        assert self.init
        return size / (self.gpu_prop["net_bandwidth"] * utils)


model_perf = ModelPerf()
