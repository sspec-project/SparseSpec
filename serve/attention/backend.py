import functools
import math
import os
import pathlib
from typing import Optional, Tuple, Union

import jinja2
import torch
from flashinfer.attention import BatchAttention, get_customize_batch_attention_module
from flashinfer.jit import gen_customize_batch_attention_score_module
from flashinfer.jit.core import JitSpec, gen_jit_spec, logger
from flashinfer.jit.utils import filename_safe_dtype_map, write_if_different
from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
from flashinfer.utils import PosEncodingMode, TensorLayout, _unpack_paged_kv_cache


class BatchTopKAttention(BatchAttention):
    core_variant_name: str = "BatchTopKAttention"
    core_variant_decl: str = """
    struct BatchTopKAttention : AttentionVariantBase {
        float sm_scale_log2;
        uint32_t qo_len, kv_len;
        uint32_t request_type;

        PROFILER_CLOSURE_PARAMS_DECL

        template <typename Params>
        __device__ __host__ BatchTopKAttention(const Params& params, uint32_t work_idx,
                                                uint8_t* smem_ptr) {
            uint32_t batch_idx = params.batch_idx_arr[work_idx];
            sm_scale_log2 = params.sm_scale * math::log2e;
            qo_len = params.get_qo_len(work_idx);
            kv_len = params.get_kv_len(work_idx);
            request_type = params.request_type[batch_idx];

        }
        REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
            return logits;
        })
        REGISTER_INPUT_TRANSFORM(params, batch_idx, kv_head_idx, indices, {
            // NORMAL(0), DRAFT(1), VERIFY(2)
            if (request_type == 1){
                indices = params.flatten_indices + kv_head_idx * {{MAX_TOTAL_DRAFT_KV_LEN}};
            }else{
                indices = params.kv_indices;
            }
            return;
        })
        REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
            return true;
        })
    };
    """
    score_variant_name: str = "BatchAttentionScore"
    score_variant_decl: str = """
    struct BatchAttentionScore : AttentionVariantBase {
        float sm_scale_log2;
        uint32_t qo_len, kv_len;
        uint32_t request_type;
        uint32_t dump_logits_offset;

        PROFILER_CLOSURE_PARAMS_DECL

        template <typename Params>
        __device__ __host__ BatchAttentionScore(const Params& params, uint32_t work_idx,
                                                uint8_t* smem_ptr) {
            uint32_t batch_idx = params.batch_idx_arr[work_idx];
            uint32_t kv_head_idx = params.kv_head_idx_arr[work_idx];

            sm_scale_log2 = params.sm_scale * math::log2e;
            qo_len = params.get_qo_len(work_idx);
            kv_len = params.get_kv_len(work_idx);
            request_type = params.request_type[batch_idx];
            dump_logits_offset = kv_head_idx * {{MAX_TOTAL_VERIFY_KV_LEN}} + params.verify_kv_indptr[batch_idx];
        }
        REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
            // NORMAL(0), DRAFT(1), VERIFY(2)
            // dump logits if verification
            if (request_type == 2){
                // dump_logits: [num_kv_heads, MAX_TOTAL_VERIFY_KV_LEN]
                // should be only have qo_idx == 0 (no multi-qo-tile supported)
                // second tile will be skipped (assert in `pillar.py`)
                uint32_t packed_qo_len = params.gqa_group_size * min(qo_len, kv_len - kv_idx);
                if (qo_idx == 0 && kv_idx < kv_len){
                    // get average score
                    // use fp16 to reduce memory footprint (safe for normalized scores)
                    params.dump_logits[dump_logits_offset + kv_idx] = __float2half(logits / packed_qo_len);
                }
            }
            return logits;
        })
        REGISTER_INPUT_TRANSFORM(params, batch_idx, kv_head_idx, indices, {
            return;
        })
        REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
            return true;
        })
    };
    """

    def __init__(
        self,
        kv_layout: str = "NHD",
        device: str = "cuda",
        # const lifetime parameters
        num_kv_heads: int = 1,
        num_qo_heads: int = 1,
        head_dim: int = 128,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: torch.dtype = torch.bfloat16,
        MAX_TOTAL_DRAFT_KV_LEN: int = 64 * 1024,
        MAX_TOTAL_VERIFY_KV_LEN: int = 512 * 1024,
        MAX_SPEC_STRIDE: int = 16,
    ):
        # init attention score kernel first
        _score_variant_name = BatchTopKAttention.score_variant_name
        _score_variant_decl = jinja2.Template(
            BatchTopKAttention.score_variant_decl
        ).render(
            MAX_TOTAL_VERIFY_KV_LEN=MAX_TOTAL_VERIFY_KV_LEN,
        )
        score_uri = (
            f"batch_attention_score_{filename_safe_dtype_map[q_data_type]}"
            f"_{filename_safe_dtype_map[kv_data_type]}"
            f"_qo_heads_{num_qo_heads}_kv_heads_{num_kv_heads}"
            f"_head_dim_{head_dim}"
            f"_verify_kv_len_{MAX_TOTAL_VERIFY_KV_LEN}"
        )
        score_jit_args = [
            score_uri,  # uri
            q_data_type,  # dtype_q
            kv_data_type,  # dtype_kv
            q_data_type,  # dtype_o
            torch.int32,  # idtype
            head_dim,  # hidden_dim_qk
            head_dim,  # hidden_dim_vo
            [
                "request_type",
                "verify_kv_indptr",
                "dump_logits",
            ],  # additional_tensor_names
            ["int32_t", "int32_t", "half"],  # additional_tensor_dtypes
            ["sm_scale"],  # additional_scalar_names
            ["double"],  # additional_scalar_dtypes
            _score_variant_name,
            _score_variant_decl,
        ]
        self.score_jit_module = gen_customize_batch_attention_score_module(
            *score_jit_args,
        ).build_and_load()

        # Two major modification compared to naive BatchAttention:
        # 1. InputTransform is customized to redirect input indices.
        # indices + flatten_indices [num_layers, num_kv_heads, MAX_TOTAL_DRAFT_KV_LEN]
        # flatten_indices will be moved to corresponding `layer_idx` and `kv_head_idx` offset
        # 2. Dump logits into [MAX_TOTAL_VERIFY_KV_LEN, num_kv_heads, gqa_group_size, MAX_SPEC_STRIDE]
        # All operations will depends on the request type [NORMAL, DRAFT, VERIFY]
        _variant_name = BatchTopKAttention.core_variant_name
        _variant_decl = jinja2.Template(BatchTopKAttention.core_variant_decl).render(
            num_qo_heads=num_qo_heads,
            gqa_group_size=num_qo_heads // num_kv_heads,
            MAX_SPEC_STRIDE=MAX_SPEC_STRIDE,
            MAX_TOTAL_DRAFT_KV_LEN=MAX_TOTAL_DRAFT_KV_LEN,
            MAX_TOTAL_VERIFY_KV_LEN=MAX_TOTAL_VERIFY_KV_LEN,
        )
        uri = (
            f"batch_topk_attention_{filename_safe_dtype_map[q_data_type]}"
            f"_{filename_safe_dtype_map[kv_data_type]}"
            f"_qo_heads_{num_qo_heads}_kv_heads_{num_kv_heads}"
            f"_head_dim_{head_dim}"
            f"_spec_stride_{MAX_SPEC_STRIDE}"
            f"_draft_kv_len_{MAX_TOTAL_DRAFT_KV_LEN}"
        )
        jit_args = [
            uri,  # uri
            q_data_type,  # dtype_q
            kv_data_type,  # dtype_kv
            q_data_type,  # dtype_o
            torch.int32,  # idtype
            head_dim,  # hidden_dim_qk
            head_dim,  # hidden_dim_vo
            [
                "flatten_indices",
                "request_type",
            ],  # additional_tensor_names
            ["int32_t", "int32_t"],  # additional_tensor_dtypes
            ["sm_scale"],  # additional_scalar_names
            ["double"],  # additional_scalar_dtypes
            _variant_name,
            _variant_decl,
        ]
        jit_kwargs = {
            "pos_encoding_mode": PosEncodingMode["NONE"].value,
            "use_logits_soft_cap": False,
            "use_profiler": False,
        }

        self.MAX_TOTAL_DRAFT_KV_LEN = MAX_TOTAL_DRAFT_KV_LEN
        self.MAX_TOTAL_VERIFY_KV_LEN = MAX_TOTAL_VERIFY_KV_LEN
        self.MAX_SPEC_STRIDE = MAX_SPEC_STRIDE

        # dispatch to PersistentAttentionScorePersistent
        # choose both accurate and efficient `CTA_TILE_Q`
        _max_len_verify_packed_qo = num_qo_heads // num_kv_heads * self.MAX_SPEC_STRIDE
        if _max_len_verify_packed_qo <= 48:
            self.DISPATCH_CTA_TILE_Q = 32
        else:
            self.DISPATCH_CTA_TILE_Q = 64

        if self.DISPATCH_CTA_TILE_Q < _max_len_verify_packed_qo:
            logger.warning(
                f"[LOGITS DUMP KERNEL]: CTA_TILE_Q={self.DISPATCH_CTA_TILE_Q} is not enough "
                f"for max_len_verify_packed_qo={_max_len_verify_packed_qo}."
                f"This potentially impact the accuracy of speculation."
            )

        super().__init__(kv_layout, device, jit_args, jit_kwargs)

    def run_attention(
        self,
        q: torch.Tensor,
        kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        kv_indices: torch.Tensor,
        *args,
    ):
        head_dim_qk = q.shape[2]
        if self._sm_scale is None:
            self._sm_scale = 1.0 / math.sqrt(head_dim_qk)
        # add sm_scale
        out, lse = super().run(
            q,
            kv_cache,
            kv_indices,
            None,
            None,
            *args,
            self._sm_scale,
        )

        return out, lse

    """
    There are essentially two ways to dump attention scores from verification attention:
    1. directly dump packed logits with shape [qo_len*spec_stride*gqa_grou_size, kv_len] into global memory,
        which is then normalized by lse and reduce across first dimension to get the TopK tokens.
    2. launch a separate kernel to load Q and K again, with the previous lse to directly compute
        the reduced attention scores.
    In practice, this two have similar performance. Therefore, we choose the second approach for maintainability,
    which is implemented as `rematerialize_scores` method below.
    """

    def rematerialize_scores(
        self,
        q: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        kv_indices: torch.Tensor,
        *args,
    ):
        k_cache, v_cache = _unpack_paged_kv_cache(kv_cache, self._kv_layout)

        head_dim_qk = q.shape[2]
        if self._sm_scale is None:
            self._sm_scale = 1.0 / math.sqrt(head_dim_qk)

        # prepare run args
        run_args = [
            self.float_workspace_buffer,
            self.int_workspace_buffer,
            self._plan_info,
            q,
            k_cache,
            v_cache,
            kv_indices,
            out,
            lse,
            self._mask_mode,
            TensorLayout[self._kv_layout].value,
            self._num_qo_heads,
            self._num_kv_heads,
            self._page_size,
            self.DISPATCH_CTA_TILE_Q,
        ]
        # append customized args
        run_args.extend(args)
        # append sm_scale
        run_args.extend([self._sm_scale])

        # execute
        self.score_jit_module.run(*run_args)


@functools.cache
def gen_topk_op_module(uri: str) -> JitSpec:
    # configure jit env variable
    from flashinfer.jit import env as jit_env

    _package_root = pathlib.Path(__file__).resolve().parents[2]
    SSPEC_CSRC_DIR = _package_root / "serve" / "attention" / "csrc"
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri

    # try to find RAFT_INCLUDE_DIR from env variable
    CONDA_INSTALL_INCLUDE_DIR = (
        os.environ.get("INSTALL_PREFIX")
        or os.environ.get("PREFIX")
        or os.environ.get("CONDA_PREFIX")
    )
    assert CONDA_INSTALL_INCLUDE_DIR is not None, "CONDA_INSTALL_INCLUDE_DIR is not set"
    CONDA_INSTALL_INCLUDE_DIR = pathlib.Path(CONDA_INSTALL_INCLUDE_DIR)
    RAFT_INCLUDE_DIR = CONDA_INSTALL_INCLUDE_DIR / "include"
    logger.info(f"RAFT_INCLUDE_DIR: {RAFT_INCLUDE_DIR}")

    # try to init rmm
    assert (
        CONDA_INSTALL_INCLUDE_DIR / "lib" / "librmm.so"
    ).exists(), "librmm.so is not found"

    # copy corresponding source files
    os.makedirs(gen_directory, exist_ok=True)
    source_paths = []
    for filename in ["ops.cu"]:
        src_path = SSPEC_CSRC_DIR / filename
        dest_path = gen_directory / filename

        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)

    return gen_jit_spec(
        uri,
        source_paths,
        extra_include_paths=[
            SSPEC_CSRC_DIR,
            RAFT_INCLUDE_DIR,
        ],
        extra_cuda_cflags=[
            "-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE",
        ],
        extra_ldflags=[f"-L{CONDA_INSTALL_INCLUDE_DIR/'lib'}", "-lrmm"],
    )


class BatchTopKWrapper:
    def __init__(
        self,
        float_workspace_buffer: Optional[torch.Tensor] = None,
        MAX_TOTAL_DRAFT_KV_LEN: int = 64 * 1024,
        MAX_TOTAL_VERIFY_KV_LEN: int = 512 * 1024,
        MAX_DRAFT_KV_LEN: int = 8192,
    ):
        if float_workspace_buffer is None:
            self.float_workspace_buffer = torch.empty(
                64 * 1024 * 1024,
                dtype=torch.uint8,
                device="cuda",
            )
        else:
            self.float_workspace_buffer = float_workspace_buffer

        # jit build topk
        self.uri = "batch_topk_csr"
        self._cached_module = gen_topk_op_module(self.uri).build_and_load()

        self.MAX_TOTAL_DRAFT_KV_LEN = MAX_TOTAL_DRAFT_KV_LEN
        self.MAX_TOTAL_VERIFY_KV_LEN = MAX_TOTAL_VERIFY_KV_LEN
        self.MAX_DRAFT_KV_LEN = MAX_DRAFT_KV_LEN

    def run(
        self,
        in_val: torch.Tensor,
        in_idx: torch.Tensor,
        in_val_start_pos: torch.Tensor,
        in_idx_start_pos: torch.Tensor,
        in_lens: torch.Tensor,
        ks: torch.Tensor,
        num_requests: int,
        out_val: Optional[torch.Tensor] = None,
        out_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        saved_batch_shapes = in_val.shape[:-1]
        batch_size = math.prod(saved_batch_shapes)

        assert in_val.shape[-1] == self.MAX_TOTAL_VERIFY_KV_LEN
        in_val = in_val.reshape(batch_size, self.MAX_TOTAL_VERIFY_KV_LEN).to(
            torch.float16
        )

        if out_val is None:
            out_val = torch.empty(
                num_requests,
                batch_size,
                self.MAX_DRAFT_KV_LEN,
                device=in_val.device,
                dtype=in_val.dtype,
            )

        if out_idx is None:
            out_idx = torch.empty(
                num_requests,
                batch_size,
                self.MAX_DRAFT_KV_LEN,
                device=in_val.device,
                dtype=in_idx.dtype,
            )

        assert out_val.shape == (num_requests, batch_size, self.MAX_DRAFT_KV_LEN)
        assert out_idx.shape == (num_requests, batch_size, self.MAX_DRAFT_KV_LEN)
        assert in_val.dtype == out_val.dtype
        assert in_idx.dtype == out_idx.dtype

        self._cached_module.batch_topk(
            in_val,
            in_idx,
            in_val_start_pos,
            in_idx_start_pos,
            in_lens,
            ks,
            num_requests,
            out_val,
            out_idx,
            self.float_workspace_buffer,
        )

        # reshape back
        out_val = out_val.reshape(
            (num_requests,) + saved_batch_shapes + (self.MAX_DRAFT_KV_LEN,)
        )
        out_idx = out_idx.reshape(
            (num_requests,) + saved_batch_shapes + (self.MAX_DRAFT_KV_LEN,)
        )

        return out_val, out_idx
