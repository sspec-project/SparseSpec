import argparse
import copy
import json
import os
import random

# Import our custom inference function
import sys
import time
from datetime import datetime
from parser import *

import torch
import wandb
from data_loader import load_data
from evaluate import evaluate
from model_utils import generate_completions, load_hf_lm_and_tokenizer
from python_executor import PythonExecutor
from tqdm import tqdm
from trajectory import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import construct_prompt, load_jsonl, save_jsonl, set_seed
from vllm import LLM, SamplingParams

from serve.request.request import Request
from serve.run import run_inference
from serve.utils import WandbLogger, is_first_rank


def parse_args():
    parser = argparse.ArgumentParser()
    # Evaluation / workload arguments
    parser.add_argument(
        "--data_names",
        "--dataset_name",
        dest="data_names",
        default="gsm8k,math",
        type=str,
        help="Comma-separated dataset names. Alias: --dataset_name (serve/run.py style, single dataset).",
    )
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument(
        "--model_name_or_path",
        "--model_name",
        dest="model_name_or_path",
        default="gpt-4",
        type=str,
    )
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument(
        "--num_test_sample",
        default=-1,
        type=int,
        help="-1 for full data; otherwise limit number of test samples.",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    # Temperature aligned with serve/run.py default (0.0)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=None, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument(
        "--use_specgen", action="store_true", help="Use specgen inference"
    )
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--repeat_dataset",
        type=int,
        default=1,
        help="Repeat the dataset this many times to simulate multiple request rounds.",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode when applying chat template.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )

    # specgen specific arguments (mirror serve/run.py CLI; keep specgen_* as aliases)
    parser.add_argument(
        "--max_batch_size",
        "--specgen_max_batch_size",
        dest="specgen_max_batch_size",
        type=int,
        default=128,
        help="Max batch size for specgen inference (same as --max_batch_size in serve/run.py).",
    )
    parser.add_argument(
        "--tp_ranks",
        "--specgen_tp_ranks",
        dest="specgen_tp_ranks",
        type=int,
        default=1,
        help="Number of tensor parallel ranks (same as --tp_ranks in serve/run.py).",
    )
    parser.add_argument(
        "--max_mem_per_rank",
        "--specgen_max_mem_per_rank",
        dest="specgen_max_mem_per_rank",
        type=int,
        default=None,
        help="Max GPU memory per rank in GB for specgen inference (None = use device total * specgen_mem_fraction).",
    )
    parser.add_argument(
        "--mem_fraction",
        "--specgen_mem_fraction",
        dest="specgen_mem_fraction",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use per rank for specgen inference (e.g., 0.9).",
    )
    parser.add_argument(
        "--kv_cache",
        "--specgen_kv_cache",
        dest="specgen_kv_cache",
        type=str,
        default="full",
        choices=["full", "streaming", "pillar"],
        help="KV cache type for specgen inference (full / streaming / pillar).",
    )
    parser.add_argument(
        "--admit_policy",
        "--specgen_admit_policy",
        dest="specgen_admit_policy",
        type=str,
        default="naive",
        choices=["naive", "oracle", "offloading"],
        help="Admission policy (for non-full kv_cache, 'naive' will be auto-updated to 'offloading').",
    )
    parser.add_argument(
        "--batch_policy",
        "--specgen_batch_policy",
        dest="specgen_batch_policy",
        type=str,
        default="greedy",
        choices=["greedy", "random", "naive"],
        help="Batch policy for specgen inference (greedy: load balance, random: random assignment, naive: max bucket).",
    )
    parser.add_argument(
        "--reduce_method",
        "--specgen_reduce_method",
        dest="specgen_reduce_method",
        type=str,
        default="customized",
        choices=["naive", "pynccl", "customized"],
        help="Tensor-parallel reduce method for specgen inference.",
    )
    parser.add_argument(
        "--dtype",
        "--specgen_dtype",
        dest="specgen_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for specgen model weights and KV cache.",
    )
    parser.add_argument(
        "--chunk_size",
        "--specgen_chunk_size",
        dest="specgen_chunk_size",
        type=int,
        default=16 * 1024,
        help="Chunk size for specgen prefill (same as --chunk_size in serve/run.py).",
    )
    parser.add_argument(
        "--clip_length",
        "--specgen_clip_length",
        dest="specgen_clip_length",
        type=int,
        default=None,
        help=(
            "Max total sequence length (input + output) for specgen. "
            "If None, will be auto-resolved from the model's max_position_embeddings."
        ),
    )
    parser.add_argument(
        "--spec_stride",
        "--specgen_stride",
        dest="specgen_stride",
        type=int,
        default=16,
        help="Speculative stride for specgen inference.",
    )
    parser.add_argument(
        "--budget_ratio",
        "--specgen_budget_ratio",
        dest="specgen_budget_ratio",
        type=float,
        default=0.05,
        help=(
            "Budget ratio (sparsity parameter) for specgen inference. "
            "Controls the fraction of KV cache to keep (e.g., 0.05 = 5%)."
        ),
    )
    parser.add_argument(
        "--num_min_budget",
        "--specgen_num_min_budget",
        dest="specgen_num_min_budget",
        type=int,
        default=128,
        help="Minimum draft-token budget for specgen inference (same as --num_min_budget in serve/run.py).",
    )
    parser.add_argument(
        "--cuda_graph",
        "--specgen_cuda_graph",
        dest="specgen_cuda_graph",
        action="store_true",
        help="Enable CUDA graph optimization.",
    )
    parser.add_argument(
        "--enable-torch-profiler",
        "--specgen_enable_torch_profiler",
        dest="specgen_enable_torch_profiler",
        action="store_true",
        help="Enable PyTorch profiler inside specgen inference.",
    )
    parser.add_argument(
        "--debug",
        "--specgen_debug",
        dest="specgen_debug",
        action="store_true",
        help="Enable debug mode for specgen.",
    )
    parser.add_argument(
        "--async_cpu",
        "--specgen_async_cpu",
        dest="specgen_async_cpu",
        action="store_true",
        help="Enable async CPU scheduling.",
    )
    parser.add_argument(
        "--vanilla-attn",
        "--specgen_vanilla_attn",
        dest="specgen_vanilla_attn",
        action="store_true",
        help="Use vanilla attention backend in specgen (no FlashAttention).",
    )
    parser.add_argument(
        "--specgen_keep_input",
        action="store_true",
        help="Keep input text in output (for debugging)",
    )
    parser.add_argument(
        "--specgen_verbose",
        action="store_true",
        help="Print detailed debugging information for specgen",
    )
    parser.add_argument(
        "--perf_sim",
        "--specgen_perf_sim",
        dest="specgen_perf_sim",
        action="store_true",
        help="(Deprecated) Perf simulation flag for specgen; kept for API compatibility.",
    )
    parser.add_argument(
        "--gemm_profile_path",
        "--specgen_gemm_profile_path",
        dest="specgen_gemm_profile_path",
        type=str,
        default=None,
        help="Optional GEMM profile path for specgen (same semantics as serve/run.py).",
    )
    parser.add_argument(
        "--result_dir",
        "--specgen_result_dir",
        dest="specgen_result_dir",
        type=str,
        default=None,
        help="Optional directory for specgen to dump per-request outputs and summary logs.",
    )
    parser.add_argument(
        "--enable_wandb",
        action="store_true",
        help="Enable wandb logging for experiment tracking (same name as serve/run.py).",
    )

    # vLLM ngram speculative decoding arguments
    parser.add_argument(
        "--vllm_enable_ngram",
        action="store_true",
        help="Enable ngram-based speculative decoding in vLLM",
    )
    parser.add_argument(
        "--vllm_num_speculative_tokens",
        type=int,
        default=5,
        help="Number of speculative tokens for vLLM ngram speculation",
    )
    parser.add_argument(
        "--vllm_ngram_prompt_lookup_min",
        type=int,
        default=1,
        help="Minimum ngram size for vLLM prompt lookup",
    )
    parser.add_argument(
        "--vllm_ngram_prompt_lookup_max",
        type=int,
        default=4,
        help="Maximum ngram size for vLLM prompt lookup",
    )

    # vLLM memory and performance arguments
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use for vLLM (default: 0.9)",
    )
    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=None,
        help="Maximum sequence length for vLLM (default: auto from model config)",
    )
    parser.add_argument(
        "--vllm_max_num_seqs",
        type=int,
        default=256,
        help="Maximum number of sequences to process in parallel (default: 256)",
    )
    parser.add_argument(
        "--vllm_enforce_eager",
        action="store_true",
        help="Disable CUDA graphs to save memory (reduces performance)",
    )

    # vLLM EAGLE3 speculative decoding arguments
    parser.add_argument(
        "--vllm_enable_eagle3",
        action="store_true",
        help="Enable EAGLE3 speculative decoding in vLLM",
    )
    parser.add_argument(
        "--vllm_draft_model_path",
        type=str,
        default=None,
        help="Path to the draft model for EAGLE3 speculative decoding in vLLM",
    )
    parser.add_argument(
        "--vllm_eagle3_num_speculative_tokens",
        type=int,
        default=None,
        help="Number of speculative tokens for EAGLE3 (default: None, uses vLLM's default)",
    )

    # vLLM self-speculative decoding arguments (MagicDec-style)
    parser.add_argument(
        "--vllm_enable_sspec",
        action="store_true",
        help="Enable self-speculative decoding in vLLM (requires V1 engine)",
    )
    parser.add_argument(
        "--vllm_sspec_num_speculative_tokens",
        type=int,
        default=16,
        help="Number of speculative tokens for self-spec (default: 16)",
    )
    parser.add_argument(
        "--vllm_sspec_sink_size",
        type=int,
        default=8,
        help="Number of sink blocks for streaming cache (default: 8 blocks = 128 tokens with block_size=16)",
    )
    parser.add_argument(
        "--vllm_sspec_recent_ratio",
        type=float,
        default=0.10,
        help="Ratio of recent blocks for streaming cache (default: 0.10 = 10%% of computed tokens)",
    )
    parser.add_argument(
        "--vllm_sspec_block_size",
        type=int,
        default=1,
        help="Block size for self-spec (default: 1, disables prefix caching)",
    )

    # vLLM self-spec with n-gram assistance (TriForce-style)
    parser.add_argument(
        "--vllm_enable_sspec_ngram",
        action="store_true",
        help="Enable self-spec with n-gram assistance in vLLM (requires V1 engine)",
    )
    parser.add_argument(
        "--vllm_sspec_ngram_num_speculative_tokens",
        type=int,
        default=16,
        help="Threshold for ACCUMULATING -> VERIFYING transition (default: 16)",
    )
    parser.add_argument(
        "--vllm_sspec_ngram_num_ngram_draft_tokens",
        type=int,
        default=3,
        help="Number of n-gram draft tokens per step during ACCUMULATING (default: 3)",
    )
    parser.add_argument(
        "--vllm_sspec_ngram_prompt_lookup_min",
        type=int,
        default=None,
        help="Minimum n-gram window size (default: same as num_ngram_draft_tokens)",
    )
    parser.add_argument(
        "--vllm_sspec_ngram_prompt_lookup_max",
        type=int,
        default=None,
        help="Maximum n-gram window size (default: same as num_ngram_draft_tokens)",
    )
    parser.add_argument(
        "--vllm_sspec_ngram_sink_size",
        type=int,
        default=8,
        help="Number of sink blocks for streaming cache (default: 8 blocks)",
    )
    parser.add_argument(
        "--vllm_sspec_ngram_recent_ratio",
        type=float,
        default=0.10,
        help="Ratio of recent blocks for streaming cache (default: 0.10)",
    )
    parser.add_argument(
        "--vllm_sspec_ngram_block_size",
        type=int,
        default=1,
        help="Block size for self-spec ngram (default: 1, disables prefix caching)",
    )

    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    repeat_suffix = "" if args.repeat_dataset == 1 else f"_repeat{args.repeat_dataset}"
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}{repeat_suffix}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = (
        f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    )
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(list(load_jsonl(f"{output_dir}/{data_name}/{f}")))

    processed_samples = {sample.get("idx"): sample for sample in processed_samples}
    processed_samples = [
        sample for sample in processed_samples.values() if sample is not None
    ]

    if args.repeat_dataset < 1:
        raise ValueError("repeat_dataset must be >= 1")

    if args.repeat_dataset > 1:
        processed_pairs = set()
        for sample in processed_samples:
            repeat_id = sample.get("repeat_id", 0)
            source_idx = sample.get("source_idx", sample.get("idx"))
            processed_pairs.add((repeat_id, source_idx))
        base_examples = list(examples)
        if not base_examples:
            examples = []
        else:
            base_stride = max(example["idx"] for example in base_examples) + 1
            repeated_examples = []
            for repeat_idx in range(args.repeat_dataset):
                for example in base_examples:
                    source_idx = example["idx"]
                    if (repeat_idx, source_idx) in processed_pairs:
                        continue
                    duplicated = copy.deepcopy(example)
                    duplicated["repeat_id"] = repeat_idx
                    duplicated["source_idx"] = source_idx
                    if repeat_idx > 0:
                        duplicated["idx"] = source_idx + repeat_idx * base_stride
                    repeated_examples.append(duplicated)
            examples = repeated_examples
    else:
        processed_idxs = {sample["idx"] for sample in processed_samples}
        examples = [
            example for example in examples if example["idx"] not in processed_idxs
        ]

    return examples, processed_samples, out_file


def setup(args):
    # load model
    if args.use_specgen:
        # For specgen, we don't need to load model here as run_inference handles it
        llm = None
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if args.use_vllm:
            # Check for mutually exclusive vLLM speculative decoding options
            spec_options = sum(
                [
                    bool(args.vllm_enable_ngram),
                    bool(args.vllm_enable_eagle3),
                    bool(getattr(args, "vllm_enable_sspec", False)),
                    bool(getattr(args, "vllm_enable_sspec_ngram", False)),
                ]
            )
            if spec_options > 1:
                raise ValueError(
                    "Cannot enable multiple vLLM speculative decoding methods at the same time. "
                    "Please choose only one: vllm_enable_ngram, vllm_enable_eagle3, "
                    "vllm_enable_sspec, or vllm_enable_sspec_ngram."
                )

            # Set environment variables for self-spec methods (must be set before vLLM import)
            if getattr(args, "vllm_enable_sspec", False) or getattr(
                args, "vllm_enable_sspec_ngram", False
            ):
                os.environ["VLLM_USE_V1"] = "1"
                os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
                os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
                method_name = (
                    "self_spec_ngram" if args.vllm_enable_sspec_ngram else "self_specs"
                )
                print(f"Enabled V1 engine for {method_name}")
                print(f"  VLLM_USE_V1=1")
                print(f"  VLLM_ENABLE_V1_MULTIPROCESSING=1")
                print(f"  VLLM_ATTENTION_BACKEND=FLASHINFER")

            # Build vLLM kwargs
            vllm_kwargs = {
                "model": args.model_name_or_path,
                "tensor_parallel_size": len(available_gpus)
                // args.pipeline_parallel_size,
                "pipeline_parallel_size": args.pipeline_parallel_size,
                "trust_remote_code": True,
                "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                "max_num_seqs": args.vllm_max_num_seqs,
            }

            # Add optional memory parameters
            if args.vllm_max_model_len is not None:
                vllm_kwargs["max_model_len"] = args.vllm_max_model_len
            if args.vllm_enforce_eager:
                vllm_kwargs["enforce_eager"] = True
                print("vLLM: CUDA graphs disabled (enforce_eager=True)")

            print(f"vLLM memory settings:")
            print(f"  - GPU memory utilization: {args.vllm_gpu_memory_utilization}")
            print(
                f"  - Max model length: {args.vllm_max_model_len if args.vllm_max_model_len else 'auto'}"
            )
            print(f"  - Max num seqs: {args.vllm_max_num_seqs}")

            # Add ngram speculative decoding config if enabled
            if args.vllm_enable_ngram:
                speculative_config = {
                    "method": "ngram",
                    "num_speculative_tokens": args.vllm_num_speculative_tokens,
                    "ngram_prompt_lookup_min": args.vllm_ngram_prompt_lookup_min,
                    "ngram_prompt_lookup_max": args.vllm_ngram_prompt_lookup_max,
                }
                vllm_kwargs["speculative_config"] = speculative_config
                # IMPORTANT: Enable stat logging to access spec decode metrics via get_metrics()
                vllm_kwargs["disable_log_stats"] = False
                print(
                    f"Enabled vLLM ngram speculative decoding with config: {speculative_config}"
                )
                print(f"Enabled vLLM stat logging for metrics collection")

            # Add EAGLE3 speculative decoding config if enabled
            if args.vllm_enable_eagle3:
                if args.vllm_draft_model_path is None:
                    raise ValueError(
                        "vllm_draft_model_path must be specified when vllm_enable_eagle3 is enabled"
                    )
                # Configure EAGLE3 using speculative_config
                eagle3_config = {
                    "method": "eagle3",
                    "model": args.vllm_draft_model_path,
                    "draft_tensor_parallel_size": 1,  # EAGLE3 requires TP=1 for draft model
                }
                # Add num_speculative_tokens if specified
                if args.vllm_eagle3_num_speculative_tokens is not None:
                    eagle3_config["num_speculative_tokens"] = (
                        args.vllm_eagle3_num_speculative_tokens
                    )

                vllm_kwargs["speculative_config"] = eagle3_config
                # Enable stat logging to access spec decode metrics
                vllm_kwargs["disable_log_stats"] = False
                print(
                    f"Enabled vLLM EAGLE3 speculative decoding with draft model: {args.vllm_draft_model_path}"
                )
                print(f"  - Method: eagle3")
                print(f"  - Draft model tensor parallel size: 1")
                if args.vllm_eagle3_num_speculative_tokens is not None:
                    print(
                        f"  - Num speculative tokens: {args.vllm_eagle3_num_speculative_tokens}"
                    )
                else:
                    print(f"  - Num speculative tokens: (using vLLM default)")
                print(f"Enabled vLLM stat logging for metrics collection")

            # Add self-speculative decoding config if enabled (MagicDec-style)
            if getattr(args, "vllm_enable_sspec", False):
                sspec_config = {
                    "method": "self_specs",
                    "model": None,
                    "num_speculative_tokens": args.vllm_sspec_num_speculative_tokens,
                }
                vllm_kwargs["speculative_config"] = sspec_config
                # Streaming cache parameters
                vllm_kwargs["sink_size"] = args.vllm_sspec_sink_size
                vllm_kwargs["recent_ratio"] = args.vllm_sspec_recent_ratio
                vllm_kwargs["block_size"] = args.vllm_sspec_block_size
                # Enable stat logging to access spec decode metrics
                vllm_kwargs["disable_log_stats"] = False
                # Additional V1-specific parameters
                vllm_kwargs["enforce_eager"] = False
                vllm_kwargs["enable_chunked_prefill"] = True
                vllm_kwargs["enable_prefix_caching"] = False
                vllm_kwargs["gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
                vllm_kwargs["max_num_batched_tokens"] = 2048
                vllm_kwargs["max_num_seqs"] = args.vllm_max_num_seqs
                vllm_kwargs["cuda_graph_sizes"] = [
                    1,
                    2,
                    4,
                    8,
                    16,
                    32,
                    64,
                    128,
                    192,
                    256,
                    320,
                    384,
                    448,
                    512,
                    768,
                    1024,
                    1536,
                ]

                print(f"Enabled vLLM self-speculative decoding with config:")
                print(f"  - Method: self_specs")
                print(
                    f"  - Num speculative tokens: {args.vllm_sspec_num_speculative_tokens}"
                )
                print(
                    f"  - Sink size (streaming cache): {args.vllm_sspec_sink_size} blocks"
                )
                print(
                    f"  - Recent ratio (streaming cache): {args.vllm_sspec_recent_ratio} ({args.vllm_sspec_recent_ratio*100:.1f}%)"
                )
                print(f"  - Block size: {args.vllm_sspec_block_size}")
                print(f"Enabled vLLM stat logging for metrics collection")

            # Add self-spec with n-gram config if enabled (TriForce-style)
            if getattr(args, "vllm_enable_sspec_ngram", False):
                sspec_ngram_config = {
                    "method": "self_spec_ngram",
                    "model": None,
                    "num_speculative_tokens": args.vllm_sspec_ngram_num_speculative_tokens,
                    "num_ngram_draft_tokens": args.vllm_sspec_ngram_num_ngram_draft_tokens,
                }
                # Set n-gram window sizes (default to num_ngram_draft_tokens if not specified)
                lookup_max = (
                    args.vllm_sspec_ngram_prompt_lookup_max
                    if args.vllm_sspec_ngram_prompt_lookup_max is not None
                    else args.vllm_sspec_ngram_num_ngram_draft_tokens
                )
                lookup_min = (
                    args.vllm_sspec_ngram_prompt_lookup_min
                    if args.vllm_sspec_ngram_prompt_lookup_min is not None
                    else args.vllm_sspec_ngram_num_ngram_draft_tokens
                )
                sspec_ngram_config["prompt_lookup_max"] = lookup_max
                sspec_ngram_config["prompt_lookup_min"] = lookup_min

                vllm_kwargs["speculative_config"] = sspec_ngram_config
                # Streaming cache parameters
                vllm_kwargs["sink_size"] = args.vllm_sspec_ngram_sink_size
                vllm_kwargs["recent_ratio"] = args.vllm_sspec_ngram_recent_ratio
                vllm_kwargs["block_size"] = args.vllm_sspec_ngram_block_size
                # Enable stat logging to access spec decode metrics
                vllm_kwargs["disable_log_stats"] = False
                # Additional V1-specific parameters
                vllm_kwargs["enable_chunked_prefill"] = True
                vllm_kwargs["enable_prefix_caching"] = False
                vllm_kwargs["gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
                vllm_kwargs["max_num_batched_tokens"] = 2048
                vllm_kwargs["max_num_seqs"] = args.vllm_max_num_seqs
                vllm_kwargs["cuda_graph_sizes"] = [
                    1,
                    2,
                    4,
                    8,
                    16,
                    32,
                    64,
                    128,
                    192,
                    256,
                    320,
                    384,
                    448,
                    512,
                    768,
                    1024,
                    1536,
                ]

                print(f"Enabled vLLM self-spec with n-gram assistance:")
                print(f"  - Method: self_spec_ngram")
                print(
                    f"  - Threshold (ACCUMULATING -> VERIFYING): {args.vllm_sspec_ngram_num_speculative_tokens} tokens"
                )
                print(
                    f"  - N-gram draft tokens per step: {args.vllm_sspec_ngram_num_ngram_draft_tokens} tokens"
                )
                print(f"  - N-gram window size: max={lookup_max}, min={lookup_min}")
                print(
                    f"  - Sink size (streaming cache): {args.vllm_sspec_ngram_sink_size} blocks"
                )
                print(
                    f"  - Recent ratio (streaming cache): {args.vllm_sspec_ngram_recent_ratio} ({args.vllm_sspec_ngram_recent_ratio*100:.1f}%)"
                )
                print(f"  - Block size: {args.vllm_sspec_ngram_block_size}")
                print(f"Enabled vLLM stat logging for metrics collection")

            llm = LLM(**vllm_kwargs)
            tokenizer = None
            if args.apply_chat_template:
                tokenizer = AutoTokenizer.from_pretrained(
                    args.model_name_or_path, trust_remote_code=True
                )
        else:
            llm, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path,
                load_in_half=True,
                use_fast_tokenizer=True,
                use_safetensors=args.use_safetensors,
            )

    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(llm, tokenizer, data_name, args))

    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\n=== Final Evaluation Results ===")
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))

    # Print speculative decoding summary if applicable
    if args.use_specgen and args.specgen_kv_cache != "full":
        spec_results = [
            result for result in results if "speculative_decoding" in result
        ]
        if spec_results:
            # Average the acceptance rates across datasets (excluding "avg" result)
            avg_acceptance_rate = sum(
                result["speculative_decoding"]["acceptance_rate"]
                for result in spec_results
            ) / len(spec_results)
            total_spec_tokens = sum(
                result["speculative_decoding"]["total_speculative_tokens"]
                for result in spec_results
            )
            total_accepted_tokens = sum(
                result["speculative_decoding"]["accepted_speculative_tokens"]
                for result in spec_results
            )

            print("\n=== Speculative Decoding Summary ===")
            print(
                f"Overall Acceptance Rate: {avg_acceptance_rate:.4f} ({avg_acceptance_rate * 100:.2f}%)"
            )
            print(f"Total Speculative Tokens: {total_spec_tokens}")
            print(f"Total Accepted Tokens: {total_accepted_tokens}")
            print("=" * 40)

    # Finish wandb run if enabled
    if args.enable_wandb and wandb.run is not None and is_first_rank():
        wandb.finish()
        print("Wandb run finished")


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(llm, tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))

    # Initialize wandb for vLLM if enabled
    inference_wandb_logger = None
    if args.use_vllm and args.enable_wandb and is_first_rank():
        inference_wandb_logger = WandbLogger()
        model_name = args.model_name_or_path.split("/")[-1]
        available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        num_gpus = len(available_gpus.split(","))
        engine_name = "vllm"
        wandb_config = {
            "model_name": model_name,
            "engine": engine_name,
            "data_name": data_name,
            "prompt_type": args.prompt_type,
            "temperature": args.temperature,
            "max_tokens_per_call": args.max_tokens_per_call,
            "num_test_sample": args.num_test_sample,
            "n_sampling": args.n_sampling,
            "seed": args.seed,
            "num_gpus": num_gpus,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "tensor_parallel_size": num_gpus // args.pipeline_parallel_size,
            "trace_name": data_name,
            "max_batch_size": "N/A",  # vLLM/SGLang manages this internally
            "num_requests": len(examples),
            "tp_ranks": num_gpus // args.pipeline_parallel_size,
            "kv_cache": "full",  # vLLM/SGLang uses full kv cache
            "async_cpu": False,
            "batch_policy": f"{engine_name}_default",
            "admit_policy": f"{engine_name}_default",
        }
        # Create wandb run name
        name = f"{model_name}-{data_name}-{engine_name}-tp{wandb_config['tensor_parallel_size']}-t{args.temperature}-num{len(examples)}"
        name += time.strftime("-%Y-%m-%d-%H-%M", time.localtime())

        if wandb.run is None:
            assert (
                os.environ.get("WANDB_API_KEY") is not None
            ), "WANDB_API_KEY environment variable must be set"
            wandb.login()
            wandb.init(
                project="specgen-benchmark",
                entity="yilongzhao-uc-berkeley",
                name=name,
                config=wandb_config,
            )
        print(f"Wandb initialized for {engine_name}: {name}")

    # Silence debug print of the first example
    # if len(examples) > 0:
    #     print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        # Do not print the full prompt/question
        # if idx == args.start:
        #     print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
            "repeat_id",
            "source_idx",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
            for prompt in input_prompts
        ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    # Throughput counters; use engine-side token counts when possible
    total_input_tokens = 0
    total_output_tokens = 0

    # EAGLE3 speculative decoding statistics
    total_draft_tokens = 0  # All tokens proposed by draft model
    total_accepted_tokens = 0  # Tokens actually accepted
    total_spec_verify_count = 0  # Number of verification steps

    # start inference
    # measure time use
    start_time = time.time()
    engine_time_s = 0.0  # measure only engine generation time for fair comparison
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        if args.use_vllm:
            time_engine_phase_start = time.time()
            vllm_outputs = llm.generate(
                prompts,
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens_per_call,
                    n=1,
                    stop=stop_words,
                ),
            )
            engine_time_s += time.time() - time_engine_phase_start

            # Sort by request_id and accumulate token counts from vLLM outputs
            vllm_outputs = sorted(vllm_outputs, key=lambda x: int(x.request_id))
            for out in vllm_outputs:
                total_input_tokens += len(getattr(out, "prompt_token_ids", []))
                total_output_tokens += len(getattr(out.outputs[0], "token_ids", []))
            outputs = [out.outputs[0].text for out in vllm_outputs]
        elif args.use_specgen:
            # Convert prompts to Request objects
            from transformers import AutoTokenizer

            specgen_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

            requests = []
            for prompt in prompts:
                # Tokenize the prompt
                token_ids = specgen_tokenizer(prompt, return_tensors="pt")["input_ids"][
                    0
                ].tolist()
                # Create Request object; max_tokens_per_call controls desired generation length
                req = Request(token_ids, args.max_tokens_per_call)
                requests.append(req)

            # Automatically adjust admit_policy if using sparse kv_cache (mirror serve/run.py)
            specgen_admit_policy = args.specgen_admit_policy
            if args.specgen_kv_cache != "full" and specgen_admit_policy == "naive":
                specgen_admit_policy = (
                    "offloading"  # Use offloading as default for non-full kv_cache
                )
                print(
                    "Warning: specgen_admit_policy 'naive' is not compatible with non-full kv_cache. "
                    "Using 'offloading' instead."
                )

            # Call our custom inference function
            time_engine_phase_start = time.time()
            inference_outputs = run_inference(
                reqs=requests,
                model_name=args.model_name_or_path,
                max_batch_size=args.specgen_max_batch_size,
                tp_ranks=args.specgen_tp_ranks,
                max_mem_per_rank=args.specgen_max_mem_per_rank,
                mem_fraction=args.specgen_mem_fraction,
                cuda_graph=args.specgen_cuda_graph,
                async_cpu=args.specgen_async_cpu,
                reduce_method=args.specgen_reduce_method,
                dtype=args.specgen_dtype,
                kv_cache=args.specgen_kv_cache,
                admit_policy=specgen_admit_policy,
                batch_policy=args.specgen_batch_policy,
                chunk_size=args.specgen_chunk_size,
                clip_length=args.specgen_clip_length,
                enable_wandb=args.enable_wandb,
                spec_stride=args.specgen_stride,
                budget_ratio=args.specgen_budget_ratio,
                num_min_budget=args.specgen_num_min_budget,
                temperature=args.temperature,
                perf_sim=args.specgen_perf_sim,
                gemm_profile_path=args.specgen_gemm_profile_path,
                debug=args.specgen_debug,
                result_dir=args.specgen_result_dir,
                vanilla_attn=args.specgen_vanilla_attn,
                dataset_name=data_name,
                enable_torch_profiler=args.specgen_enable_torch_profiler,
            )
            engine_time_s += time.time() - time_engine_phase_start

            # Capture speculative decoding acceptance rate statistics
            if args.specgen_kv_cache != "full":
                from serve.utils import wandb_logger

                accepted_speculative_tokens = wandb_logger.log_data.get(
                    "spec/accepted_speculative_tokens", 0
                )
                total_speculative_tokens = wandb_logger.log_data.get(
                    "spec/total_speculative_tokens", 1
                )
                acceptance_rate = (
                    accepted_speculative_tokens / total_speculative_tokens
                    if total_speculative_tokens > 0
                    else 0.0
                )

                print(f"=== Speculative Decoding Statistics ===")
                print(f"Total Speculative Tokens: {total_speculative_tokens}")
                print(f"Accepted Speculative Tokens: {accepted_speculative_tokens}")
                print(
                    f"Acceptance Rate: {acceptance_rate:.4f} ({acceptance_rate * 100:.2f}%)"
                )
                print("=" * 40)

            # Accumulate engine-side token counts if provided
            for o in inference_outputs:
                if "len_input" in o and "len_output" in o:
                    total_input_tokens += (
                        int(o["len_input"]) if o["len_input"] is not None else 0
                    )
                    total_output_tokens += (
                        int(o["len_output"]) if o["len_output"] is not None else 0
                    )
                elif "output_token_ids" in o:
                    total_output_tokens += len(o["output_token_ids"]) or 0

            # Extract text outputs
            outputs = []
            for i, inference_output in enumerate(inference_outputs):
                full_text = inference_output["tokens"]
                # Use the pre-decoded generated part if available
                if "generated_tokens" in inference_output:
                    generated_text = inference_output["generated_tokens"]
                    if args.specgen_verbose and i == 0:
                        print(f"Debug - Using pre-decoded generated tokens")
                else:
                    # Fallback to old method
                    input_text = prompts[i]
                    if args.specgen_keep_input:
                        generated_text = full_text
                        if args.specgen_verbose and i == 0:
                            print(f"Debug - Keeping full text (input + generated)")
                    else:
                        # Try to extract generated part manually
                        if full_text.startswith(input_text):
                            generated_text = full_text[len(input_text) :]
                            if args.specgen_verbose and i == 0:
                                print(f"Debug - Manual extraction: direct removal")
                        else:
                            generated_text = full_text
                            if args.specgen_verbose and i == 0:
                                print(f"Debug - Manual extraction: using full text")

                # Debug: print details for first sample
                if args.specgen_verbose and i == 0:
                    print(f"Debug - Full text length: {len(full_text)}")
                    print(f"Debug - Generated text length: {len(generated_text)}")
                    print(f"Debug - Full text preview: {repr(full_text[:200])}")
                    print(
                        f"Debug - Generated text preview: {repr(generated_text[:200])}"
                    )

                # Apply stop words filtering
                original_generated = generated_text
                for stop_word in stop_words:
                    if stop_word in generated_text:
                        generated_text = generated_text.split(stop_word)[0]

                # Debug: show the result after stop words
                if args.specgen_verbose and i == 0:
                    print(
                        f"Debug - Generated text length before stop words: {len(original_generated)}"
                    )
                    print(
                        f"Debug - Generated text length after stop words: {len(generated_text)}"
                    )
                    print(
                        f"Debug - Final generated text preview: {repr(generated_text[:200])}"
                    )

                # Additional check: if generated text is too short, warn
                if len(generated_text.strip()) == 0:
                    if args.specgen_verbose and i == 0:
                        print(f"Warning - Generated text is empty for sample {i}")
                        print(f"Full text was: {repr(full_text[:500])}")
                        print(
                            f"Original generated was: {repr(original_generated[:500])}"
                        )
                elif (
                    len(generated_text.strip()) < 10 and args.specgen_verbose and i == 0
                ):
                    print(
                        f"Warning - Generated text is very short: {repr(generated_text)}"
                    )

                outputs.append(generated_text)
        else:
            time_engine_phase_start = time.time()
            outputs = generate_completions(
                model=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_tokens_per_call,
                batch_size=16,
                stop_id_sequences=stop_words,
            )
            engine_time_s += time.time() - time_engine_phase_start

            # As HF path doesn't expose token ids here, approximate counts via tokenizer
            if tokenizer is not None:
                total_input_tokens += sum(
                    len(tokenizer(p, return_tensors="pt")["input_ids"][0])
                    for p in prompts
                )
                total_output_tokens += sum(
                    len(tokenizer(t, return_tensors="pt")["input_ids"][0])
                    for t in outputs
                )

        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]
    time_use = time.time() - start_time

    # Get vLLM speculation statistics after all generation is done (ngram or EAGLE3)
    if args.use_vllm and (args.vllm_enable_ngram or args.vllm_enable_eagle3):
        try:
            spec_method = "ngram" if args.vllm_enable_ngram else "EAGLE3"
            if hasattr(llm, "get_metrics"):
                metrics = llm.get_metrics()

                # Extract spec decode metrics from Prometheus snapshot
                for metric in metrics:
                    metric_name = (
                        metric.name if hasattr(metric, "name") else str(metric)
                    )

                    if "spec_decode_num_draft_tokens" in metric_name and hasattr(
                        metric, "value"
                    ):
                        total_draft_tokens = int(metric.value)
                    elif (
                        "spec_decode_num_accepted_tokens" in metric_name
                        and "per_pos" not in metric_name
                        and hasattr(metric, "value")
                    ):
                        total_accepted_tokens = int(metric.value)
                    elif "spec_decode_num_drafts" in metric_name and hasattr(
                        metric, "value"
                    ):
                        total_spec_verify_count = int(metric.value)

                if total_draft_tokens > 0 and total_accepted_tokens > 0:
                    print(
                        f"\n[vLLM {spec_method}] Successfully retrieved speculation statistics:"
                    )
                    print(f"  Draft tokens: {total_draft_tokens:,}")
                    print(f"  Accepted tokens: {total_accepted_tokens:,}")
                    print(
                        f"  Acceptance rate: {(total_accepted_tokens/total_draft_tokens)*100:.2f}%"
                    )

        except Exception as e:
            if "Stat logging disabled" in str(e):
                print(
                    f"\n[vLLM] Warning: Stat logging is disabled. Cannot retrieve spec decode statistics."
                )
            else:
                print(f"\n[vLLM] Could not retrieve spec decode statistics: {e}")

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    # For fair comparison across engines, report time excluding one-time init/capture
    result_json["time_use_in_second"] = engine_time_s
    result_json["time_use_in_minutes"] = (
        f"{int(engine_time_s // 60)}:{int(engine_time_s % 60):02d}"
    )
    # Record engine-only generation time for fair throughput comparison
    result_json["engine_inference_seconds"] = engine_time_s

    # Add speculative decoding statistics to metrics if available
    if args.use_specgen and args.specgen_kv_cache != "full":
        from serve.utils import wandb_logger

        accepted_speculative_tokens = wandb_logger.log_data.get(
            "spec/accepted_speculative_tokens", 0
        )
        total_speculative_tokens = wandb_logger.log_data.get(
            "spec/total_speculative_tokens", 1
        )
        acceptance_rate = (
            accepted_speculative_tokens / total_speculative_tokens
            if total_speculative_tokens > 0
            else 0.0
        )

        result_json["speculative_decoding"] = {
            "total_speculative_tokens": total_speculative_tokens,
            "accepted_speculative_tokens": accepted_speculative_tokens,
            "acceptance_rate": acceptance_rate,
            "acceptance_rate_percentage": acceptance_rate * 100,
        }

    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        # Add throughput metrics (engine-only time)
        result_json["throughput"] = {
            "input_tps": (
                (total_input_tokens / engine_time_s) if engine_time_s > 0 else None
            ),
            "output_tps": (
                (total_output_tokens / engine_time_s) if engine_time_s > 0 else None
            ),
            "total_tps": (
                ((total_input_tokens + total_output_tokens) / engine_time_s)
                if engine_time_s > 0
                else None
            ),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "elapsed_seconds": time_use,
            "engine_seconds": engine_time_s,
            "engine": (
                "vllm" if args.use_vllm else ("specgen" if args.use_specgen else "hf")
            ),
        }

        # Add speculative decoding statistics if available (vLLM)
        # Only add if we have REAL data, not fake estimates
        if args.use_vllm and total_draft_tokens > 0 and total_accepted_tokens > 0:
            acceptance_rate = (
                total_accepted_tokens / total_draft_tokens
                if total_draft_tokens > 0
                else 0.0
            )
            # Determine algorithm name
            if args.vllm_enable_eagle3:
                algorithm = "EAGLE3"
            elif args.vllm_enable_ngram:
                algorithm = "ngram"
            else:
                algorithm = "Unknown"

            result_json["speculative_decoding"] = {
                "algorithm": algorithm,
                "total_draft_tokens": total_draft_tokens,
                "total_accepted_tokens": total_accepted_tokens,
                "total_rejected_tokens": total_draft_tokens - total_accepted_tokens,
                "acceptance_rate": acceptance_rate,
                "acceptance_rate_percentage": acceptance_rate * 100,
                "verification_count": total_spec_verify_count,
                "avg_draft_per_verification": (
                    total_draft_tokens / total_spec_verify_count
                    if total_spec_verify_count > 0
                    else 0.0
                ),
                "avg_accepted_per_verification": (
                    total_accepted_tokens / total_spec_verify_count
                    if total_spec_verify_count > 0
                    else 0.0
                ),
            }

            # Add algorithm-specific notes
            if args.vllm_enable_eagle3:
                result_json["speculative_decoding"][
                    "note"
                ] = "EAGLE3 speculative decoding"
            elif args.vllm_enable_ngram:
                result_json["speculative_decoding"][
                    "note"
                ] = "ngram prompt lookup speculation"

            print(f"\n[{algorithm} Speculation Statistics]")
            print(f"  Draft tokens: {total_draft_tokens:,}")
            print(f"  Accepted tokens: {total_accepted_tokens:,}")
            print(f"  Rejected tokens: {total_draft_tokens - total_accepted_tokens:,}")
            print(f"  Acceptance rate: {acceptance_rate * 100:.2f}%")
            if total_spec_verify_count > 0:
                print(f"  Verifications: {total_spec_verify_count:,}")
                print(
                    f"  Avg draft per verification: {total_draft_tokens / total_spec_verify_count:.2f}"
                )
                print(
                    f"  Avg accepted per verification: {total_accepted_tokens / total_spec_verify_count:.2f}"
                )

        json.dump(result_json, f, indent=4)

    # Log final metrics to wandb for vLLM
    if (
        args.use_vllm
        and args.enable_wandb
        and inference_wandb_logger is not None
        and is_first_rank()
    ):
        wandb_metrics = {
            "accuracy": result_json["acc"],
            "input_tps": result_json["throughput"]["input_tps"],
            "output_tps": result_json["throughput"]["output_tps"],
            "total_tps": result_json["throughput"]["total_tps"],
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "engine_time_seconds": engine_time_s,
            "total_time_seconds": time_use,
            "num_samples": len(all_samples),
        }
        # Log to wandb
        wandb.log(wandb_metrics)
        print(f"Logged metrics to wandb: {wandb_metrics}")

    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
