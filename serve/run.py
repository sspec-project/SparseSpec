import argparse
import json
import logging
import math
import os
import time
from typing import List

import torch
from datasets import load_dataset
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from serve.distribute.tp import end_tp_worker, setup_tp_worker
from serve.model.config import ModelArgs
from serve.model.graph_model import CUDAGraphModelWrapper
from serve.model.model import KVPool, Transformer
from serve.model.model_perf import *
from serve.request.kv_cache_ptr.base import FullCachePtr
from serve.request.kv_cache_ptr.pillar import PillarCachePtr
from serve.request.kv_cache_ptr.streaming import StreamingCachePtr
from serve.request.request import Request
from serve.sampling.rejection_sampler import RejectionSampler
from serve.scheduler.scheduler import BaseScheduler
from serve.scheduler.spec_scheduler import SpecScheduler
from serve.utils import *

logger = logging.getLogger(__name__)


def dispatch_str_to_cache_ptr(kv_cache: str):
    if kv_cache == "full":
        return FullCachePtr
    elif kv_cache == "streaming":
        return StreamingCachePtr
    elif kv_cache == "pillar":
        return PillarCachePtr
    else:
        raise ValueError(f"Invalid kv_cache: {kv_cache}")


def run_debug_scheduler(
    reqs: List[Request],
    **args,
):
    model = AutoModelForCausalLM.from_pretrained(
        args["model_name"],
        device_map="cuda:0",
    )

    for req in reqs:
        # generate the output by the new model
        with torch.no_grad():
            input_ids = torch.tensor(req.token_ids, device="cuda").unsqueeze(0)
            if args["temperature"] > 0:
                output = model.generate(
                    input_ids=torch.tensor(input_ids, device="cuda"),
                    max_new_tokens=req.len_output,
                    do_sample=True,
                    temperature=args["temperature"],
                )
            else:
                output = model.generate(
                    input_ids=torch.tensor(input_ids, device="cuda"),
                    max_new_tokens=req.len_output,
                    do_sample=False,
                )

            # output = model(input_ids=input_ids, temperature=args["temperature"])
        req.token_ids = output[0].tolist()
    logger.info(f"DebugScheduler finished with {len(reqs)} requests.")
    return reqs


def run_base_scheduler(
    model,
    kv_pool: KVPool,
    reqs: List[Request],
    **args,
):
    if args["enable_wandb"]:
        wandb_logger.init_wandb(config=args)
    logger.info(f"Running BaseScheduler with args: {args}")

    worker = BaseScheduler(kv_pool, model, **args)
    ret_reqs = worker.run(reqs)
    return ret_reqs


def run_spec_scheduler(
    model,
    kv_pool: KVPool,
    reqs: List[Request],
    **args,
):
    if args["enable_wandb"]:
        wandb_logger.init_wandb(config=args)
    logger.info(f"Running SpecScheduler with args: {args}")

    worker = SpecScheduler(kv_pool, model, **args)
    ret_reqs = worker.run(reqs)
    return ret_reqs


def get_req_queue(
    num_requests: int,
    dataset_name: str,
    tokenizer,
    clip_length: int = 1024,
):
    if dataset_name == "aime":
        dataset = load_dataset("AI-MO/aimo-validation-aime")
    elif dataset_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500")
    elif dataset_name == "gpqa-diamond":
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
    else:
        raise ValueError("Invalid dataset name")

    reqs = []
    for i in range(num_requests):
        data = dataset["train"][i % len(dataset["train"])]
        problem = data["problem"]

        # tokenize the problem
        token_ids = tokenizer(problem)["input_ids"]
        req = Request(token_ids=token_ids, desired_output_length=clip_length)
        reqs.append(req)

    return reqs


def run_inference(
    reqs: List[Request],
    model_name: str,
    max_batch_size: int = 128,
    tp_ranks: int = 1,
    max_mem_per_rank: int = None,
    mem_fraction: float = 0.9,
    cuda_graph: bool = False,
    async_cpu: bool = False,
    reduce_method: str = "customized",
    dtype: str = "bfloat16",
    kv_cache: str = "full",
    admit_policy: str = "naive",
    batch_policy: str = "greedy",
    chunk_size: int = 256,
    clip_length: int = 32 * 1024,
    enable_wandb: bool = False,
    spec_stride: int = 16,
    budget_ratio: float = 0.05,
    num_min_budget: int = 128,
    temperature: float = 0.0,
    perf_sim: bool = False,
    gemm_profile_path: str = None,
    debug: bool = False,
    result_dir: str = None,
    vanilla_attn: bool = False,
    dataset_name: str = "unknown",
    enable_torch_profiler: bool = False,
):
    # Ensure logging is configured for library use
    if not logging.getLogger().hasHandlers():
        global_logging_config()

    # only enable first rank
    if is_first_rank():
        logger.info(f"Tensor parallelism with {tp_ranks} ranks.")
    else:
        # disable logger globally
        logging.disable(logging.ERROR)

    # loading model
    default_dtype = str_to_dtype(dtype)
    # Use model_config_name if provided, otherwise use model_name
    config_name = model_name
    model_config = ModelArgs.from_name(config_name)
    # Auto-resolve clip_length to HF max_position_embeddings if not set
    if clip_length is None:
        if getattr(model_config, "max_position_embeddings", None) is not None:
            clip_length = int(model_config.max_position_embeddings)
        else:
            raise ValueError(
                "clip_length is not set and model_config.max_position_embeddings is not set; "
                "please pass --sspec_clip_length explicitly or provide HF max_position_embeddings in config.json"
            )
    else:
        # Clip length cannot exceed model's max position embeddings
        if getattr(model_config, "max_position_embeddings", None) is not None:
            original_clip = clip_length
            clip_length = min(clip_length, int(model_config.max_position_embeddings))
            if original_clip > clip_length:
                logger.warning(
                    f"clip_length ({original_clip}) exceeds model's max_position_embeddings "
                    f"({model_config.max_position_embeddings}). Using {clip_length} instead."
                )

    model_size = model_config.element_size() * dtype_to_bytes(default_dtype)
    page_size = 1  # hardcoded

    # Automatically bound all requests' output length by clip_length
    # This ensures no request exceeds the model's max_position_embeddings or clip_length
    num_clipped = 0
    for req in reqs:
        max_output_length = clip_length - req.len_input
        if req.len_output is None or req.len_output > max_output_length:
            if req.len_output is not None and req.len_output > max_output_length:
                num_clipped += 1
            req.len_output = max_output_length

    if num_clipped > 0:
        logger.info(
            f"Clipped {num_clipped} requests to fit within clip_length={clip_length}"
        )

    logger.info(f"[debug]: enabled={debug}, torch_profiler={enable_torch_profiler}")
    logger.info(f"[Model]: {model_config}")
    logger.info(
        f"[Specgen]: clip_length={clip_length}, max_position_embeddings={model_config.max_position_embeddings}"
    )
    logging.info(
        f"[Specgen Config] clip_length={clip_length}, max_position_embeddings={model_config.max_position_embeddings}"
    )
    logging.info(
        f"[Specgen Config] All requests bounded: max_total_length (input+output) <= {clip_length}"
    )

    # setup default device and dtype
    # assume intra-node sharding
    device = f"cuda:{get_local_rank()}"
    torch.set_default_device(device)
    torch.set_default_dtype(default_dtype)

    # reproducibility
    setup_seed(330)

    # allocate kv cache memory
    if max_mem_per_rank is None:
        max_mem_per_rank = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    gpu_size_per_rank = min(
        torch.cuda.get_device_properties(0).total_memory * mem_fraction,
        max_mem_per_rank * (1024**3),
    )
    kv_token_sz = (
        model_config.head_dim
        * model_config.num_kv_heads
        * model_config.num_layers
        * dtype_to_bytes(default_dtype)
        * 2
    )

    # assume model weights are evenly sharded
    available_kv_mem = gpu_size_per_rank * tp_ranks - model_size
    num_pages_per_rank = int(available_kv_mem / kv_token_sz / page_size)
    logging.info(
        f"GPU Size: {gpu_size_per_rank / (1024**3):.2f} GB, Model Size: {model_size / (1024**3):.2f} GB, "
        f"Available KV Memory: {available_kv_mem / (1024**3):.2f} GB, "
        f"tokens per rank: {num_pages_per_rank * page_size // 1024}K"
    )
    assert num_pages_per_rank > 0
    assert temperature >= 0.0

    # Get tokenizer for eos_token_id computation
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # collect job args and initialize kv-cache-ptr
    job_args = {
        "model_name": model_name,
        "max_batch_size": max_batch_size,
        "num_requests": len(reqs),
        "admit_policy": admit_policy,
        "tp_ranks": tp_ranks,
        "kv_cache": kv_cache,
        "eos_token_id": [
            model_config.eos_token_id,  # read from hf-config
            tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
            tokenizer.convert_tokens_to_ids("<\uff5cend\u2581of\u2581sentence\uff5c>"),
        ],
        "enable_wandb": enable_wandb,
        "async_cpu": async_cpu,
        "enable_cuda_graph": cuda_graph,
        "chunk_size": chunk_size,
        "batch_policy": batch_policy,
        "trace_name": dataset_name,  # Use trace_name for wandb compatibility
        "temperature": temperature,
        "enable_torch_profiler": enable_torch_profiler,
        "profiler_result_dir": result_dir,
    }

    additional_args = {
        "spec_stride": spec_stride,
        "budget_ratio": budget_ratio,
        "num_min_budget": num_min_budget,
        "num_sink_tokens": 4,
        "max_batch_size": max_batch_size,
    }
    logger.info(f"eos_token_id: {job_args['eos_token_id']}")

    # Below this point, actual model is created.
    with torch.inference_mode():
        if not debug:
            assert torch.cuda.device_count() >= tp_ranks

            # load
            time_init_start = time.time()
            with ctx_log(tag="model loading", enable_timing=True):
                model = Transformer.from_pretrained(
                    repo_id=model_name,
                    device=device,
                    dtype=default_dtype,
                )

            # shard
            with ctx_log(tag="model sharding", enable_timing=True):
                model = setup_tp_worker(
                    model=model,
                    tp_ranks=tp_ranks,
                    device=device,
                    dtype=default_dtype,
                    reduce_method=reduce_method,
                )
                # config is updated according to sharding policy
                model_config = model.config

            # init kv_cache_class
            kv_cache_class = dispatch_str_to_cache_ptr(kv_cache)
            kv_cache_class.initialize_class_metadata(
                dtype=default_dtype,
                head_dim=model_config.head_dim,
                num_qo_heads=model_config.num_qo_heads,
                num_kv_heads=model_config.num_kv_heads,
                num_layers=model_config.num_layers,
                capacity=num_pages_per_rank,
                device=device,
                max_batch_size=max_batch_size,  # for paged tensor
                spec_cache_args=additional_args,
            )

            # raise warning w/ corresponding check
            if kv_cache_class._MAX_KV_CONTEXT_LEN < num_pages_per_rank * page_size:
                logger.warning(
                    f"kv-indices graph buffer is too small, consider increasing the capacity"
                )
            if (
                spec_stride + 1
            ) * model_config.num_qo_heads // model_config.num_kv_heads >= 64:
                logger.warning(
                    f"Packed tokens in logits dump kernel is "
                    f"{((spec_stride + 1) * model_config.num_qo_heads // model_config.num_kv_heads)}"
                    f"which exceeds the limit of 64. This will impact the accuracy of speculation."
                )
            if (spec_stride + 1) >= RejectionSampler.MAX_SPEC_LEN:
                # round up to the nearest multiple of 4
                RejectionSampler.MAX_SPEC_LEN = math.ceil((spec_stride + 1) / 4) * 4
                logger.warning(
                    f"Adjust RejectionSampler.MAX_SPEC_LEN to {RejectionSampler.MAX_SPEC_LEN} "
                    f"to fit the spec_stride {spec_stride}"
                )
            if max_batch_size >= kv_cache_class._MAX_BATCH_SIZE:
                logger.warning(
                    f"max_batch_size is too large, consider decreasing the max_batch_size"
                )
            if (
                kv_cache != "full"
                and clip_length * budget_ratio
                >= kv_cache_class.MAX_LEN_SELECTED_KV_CACHE
            ):
                logger.warning(
                    f"budget_ratio is too large, consider decreasing the clip_length or budget_ratio"
                )

            # Model are sharded beyond this point, with sharded configuration.
            # assume distribute `num_heads` dimension to each rank
            with ctx_log(tag="kv_pool initialization", enable_timing=True):
                kv_pool = KVPool(
                    config=model_config,
                    capacity=num_pages_per_rank,
                    device=device,
                    dtype=default_dtype,
                    kv_cache_class=kv_cache_class,
                    page_size=page_size,
                    vanilla_backend=vanilla_attn,
                )

            with ctx_log(tag="attaching kv_pool", enable_timing=True):
                # init kv_pool
                model.attach_kv_pool(kv_pool)

            # Try to compile the model
            with ctx_log(tag="model compilation", enable_timing=True):
                model = CUDAGraphModelWrapper(model)
                if cuda_graph:
                    model.eager_cuda_graph_mode(
                        max_batch_size=int(max_batch_size * 2.5),
                        num_cuda_graphs=14,
                        max_padded_ratio=1,  # Maximal double the batch size
                    )

            # record init elapsed (local variable; not exported globally)
            init_elapsed_seconds = time.time() - time_init_start

        # start executing requests
        time_engine_start = time.time()
        if not debug:
            if kv_cache != "full":
                job_args.update(additional_args)
                ret_reqs = run_spec_scheduler(model, kv_pool, reqs, **job_args)
            else:
                # base
                ret_reqs = run_base_scheduler(model, kv_pool, reqs, **job_args)
        else:
            # debug scheduler
            logger.info("Running in debug mode, using DebugScheduler.")
            ret_reqs = run_debug_scheduler(reqs, **job_args)
        time_engine_end = time.time()

    # post-processing for statistics
    engine_elapsed_seconds = time_engine_end - time_engine_start

    num_finished_req = len(ret_reqs)
    num_total_input_tokens = sum([req.len_input for req in ret_reqs])
    num_total_output_tokens = sum(
        [len(req.token_ids) - req.len_input for req in ret_reqs]
    )
    average_input_tokens = num_total_input_tokens / num_finished_req
    average_output_tokens = num_total_output_tokens / num_finished_req

    input_throughput = num_total_input_tokens / engine_elapsed_seconds
    output_throughput = num_total_output_tokens / engine_elapsed_seconds
    total_throughput = (
        num_total_input_tokens + num_total_output_tokens
    ) / engine_elapsed_seconds

    table = [
        ["time (s)", f"{engine_elapsed_seconds:.4f}"],
        ["Finished Requests", num_finished_req],
        ["Total Input Tokens", num_total_input_tokens],
        ["Total Output Tokens", num_total_output_tokens],
        ["Avg. Input Tokens", f"{average_input_tokens:.2f}"],
        ["Avg. Output Tokens", f"{average_output_tokens:.2f}"],
        ["Input Throughput (tok/s)", f"{input_throughput:.2f}"],
        ["Output Throughput (tok/s)", f"{output_throughput:.2f}"],
        ["Total Throughput (tok/s)", f"{total_throughput:.2f}"],
    ]

    # Add CUDA graph statistics only if available
    if hasattr(model, "num_captured_tokens"):
        table.extend(
            [
                ["Graph Captured Tokens", model.num_captured_tokens],
                ["Graph Padded Tokens", model.num_padded_tokens],
                [
                    "Graph Captured Ratio",
                    f"{model.num_captured_tokens / model.num_total_tokens * 100:.2f}",
                ],
                [
                    "Graph Padded Ratio",
                    f"{model.num_padded_tokens / model.num_total_tokens * 100:.2f}",
                ],
            ]
        )
    else:
        table.extend(
            [
                ["Graph Captured Tokens", "N/A (EagerModelRunner)"],
                ["Graph Padded Tokens", "N/A (EagerModelRunner)"],
                ["Graph Captured Ratio", "N/A (EagerModelRunner)"],
                ["Graph Padded Ratio", "N/A (EagerModelRunner)"],
            ]
        )
    # Add spec scheduler metrics if available
    if kv_cache != "full":
        accepted_speculative_tokens = wandb_logger.log_data.get(
            "spec/accepted_speculative_tokens", 1
        )
        total_speculative_tokens = wandb_logger.log_data.get(
            "spec/total_speculative_tokens", 1
        )
        table.extend(
            [
                ["Accepted Speculative Tokens", accepted_speculative_tokens],
                ["Total Speculative Tokens", total_speculative_tokens],
                [
                    "Acceptance Rate",
                    f"{accepted_speculative_tokens / total_speculative_tokens * 100:.2f}",
                ],
            ]
        )
    else:
        table.extend(
            [
                ["Accepted Speculative Tokens", "N/A (BaseScheduler)"],
                ["Total Speculative Tokens", "N/A (BaseScheduler)"],
                ["Acceptance Rate", "N/A (BaseScheduler)"],
            ]
        )
    table_str = tabulate(table, headers=["Metric", "Value"], tablefmt="grid")
    logger.info(f"Benchmarking Results:\n{table_str}")

    for req in ret_reqs:
        assert len(req.token_ids) > 0, f"Request {req.token_ids} has no tokens"
        if req.token_ids[-1] == -233:
            req.token_ids = req.token_ids[:-1]

    outputs = []
    if result_dir is not None:
        # only first rank saves results
        if is_first_rank():
            # mkdir if not exists
            os.makedirs(result_dir, exist_ok=True)

            # save summary result
            with open(os.path.join(result_dir, "summary.log"), "w") as f:
                f.write(f"Job Args:\n{job_args}\n")
                f.write(f"Model Config:\n{model_config}\n")
                f.write(f"Benchmarking Results:\n{table_str}\n")

            # save per-request output
            for req in ret_reqs:
                input_tokens = tokenizer.decode(req.token_ids[: req.len_input])
                output_tokens = tokenizer.decode(req.token_ids[req.len_input :])
                full_tokens = tokenizer.decode(req.token_ids)
                outputs.append(
                    {
                        "id": req.request_id,
                        "len_input": req.len_input,
                        "len_output": len(req.token_ids) - req.len_input,
                        "output_token_ids": list(req.token_ids[req.len_input :]),
                        # following is expected by math_eval.py
                        "tokens": full_tokens,
                        "generated_tokens": output_tokens,
                    }
                )

            with open(os.path.join(result_dir, "requests.json"), "w") as f:
                json.dump(outputs, f, indent=4, ensure_ascii=False)
                f.write("\n")
    else:
        # Create outputs even if result_dir is None
        for req in ret_reqs:
            input_tokens = tokenizer.decode(req.token_ids[: req.len_input])
            output_tokens = tokenizer.decode(req.token_ids[req.len_input :])
            full_tokens = tokenizer.decode(req.token_ids)
            outputs.append(
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "len_input": req.len_input,
                    "len_output": len(req.token_ids) - req.len_input,
                    "output_token_ids": list(req.token_ids[req.len_input :]),
                    "id": req.request_id,
                    # Add format expected by math_eval.py
                    "tokens": full_tokens,
                    "generated_tokens": output_tokens,
                }
            )

    return outputs


if __name__ == "__main__":
    """
    Example usage:
        - ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=2 run.py --kv_cache full --max_batch_size 128 --tp_ranks 2 --num_requests 1024 --dtype bfloat16 --admit_policy lazy-stack-offload --cuda_graph --model_name Qwen2.5-7b
        - NSYS Prefix: nsys profile --cuda-graph-trace node --trace=cuda,nvtx,osrt --delay=180 --duration=3 --force-overwrite true -o my_profile
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_name", type=str, default="Qwen2.5-7b")
    argparser.add_argument("--max_batch_size", type=int, default=128)
    argparser.add_argument("--tp_ranks", type=int, default=1)
    argparser.add_argument("--max_mem_per_rank", type=int, default=None, help="GB")
    argparser.add_argument("--mem_fraction", type=float, default=0.9)
    argparser.add_argument(
        "--cuda_graph", action="store_true", help="Enable CUDA graph"
    )
    argparser.add_argument(
        "--async_cpu",
        action="store_true",
        help="Enable async CPU schedule to boost throughput",
    )
    argparser.add_argument(
        "--reduce_method",
        type=str,
        default="customized",
        choices=["naive", "pynccl", "customized"],
    )
    argparser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
    )
    argparser.add_argument(
        "--kv_cache", type=str, default="full", choices=["full", "streaming", "pillar"]
    )
    argparser.add_argument(
        "--admit_policy",
        type=str,
        default="naive",
        choices=[
            "naive",
            "oracle",
            "offloading",
        ],
    )
    argparser.add_argument(
        "--batch_policy",
        type=str,
        default="random",
        choices=["naive", "random", "greedy"],
    )
    argparser.add_argument(
        "--chunk_size", type=int, default=16 * 1024, help="chunked prefill"
    )

    # workload args
    argparser.add_argument("--num_requests", type=int, default=1024)
    argparser.add_argument("--clip_length", type=int, default=32 * 1024)
    argparser.add_argument("--dataset_name", type=str, default="aime")
    argparser.add_argument("--result_dir", type=str, default=None)
    argparser.add_argument("--enable_wandb", action="store_true")

    # spec args
    argparser.add_argument("--spec_stride", type=int, default=16)
    argparser.add_argument("--budget_ratio", type=float, default=0.05)
    argparser.add_argument("--num_min_budget", type=int, default=128)
    argparser.add_argument("--temperature", type=float, default=0.0)

    argparser.add_argument("--perf_sim", action="store_true")
    argparser.add_argument("--gemm_profile_path", type=str, default=None)
    argparser.add_argument("--debug", action="store_true", help="Transformer")
    argparser.add_argument("--vanilla-attn", action="store_true")
    argparser.add_argument("--enable-torch-profiler", action="store_true")
    args = argparser.parse_args()

    assert args.admit_policy != "naive" or args.kv_cache == "full"
    assert not args.cuda_graph or not args.vanilla_attn, "not compatible"
    assert not args.perf_sim, "perf simulation is deprecated"

    # setup logging config
    global_logging_config()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    reqs = get_req_queue(
        num_requests=args.num_requests,
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        clip_length=args.clip_length,
    )

    # Call the new run_inference function
    outputs = run_inference(
        reqs=reqs,
        model_name=args.model_name,
        max_batch_size=args.max_batch_size,
        tp_ranks=args.tp_ranks,
        max_mem_per_rank=args.max_mem_per_rank,
        mem_fraction=args.mem_fraction,
        cuda_graph=args.cuda_graph,
        async_cpu=args.async_cpu,
        reduce_method=args.reduce_method,
        dtype=args.dtype,
        kv_cache=args.kv_cache,
        admit_policy=args.admit_policy,
        batch_policy=args.batch_policy,
        chunk_size=args.chunk_size,
        clip_length=args.clip_length,
        enable_wandb=args.enable_wandb,
        spec_stride=args.spec_stride,
        budget_ratio=args.budget_ratio,
        num_min_budget=args.num_min_budget,
        temperature=args.temperature,
        perf_sim=args.perf_sim,
        gemm_profile_path=args.gemm_profile_path,
        debug=args.debug,
        result_dir=args.result_dir,
        vanilla_attn=args.vanilla_attn,
        dataset_name=args.dataset_name,
        enable_torch_profiler=args.enable_torch_profiler,
    )
