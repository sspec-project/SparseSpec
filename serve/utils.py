import array
import contextlib
import gc
import logging
import os
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps

import numpy as np
import torch
import triton
import triton.language as tl
import wandb
from torch._subclasses.fake_tensor import FakeTensor

# import serve.kernels

logger = logging.getLogger(__name__)

_INTERVAL_COUNTERS = defaultdict(int)


@triton.jit
def index_copy_kernel(
    dst_ptr,
    src_ptr,
    src_idx_ptr,
    des_idx_ptr,
    n_indices,
    src_len,
    dst_len,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    m = offs < n_indices

    sidx = tl.load(src_idx_ptr + offs, mask=m, other=0)
    didx = tl.load(des_idx_ptr + offs, mask=m, other=0)

    m = m & (sidx >= 0) & (sidx < src_len) & (didx >= 0) & (didx < dst_len)
    vals = tl.load(src_ptr + sidx, mask=m, other=0)

    tl.store(dst_ptr + didx, vals, mask=m)


def index_copy(
    dst: torch.Tensor,
    src: torch.Tensor,
    src_idx: torch.Tensor,
    des_idx: torch.Tensor,
):
    assert dst.dtype == src.dtype
    assert dst.ndim == src.ndim == 1
    assert dst.is_contiguous() and src.is_contiguous()
    assert src_idx.ndim == des_idx.ndim == 1
    assert src_idx.numel() == des_idx.numel()
    assert src_idx.is_contiguous() and des_idx.is_contiguous()

    n = src_idx.numel()
    if n <= 0:
        return

    block_size = 32
    grid = (triton.cdiv(n, block_size),)
    index_copy_kernel[grid](
        dst,
        src,
        src_idx,
        des_idx,
        n,
        src.numel(),
        dst.numel(),
        BLOCK=block_size,
        num_warps=1,
    )


@contextmanager
def interval_trigger(tag: str, ntimes: int):
    if ntimes <= 0:
        raise ValueError("ntimes must be a positive integer")

    _INTERVAL_COUNTERS[tag] += 1
    fire = _INTERVAL_COUNTERS[tag] % ntimes == 0

    try:
        yield fire
    finally:
        pass


@torch.compile(dynamic=True)
def get_masked_input_and_mask(
    input_: torch.Tensor, org_vocab_start_index: int, org_vocab_end_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Credit: https://github.com/vllm-project/vllm
    """
    # torch.compile will fuse all of the pointwise ops below
    # into a single kernel, making it very fast
    org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
    valid_offset = org_vocab_start_index * org_vocab_mask
    vocab_mask = org_vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask


@contextmanager
def ctx_torch_default(device=None, dtype=None):
    old_device = torch.get_default_device()
    old_dtype = torch.get_default_dtype()

    if device is not None:
        torch.set_default_device(device)
    if dtype is not None:
        torch.set_default_dtype(dtype)

    try:
        yield
    finally:
        torch.set_default_device(old_device)
        torch.set_default_dtype(old_dtype)


@contextmanager
def ctx_log(tag: str = "", enable_timing: bool = False):
    logger.info(f"{tag} start")
    start_time = time.time()
    try:
        yield
    finally:
        if enable_timing:
            elapsed = time.time() - start_time
            logger.info(f"{tag} done, elapsed {elapsed:.4f}s")


def cleanup(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

    return wrapper


def find_nccl_library() -> str:
    """Return the path to the NCCL library if available, otherwise raise an error."""
    if torch.version.cuda is not None:
        so_file = "libnccl.so.2"
    else:
        raise ValueError("NCCL only supports CUDA and ROCm backends.")
    logger.info("Found nccl from library %s", so_file)
    return so_file


@contextlib.contextmanager
def timing_context(name):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    logger.info(f"{name} took {elapsed_time:.4f} seconds")


def str_to_dtype(dtype: str):
    match dtype:
        case "float32":
            return torch.float32
        case "float16":
            return torch.float16
        case "bfloat16":
            return torch.bfloat16
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def dtype_to_bytes(dtype: torch.dtype):
    match dtype:
        case torch.float32:
            return 4
        case torch.float16:
            return 2
        case torch.bfloat16:
            return 2
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def setup_seed(seed):
    """Setup all seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_local_rank() -> int:
    """Get the local rank of the current process."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_first_rank() -> bool:
    """Check if the current process is the first process."""
    return get_local_rank() == 0


def get_world_size() -> int:
    """Get the world size of the current process."""
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


def lists_to_tensors(device, *lists, fake_tensor=False):
    """
    Convert multiple lists of integers to tensors on the specified device.

    :param device: The target device (e.g., self.model.device)
    :param lists: Arbitrary number of lists containing integers
    :return: A tuple of tensors
    """
    if fake_tensor:
        # return torch.empty with the same shape and dtype
        return tuple(
            torch.empty(len(lst), dtype=torch.int32, device=device) for lst in lists
        )

    return tuple(torch.tensor(lst, dtype=torch.int32, device=device) for lst in lists)


def tensors_to_lists(*tensors):
    """
    Convert multiple tensors to lists of integers.

    :param tensors: Arbitrary number of tensors
    :return: A tuple of lists
    """

    if not isinstance(tensors[0], FakeTensor):
        return tuple(t.cpu().tolist() for t in tensors)
    else:
        ret_list = []
        for t in tensors:
            assert (
                t.dtype == torch.long or t.dtype == torch.int32
            ) and t.dim() == 1, f"Invalid tensor: {t}, {t.dtype}, {t.dim()}"
            lst = [random.randint(200, 500) for _ in range(t.numel())]
            ret_list.append(lst)

        return tuple(ret_list)


def global_logging_config():
    """Called once in main entry."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - (%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def num_status_req(reqs, status) -> int:
    return sum(1 for req in reqs if req.status == status)


class MetricBucket:

    def __init__(self):
        self.log_data = {}

    def get_op_func(self, op: str):
        match op:
            case "sum":
                return lambda x, y: x + y
            case "max":
                return max
            case "replace":
                return lambda x, y: y
            case _:
                raise ValueError(f"Unsupported operation: {op}")

    def merge(self, metrics: dict, op: str = "replace"):
        op_func = self.get_op_func(op)

        for k, v in metrics.items():
            if k in self.log_data:
                self.log_data[k] = op_func(self.log_data[k], v)
            else:
                self.log_data[k] = v

    def merge_bucket(self, _bucket: "MetricBucket", op: str = "replace"):
        self.merge(_bucket.log_data, op)

    def get(self, key, default=None):
        return self.log_data.get(key, default)

    @property
    def data(self) -> dict:
        return self.log_data


class WandbLogger:

    def __init__(self):
        self.step_counter = 0
        self.log_data = MetricBucket()

    def init_wandb(
        self, config, project_name="Specgen-benchmark", entity="yilongzhao-uc-berkeley"
    ):
        if wandb.run is None and is_first_rank():
            assert os.environ.get("WANDB_API_KEY") is not None
            wandb.login()
            model_name = config["model_name"]
            admit_policy = config["admit_policy"]
            max_batch_size = config["max_batch_size"]
            num_requests = config["num_requests"]
            tp_ranks = config["tp_ranks"]
            kv_cache = config["kv_cache"]
            async_mode = config["async_cpu"]
            batch_policy = config["batch_policy"]
            trace_name = config["trace_name"]
            if kv_cache != "full":
                spec_stride = config["spec_stride"]
                budget_ratio = config["budget_ratio"]
                name = (
                    f"{model_name}-{trace_name}-{admit_policy}-{batch_policy}-bs{max_batch_size}-tp{tp_ranks}-spec-stride{spec_stride}-budget{budget_ratio}-num{num_requests}"
                    + ("-async" if async_mode else "")
                )
            else:
                name = (
                    f"{model_name}-{trace_name}-{admit_policy}-bs{max_batch_size}-tp{tp_ranks}-full-num{num_requests}"
                    + ("-async" if async_mode else "")
                )
            # name += year-mon-day-hour-min
            name += time.strftime("-%Y-%m-%d-%H-%M", time.localtime())
            wandb.init(project=project_name, entity=entity, name=name, config=config)

    def log(self, metrics: dict, categories: str = None, op: str = "replace"):
        if categories is not None:
            metrics = {f"{categories}/{k}": v for k, v in metrics.items()}
        else:
            metrics = metrics

        self.log_data.merge(metrics, op)

    def step(self):
        """Called each decoding step. Log the current metrics."""
        if not wandb.run or not is_first_rank():
            return
        wandb.log(self.log_data.data, step=self.step_counter)

        # Store the current log data for comparison in the next step
        self.step_counter += 1


wandb_logger = WandbLogger()
