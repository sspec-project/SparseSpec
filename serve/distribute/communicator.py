from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from serve.distribute.custom_all_reduce import CustomAllreduce
from serve.distribute.pynccl import PyNcclCommunicator
from serve.utils import get_local_rank, get_world_size


class CudaCommunicator:

    def __init__(self, device_group: ProcessGroup, method: str = "customized"):
        assert device_group is not None
        assert dist.get_backend(device_group) == dist.Backend.NCCL
        self.device_group = device_group
        self.method = method

        # NOTE: always use default_process_group's rank assignment
        self.rank = get_local_rank()

        self.world_size = get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")

        match method:
            case "naive":
                self.communicator = None
                self.reducer = lambda x: (
                    dist.all_reduce(x, group=self.device_group),
                    x,
                )[1]
            case "pynccl":
                ranks = list(range(self.world_size))
                customized_group = torch.distributed.new_group(ranks, backend="gloo")
                self.communicator = PyNcclCommunicator(customized_group, self.device)
                self.reducer = lambda x: self.communicator.all_reduce(x)
            case "customized":
                ranks = list(range(self.world_size))
                customized_group = torch.distributed.new_group(ranks, backend="gloo")
                self.communicator = CustomAllreduce(
                    group=customized_group, device=self.device
                )
                self.reducer = lambda x: self.communicator.custom_all_reduce(x)
            case _:
                raise ValueError(f"Unsupported method: {method}")

    @contextmanager
    def capture_context(self):
        match self.method:
            case "naive":
                # No special context to capture for naive method
                yield
            case "pynccl":
                # PyNcclCommunicator does not have a specific context manager
                yield
            case "customized":
                # CustomAllreduce may have its own context management
                with self.communicator.capture():
                    yield
            case _:
                raise ValueError(f"Unsupported method: {self.method}")

    def all_reduce(self, input_):
        assert self.reducer is not None, "Reducer is not initialized."
        out = self.reducer(input_)
        return out

    def __del__(self):
        if self.communicator is not None:
            del self.communicator
            self.communicator = None
            self.reducer = None
