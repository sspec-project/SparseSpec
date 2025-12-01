import os
from typing import Optional

import torch
from torch.profiler.profiler import ProfilerAction


class Profiler:
    def __init__(
        self,
        tag: str,
        enable: bool,
        wait: int = 10,
        warmup: int = 50,
        active: int = 10,
        repeat: int = 2,
        result_dir: Optional[str] = None,
    ):
        self.tag = tag
        self.enable = enable
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat

        # setup result directory
        if result_dir is None:
            self.result_dir = f"{os.getcwd()}/profile"
        else:
            self.result_dir = result_dir

        self._step = 0
        self._start()

    def step(self):
        self._step += 1

        if self.prof is not None and self.enable:
            # record cuda profile
            if self.prof.schedule(self._step) in [
                ProfilerAction.RECORD_AND_SAVE,
                ProfilerAction.RECORD,
            ]:
                torch.cuda.synchronize()

            self.prof.step()

            if self.prof.schedule(self._step) in [
                ProfilerAction.RECORD_AND_SAVE,
                ProfilerAction.RECORD,
            ]:
                torch.cuda.synchronize()

            if self._step == self._profile_step():
                self._finalize()

            # record memory profile
            # only record once
            if self._step == self.wait:
                torch.cuda.memory._record_memory_history()

            if self._step == self.wait + self.active:
                local_file = f"{self.result_dir}/memory_viz.{self.tag}.pickle"
                torch.cuda.memory._dump_snapshot(local_file)
                torch.cuda.memory._record_memory_history(enabled=None)

    def _start(self):
        if not self.enable:
            self.prof = None
            return

        # create profile directory
        os.makedirs(self.result_dir, exist_ok=True)

        # create profiler
        self.prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
            ),
            on_trace_ready=self._trace_handler,
            record_shapes=True,
            with_stack=True,
            with_modules=True,
            profile_memory=False,
        )
        self.prof.start()

    def _profile_step(self):
        return (self.wait + self.warmup + self.active) * self.repeat

    def _finalize(self):
        self.prof.__exit__(None, None, None)
        self.prof = None

    def _trace_handler(self, prof):
        local_file = f"{self.result_dir}/{self.tag}.json.gz"
        prof.export_chrome_trace(local_file)
