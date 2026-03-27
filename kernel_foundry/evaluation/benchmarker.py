from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class BenchmarkResult:
    mean_ms: float
    std_ms: float
    min_ms: float
    median_ms: float
    num_iters: int


class Benchmarker:
    """
    Measures kernel execution time using the paper's adaptive warmup/benchmark loop.

    Algorithm:
    1. Run 3 probe iterations to estimate rough runtime.
    2. Warmup until warmup_min_time elapsed AND warmup_min_iters reached.
    3. Determine inner_loop_iters = max(1, ceil(inner_loop_min_time / rough_runtime))
       to amortize synchronization overhead for fast kernels.
    4. Main measurement: run until benchmark_min_time elapsed AND benchmark_min_iters reached.
    """

    def __init__(
        self,
        warmup_min_time: float = 1.0,
        warmup_min_iters: int = 10,
        benchmark_min_time: float = 1.0,
        benchmark_min_iters: int = 10,
        inner_loop_min_time: float = 0.01,
    ) -> None:
        self._warmup_min_time = warmup_min_time
        self._warmup_min_iters = warmup_min_iters
        self._benchmark_min_time = benchmark_min_time
        self._benchmark_min_iters = benchmark_min_iters
        self._inner_loop_min_time = inner_loop_min_time

    def measure(self, fn: Callable, inputs: tuple) -> BenchmarkResult:
        """Time fn(*inputs) on CUDA. Returns BenchmarkResult in milliseconds."""
        device = self._find_cuda_device(inputs)

        # Probe phase: 3 runs to estimate rough runtime
        rough_ms = self._probe(fn, inputs, device)

        # Warmup
        warmup_iters = max(self._warmup_min_iters, int(self._warmup_min_time / (rough_ms / 1000) + 1))
        for _ in range(warmup_iters):
            fn(*inputs)
        if device is not None:
            torch.cuda.synchronize(device)

        # Determine inner loop count (amortize synchronize overhead)
        inner_iters = max(1, int(self._inner_loop_min_time / (rough_ms / 1000) + 1))

        # Main benchmark loop
        timings: list[float] = []
        elapsed = 0.0
        main_iters = 0
        while elapsed < self._benchmark_min_time or main_iters < self._benchmark_min_iters:
            t0 = time.perf_counter()
            for _ in range(inner_iters):
                fn(*inputs)
            if device is not None:
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            per_iter_ms = (t1 - t0) * 1000 / inner_iters
            timings.append(per_iter_ms)
            elapsed += t1 - t0
            main_iters += 1

        import statistics

        return BenchmarkResult(
            mean_ms=statistics.mean(timings),
            std_ms=statistics.stdev(timings) if len(timings) > 1 else 0.0,
            min_ms=min(timings),
            median_ms=statistics.median(timings),
            num_iters=main_iters * inner_iters,
        )

    def _probe(self, fn: Callable, inputs: tuple, device) -> float:
        """Run 3 iterations and return mean time in ms."""
        if device is not None:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(3):
            fn(*inputs)
        if device is not None:
            torch.cuda.synchronize(device)
        return (time.perf_counter() - t0) * 1000 / 3

    @staticmethod
    def _find_cuda_device(inputs: tuple):
        """Return the CUDA device of the first tensor input, or None."""
        for x in inputs:
            if isinstance(x, torch.Tensor) and x.is_cuda:
                return x.device
        return None
