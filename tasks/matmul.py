"""
Matrix multiplication task: C = A @ B

Reference: torch.matmul on float16 inputs.
Classic Triton optimization target: tiled matmul with register blocking.
"""

from __future__ import annotations

import inspect

import torch

from kernel_foundry.evaluation.benchmarker import Benchmarker
from kernel_foundry.task.spec import TaskSpec, detect_hardware_spec


M, N, K = 1024, 1024, 1024


def reference_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)


def input_generator() -> tuple:
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    return (a, b)


def build(config=None) -> TaskSpec:
    hardware_spec = detect_hardware_spec()
    baseline_time_ms = _measure_baseline()

    return TaskSpec(
        name="matmul_fp16",
        description=(
            f"Matrix multiplication: C = A @ B\n"
            f"A shape: ({M}, {K}), B shape: ({K}, {N}), dtype: float16, device: CUDA\n"
            f"Output C shape: ({M}, {N})\n"
            f"Baseline (torch.matmul): {baseline_time_ms:.3f}ms"
        ),
        reference_code=inspect.getsource(reference_fn),
        reference_fn=reference_fn,
        input_generator=input_generator,
        hardware_spec=hardware_spec,
        baseline_time_ms=baseline_time_ms,
    )


def _measure_baseline() -> float:
    if not torch.cuda.is_available():
        return 1.0

    benchmarker = Benchmarker(
        warmup_min_time=0.5,
        warmup_min_iters=5,
        benchmark_min_time=0.5,
        benchmark_min_iters=5,
    )
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    result = benchmarker.measure(lambda x, y: torch.matmul(x, y), (a, b))
    return result.mean_ms
