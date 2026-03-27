"""
Row-wise softmax task.

Reference: torch.softmax(x, dim=-1) over a 2D input.
This is a classic Triton tutorial kernel with known optimization strategies:
- Flash-attention-style online softmax (2-pass → 1-pass via running max/sum)
- Coalesced row access with BLOCK_SIZE tiling
"""

from __future__ import annotations

import inspect

import torch

from kernel_foundry.evaluation.benchmarker import Benchmarker
from kernel_foundry.task.spec import TaskSpec, detect_hardware_spec


ROWS = 1024
COLS = 4096


def reference_fn(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def input_generator() -> tuple:
    x = torch.randn(ROWS, COLS, device="cuda", dtype=torch.float32)
    return (x,)


def build(config=None) -> TaskSpec:
    hardware_spec = detect_hardware_spec()

    # Measure baseline once
    baseline_time_ms = _measure_baseline()

    return TaskSpec(
        name="row_softmax",
        description=(
            f"Compute row-wise softmax: output[i, j] = exp(x[i, j]) / sum_k(exp(x[i, k]))\n"
            f"Input shape: ({ROWS}, {COLS}), dtype: float32, device: CUDA\n"
            f"Each output row must sum to 1.0.\n"
            f"Baseline (torch.softmax): {baseline_time_ms:.3f}ms"
        ),
        reference_code=inspect.getsource(reference_fn),
        reference_fn=reference_fn,
        input_generator=input_generator,
        hardware_spec=hardware_spec,
        baseline_time_ms=baseline_time_ms,
    )


def _measure_baseline() -> float:
    if not torch.cuda.is_available():
        return 1.0  # dummy for CPU testing

    benchmarker = Benchmarker(
        warmup_min_time=0.5,
        warmup_min_iters=5,
        benchmark_min_time=0.5,
        benchmark_min_iters=5,
    )
    x = torch.randn(ROWS, COLS, device="cuda", dtype=torch.float32)
    result = benchmarker.measure(lambda t: torch.softmax(t, dim=-1), (x,))
    return result.mean_ms
