"""
Depthwise causal 1D convolution task.

Reference: F.conv1d with left-only (causal) padding and groups=dim,
equivalent to the mamba-ssm causal_conv1d kernel.
This is a classic optimization target:
- Depthwise (one filter per channel), so no cross-channel reduction
- Causal masking: output[b, c, t] depends only on x[b, c, t-w+1 .. t]
- Optional SiLU activation fused into the kernel
"""

from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F

from kernel_foundry.evaluation.benchmarker import Benchmarker
from kernel_foundry.task.spec import TaskSpec, detect_hardware_spec


BATCH = 2
DIM = 1024
SEQLEN = 2048
WIDTH = 4


def reference_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    # x: (batch, dim, seqlen)
    # weight: (dim, width) — depthwise filter per channel
    # Causal padding: pad (width-1) on the left, 0 on the right
    out = F.conv1d(
        x,
        weight.unsqueeze(1),  # (dim, 1, width)
        bias=bias,
        groups=DIM,
        padding=(WIDTH - 1, 0),
    )
    return out[..., :SEQLEN]


def input_generator() -> tuple:
    x = torch.randn(BATCH, DIM, SEQLEN, device="cuda", dtype=torch.float32)
    weight = torch.randn(DIM, WIDTH, device="cuda", dtype=torch.float32)
    bias = torch.randn(DIM, device="cuda", dtype=torch.float32)
    return (x, weight, bias)


def build(config=None) -> TaskSpec:
    hardware_spec = detect_hardware_spec()
    baseline_time_ms = _measure_baseline()

    return TaskSpec(
        name="causal_conv1d",
        description=(
            f"Depthwise causal 1D convolution: out[b, c, t] = bias[c] + sum_w(weight[c, w] * x[b, c, t - (WIDTH-1) + w])\n"
            f"Input x shape: ({BATCH}, {DIM}, {SEQLEN}), dtype: float32, device: CUDA\n"
            f"Weight shape: ({DIM}, {WIDTH}), Bias shape: ({DIM},)\n"
            f"Causal: output at position t depends only on x[..., t-{WIDTH-1}:t+1]\n"
            f"No SiLU activation.\n"
            f"Baseline (F.conv1d depthwise): {baseline_time_ms:.3f}ms"
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
    x = torch.randn(BATCH, DIM, SEQLEN, device="cuda", dtype=torch.float32)
    weight = torch.randn(DIM, WIDTH, device="cuda", dtype=torch.float32)
    bias = torch.randn(DIM, device="cuda", dtype=torch.float32)
    result = benchmarker.measure(
        lambda x, w, b: F.conv1d(
            x,
            w.unsqueeze(1),
            bias=b,
            groups=DIM,
            padding=(WIDTH - 1, 0),
        )[..., :SEQLEN],
        (x, weight, bias),
    )
    return result.mean_ms
