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

BATCH = 4
DIM = 2048
SEQLEN = 4096
WIDTH = 4


def reference_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    # x: (batch, dim, seqlen)
    # weight: (dim, width) — depthwise filter per channel
    # Causal padding: pad (width-1) on the left, 0 on the right
    x_padded = F.pad(x, (WIDTH - 1, 0))
    return F.conv1d(
        x_padded,
        weight.unsqueeze(1),  # (dim, 1, width)
        bias=bias,
        groups=DIM,
    )


def input_generator() -> tuple:
    x = torch.randn(BATCH, DIM, SEQLEN, device="cuda", dtype=torch.float32)
    weight = torch.randn(DIM, WIDTH, device="cuda", dtype=torch.float32)
    bias = torch.randn(DIM, device="cuda", dtype=torch.float32)
    return (x, weight, bias)


_SEED_KERNEL = """\
import torch
import triton
import triton.language as tl


@triton.jit
def _silu(acc):
    return acc / (1 + tl.exp2(-1.44269504089 * acc))


@triton.jit
def _causal_conv1d_fwd_kernel(
    X, WEIGHT, BIAS, OUT,
    seqlen, dim,
    stride_x_batch, stride_x_channel, stride_x_seqlen,
    stride_weight_channel, stride_weight_width,
    stride_bias_channel,
    stride_out_batch, stride_out_channel, stride_out_seqlen,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_l = tl.program_id(0).to(tl.int64)
    pid_c = tl.program_id(1).to(tl.int64)
    pid_b = tl.program_id(2).to(tl.int64)

    channels = pid_c * BLOCK_C + tl.arange(0, BLOCK_C).to(tl.int64)
    out_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L).to(tl.int64)

    channel_mask = channels < dim
    out_mask = out_offsets < seqlen
    full_mask = channel_mask[:, None] & out_mask[None, :]

    x_ptrs = (
        X
        + pid_b * stride_x_batch
        + channels[:, None] * stride_x_channel
        + out_offsets[None, :] * stride_x_seqlen
    )
    out_ptrs = (
        OUT
        + pid_b * stride_out_batch
        + channels[:, None] * stride_out_channel
        + out_offsets[None, :] * stride_out_seqlen
    )
    weight_ptrs = WEIGHT + channels * stride_weight_channel

    acc = tl.zeros((BLOCK_C, BLOCK_L), dtype=tl.float32)
    if HAS_BIAS:
        bias_vals = tl.load(BIAS + channels * stride_bias_channel, mask=channel_mask, other=0.0).to(tl.float32)
        acc += bias_vals[:, None]

    w0 = tl.load(weight_ptrs + 0 * stride_weight_width, mask=channel_mask, other=0.0).to(tl.float32)
    x0_ptrs = x_ptrs - (WIDTH - 1) * stride_x_seqlen
    x0_mask = channel_mask[:, None] & (out_offsets[None, :] >= (WIDTH - 1)) & out_mask[None, :]
    x0 = tl.load(x0_ptrs, mask=x0_mask, other=0.0).to(tl.float32)
    acc += w0[:, None] * x0

    w1 = tl.load(weight_ptrs + 1 * stride_weight_width, mask=channel_mask, other=0.0).to(tl.float32)
    x1_ptrs = x_ptrs - (WIDTH - 2) * stride_x_seqlen
    x1_mask = channel_mask[:, None] & (out_offsets[None, :] >= (WIDTH - 2)) & out_mask[None, :]
    x1 = tl.load(x1_ptrs, mask=x1_mask, other=0.0).to(tl.float32)
    acc += w1[:, None] * x1

    if WIDTH >= 3:
        w2 = tl.load(weight_ptrs + 2 * stride_weight_width, mask=channel_mask, other=0.0).to(tl.float32)
        x2_ptrs = x_ptrs - (WIDTH - 3) * stride_x_seqlen
        x2_mask = channel_mask[:, None] & (out_offsets[None, :] >= (WIDTH - 3)) & out_mask[None, :]
        x2 = tl.load(x2_ptrs, mask=x2_mask, other=0.0).to(tl.float32)
        acc += w2[:, None] * x2

    if WIDTH >= 4:
        w3 = tl.load(weight_ptrs + 3 * stride_weight_width, mask=channel_mask, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptrs, mask=full_mask, other=0.0).to(tl.float32)
        acc += w3[:, None] * x3

    if SILU_ACTIVATION:
        acc = _silu(acc)

    tl.store(out_ptrs, acc, mask=full_mask)


def kernel_fn(x, weight, bias):
    batch, dim, seqlen = x.shape
    out = torch.empty_like(x)
    BLOCK_C = 32
    BLOCK_L = 128
    width = weight.shape[1]
    grid = (triton.cdiv(seqlen, BLOCK_L), triton.cdiv(dim, BLOCK_C), batch)
    _causal_conv1d_fwd_kernel[grid](
        x, weight, bias, out,
        seqlen, dim,
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(0), weight.stride(1),
        bias.stride(0),
        out.stride(0), out.stride(1), out.stride(2),
        HAS_BIAS=True,
        SILU_ACTIVATION=False,
        WIDTH=width,
        BLOCK_C=BLOCK_C,
        BLOCK_L=BLOCK_L,
    )
    return out
"""


def build(config=None) -> TaskSpec:
    hardware_spec = detect_hardware_spec()
    baseline_time_ms = _measure_baseline()

    return TaskSpec(
        name="causal_conv1d",
        description=(
            f"Depthwise causal 1D convolution: out[b, c, t] = bias[c] + sum_w(weight[c, w] * x[b, c, t - (WIDTH-1) + w])\n"
            f"Input x shape: ({BATCH}, {DIM}, {SEQLEN}), dtype: float32, device: CUDA\n"
            f"Weight shape: ({DIM}, {WIDTH}), Bias shape: ({DIM},)\n"
            f"Causal: output at position t depends only on x[..., t-{WIDTH - 1}:t+1]\n"
            f"No SiLU activation.\n"
            f"Baseline (F.conv1d depthwise): {baseline_time_ms:.3f}ms"
        ),
        reference_code=inspect.getsource(reference_fn),
        reference_fn=reference_fn,
        input_generator=input_generator,
        hardware_spec=hardware_spec,
        baseline_time_ms=baseline_time_ms,
        seed_kernel=_SEED_KERNEL,
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
            F.pad(x, (WIDTH - 1, 0)),
            w.unsqueeze(1),
            bias=b,
            groups=DIM,
        ),
        (x, weight, bias),
    )
    return result.mean_ms
