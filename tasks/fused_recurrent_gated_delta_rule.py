"""
Fused recurrent gated delta rule forward task.

This task is intentionally close to vLLM's
`fused_recurrent_gated_delta_rule_fwd_kernel` and its Python wrapper:
- same high-level `kernel_fn` arguments
- continuous batching via `ssm_state_indices`
- optional speculative decoding via `num_accepted_tokens`
- varlen sequences via `cu_seqlens`
- headwise-vector `beta`
- optional Q/K L2 normalization flag

To keep evaluation practical, the task checks the recurrent output tensor only.
"""

from __future__ import annotations

import inspect
import math
from functools import partial

import torch
import torch.nn.functional as F

from kernel_foundry.evaluation.benchmarker import Benchmarker
from kernel_foundry.task.spec import BenchmarkCase, TaskSpec, detect_hardware_spec

DTYPE = torch.bfloat16

CASE_SPECS = [
    {
        "name": "decode",
        "num_reqs": 16,
        "tokens_per_req": 1,
        "num_q_heads": 4,
        "num_v_heads": 8,
        "head_k_dim": 64,
        "head_v_dim": 64,
        "state_pool": 32,
    },
    {
        "name": "spec_decode",
        "num_reqs": 8,
        "tokens_per_req": 4,
        "num_q_heads": 4,
        "num_v_heads": 8,
        "head_k_dim": 64,
        "head_v_dim": 64,
        "state_pool": 48,
    },
]


def reference_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    ssm_state_indices: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
) -> torch.Tensor:
    _, total_tokens, num_q_heads, head_k_dim = q.shape
    _, _, num_v_heads, head_v_dim = v.shape
    assert num_v_heads % num_q_heads == 0
    groups = num_v_heads // num_q_heads

    q_flat = q.reshape(total_tokens, num_q_heads, head_k_dim).float()
    k_flat = k.reshape(total_tokens, num_q_heads, head_k_dim).float()
    v_flat = v.reshape(total_tokens, num_v_heads, head_v_dim).float()
    g_flat = g.reshape(total_tokens, num_v_heads).float()
    beta_flat = beta.reshape(total_tokens, num_v_heads, head_v_dim).float()

    if use_qk_l2norm_in_kernel:
        q_flat = q_flat / (torch.linalg.vector_norm(q_flat, dim=-1, keepdim=True) + 1e-6)
        k_flat = k_flat / (torch.linalg.vector_norm(k_flat, dim=-1, keepdim=True) + 1e-6)
    q_flat = q_flat * scale

    if cu_seqlens is None:
        num_sequences = q.shape[0]
        seq_len = q.shape[1]
        boundaries = [(i * seq_len, (i + 1) * seq_len) for i in range(num_sequences)]
    else:
        cu = cu_seqlens.tolist()
        boundaries = [(int(cu[i]), int(cu[i + 1])) for i in range(len(cu) - 1)]

    out = torch.zeros(total_tokens, num_v_heads, head_v_dim, device=q.device, dtype=torch.float32)

    for seq_idx, (bos, eos) in enumerate(boundaries):
        if eos <= bos:
            continue

        if ssm_state_indices is None:
            state = initial_state[seq_idx].float().clone()
        else:
            start_slot = 0
            if num_accepted_tokens is not None:
                start_slot = int(num_accepted_tokens[seq_idx].item()) - 1
            if ssm_state_indices.ndim == 1:
                state_idx = int(ssm_state_indices[seq_idx].item())
            else:
                state_idx = int(ssm_state_indices[seq_idx, start_slot].item())
            if state_idx < 0:
                continue
            state = initial_state[state_idx].float().clone()

        for token_slot, token_idx in enumerate(range(bos, eos)):
            q_t = q_flat[token_idx]
            k_t = k_flat[token_idx]
            v_t = v_flat[token_idx]
            g_t = g_flat[token_idx]
            beta_t = beta_flat[token_idx]

            for hv in range(num_v_heads):
                q_head = hv // groups
                state_h = state[hv]
                state_h = state_h * torch.exp(g_t[hv])
                projected = torch.matmul(state_h, k_t[q_head].unsqueeze(-1)).squeeze(-1)
                delta = (v_t[hv] - projected) * beta_t[hv]
                state_h = state_h + delta.unsqueeze(-1) * k_t[q_head].unsqueeze(0)
                out[token_idx, hv] = torch.matmul(state_h, q_t[q_head].unsqueeze(-1)).squeeze(-1)
                state[hv] = state_h

    return out.view(1, total_tokens, num_v_heads, head_v_dim).to(q.dtype)


def input_generator() -> tuple:
    idx = torch.randint(0, len(CASE_SPECS), (), device="cpu").item()
    return _generate_inputs(CASE_SPECS[idx])


_SEED_KERNEL = """\
import torch
import triton
import triton.language as tl


@triton.jit
def exp(x):
    return tl.exp2(1.4426950408889634 * x)


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N: tl.int64,
    T: tl.int64,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
    p_g = g + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(tl.int64)
            if state_idx < 0:
                return
            p_h0 = h0 + state_idx * stride_init_state_token
        else:
            p_h0 = h0 + bos * stride_init_state_token
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_g = tl.load(p_g).to(tl.float32)
        b_h *= exp(b_g)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        if INPLACE_FINAL_STATE:
            final_state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t * stride_indices_tok
            ).to(tl.int64)
            if final_state_idx >= 0:
                p_ht = ht + final_state_idx * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_g += HV
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)


def kernel_fn(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    use_qk_l2norm_in_kernel,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"

    o = q.new_empty(NK, *v.shape)
    final_state = initial_state

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (NK, NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=True,
        IS_KDA=False,
        num_warps=1,
        num_stages=3,
    )
    return o.squeeze(0)
"""


def build(config=None) -> TaskSpec:
    hardware_spec = detect_hardware_spec()
    benchmark_cases = _build_benchmark_cases()
    baseline_time_ms = _geometric_mean(case.baseline_time_ms for case in benchmark_cases)
    case_summary = ", ".join(
        (
            f"{case['name']}: q/k=(1, {case['num_reqs'] * case['tokens_per_req']}, "
            f"{case['num_q_heads']}, {case['head_k_dim']}), "
            f"v/out=(1, {case['num_reqs'] * case['tokens_per_req']}, "
            f"{case['num_v_heads']}, {case['head_v_dim']})"
        )
        for case in CASE_SPECS
    )

    return TaskSpec(
        name="fused_recurrent_gated_delta_rule",
        description=(
            "Forward fused recurrent gated delta rule task modeled after vLLM's decode path.\n"
            "Inputs follow the vLLM wrapper: q, k, v, g, beta, scale, initial_state, "
            "cu_seqlens, ssm_state_indices, num_accepted_tokens, use_qk_l2norm_in_kernel.\n"
            "This task includes continuous batching through state-index indirection, "
            "varlen sequence boundaries through cu_seqlens, and speculative decoding via "
            "num_accepted_tokens.\n"
            "The checked output is the recurrent attention tensor with shape (1, total_tokens, HV, V).\n"
            f"Evaluate across shape suite: {case_summary}\n"
            f"Input dtype: {DTYPE}, g dtype: float32, device: CUDA\n"
            f"Aggregate baseline (geometric mean, PyTorch reference): {baseline_time_ms:.3f}ms"
        ),
        reference_code=inspect.getsource(reference_fn),
        reference_fn=reference_fn,
        input_generator=input_generator,
        hardware_spec=hardware_spec,
        baseline_time_ms=baseline_time_ms,
        benchmark_cases=benchmark_cases,
        seed_kernel=_SEED_KERNEL,
    )


def _build_benchmark_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            name=case["name"],
            input_generator=partial(_generate_inputs, case),
            baseline_time_ms=_measure_baseline(case),
        )
        for case in CASE_SPECS
    ]


def _generate_inputs(case: dict) -> tuple:
    num_reqs = case["num_reqs"]
    tokens_per_req = case["tokens_per_req"]
    total_tokens = num_reqs * tokens_per_req
    num_q_heads = case["num_q_heads"]
    num_v_heads = case["num_v_heads"]
    head_k_dim = case["head_k_dim"]
    head_v_dim = case["head_v_dim"]
    state_pool = case["state_pool"]

    q = torch.randn(
        1,
        total_tokens,
        num_q_heads,
        head_k_dim,
        device="cuda",
        dtype=DTYPE,
    )
    k = torch.randn_like(q)
    v = torch.randn(
        1,
        total_tokens,
        num_v_heads,
        head_v_dim,
        device="cuda",
        dtype=DTYPE,
    )

    a = torch.randn(total_tokens, num_v_heads, device="cuda", dtype=torch.float32)
    b = torch.randn(total_tokens, num_v_heads, head_v_dim, device="cuda", dtype=torch.float32)
    a_log = torch.randn(num_v_heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(num_v_heads, device="cuda", dtype=torch.float32)

    g = (-torch.exp(a_log)[None, :] * F.softplus(a + dt_bias[None, :])).view(
        1, total_tokens, num_v_heads
    )
    beta = torch.sigmoid(b).to(DTYPE).view(1, total_tokens, num_v_heads, head_v_dim)
    scale = float(head_k_dim**-0.5)

    initial_state = torch.randn(
        state_pool,
        num_v_heads,
        head_v_dim,
        head_k_dim,
        device="cuda",
        dtype=DTYPE,
    )

    cu_seqlens = torch.arange(
        0,
        total_tokens + 1,
        tokens_per_req,
        device="cuda",
        dtype=torch.int32,
    )
    if tokens_per_req == 1:
        ssm_state_indices = torch.randperm(state_pool, device="cuda", dtype=torch.int32)[:num_reqs]
        num_accepted_tokens = None
    else:
        ssm_state_indices = torch.randint(
            0,
            state_pool,
            (num_reqs, tokens_per_req),
            device="cuda",
            dtype=torch.int32,
        )
        num_accepted_tokens = torch.randint(
            1,
            tokens_per_req + 1,
            (num_reqs,),
            device="cuda",
            dtype=torch.int32,
        )

    use_qk_l2norm_in_kernel = True
    return (
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        use_qk_l2norm_in_kernel,
    )


def _measure_baseline(case: dict) -> float:
    if not torch.cuda.is_available():
        return 1.0

    benchmarker = Benchmarker(
        warmup_min_time=0.5,
        warmup_min_iters=5,
        benchmark_min_time=0.5,
        benchmark_min_iters=5,
    )
    inputs = _generate_inputs(case)
    result = benchmarker.measure(reference_fn, inputs)
    return result.mean_ms


def _geometric_mean(values) -> float:
    values = [float(v) for v in values]
    if not values:
        return 0.0
    return math.exp(sum(math.log(max(v, 1e-12)) for v in values) / len(values))
