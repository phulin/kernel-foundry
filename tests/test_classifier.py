import pytest
from kernel_foundry.classifier.triton_classifier import TritonBehaviorClassifier
from kernel_foundry.types import BehavioralCoords


@pytest.fixture
def clf():
    return TritonBehaviorClassifier()


# ── helper to build minimal valid Triton source

def _kernel(body: str, extra_fns: str = "") -> str:
    return f"""
import triton
import triton.language as tl
import torch

@triton.jit
def the_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
{body}

{extra_fns}
def kernel_fn(x):
    return torch.zeros(1)
""".strip()


class TestFeatureExtraction:
    def test_extract_features_for_coalesced_kernel(self, clf):
        code = _kernel(
            "    pid = tl.program_id(0)\n"
            "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
            "    tl.multiple_of(offs, BLOCK)\n"
            "    x = tl.load(x_ptr + offs)\n"
            "    tl.store(out_ptr + offs, x)"
        )
        features = clf.extract_features(code)
        assert features.has_arange_indexing is True
        assert features.has_alignment_hint is True
        assert features.load_count == 1
        assert features.store_count == 1

    def test_extract_features_for_tiled_kernel(self, clf):
        code = """
import triton, triton.language as tl, torch
@triton.autotune(configs=[triton.Config({'BK': 32}, num_stages=3)], key=['K'])
@triton.jit
def matmul(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr + k)
        b = tl.load(b_ptr + k)
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr, acc)
def kernel_fn(a, b):
    return torch.zeros(1)
"""
        features = clf.extract_features(code)
        assert features.has_tiled_dot_loop is True
        assert features.has_pipeline is True
        assert features.has_multi_dim_accumulator is True


# ──────────────────────────────────────────── d_mem

class TestMemoryDimension:
    def test_no_loads_is_zero(self, clf):
        code = _kernel("    pass")
        assert clf.classify(code).d_mem == 0

    def test_basic_load_store_is_one(self, clf):
        code = _kernel(
            "    pid = tl.program_id(0)\n"
            "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
            "    x = tl.load(x_ptr + offs, mask=offs < n)\n"
            "    tl.store(out_ptr + offs, x, mask=offs < n)"
        )
        assert clf.classify(code).d_mem == 1

    def test_scalar_load_without_contiguous_signal_is_zero(self, clf):
        code = _kernel(
            "    x = tl.load(x_ptr)\n"
            "    tl.store(out_ptr, x)"
        )
        assert clf.classify(code).d_mem == 0

    def test_tiled_dot_loop_is_two(self, clf):
        code = """
import triton, triton.language as tl, torch
@triton.jit
def matmul(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr + k)
        b = tl.load(b_ptr + k)
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr, acc)
def kernel_fn(a, b):
    return torch.zeros(1)
"""
        assert clf.classify(code).d_mem == 2

    def test_pipeline_with_tiling_is_three(self, clf):
        code = """
import triton, triton.language as tl, torch
@triton.autotune(configs=[triton.Config({'BK': 32}, num_stages=3)], key=['K'])
@triton.jit
def pipelined(a_ptr, b_ptr, c_ptr, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr + k)
        b = tl.load(b_ptr + k)
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr, acc)
def kernel_fn(a, b):
    return torch.zeros(1)
"""
        assert clf.classify(code).d_mem == 3

    def test_block_ptr_counts_as_tiled_memory(self, clf):
        code = """
import triton, triton.language as tl, torch
@triton.jit
def blocked(a_ptr, out_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    block = tl.make_block_ptr(
        base=a_ptr, shape=(M, N), strides=(N, 1), offsets=(0, 0),
        block_shape=(BM, BN), order=(1, 0)
    )
    x = tl.load(block)
    tl.store(out_ptr, x)
def kernel_fn(a):
    return torch.zeros(1)
"""
        assert clf.classify(code).d_mem >= 2

    def test_static_range_tiling_counts_as_tiled_memory(self, clf):
        code = """
import triton, triton.language as tl, torch
@triton.jit
def blocked_loop(x_ptr, out_ptr, C, T, stride_xc, stride_xt, stride_oc, stride_ot, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_bc = tl.program_id(1)
    c0 = pid_bc * BLOCK_C
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T
    for i in tl.static_range(0, BLOCK_C):
        c = c0 + i
        if c < C:
            x = tl.load(x_ptr + c * stride_xc + offs_t * stride_xt, mask=mask_t, other=0.0)
            tl.store(out_ptr + c * stride_oc + offs_t * stride_ot, x, mask=mask_t)
def kernel_fn(x):
    return torch.zeros(1)
"""
        assert clf.classify(code).d_mem >= 2


# ──────────────────────────────────────────── d_algo (specialization)

class TestAlgoDimension:
    def test_minimal_kernel_is_zero(self, clf):
        code = _kernel(
            "    x = tl.load(x_ptr)\n"
            "    tl.store(out_ptr, x)"
        )
        assert clf.classify(code).d_algo == 0

    def test_tuned_single_axis_kernel_is_one(self, clf):
        code = """
import triton, triton.language as tl, torch
@triton.autotune(configs=[triton.Config({'BLOCK_T': 128}, num_warps=4, num_stages=2)], key=['T'])
@triton.jit
def tuned(x_ptr, out_ptr, T, BLOCK_T: tl.constexpr):
    offs = tl.arange(0, BLOCK_T)
    x = tl.load(x_ptr + offs, mask=offs < T, other=0.0)
    tl.store(out_ptr + offs, x, mask=offs < T)
def kernel_fn(x):
    return torch.zeros(1)
"""
        assert clf.classify(code).d_algo == 1

    def test_multi_axis_specialization_is_two(self, clf):
        code = """
import triton, triton.language as tl, torch
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 128, 'BLOCK_C': 1}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T': 256, 'BLOCK_C': 2}, num_warps=8, num_stages=2),
    ],
    key=['T', 'C'],
)
@triton.jit
def blocked(x_ptr, out_ptr, C, T, stride_xc, stride_xt, stride_oc, stride_ot, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_bc = tl.program_id(1)
    c0 = (pid_bc % tl.cdiv(C, BLOCK_C)) * BLOCK_C
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T
    for i in tl.static_range(0, BLOCK_C):
        c = c0 + i
        if c < C:
            x = tl.load(x_ptr + c * stride_xc + offs_t * stride_xt, mask=mask_t, other=0.0)
            tl.store(out_ptr + c * stride_oc + offs_t * stride_ot, x, mask=mask_t)
def kernel_fn(x):
    return torch.zeros(1)
"""
        assert clf.classify(code).d_algo == 2

    def test_helper_and_multi_kernel_pipeline_is_three(self, clf):
        code = """
import triton, triton.language as tl, torch
@triton.jit
def helper(x):
    return x
@triton.jit
def stage1(x_ptr, tmp_ptr, n, BLOCK: tl.constexpr):
    pass
@triton.jit
def stage2(tmp_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pass
def kernel_fn(x):
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)
    stage1[(1,)](x, tmp, x.numel(), BLOCK=128)
    stage2[(1,)](tmp, out, x.numel(), BLOCK=128)
    return out
"""
        assert clf.classify(code).d_algo == 3


# ──────────────────────────────────────────── d_sync (shape)

class TestSyncDimension:
    def test_scalar_kernel_is_zero(self, clf):
        code = _kernel(
            "    x = tl.load(x_ptr)\n"
            "    tl.store(out_ptr, x)"
        )
        assert clf.classify(code).d_sync == 0

    def test_single_blocked_axis_is_one(self, clf):
        code = _kernel(
            "    pid = tl.program_id(0)\n"
            "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
            "    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0)\n"
            "    tl.store(out_ptr + offs, x, mask=offs < n)"
        )
        assert clf.classify(code).d_sync == 1

    def test_two_dimensional_tile_is_two(self, clf):
        code = """
import triton, triton.language as tl, torch
@triton.jit
def tiled(x_ptr, out_ptr, C, T, stride_xc, stride_xt, stride_oc, stride_ot, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_bc = tl.program_id(1)
    c0 = pid_bc * BLOCK_C
    offs_c = c0 + tl.arange(0, BLOCK_C)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = (offs_c[:, None] < C) & (offs_t[None, :] < T)
    x = tl.load(x_ptr + offs_c[:, None] * stride_xc + offs_t[None, :] * stride_xt, mask=mask, other=0.0)
    tl.store(out_ptr + offs_c[:, None] * stride_oc + offs_t[None, :] * stride_ot, x, mask=mask)
def kernel_fn(x):
    return torch.zeros(1)
"""
        assert clf.classify(code).d_sync == 2

    def test_static_range_hierarchy_is_three(self, clf):
        code = """
import triton, triton.language as tl, torch
@triton.jit
def blocked_loop(x_ptr, out_ptr, C, T, stride_xc, stride_xt, stride_oc, stride_ot, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_bc = tl.program_id(1)
    c0 = pid_bc * BLOCK_C
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T
    for i in tl.static_range(0, BLOCK_C):
        c = c0 + i
        if c < C:
            x = tl.load(x_ptr + c * stride_xc + offs_t * stride_xt, mask=mask_t, other=0.0)
            tl.store(out_ptr + c * stride_oc + offs_t * stride_ot, x, mask=mask_t)
def kernel_fn(x):
    return torch.zeros(1)
"""
        assert clf.classify(code).d_sync == 3

    def test_three_program_axes_is_three(self, clf):
        code = """
import triton, triton.language as tl, torch
@triton.jit
def three_axis(x_ptr, out_ptr, B, C, T, stride_xb, stride_xc, stride_xt, stride_ob, stride_oc, stride_ot, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = (offs_c[:, None] < C) & (offs_t[None, :] < T)
    x = tl.load(x_ptr + pid_b * stride_xb + offs_c[:, None] * stride_xc + offs_t[None, :] * stride_xt, mask=mask, other=0.0)
    tl.store(out_ptr + pid_b * stride_ob + offs_c[:, None] * stride_oc + offs_t[None, :] * stride_ot, x, mask=mask)
def kernel_fn(x):
    return torch.zeros(1)
"""
        assert clf.classify(code).d_sync == 3


# ──────────────────────────────────────────── edge cases

class TestClassifierEdgeCases:
    def test_syntax_error_returns_zeros(self, clf):
        c = clf.classify("this is not valid python !!!{}")
        assert c == BehavioralCoords(0, 0, 0)

    def test_empty_string_returns_zeros(self, clf):
        c = clf.classify("")
        assert c == BehavioralCoords(0, 0, 0)

    def test_coords_bounded_to_three(self, clf):
        # Even a very complex kernel shouldn't exceed level 3
        c = clf.classify(_kernel("    tl.atomic_add(x_ptr, 1.0)"))
        assert 0 <= c.d_mem <= 3
        assert 0 <= c.d_algo <= 3
        assert 0 <= c.d_sync <= 3
