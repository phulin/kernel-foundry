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


# ──────────────────────────────────────────── d_algo

class TestAlgoDimension:
    def test_single_elementwise_is_zero(self, clf):
        code = _kernel(
            "    pid = tl.program_id(0)\n"
            "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
            "    x = tl.load(x_ptr + offs)\n"
            "    tl.store(out_ptr + offs, x)"
        )
        assert clf.classify(code).d_algo == 0

    def test_tl_dot_alone_is_one(self, clf):
        code = """
import triton, triton.language as tl, torch
@triton.jit
def dotk(a_ptr, b_ptr, c_ptr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    c = tl.dot(a, b)
    tl.store(c_ptr, c)
def kernel_fn(a, b):
    return torch.zeros(1)
"""
        assert clf.classify(code).d_algo == 1

    def test_exp_plus_loads_is_fused(self, clf):
        # exp + load/store = 2 categories → fused (level 1)
        code = _kernel(
            "    pid = tl.program_id(0)\n"
            "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
            "    x = tl.load(x_ptr + offs)\n"
            "    y = tl.exp(x)\n"
            "    tl.store(out_ptr + offs, y)"
        )
        assert clf.classify(code).d_algo == 1

    def test_flash_attention_pattern_is_two(self, clf):
        code = """
import triton, triton.language as tl, torch
@triton.jit
def flash(q_ptr, k_ptr, v_ptr, out_ptr, N, BLOCK: tl.constexpr):
    m_i = tl.zeros([BLOCK], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK], dtype=tl.float32)
    acc = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
    for j in range(N):
        s = tl.load(k_ptr + j)
        m_ij = tl.maximum(m_i, s)
        p = tl.exp(s - m_ij)
        l_i = tl.exp(m_i - m_ij) * l_i + p
        m_i = m_ij
    tl.store(out_ptr, acc / l_i[:, None])
def kernel_fn(q, k, v):
    return torch.zeros(1)
"""
        assert clf.classify(code).d_algo == 2

    def test_running_max_variable_name_triggers_two(self, clf):
        # Names alone should not trigger reformulation.
        code = _kernel(
            "    m_i = tl.zeros([BLOCK], dtype=tl.float32)\n"
            "    l_i = tl.zeros([BLOCK], dtype=tl.float32)\n"
            "    x = tl.load(x_ptr)\n"
            "    tl.store(out_ptr, m_i)"
        )
        assert clf.classify(code).d_algo == 0

    def test_multi_kernel_pipeline_is_novel(self, clf):
        code = """
import triton, triton.language as tl, torch
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


# ──────────────────────────────────────────── d_sync

class TestSyncDimension:
    def test_no_sync_is_zero(self, clf):
        code = _kernel(
            "    pid = tl.program_id(0)\n"
            "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
            "    x = tl.load(x_ptr + offs)\n"
            "    tl.store(out_ptr + offs, x)"
        )
        assert clf.classify(code).d_sync == 0

    def test_debug_barrier_is_one(self, clf):
        code = _kernel(
            "    x = tl.load(x_ptr)\n"
            "    tl.debug_barrier()\n"
            "    tl.store(out_ptr, x)"
        )
        assert clf.classify(code).d_sync == 1

    def test_serial_accumulation_loop_is_one(self, clf):
        code = """
import triton, triton.language as tl, torch
@triton.jit
def seq_red(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    acc = 0.0
    for i in range(0, BLOCK):
        acc += tl.load(x_ptr + i)
    tl.store(out_ptr, acc)
def kernel_fn(x):
    return torch.zeros(1)
"""
        assert clf.classify(code).d_sync == 1

    def test_associative_scan_is_two(self, clf):
        code = """
import triton, triton.language as tl, torch
@triton.jit
def scan_k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.associative_scan(x, 0, lambda a, b: a + b)
    tl.store(out_ptr + offs, y)
def kernel_fn(x):
    return torch.zeros(1)
"""
        assert clf.classify(code).d_sync == 2

    def test_atomic_add_is_three(self, clf):
        code = """
import triton, triton.language as tl, torch
@triton.jit
def atomic_k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n)
    tl.atomic_add(out_ptr, tl.sum(x, axis=0))
def kernel_fn(x):
    return torch.zeros(1)
"""
        assert clf.classify(code).d_sync == 3

    def test_atomic_cas_is_three(self, clf):
        code = _kernel(
            "    tl.atomic_cas(x_ptr, 0, 1)"
        )
        assert clf.classify(code).d_sync == 3

    def test_tl_reduce_without_tiling_is_two(self, clf):
        code = """
import triton, triton.language as tl, torch
@triton.jit
def red(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    s = tl.reduce(x, 0, lambda a, b: a + b)
    tl.store(out_ptr, s)
def kernel_fn(x):
    return torch.zeros(1)
"""
        assert clf.classify(code).d_sync == 2

    def test_tl_reduce_inside_tiling_loop_not_double_counted(self, clf):
        # Has both tl.dot (tiling) and tl.reduce — d_sync should NOT be 2 from reduce alone
        code = """
import triton, triton.language as tl, torch
@triton.jit
def dotred(a_ptr, b_ptr, c_ptr, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        acc = tl.dot(a, b, acc)
    s = tl.reduce(acc, 0, lambda x, y: x + y)
    tl.store(c_ptr, s)
def kernel_fn(a, b):
    return torch.zeros(1)
"""
        c = clf.classify(code)
        # d_mem=2 (tiled), d_sync=0 (single reduce inside tiling loop, anti-double-count)
        assert c.d_mem == 2
        assert c.d_sync == 0  # single reduce → anti-double-count suppresses it

    def test_multiple_reduces_with_tiling_is_two(self, clf):
        # Two tl.reduce calls with tiling → suspicious, likely global → d_sync=2
        code = """
import triton, triton.language as tl, torch
@triton.jit
def dotred2(a_ptr, b_ptr, c_ptr, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        acc = tl.dot(a, b, acc)
    s1 = tl.reduce(acc, 0, lambda x, y: x + y)
    s2 = tl.reduce(acc, 1, lambda x, y: tl.maximum(x, y))
    tl.store(c_ptr, s1)
    tl.store(c_ptr + 1, s2)
def kernel_fn(a, b):
    return torch.zeros(1)
"""
        c = clf.classify(code)
        assert c.d_sync == 2

    def test_multi_kernel_launches_are_global_sync(self, clf):
        code = """
import triton, triton.language as tl, torch
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
