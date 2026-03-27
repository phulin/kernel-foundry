from kernel_foundry.evaluation.template_optimizer import TemplateOptimizer


def _make_optimizer():
    return TemplateOptimizer(
        compiler=None,  # type: ignore[arg-type]
        checker=None,  # type: ignore[arg-type]
        benchmarker=None,  # type: ignore[arg-type]
        baseline_time_ms=1.0,
    )


AUTOTUNE_SOURCE = """\
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256, 'GROUP_SIZE': 8}, num_warps=8, num_stages=3),
    ],
    key=['n'],
)
@triton.jit
def my_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr = 1):
    pass
"""


class TestTemplateOptimizerParsing:
    def test_extract_autotune_configs(self):
        optimizer = _make_optimizer()
        configs = optimizer._extract_autotune_configs(AUTOTUNE_SOURCE)
        assert len(configs) == 2
        assert configs[0].kwargs == {"BLOCK_SIZE": 128}
        assert configs[0].num_warps == 4
        assert configs[0].num_stages == 2
        assert configs[1].kwargs == {"BLOCK_SIZE": 256, "GROUP_SIZE": 8}
        assert configs[1].num_warps == 8
        assert configs[1].num_stages == 3

    def test_rewrite_with_single_config(self):
        optimizer = _make_optimizer()
        configs = optimizer._extract_autotune_configs(AUTOTUNE_SOURCE)
        rewritten = optimizer._rewrite_with_single_config(AUTOTUNE_SOURCE, configs[1].raw_config)
        assert rewritten.count("triton.Config(") == 1
        assert "'GROUP_SIZE': 8" in rewritten
        assert "'BLOCK_SIZE': 128" not in rewritten
