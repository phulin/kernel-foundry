from types import SimpleNamespace

from kernel_foundry.evaluation.benchmarker import BenchmarkResult
from kernel_foundry.evaluation.template_optimizer import TemplateOptimizer
from kernel_foundry.task.spec import BenchmarkCase


def _make_optimizer():
    return TemplateOptimizer(
        compiler=None,  # type: ignore[arg-type]
        checker=None,  # type: ignore[arg-type]
        benchmarker=None,  # type: ignore[arg-type]
        baseline_time_ms=1.0,
        input_generator=lambda: tuple(),
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


class _FakeBenchmarker:
    def __init__(self, timings_by_case):
        self.timings_by_case = timings_by_case
        self.seen = []

    def measure(self, fn, inputs):
        case_name = inputs[0]
        self.seen.append(case_name)
        mean_ms = self.timings_by_case[case_name]
        return BenchmarkResult(
            mean_ms=mean_ms,
            std_ms=0.0,
            min_ms=mean_ms,
            median_ms=mean_ms,
            num_iters=10,
        )


class TestTemplateOptimizerBenchmarking:
    def test_benchmark_kernel_uses_all_benchmark_cases_and_geometric_aggregation(self):
        benchmarker = _FakeBenchmarker({"medium": 0.5, "large": 2.0})
        optimizer = TemplateOptimizer(
            compiler=None,  # type: ignore[arg-type]
            checker=None,  # type: ignore[arg-type]
            benchmarker=benchmarker,  # type: ignore[arg-type]
            baseline_time_ms=999.0,
            benchmark_cases=[
                BenchmarkCase(
                    name="medium",
                    input_generator=lambda: ("medium",),
                    baseline_time_ms=2.0,
                ),
                BenchmarkCase(
                    name="large",
                    input_generator=lambda: ("large",),
                    baseline_time_ms=8.0,
                ),
            ],
        )

        result = optimizer._benchmark_kernel(lambda *args: None)

        assert benchmarker.seen == ["medium", "large"]
        assert result["kernel_time_ms"] == 1.0
        assert result["baseline_time_ms"] == 4.0
        assert result["speedup"] == 4.0
        assert "agg: mean=1.000ms speedup=4.00x" in result["profiling_summary"]

    def test_evaluate_single_config_uses_aggregated_benchmark_result(self):
        benchmarker = _FakeBenchmarker({"medium": 0.5, "large": 2.0})
        compiler = SimpleNamespace(
            compile=lambda source_code, kernel_id: SimpleNamespace(
                success=True,
                module=SimpleNamespace(kernel_fn=lambda *args: None),
            )
        )
        checker = SimpleNamespace(
            check=lambda module: SimpleNamespace(correct=True),
        )
        optimizer = TemplateOptimizer(
            compiler=compiler,  # type: ignore[arg-type]
            checker=checker,  # type: ignore[arg-type]
            benchmarker=benchmarker,  # type: ignore[arg-type]
            baseline_time_ms=999.0,
            target_speedup=10.0,
            benchmark_cases=[
                BenchmarkCase(
                    name="medium",
                    input_generator=lambda: ("medium",),
                    baseline_time_ms=2.0,
                ),
                BenchmarkCase(
                    name="large",
                    input_generator=lambda: ("large",),
                    baseline_time_ms=8.0,
                ),
            ],
        )

        result = optimizer._evaluate_single_config("source", kernel_id="tpl_0")

        assert result is not None
        assert benchmarker.seen == ["medium", "large"]
        assert result.kernel_time_ms == 1.0
        assert result.baseline_time_ms == 4.0
        assert result.speedup == 4.0
        assert result.profiling_summary is not None
        assert "medium:" in result.profiling_summary
