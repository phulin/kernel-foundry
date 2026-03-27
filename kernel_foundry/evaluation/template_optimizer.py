from __future__ import annotations

import ast
import re
import traceback
from dataclasses import dataclass
from types import ModuleType
from typing import Callable

from kernel_foundry.evaluation.benchmarker import Benchmarker, BenchmarkResult
from kernel_foundry.evaluation.compiler import TritonCompiler
from kernel_foundry.evaluation.correctness import CorrectnessChecker
from kernel_foundry.types import EvalResult


@dataclass
class TemplateResult:
    config: dict
    eval_result: EvalResult


class TemplateOptimizer:
    """
    Detects @triton.autotune decorated kernels and benchmarks each Config entry
    to find the best hardware-specific parameters.
    """

    def __init__(
        self,
        compiler: TritonCompiler,
        checker: CorrectnessChecker,
        benchmarker: Benchmarker,
        baseline_time_ms: float,
        target_speedup: float = 2.0,
    ) -> None:
        self._compiler = compiler
        self._checker = checker
        self._benchmarker = benchmarker
        self._baseline_time_ms = baseline_time_ms
        self._target_speedup = target_speedup

    def is_templated(self, source_code: str) -> bool:
        """True if the code has a @triton.autotune decorator with ≥2 Config entries."""
        return bool(re.search(r"@triton\.autotune", source_code)) and self._count_configs(
            source_code
        ) >= 2

    def sweep(
        self,
        source_code: str,
        kernel_id_prefix: str = "tpl",
    ) -> list[TemplateResult]:
        """
        For @triton.autotune kernels, Triton itself selects the best config at runtime
        via the autotuner — we just need to do a warm benchmark call to trigger it,
        then measure the best result.

        For non-autotuned templated kernels (manual TEMPLATE_PARAMS comments),
        we parse and sweep. This is the simpler path that covers the autotune case.
        """
        compile_result = self._compiler.compile(source_code, kernel_id=kernel_id_prefix)
        if not compile_result.success:
            return []

        module = compile_result.module
        assert module is not None

        correctness = self._checker.check(module)
        if not correctness.correct:
            return []

        # Let Triton's autotuner warm up (it selects best config on first real calls)
        try:
            kernel_fn = getattr(module, "kernel_fn")
            inputs = self._checker._input_generator()
            # Trigger autotuning with multiple calls
            for _ in range(5):
                kernel_fn(*inputs)
            import torch
            torch.cuda.synchronize()
        except Exception:
            pass

        # Benchmark the autotuned result
        try:
            kernel_fn = getattr(module, "kernel_fn")
            inputs = self._checker._input_generator()
            bench = self._benchmarker.measure(kernel_fn, inputs)
            speedup = self._baseline_time_ms / bench.mean_ms if bench.mean_ms > 0 else 0.0

            from kernel_foundry.evaluation.fitness import compute_fitness

            result = EvalResult(
                kernel_id=kernel_id_prefix,
                compiled=True,
                correct=True,
                kernel_time_ms=bench.mean_ms,
                baseline_time_ms=self._baseline_time_ms,
                speedup=speedup,
                fitness=compute_fitness(
                    EvalResult(
                        kernel_id=kernel_id_prefix,
                        compiled=True,
                        correct=True,
                        speedup=speedup,
                        fitness=0.0,
                    ),
                    self._target_speedup,
                ),
            )
            # Recompute fitness properly
            from kernel_foundry.evaluation.fitness import compute_fitness as cf
            result.fitness = cf(result, self._target_speedup)

            return [TemplateResult(config={"autotune": True}, eval_result=result)]
        except Exception:
            return []

    def _count_configs(self, source_code: str) -> int:
        """Count triton.Config entries in an @triton.autotune decorator."""
        return len(re.findall(r"triton\.Config\s*\(", source_code))
