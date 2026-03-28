from __future__ import annotations

import traceback
import uuid

from kernel_foundry.config import EvolutionConfig
from kernel_foundry.evaluation.benchmarker import Benchmarker
from kernel_foundry.evaluation.compiler import TritonCompiler
from kernel_foundry.evaluation.correctness import CorrectnessChecker
from kernel_foundry.evaluation.fitness import compute_fitness
from kernel_foundry.evaluation.isolation import (
    run_benchmark_in_subprocess,
    run_correctness_in_subprocess,
)
from kernel_foundry.evaluation.profiler import find_ncu
from kernel_foundry.evaluation.template_optimizer import TemplateOptimizer
from kernel_foundry.task.spec import TaskSpec
from kernel_foundry.types import EvalResult, KernelRecord


class CandidateEvaluator:
    """
    Shared compile/correctness/benchmark/classify pipeline used by both the
    legacy evolutionary loop and the world-model search path.
    """

    def __init__(
        self,
        task: TaskSpec,
        config: EvolutionConfig,
        compiler: TritonCompiler | None = None,
        benchmarker: Benchmarker | None = None,
        checker: CorrectnessChecker | None = None,
        template_optimizer: TemplateOptimizer | None = None,
    ) -> None:
        self.task = task
        self.config = config
        self.compiler = compiler or TritonCompiler()
        self.benchmarker = benchmarker or Benchmarker(
            warmup_min_time=config.warmup_min_time,
            warmup_min_iters=config.warmup_min_iters,
            benchmark_min_time=config.benchmark_min_time,
            benchmark_min_iters=config.benchmark_min_iters,
            inner_loop_min_time=config.inner_loop_min_time,
        )
        self.checker = checker or CorrectnessChecker(
            reference_fn=task.reference_fn,
            input_generator=task.input_generator,
        )
        self.template_optimizer = template_optimizer or TemplateOptimizer(
            self.compiler,
            self.checker,
            self.benchmarker,
            baseline_time_ms=task.baseline_time_ms,
            target_speedup=config.target_speedup,
            input_generator=task.input_generator,
            benchmark_cases=task.benchmark_cases,
        )
        self._ncu_path = find_ncu()
        if self._ncu_path:
            print(f"NCU found: {self._ncu_path}")

    def evaluate_candidate(
        self, source_code: str, generation: int, parent_id: str | None
    ) -> KernelRecord:
        kernel_id = str(uuid.uuid4())[:8]
        result = EvalResult(kernel_id=kernel_id, compiled=False, correct=False)

        compile_result = self.compiler.compile(source_code, kernel_id=kernel_id)
        result.compiled = compile_result.success
        if not compile_result.success:
            result.error_log = compile_result.error_log
            result.fitness = compute_fitness(result, self.config.target_speedup)
            return self._make_record(kernel_id, generation, parent_id, source_code, result)

        correctness = run_correctness_in_subprocess(
            source_code,
            kernel_id=kernel_id,
            reference_fn=self.task.reference_fn,
            input_generator=self.task.input_generator,
            threshold=self.checker._threshold,
            pass_fraction=self.checker._pass_fraction,
            num_trials=self.checker._num_trials,
            eps=self.checker._eps,
        )
        result.correct = correctness.correct
        if not correctness.correct:
            result.error_log = correctness.error_log
            result.fitness = compute_fitness(result, self.config.target_speedup)
            return self._make_record(kernel_id, generation, parent_id, source_code, result)

        try:
            benchmark = self.benchmark_candidate(source_code, kernel_id)
            result.kernel_time_ms = benchmark["kernel_time_ms"]
            result.baseline_time_ms = benchmark["baseline_time_ms"]
            result.speedup = benchmark["speedup"]
            result.profiling_summary = benchmark["profiling_summary"]
            result.ncu_output = benchmark.get("ncu_output")
        except Exception:
            result.error_log = traceback.format_exc(limit=6)
            result.fitness = compute_fitness(result, self.config.target_speedup)
            return self._make_record(kernel_id, generation, parent_id, source_code, result)

        result.fitness = compute_fitness(result, self.config.target_speedup)

        is_templated = False
        template_configs: list[dict] | None = None
        if self.template_optimizer.is_templated(source_code):
            is_templated = True
            sweep_results = self.template_optimizer.sweep(
                source_code, kernel_id_prefix=kernel_id
            )
            if sweep_results:
                template_configs = [
                    {
                        "config": r.config,
                        "speedup": r.eval_result.speedup,
                        "correct": r.eval_result.correct,
                        "kernel_time_ms": r.eval_result.kernel_time_ms,
                    }
                    for r in sweep_results
                ]
                best_sweep = max(sweep_results, key=lambda r: r.eval_result.fitness)
                if best_sweep.eval_result.fitness > result.fitness:
                    result.kernel_time_ms = best_sweep.eval_result.kernel_time_ms
                    result.speedup = best_sweep.eval_result.speedup
                    result.fitness = best_sweep.eval_result.fitness

        return self._make_record(
            kernel_id,
            generation,
            parent_id,
            source_code,
            result,
            is_templated=is_templated,
            template_configs=template_configs,
        )

    def benchmark_candidate(self, source_code: str, kernel_id: str) -> dict[str, float | str]:
        return run_benchmark_in_subprocess(
            source_code,
            kernel_id=kernel_id,
            input_generator=self.task.input_generator,
            benchmark_cases=self.task.benchmark_cases,
            baseline_time_ms=self.task.baseline_time_ms,
            warmup_min_time=self.config.warmup_min_time,
            warmup_min_iters=self.config.warmup_min_iters,
            benchmark_min_time=self.config.benchmark_min_time,
            benchmark_min_iters=self.config.benchmark_min_iters,
            inner_loop_min_time=self.config.inner_loop_min_time,
            ncu_path=self._ncu_path,
        )

    @staticmethod
    def _make_record(
        kernel_id: str,
        generation: int,
        parent_id: str | None,
        source_code: str,
        result: EvalResult,
        is_templated: bool = False,
        template_configs: list[dict] | None = None,
    ) -> KernelRecord:
        return KernelRecord(
            kernel_id=kernel_id,
            generation=generation,
            parent_id=parent_id,
            source_code=source_code,
            eval_result=result,
            is_templated=is_templated,
            template_configs=template_configs,
        )
