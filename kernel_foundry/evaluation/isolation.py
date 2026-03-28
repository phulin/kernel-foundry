from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import asdict
from typing import Any, Callable

import torch

from kernel_foundry.evaluation.benchmarker import Benchmarker
from kernel_foundry.evaluation.compiler import TritonCompiler
from kernel_foundry.evaluation.correctness import CorrectnessChecker, CorrectnessResult


def run_correctness_in_subprocess(
    source_code: str,
    kernel_id: str,
    reference_fn: Callable,
    input_generator: Callable[[], tuple],
    *,
    threshold: float,
    pass_fraction: float,
    num_trials: int,
    eps: float,
    timeout_s: float = 300.0,
) -> CorrectnessResult:
    payload = _run_in_subprocess(
        _correctness_worker,
        (
            source_code,
            kernel_id,
            reference_fn,
            input_generator,
            threshold,
            pass_fraction,
            num_trials,
            eps,
        ),
        timeout_s=timeout_s,
    )
    if payload["status"] != "ok":
        return CorrectnessResult(False, float("inf"), 0.0, payload["error_log"])
    return CorrectnessResult(**payload["result"])


def run_benchmark_in_subprocess(
    source_code: str,
    kernel_id: str,
    input_generator: Callable[[], tuple] | None,
    benchmark_cases,
    *,
    baseline_time_ms: float,
    warmup_min_time: float,
    warmup_min_iters: int,
    benchmark_min_time: float,
    benchmark_min_iters: int,
    inner_loop_min_time: float,
    timeout_s: float = 300.0,
) -> dict[str, float | str]:
    payload = _run_in_subprocess(
        _benchmark_worker,
        (
            source_code,
            kernel_id,
            input_generator,
            benchmark_cases,
            baseline_time_ms,
            warmup_min_time,
            warmup_min_iters,
            benchmark_min_time,
            benchmark_min_iters,
            inner_loop_min_time,
        ),
        timeout_s=timeout_s,
    )
    if payload["status"] != "ok":
        raise RuntimeError(payload["error_log"])
    return payload["result"]


def _run_in_subprocess(worker: Callable, args: tuple[Any, ...], *, timeout_s: float):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=worker, args=(queue, *args))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {
            "status": "error",
            "error_log": f"Subprocess timed out after {timeout_s:.1f}s",
        }
    if proc.exitcode not in (0, None) and queue.empty():
        return {
            "status": "error",
            "error_log": f"Subprocess exited with code {proc.exitcode}",
        }
    if queue.empty():
        return {
            "status": "error",
            "error_log": "Subprocess produced no result",
        }
    return queue.get()


def _correctness_worker(
    queue,
    source_code: str,
    kernel_id: str,
    reference_fn: Callable,
    input_generator: Callable[[], tuple],
    threshold: float,
    pass_fraction: float,
    num_trials: int,
    eps: float,
) -> None:
    try:
        compile_result = TritonCompiler().compile(source_code, kernel_id=kernel_id)
        if not compile_result.success or compile_result.module is None:
            queue.put(
                {
                    "status": "ok",
                    "result": asdict(
                        CorrectnessResult(
                            False,
                            float("inf"),
                            0.0,
                            compile_result.error_log or "Compilation failed in subprocess",
                        )
                    ),
                }
            )
            return

        checker = CorrectnessChecker(
            reference_fn=reference_fn,
            input_generator=input_generator,
            threshold=threshold,
            pass_fraction=pass_fraction,
            num_trials=num_trials,
            eps=eps,
        )
        result = checker.check(compile_result.module)
        queue.put({"status": "ok", "result": asdict(result)})
    except Exception:
        queue.put({"status": "error", "error_log": traceback.format_exc(limit=12)})


def _benchmark_worker(
    queue,
    source_code: str,
    kernel_id: str,
    input_generator: Callable[[], tuple] | None,
    benchmark_cases,
    baseline_time_ms: float,
    warmup_min_time: float,
    warmup_min_iters: int,
    benchmark_min_time: float,
    benchmark_min_iters: int,
    inner_loop_min_time: float,
) -> None:
    try:
        compile_result = TritonCompiler().compile(source_code, kernel_id=kernel_id)
        if not compile_result.success or compile_result.module is None:
            raise RuntimeError(compile_result.error_log or "Compilation failed in subprocess")
        kernel_fn = getattr(compile_result.module, "kernel_fn")
        benchmarker = Benchmarker(
            warmup_min_time=warmup_min_time,
            warmup_min_iters=warmup_min_iters,
            benchmark_min_time=benchmark_min_time,
            benchmark_min_iters=benchmark_min_iters,
            inner_loop_min_time=inner_loop_min_time,
        )
        result = _benchmark_kernel(
            benchmarker,
            kernel_fn,
            input_generator,
            benchmark_cases,
            baseline_time_ms,
        )
        queue.put({"status": "ok", "result": result})
    except Exception:
        queue.put({"status": "error", "error_log": traceback.format_exc(limit=12)})


def _benchmark_kernel(
    benchmarker: Benchmarker,
    kernel_fn,
    input_generator: Callable[[], tuple] | None,
    benchmark_cases,
    baseline_time_ms: float,
) -> dict[str, float | str]:
    cases = benchmark_cases
    if not cases:
        if input_generator is None:
            raise ValueError("Input generator is required when benchmark_cases are absent")
        inputs = input_generator()
        bench = benchmarker.measure(kernel_fn, inputs)
        input_bytes = sum(x.nbytes for x in inputs if isinstance(x, torch.Tensor))
        eff_bw_gbs = input_bytes / (bench.mean_ms / 1000) / 1e9
        return {
            "kernel_time_ms": bench.mean_ms,
            "baseline_time_ms": baseline_time_ms,
            "speedup": baseline_time_ms / bench.mean_ms if bench.mean_ms > 0 else 0.0,
            "profiling_summary": (
                f"mean={bench.mean_ms:.3f}ms std={bench.std_ms:.3f}ms "
                f"(n={bench.num_iters}) | "
                f"input={input_bytes/1e6:.2f}MB | "
                f"eff_bw={eff_bw_gbs:.1f}GB/s"
            ),
        }

    case_times: list[float] = []
    case_baselines: list[float] = []
    summary_parts: list[str] = []
    for case in cases:
        inputs = case.input_generator()
        bench = benchmarker.measure(kernel_fn, inputs)
        case_times.append(bench.mean_ms)
        case_baselines.append(case.baseline_time_ms)
        input_bytes = sum(x.nbytes for x in inputs if isinstance(x, torch.Tensor))
        eff_bw_gbs = input_bytes / (bench.mean_ms / 1000) / 1e9
        speedup = case.baseline_time_ms / bench.mean_ms if bench.mean_ms > 0 else 0.0
        summary_parts.append(
            f"{case.name}: mean={bench.mean_ms:.3f}ms std={bench.std_ms:.3f}ms "
            f"(n={bench.num_iters}, speedup={speedup:.2f}x, input={input_bytes/1e6:.2f}MB, "
            f"eff_bw={eff_bw_gbs:.1f}GB/s)"
        )

    kernel_time_ms = _geometric_mean(case_times)
    baseline_time_ms = _geometric_mean(case_baselines)
    speedup = baseline_time_ms / kernel_time_ms if kernel_time_ms > 0 else 0.0
    return {
        "kernel_time_ms": kernel_time_ms,
        "baseline_time_ms": baseline_time_ms,
        "speedup": speedup,
        "profiling_summary": (
            f"agg: mean={kernel_time_ms:.3f}ms speedup={speedup:.2f}x | "
            + " ; ".join(summary_parts)
        ),
    }


def _geometric_mean(values: list[float]) -> float:
    safe_values = [max(float(v), 1e-12) for v in values]
    import math

    return math.exp(sum(math.log(v) for v in safe_values) / len(safe_values))
