from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from kernel_foundry.evaluation.benchmarker import Benchmarker
from kernel_foundry.evaluation.compiler import TritonCompiler
from kernel_foundry.evaluation.correctness import CorrectnessChecker
from kernel_foundry.evaluation.fitness import compute_fitness
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
        configs = self._extract_autotune_configs(source_code)
        if not configs:
            return []

        results: list[TemplateResult] = []
        for idx, config in enumerate(configs):
            configured_source = self._rewrite_with_single_config(source_code, config.raw_config)
            result = self._evaluate_single_config(
                configured_source,
                kernel_id=f"{kernel_id_prefix}_{idx}",
            )
            if result is None:
                continue
            results.append(
                TemplateResult(
                    config={
                        "kwargs": config.kwargs,
                        "num_warps": config.num_warps,
                        "num_stages": config.num_stages,
                        "num_ctas": config.num_ctas,
                        "maxnreg": config.maxnreg,
                        "pre_hook": config.pre_hook,
                    },
                    eval_result=result,
                )
            )
        return results

    def _count_configs(self, source_code: str) -> int:
        """Count triton.Config entries in an @triton.autotune decorator."""
        return len(re.findall(r"triton\.Config\s*\(", source_code))

    def _evaluate_single_config(self, source_code: str, kernel_id: str) -> EvalResult | None:
        compile_result = self._compiler.compile(source_code, kernel_id=kernel_id)
        if not compile_result.success or compile_result.module is None:
            return None

        correctness = self._checker.check(compile_result.module)
        if not correctness.correct:
            return None

        kernel_fn = getattr(compile_result.module, "kernel_fn")
        inputs = self._checker._input_generator()
        bench = self._benchmarker.measure(kernel_fn, inputs)
        speedup = self._baseline_time_ms / bench.mean_ms if bench.mean_ms > 0 else 0.0

        result = EvalResult(
            kernel_id=kernel_id,
            compiled=True,
            correct=True,
            kernel_time_ms=bench.mean_ms,
            baseline_time_ms=self._baseline_time_ms,
            speedup=speedup,
        )
        result.fitness = compute_fitness(result, self._target_speedup)
        return result

    @dataclass
    class ParsedConfig:
        raw_config: str
        kwargs: dict
        num_warps: int | None
        num_stages: int | None
        num_ctas: int | None
        maxnreg: int | None
        pre_hook: str | None

    def _extract_autotune_configs(self, source_code: str) -> list[ParsedConfig]:
        marker = "configs=["
        start = source_code.find(marker)
        if start == -1:
            return []
        start += len(marker)
        end = self._find_matching_bracket(source_code, start - 1, "[", "]")
        if end == -1:
            return []

        configs_block = source_code[start:end]
        raw_configs = self._split_top_level_configs(configs_block)
        parsed: list[TemplateOptimizer.ParsedConfig] = []
        for raw in raw_configs:
            raw = raw.strip()
            if not raw:
                continue
            parsed.append(
                self.ParsedConfig(
                    raw_config=raw,
                    kwargs=self._parse_config_kwargs(raw),
                    num_warps=self._parse_kwarg_int(raw, "num_warps"),
                    num_stages=self._parse_kwarg_int(raw, "num_stages"),
                    num_ctas=self._parse_kwarg_int(raw, "num_ctas"),
                    maxnreg=self._parse_kwarg_int(raw, "maxnreg"),
                    pre_hook=self._parse_kwarg_name(raw, "pre_hook"),
                )
            )
        return parsed

    def _rewrite_with_single_config(self, source_code: str, config: str) -> str:
        marker = "configs=["
        start = source_code.find(marker)
        if start == -1:
            return source_code
        content_start = start + len(marker)
        content_end = self._find_matching_bracket(source_code, content_start - 1, "[", "]")
        if content_end == -1:
            return source_code
        return f"{source_code[:content_start]}{config}{source_code[content_end:]}"

    def _find_matching_bracket(
        self,
        text: str,
        start_idx: int,
        open_char: str,
        close_char: str,
    ) -> int:
        depth = 0
        for idx in range(start_idx, len(text)):
            ch = text[idx]
            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return idx
        return -1

    def _split_top_level_configs(self, block: str) -> list[str]:
        configs: list[str] = []
        start = 0
        depth_paren = 0
        depth_brace = 0
        depth_bracket = 0
        for idx, ch in enumerate(block):
            if ch == "(":
                depth_paren += 1
            elif ch == ")":
                depth_paren -= 1
            elif ch == "{":
                depth_brace += 1
            elif ch == "}":
                depth_brace -= 1
            elif ch == "[":
                depth_bracket += 1
            elif ch == "]":
                depth_bracket -= 1
            elif ch == "," and depth_paren == depth_brace == depth_bracket == 0:
                configs.append(block[start:idx])
                start = idx + 1
        tail = block[start:]
        if tail.strip():
            configs.append(tail)
        return configs

    def _parse_config_kwargs(self, raw_config: str) -> dict:
        match = re.search(r"triton\.Config\s*\(\s*(\{.*?\})", raw_config, re.DOTALL)
        if not match:
            return {}
        try:
            return ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            return {}

    def _parse_kwarg_int(self, raw_config: str, key: str) -> int | None:
        match = re.search(rf"{key}\s*=\s*([0-9]+)", raw_config)
        return int(match.group(1)) if match else None

    def _parse_kwarg_name(self, raw_config: str, key: str) -> str | None:
        match = re.search(rf"{key}\s*=\s*([A-Za-z_][A-Za-z0-9_\.]*)", raw_config)
        return match.group(1) if match else None
