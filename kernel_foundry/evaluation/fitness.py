from __future__ import annotations

from kernel_foundry.types import EvalResult


def compute_fitness(result: EvalResult, target_speedup: float = 2.0) -> float:
    """
    f(k) = 0                         if compilation failed
    f(k) = 0.1                       if compiled but numerically incorrect
    f(k) = 0.5 + 0.5 * s_norm        if correct
    where s_norm = min(1, speedup / target_speedup)

    This ensures correctness is a prerequisite for high fitness while providing
    a continuous gradient for performance optimization.
    """
    if not result.compiled:
        return 0.0
    if not result.correct:
        return 0.1
    speedup = result.speedup or 0.0
    s_norm = min(1.0, speedup / target_speedup)
    return 0.5 + 0.5 * s_norm
