from __future__ import annotations

from kernel_foundry.types import EvalResult


def compute_fitness(result: EvalResult, target_speedup: float = 2.0) -> float:
    """
    J(x) = s * (p_ref / p) * 100

    where s ∈ {0,1} is correctness and p_ref/p is the speedup ratio.
    Matches the K-Search paper's objective function.
    """
    if not result.compiled:
        return 0.0
    if not result.correct:
        return 0.0
    speedup = result.speedup or 0.0
    return speedup * 100.0
