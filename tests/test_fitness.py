import pytest
from kernel_foundry.types import EvalResult
from kernel_foundry.evaluation.fitness import compute_fitness


def _result(compiled, correct, speedup=None):
    return EvalResult(kernel_id="x", compiled=compiled, correct=correct, speedup=speedup, fitness=0.0)


def test_compile_failure_is_zero():
    assert compute_fitness(_result(compiled=False, correct=False)) == 0.0


def test_incorrect_is_zero():
    assert compute_fitness(_result(compiled=True, correct=False)) == 0.0


def test_correct_speedup_scaled():
    # J(x) = s * speedup * 100 = 1 * 2.0 * 100 = 200
    assert compute_fitness(_result(compiled=True, correct=True, speedup=2.0)) == 200.0


def test_correct_half_speedup():
    assert compute_fitness(_result(compiled=True, correct=True, speedup=0.5)) == 50.0


def test_correct_high_speedup():
    # No cap — linear scaling
    assert compute_fitness(_result(compiled=True, correct=True, speedup=10.0)) == 1000.0


def test_zero_speedup():
    assert compute_fitness(_result(compiled=True, correct=True, speedup=0.0)) == 0.0


def test_none_speedup_treated_as_zero():
    assert compute_fitness(_result(compiled=True, correct=True, speedup=None)) == 0.0
