import pytest
from kernel_foundry.types import EvalResult
from kernel_foundry.evaluation.fitness import compute_fitness


def _result(compiled, correct, speedup=None):
    return EvalResult(kernel_id="x", compiled=compiled, correct=correct, speedup=speedup, fitness=0.0)


def test_compile_failure_is_zero():
    assert compute_fitness(_result(compiled=False, correct=False)) == 0.0


def test_incorrect_is_point_one():
    assert compute_fitness(_result(compiled=True, correct=False)) == 0.1


def test_correct_at_target_is_one():
    assert compute_fitness(_result(compiled=True, correct=True, speedup=2.0), target_speedup=2.0) == 1.0


def test_correct_at_half_target():
    f = compute_fitness(_result(compiled=True, correct=True, speedup=1.0), target_speedup=2.0)
    assert abs(f - 0.75) < 1e-9


def test_speedup_beyond_target_is_capped():
    f = compute_fitness(_result(compiled=True, correct=True, speedup=10.0), target_speedup=2.0)
    assert f == 1.0


def test_zero_speedup():
    f = compute_fitness(_result(compiled=True, correct=True, speedup=0.0), target_speedup=2.0)
    assert f == 0.5


def test_none_speedup_treated_as_zero():
    f = compute_fitness(_result(compiled=True, correct=True, speedup=None), target_speedup=2.0)
    assert f == 0.5
