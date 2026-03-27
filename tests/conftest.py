import sys
import os
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernel_foundry.types import BehavioralCoords, EvalResult, KernelRecord, TransitionRecord
from kernel_foundry.evaluation.fitness import compute_fitness


def make_record(d_mem, d_algo, d_sync, speedup=1.0, correct=True, compiled=True, kid=None):
    coords = BehavioralCoords(d_mem, d_algo, d_sync)
    fitness = compute_fitness(
        EvalResult(kernel_id="x", compiled=compiled, correct=correct, speedup=speedup, fitness=0.0)
    )
    result = EvalResult(
        kernel_id=kid or f"{d_mem}{d_algo}{d_sync}",
        compiled=compiled,
        correct=correct,
        speedup=speedup if correct else None,
        fitness=fitness,
    )
    return KernelRecord(
        kernel_id=kid or f"{d_mem}{d_algo}{d_sync}",
        generation=0,
        parent_id=None,
        source_code="# dummy",
        coords=coords,
        eval_result=result,
    )


def make_transition(parent, child, delta_f, outcome, generation=0):
    return TransitionRecord(
        parent_coords=BehavioralCoords(*parent),
        child_coords=BehavioralCoords(*child),
        delta_fitness=delta_f,
        outcome=outcome,
        generation=generation,
        timestamp=time.time(),
    )


@pytest.fixture
def rng():
    return np.random.default_rng(42)
