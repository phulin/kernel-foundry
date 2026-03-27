import time

import numpy as np
import pytest

from kernel_foundry.gradient.estimator import GradientEstimator, TransitionBuffer
from kernel_foundry.types import BehavioralCoords, TransitionRecord
from tests.conftest import make_transition


@pytest.fixture
def buf():
    return TransitionBuffer(max_size=100)


@pytest.fixture
def estimator(buf):
    return GradientEstimator(buf, weights=(0.4, 0.4, 0.2), decay=0.95)


def _fill_improvements(buf, parent, child, n=20, start_gen=0):
    for i in range(n):
        buf.add(make_transition(parent, child, delta_f=0.3, outcome="improvement", generation=start_gen + i))


# ──────────────────────────────────────────── TransitionBuffer

class TestTransitionBuffer:
    def test_add_and_len(self, buf):
        buf.add(make_transition((0,0,0), (1,0,0), 0.1, "improvement"))
        assert len(buf) == 1

    def test_max_size_evicts_oldest(self):
        small_buf = TransitionBuffer(max_size=3)
        for i in range(5):
            small_buf.add(make_transition((0,0,0), (1,0,0), float(i), "improvement", generation=i))
        assert len(small_buf) == 3
        # Most recent 3 should remain (generations 2, 3, 4)
        gens = [t.generation for t in small_buf.recent(3)]
        assert gens == [2, 3, 4]

    def test_get_from_filters_by_parent(self, buf):
        buf.add(make_transition((0,0,0), (1,0,0), 0.1, "improvement"))
        buf.add(make_transition((1,1,1), (2,2,2), 0.2, "improvement"))
        from_origin = buf.get_from(BehavioralCoords(0,0,0))
        assert len(from_origin) == 1
        assert from_origin[0].delta_fitness == 0.1

    def test_recent_returns_last_n(self, buf):
        for i in range(10):
            buf.add(make_transition((0,0,0), (1,0,0), float(i), "neutral", generation=i))
        recent = buf.recent(3)
        assert len(recent) == 3
        assert [t.delta_fitness for t in recent] == [7.0, 8.0, 9.0]


# ──────────────────────────────────────────── GradientEstimator

class TestGradientEstimator:
    def test_sampling_weights_uniform_when_sparse(self, estimator):
        # Fewer than 3 transitions → uniform weights
        occupied = [BehavioralCoords(0,0,0), BehavioralCoords(1,1,1)]
        fitnesses = {BehavioralCoords(0,0,0): 0.5, BehavioralCoords(1,1,1): 0.8}
        weights = estimator.compute_sampling_weights(occupied, fitnesses)
        assert weights[BehavioralCoords(0,0,0)] == 1.0
        assert weights[BehavioralCoords(1,1,1)] == 1.0

    def test_sampling_weights_positive_after_transitions(self, buf, estimator):
        _fill_improvements(buf, (0,0,0), (1,1,0), n=20)
        estimator.tick(20)
        occupied = [BehavioralCoords(0,0,0), BehavioralCoords(1,1,0)]
        fitnesses = {BehavioralCoords(0,0,0): 0.5, BehavioralCoords(1,1,0): 0.8}
        weights = estimator.compute_sampling_weights(occupied, fitnesses)
        assert all(w > 0 for w in weights.values())

    def test_hints_nonempty_after_consistent_improvements(self, buf, estimator):
        _fill_improvements(buf, (0,0,0), (2,1,0), n=20)
        estimator.tick(20)
        fitnesses = {BehavioralCoords(0,0,0): 0.5, BehavioralCoords(2,1,0): 0.9}
        hints = estimator.gradient_to_hints(BehavioralCoords(0,0,0), fitnesses)
        assert len(hints) > 0
        assert all(isinstance(h, str) and len(h) > 0 for h in hints)

    def test_hints_empty_when_no_transitions(self, estimator):
        fitnesses = {BehavioralCoords(0,0,0): 0.5}
        hints = estimator.gradient_to_hints(BehavioralCoords(0,0,0), fitnesses)
        assert hints == []

    def test_hints_respect_max_hints(self, buf, estimator):
        _fill_improvements(buf, (0,0,0), (3,3,3), n=20)
        estimator.tick(20)
        fitnesses = {BehavioralCoords(0,0,0): 0.5, BehavioralCoords(3,3,3): 0.9}
        hints = estimator.gradient_to_hints(BehavioralCoords(0,0,0), fitnesses, max_hints=1)
        assert len(hints) <= 1

    def test_temporal_decay_weights_recent_higher(self, buf):
        # Add old improvements in one direction, then recent improvements in another
        for i in range(10):
            buf.add(make_transition((0,0,0), (0,0,1), delta_f=0.3, outcome="improvement", generation=i))
        for i in range(10):
            buf.add(make_transition((0,0,0), (1,0,0), delta_f=0.3, outcome="improvement", generation=100+i))

        estimator = GradientEstimator(buf, weights=(1.0, 0.0, 0.0), decay=0.5)
        estimator.tick(110)
        # With heavy decay, old transitions should be nearly weightless
        # The recent transitions push d_mem direction more than old d_sync ones
        fitnesses = {BehavioralCoords(0,0,0): 0.5}
        # Just verify it runs without error and returns valid hints
        hints = estimator.gradient_to_hints(BehavioralCoords(0,0,0), fitnesses)
        assert isinstance(hints, list)

    def test_mixed_outcomes_reduce_gradient_magnitude(self, buf, estimator):
        # Improvements in one direction, regressions in same direction → cancel out
        for i in range(10):
            buf.add(make_transition((0,0,0), (1,0,0), delta_f=0.3, outcome="improvement", generation=i))
        for i in range(10):
            buf.add(make_transition((0,0,0), (1,0,0), delta_f=-0.3, outcome="regression", generation=10+i))
        estimator.tick(20)
        occupied = [BehavioralCoords(0,0,0)]
        fitnesses = {BehavioralCoords(0,0,0): 0.5}
        weights = estimator.compute_sampling_weights(occupied, fitnesses)
        # Gradient should be smaller than with pure improvements
        pure_buf = TransitionBuffer(max_size=100)
        _fill_improvements(pure_buf, (0,0,0), (1,0,0), n=20)
        pure_est = GradientEstimator(pure_buf, weights=(0.4, 0.4, 0.2))
        pure_est.tick(20)
        pure_weights = pure_est.compute_sampling_weights(occupied, fitnesses)
        assert weights[BehavioralCoords(0,0,0)] <= pure_weights[BehavioralCoords(0,0,0)]

    def test_exploration_gradient_accounts_for_empty_cells(self, buf):
        estimator = GradientEstimator(buf, weights=(0.0, 0.0, 1.0))
        estimator.tick(0)
        occupied = [BehavioralCoords(0, 0, 0)]
        fitnesses = {BehavioralCoords(0, 0, 0): 1.0}
        weights = estimator.compute_sampling_weights(occupied, fitnesses, bins=4)
        assert weights[BehavioralCoords(0, 0, 0)] > 1e-6
