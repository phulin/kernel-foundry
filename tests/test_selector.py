import numpy as np
import pytest

from kernel_foundry.archive.map_elites import MAPElitesArchive
from kernel_foundry.evolution.selector import (
    CuriosityDrivenSelector,
    FitnessProportionateSelector,
    IslandSelector,
    UniformSelector,
    make_selector,
)
from kernel_foundry.gradient.estimator import GradientEstimator, TransitionBuffer
from kernel_foundry.config import EvolutionConfig
from tests.conftest import make_record


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def empty_archive():
    return MAPElitesArchive()


@pytest.fixture
def estimator():
    return GradientEstimator(TransitionBuffer())


@pytest.fixture
def populated_archive():
    a = MAPElitesArchive()
    for dm in range(3):
        for da in range(3):
            a.insert(make_record(dm, da, 0, speedup=float(dm + da + 1)))
    return a


class TestUniformSelector:
    def test_empty_returns_none(self, empty_archive, estimator, rng):
        assert UniformSelector().select(empty_archive, estimator, rng) is None

    def test_single_cell_returns_that_cell(self, estimator, rng):
        a = MAPElitesArchive()
        a.insert(make_record(2, 1, 0))
        from kernel_foundry.types import BehavioralCoords
        result = UniformSelector().select(a, estimator, rng)
        assert result == BehavioralCoords(2, 1, 0)

    def test_returns_occupied_cell(self, populated_archive, estimator, rng):
        sel = UniformSelector()
        occupied = set(c.to_tuple() for c in populated_archive.get_occupied_cells())
        for _ in range(20):
            result = sel.select(populated_archive, estimator, rng)
            assert result is not None
            assert result.to_tuple() in occupied


class TestFitnessProportionateSelector:
    def test_empty_returns_none(self, empty_archive, estimator, rng):
        assert FitnessProportionateSelector().select(empty_archive, estimator, rng) is None

    def test_prefers_high_fitness(self, estimator, rng):
        a = MAPElitesArchive()
        # One very high fitness cell, many low fitness cells
        a.insert(make_record(0, 0, 0, speedup=0.1))  # low fitness
        a.insert(make_record(3, 3, 3, speedup=5.0, kid="best"))  # high fitness
        sel = FitnessProportionateSelector()
        from kernel_foundry.types import BehavioralCoords
        counts = {}
        for _ in range(100):
            c = sel.select(a, estimator, rng)
            counts[c.to_tuple()] = counts.get(c.to_tuple(), 0) + 1
        # High fitness cell should be selected more often
        assert counts.get((3,3,3), 0) > counts.get((0,0,0), 0)

    def test_returns_occupied_cell(self, populated_archive, estimator, rng):
        occupied = set(c.to_tuple() for c in populated_archive.get_occupied_cells())
        for _ in range(20):
            result = FitnessProportionateSelector().select(populated_archive, estimator, rng)
            assert result.to_tuple() in occupied


class TestCuriosityDrivenSelector:
    def test_empty_returns_none(self, empty_archive, estimator, rng):
        assert CuriosityDrivenSelector().select(empty_archive, estimator, rng) is None

    def test_falls_back_to_uniform_when_sparse(self, estimator, rng):
        a = MAPElitesArchive()
        a.insert(make_record(1, 0, 0))
        # Only 1 cell — should still return a valid cell without error
        result = CuriosityDrivenSelector().select(a, estimator, rng)
        assert result is not None

    def test_returns_occupied_cell(self, populated_archive, rng):
        from kernel_foundry.gradient.estimator import TransitionBuffer
        from tests.conftest import make_transition
        buf = TransitionBuffer()
        for i in range(15):
            buf.add(make_transition((0,0,0), (1,1,0), 0.3, "improvement", generation=i))
        est = GradientEstimator(buf)
        est.tick(15)
        occupied = set(c.to_tuple() for c in populated_archive.get_occupied_cells())
        for _ in range(20):
            result = CuriosityDrivenSelector().select(populated_archive, est, rng)
            assert result.to_tuple() in occupied


class TestIslandSelector:
    def test_empty_returns_none(self, empty_archive, estimator, rng):
        assert IslandSelector().select(empty_archive, estimator, rng) is None

    def test_returns_occupied_cell(self, populated_archive, estimator, rng):
        # Island rotation/migration is now managed by IslandArchive.advance_generation();
        # the selector just picks uniformly from the current archive's occupied cells.
        occupied = set(c.to_tuple() for c in populated_archive.get_occupied_cells())
        sel = IslandSelector()
        for _ in range(20):
            result = sel.select(populated_archive, estimator, rng)
            assert result.to_tuple() in occupied


class TestMakeSelector:
    @pytest.mark.parametrize("strategy,expected_type", [
        ("uniform", UniformSelector),
        ("fitness", FitnessProportionateSelector),
        ("curiosity", CuriosityDrivenSelector),
        ("island", IslandSelector),
    ])
    def test_returns_correct_type(self, strategy, expected_type):
        config = EvolutionConfig(selection_strategy=strategy)
        sel = make_selector(config)
        assert isinstance(sel, expected_type)

    def test_unknown_strategy_raises(self):
        config = EvolutionConfig(selection_strategy="unknown")  # type: ignore
        with pytest.raises(ValueError, match="Unknown selection strategy"):
            make_selector(config)
