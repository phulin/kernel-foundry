from __future__ import annotations

from typing import Protocol

import numpy as np

from kernel_foundry.archive.map_elites import MAPElitesArchive
from kernel_foundry.config import EvolutionConfig
from kernel_foundry.gradient.estimator import GradientEstimator
from kernel_foundry.types import BehavioralCoords


class Selector(Protocol):
    def select(
        self,
        archive: MAPElitesArchive,
        gradient_estimator: GradientEstimator,
        rng: np.random.Generator,
    ) -> BehavioralCoords | None: ...


class UniformSelector:
    """Random from occupied cells."""

    def select(self, archive, gradient_estimator, rng) -> BehavioralCoords | None:
        occupied = archive.get_occupied_cells()
        if not occupied:
            return None
        return occupied[rng.integers(len(occupied))]


class FitnessProportionateSelector:
    """Weighted by fitness."""

    def select(self, archive, gradient_estimator, rng) -> BehavioralCoords | None:
        occupied = archive.get_occupied_cells()
        if not occupied:
            return None
        fitnesses = np.array([archive.get_fitness(c) for c in occupied])
        fitnesses = np.maximum(fitnesses, 1e-6)
        probs = fitnesses / fitnesses.sum()
        idx = rng.choice(len(occupied), p=probs)
        return occupied[idx]


class CuriosityDrivenSelector:
    """Weighted by |combined gradient| — prioritizes cells with high improvement potential."""

    def select(self, archive, gradient_estimator, rng) -> BehavioralCoords | None:
        occupied = archive.get_occupied_cells()
        if not occupied:
            return None
        # Fall back to uniform when archive is too small for meaningful gradients
        if len(occupied) < 3 or len(gradient_estimator._buffer) < 3:
            return occupied[rng.integers(len(occupied))]

        fitnesses = {c: archive.get_fitness(c) for c in occupied}
        weights_dict = gradient_estimator.compute_sampling_weights(
            occupied,
            fitnesses,
            bins=archive.bins,
        )
        weights = np.array([weights_dict.get(c, 1.0) for c in occupied])
        weights = np.maximum(weights, 1e-6)
        probs = weights / weights.sum()
        idx = rng.choice(len(occupied), p=probs)
        return occupied[idx]


class IslandSelector:
    """
    Uniform selection from the currently active island's occupied cells.

    Island rotation and cross-island migration are managed by
    ``IslandArchive.advance_generation()``, which is called by the evolution loop
    once per generation.  This selector therefore just needs to pick uniformly from
    whatever cells the archive exposes — ``archive.get_occupied_cells()`` already
    returns only the current island's cells when an ``IslandArchive`` is in use.
    """

    def select(self, archive, gradient_estimator, rng) -> BehavioralCoords | None:
        occupied = archive.get_occupied_cells()
        if not occupied:
            return None
        return occupied[rng.integers(len(occupied))]


class MixedSelector:
    """Sample a selection policy using configurable weights, then delegate."""

    def __init__(self, config: EvolutionConfig) -> None:
        self._selectors = [
            UniformSelector(),
            FitnessProportionateSelector(),
            CuriosityDrivenSelector(),
            IslandSelector(),
        ]
        weights = np.array(
            [
                config.selection_weight_uniform,
                config.selection_weight_fitness,
                config.selection_weight_curiosity,
                config.selection_weight_island,
            ],
            dtype=float,
        )
        weights = np.maximum(weights, 0.0)
        if weights.sum() == 0:
            raise ValueError("At least one mixed selection weight must be positive")
        self._probs = weights / weights.sum()

    def select(self, archive, gradient_estimator, rng) -> BehavioralCoords | None:
        selector = self._selectors[int(rng.choice(len(self._selectors), p=self._probs))]
        return selector.select(archive, gradient_estimator, rng)


def make_selector(config: EvolutionConfig) -> Selector:
    strategy = config.selection_strategy
    if strategy == "uniform":
        return UniformSelector()
    elif strategy == "fitness":
        return FitnessProportionateSelector()
    elif strategy == "curiosity":
        return CuriosityDrivenSelector()
    elif strategy == "island":
        return IslandSelector()
    elif strategy == "mixed":
        return MixedSelector(config)
    else:
        raise ValueError(f"Unknown selection strategy: {strategy!r}")
