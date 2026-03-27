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
        weights_dict = gradient_estimator.compute_sampling_weights(occupied, fitnesses)
        weights = np.array([weights_dict.get(c, 1.0) for c in occupied])
        weights = np.maximum(weights, 1e-6)
        probs = weights / weights.sum()
        idx = rng.choice(len(occupied), p=probs)
        return occupied[idx]


class IslandSelector:
    """
    Maintains K independent sub-populations with periodic migration.
    Each call selects from the current island; islands rotate every migration_freq calls.
    """

    def __init__(self, n_islands: int = 4, migration_freq: int = 10) -> None:
        self._n_islands = n_islands
        self._migration_freq = migration_freq
        self._call_count = 0
        self._current_island = 0
        # Island assignments are rebuilt lazily on each call
        self._island_cells: list[list[BehavioralCoords]] = []

    def select(self, archive, gradient_estimator, rng) -> BehavioralCoords | None:
        occupied = archive.get_occupied_cells()
        if not occupied:
            return None

        self._call_count += 1
        if self._call_count % self._migration_freq == 0:
            self._current_island = (self._current_island + 1) % self._n_islands
            self._island_cells = []  # force rebuild

        # Rebuild island partition if stale
        if not self._island_cells or sum(len(i) for i in self._island_cells) != len(occupied):
            shuffled = occupied.copy()
            rng.shuffle(shuffled)
            self._island_cells = [[] for _ in range(self._n_islands)]
            for i, cell in enumerate(shuffled):
                self._island_cells[i % self._n_islands].append(cell)

        island_cells = self._island_cells[self._current_island % len(self._island_cells)]
        if not island_cells:
            island_cells = occupied  # fallback

        return island_cells[rng.integers(len(island_cells))]


def make_selector(config: EvolutionConfig) -> Selector:
    strategy = config.selection_strategy
    if strategy == "uniform":
        return UniformSelector()
    elif strategy == "fitness":
        return FitnessProportionateSelector()
    elif strategy == "curiosity":
        return CuriosityDrivenSelector()
    elif strategy == "island":
        return IslandSelector(config.island_count, config.island_migration_freq)
    else:
        raise ValueError(f"Unknown selection strategy: {strategy!r}")
