from __future__ import annotations

import math
from collections import deque

import numpy as np

from kernel_foundry.types import BehavioralCoords, TransitionRecord

_DIMS = 3  # d_mem, d_algo, d_sync

# Natural-language hint templates keyed by (dim_index, direction)
# direction > 0 means "increase the level for this dimension"
_HINTS: dict[tuple[int, int], str] = {
    (0, +1): "consider adding shared memory tiling with tl.dot for data reuse",
    (0, +2): "implement multi-stage software pipelining (num_stages > 1 in triton.Config)",
    (1, +1): "explore fusing multiple operations into a single kernel pass",
    (1, +2): "consider online normalization or flash-attention-style algorithm reformulation",
    (2, +1): "use tl.reduce for intra-block reductions instead of sequential accumulation",
    (2, +2): "consider atomic operations for global coordination or multi-pass strategies",
    (0, -1): "the current tiling strategy may be over-engineering; try a simpler coalesced approach",
    (1, -1): "simplify the algorithm to a direct translation and verify correctness first",
    (2, -1): "reduce synchronization overhead; the problem may be embarrassingly parallel",
}


class TransitionBuffer:
    def __init__(self, max_size: int = 500) -> None:
        self._records: deque[TransitionRecord] = deque(maxlen=max_size)

    def add(self, record: TransitionRecord) -> None:
        self._records.append(record)

    def get_from(self, coords: BehavioralCoords) -> list[TransitionRecord]:
        return [r for r in self._records if r.parent_coords == coords]

    def recent(self, n: int) -> list[TransitionRecord]:
        return list(self._records)[-n:]

    def __len__(self) -> int:
        return len(self._records)


class GradientEstimator:
    """
    Estimates gradient signals over the MAP-Elites behavioral space from the transition history.

    Three components (paper §3.3):
      ∇F  — which directions in behavioral space improve fitness
      ∇R  — which directions yield higher improvement probability (magnitude-independent)
      ∇E  — which directions point toward empty/low-fitness cells (exploration pull)

    Combined: ∇ = w_F·∇F + w_R·∇R + w_E·∇E
    """

    def __init__(
        self,
        buffer: TransitionBuffer,
        weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
        decay: float = 0.95,
    ) -> None:
        self._buffer = buffer
        self._w_F, self._w_R, self._w_E = weights
        self._decay = decay
        self._current_generation: int = 0

    def tick(self, generation: int) -> None:
        self._current_generation = generation

    def compute_sampling_weights(
        self,
        occupied: list[BehavioralCoords],
        archive_fitnesses: dict[BehavioralCoords, float],
        bins: int = 4,
    ) -> dict[BehavioralCoords, float]:
        """
        Return per-cell sampling weight proportional to |combined gradient|.
        Falls back to uniform (equal weight 1.0) when fewer than 3 transitions exist.
        """
        if len(self._buffer) < 3:
            return {c: 1.0 for c in occupied}

        weights = {}
        for coords in occupied:
            grad = self._combined_gradient(coords, archive_fitnesses, bins=bins)
            weights[coords] = float(np.linalg.norm(grad)) + 1e-6  # avoid zero weights
        return weights

    def gradient_to_hints(
        self,
        coords: BehavioralCoords,
        archive_fitnesses: dict[BehavioralCoords, float],
        bins: int = 4,
        max_hints: int = 2,
    ) -> list[str]:
        """Convert gradient direction into natural-language mutation hints."""
        grad = self._combined_gradient(coords, archive_fitnesses, bins=bins)
        hints = []
        # Sort dimensions by absolute gradient magnitude, descending
        for dim in np.argsort(np.abs(grad))[::-1]:
            val = grad[dim]
            if abs(val) < 0.05:
                continue
            direction = +1 if val > 0 else -1
            # Map to hint at level +1 or +2 based on strength
            strength = +2 if abs(val) > 0.5 else +1
            hint = _HINTS.get((int(dim), direction * strength)) or _HINTS.get((int(dim), direction))
            if hint:
                hints.append(hint)
            if len(hints) >= max_hints:
                break
        return hints

    # ------------------------------------------------------------------ internals

    def _temporal_weight(self, record: TransitionRecord) -> float:
        age = self._current_generation - record.generation
        return math.pow(self._decay, max(age, 0))

    def _combined_gradient(
        self,
        coords: BehavioralCoords,
        archive_fitnesses: dict[BehavioralCoords, float],
        bins: int = 4,
    ) -> np.ndarray:
        transitions = self._buffer.get_from(coords)
        grad_F = self._fitness_gradient(transitions)
        grad_R = self._improvement_rate_gradient(transitions)
        grad_E = self._exploration_gradient(coords, archive_fitnesses, bins=bins)
        return self._w_F * grad_F + self._w_R * grad_R + self._w_E * grad_E

    def _fitness_gradient(self, transitions: list[TransitionRecord]) -> np.ndarray:
        """∇F: weighted mean of (Δf × sign(Δb_d)) across transitions."""
        if not transitions:
            return np.zeros(_DIMS)
        grad = np.zeros(_DIMS)
        total_w = 0.0
        for t in transitions:
            w = self._temporal_weight(t)
            total_w += w
            delta_b = np.array(t.child_coords.to_tuple()) - np.array(t.parent_coords.to_tuple())
            grad += w * t.delta_fitness * np.sign(delta_b)
        return grad / (total_w + 1e-8)

    def _improvement_rate_gradient(self, transitions: list[TransitionRecord]) -> np.ndarray:
        """∇R: P(improve | Δb_d > 0) - P(improve | Δb_d < 0) per dimension."""
        if not transitions:
            return np.zeros(_DIMS)
        grad = np.zeros(_DIMS)
        for dim in range(_DIMS):
            pos_improvements = pos_total = neg_improvements = neg_total = 0.0
            for t in transitions:
                delta = t.child_coords.to_tuple()[dim] - t.parent_coords.to_tuple()[dim]
                improved = 1.0 if t.outcome == "improvement" else 0.0
                if delta > 0:
                    pos_improvements += improved
                    pos_total += 1
                elif delta < 0:
                    neg_improvements += improved
                    neg_total += 1
            p_pos = pos_improvements / pos_total if pos_total > 0 else 0.0
            p_neg = neg_improvements / neg_total if neg_total > 0 else 0.0
            grad[dim] = p_pos - p_neg
        return grad

    def _exploration_gradient(
        self,
        coords: BehavioralCoords,
        archive_fitnesses: dict[BehavioralCoords, float],
        bins: int = 4,
    ) -> np.ndarray:
        """∇E: pull toward low-fitness / empty cells, weighted by inverse Manhattan distance."""
        if bins <= 0:
            return np.zeros(_DIMS)
        max_f = max(archive_fitnesses.values(), default=1.0) or 1.0
        b = np.array(coords.to_tuple(), dtype=float)
        grad = np.zeros(_DIMS)
        all_cells = [
            BehavioralCoords(d_mem, d_algo, d_sync)
            for d_mem in range(bins)
            for d_algo in range(bins)
            for d_sync in range(bins)
        ]
        for other in all_cells:
            fitness = archive_fitnesses.get(other, 0.0)
            if other == coords and fitness >= max_f:
                continue
            other_b = np.array(other.to_tuple(), dtype=float)
            diff = other_b - b
            dist = np.sum(np.abs(diff)) + 1e-8
            improvement_potential = max_f - fitness
            grad += (improvement_potential / dist) * (diff / dist)
        return grad / (len(all_cells) + 1e-8)
