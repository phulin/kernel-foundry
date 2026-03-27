from __future__ import annotations

import uuid

import numpy as np

from kernel_foundry.types import EvolvableSections, PromptVariant


class PromptArchive:
    """
    Capacity-bounded store for evolved prompt variants.
    Tracks which prompt variant achieved the best kernel performance.
    """

    def __init__(self, capacity: int = 16) -> None:
        self.capacity = capacity
        self._variants: dict[str, PromptVariant] = {}

    def insert(self, sections: EvolvableSections, generation: int) -> PromptVariant:
        """Create a new variant from sections and insert it. Evicts lowest-fitness if full."""
        variant = PromptVariant(
            variant_id=str(uuid.uuid4())[:8],
            sections=sections,
            generation_created=generation,
        )
        if len(self._variants) >= self.capacity:
            # Evict the lowest-fitness variant
            worst_id = min(self._variants, key=lambda vid: self._variants[vid].best_fitness)
            del self._variants[worst_id]
        self._variants[variant.variant_id] = variant
        return variant

    def update_fitness(self, variant_id: str, fitness: float) -> None:
        if variant_id in self._variants:
            v = self._variants[variant_id]
            if fitness > v.best_fitness:
                v.best_fitness = fitness

    def get_best_variant(self) -> PromptVariant | None:
        if not self._variants:
            return None
        return max(self._variants.values(), key=lambda v: v.best_fitness)

    def get_active_variant(
        self, rng: np.random.Generator, epsilon: float = 0.2
    ) -> PromptVariant | None:
        """Epsilon-greedy selection: explore a random variant with probability ``epsilon``,
        otherwise exploit the highest-fitness variant.

        This gives newly evolved prompt variants a chance to be used before they have
        accumulated any fitness signal, preventing the prompt archive from collapsing
        to the first successful variant.
        """
        if not self._variants:
            return None
        if rng.random() < epsilon:
            variants = list(self._variants.values())
            return variants[int(rng.integers(len(variants)))]
        return max(self._variants.values(), key=lambda v: v.best_fitness)

    def get_active_variant_id(self) -> str | None:
        best = self.get_best_variant()
        return best.variant_id if best else None

    def record_usage(self, variant_id: str) -> None:
        if variant_id in self._variants:
            self._variants[variant_id].usage_count += 1

    def all_variants(self) -> list[PromptVariant]:
        return list(self._variants.values())

    def size(self) -> int:
        return len(self._variants)
