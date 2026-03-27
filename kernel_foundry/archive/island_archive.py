from __future__ import annotations

from kernel_foundry.archive.map_elites import MAPElitesArchive
from kernel_foundry.types import BehavioralCoords, KernelRecord


class IslandArchive:
    """
    K independent MAP-Elites archives with periodic migration (§3.2 island selection).

    Each island evolves independently; every ``migration_freq`` generations all elites
    from every island are broadcast to every other island (full migration), then the
    active island rotates.  The selector sees only the current island's occupied cells,
    providing the isolation that prevents mode collapse across islands.
    """

    def __init__(
        self,
        n_islands: int = 4,
        migration_freq: int = 10,
        bins: int = 4,
    ) -> None:
        self.bins = bins
        self._n_islands = n_islands
        self._migration_freq = migration_freq
        self._islands: list[MAPElitesArchive] = [
            MAPElitesArchive(bins=bins) for _ in range(n_islands)
        ]
        self._current: int = 0

    # ------------------------------------------------------------------ current-island API

    def insert(self, record: KernelRecord) -> bool:
        return self._islands[self._current].insert(record)

    def get_elite(self, coords: BehavioralCoords) -> KernelRecord | None:
        return self._islands[self._current].get_elite(coords)

    def get_occupied_cells(self) -> list[BehavioralCoords]:
        return self._islands[self._current].get_occupied_cells()

    def get_fitness(self, coords: BehavioralCoords) -> float:
        return self._islands[self._current].get_fitness(coords)

    def size(self) -> int:
        return self._islands[self._current].size()

    # ------------------------------------------------------------------ global API

    def get_best_overall(self) -> KernelRecord | None:
        candidates = [island.get_best_overall() for island in self._islands]
        valid = [c for c in candidates if c is not None]
        if not valid:
            return None
        return max(valid, key=lambda r: (r.eval_result.fitness, r.eval_result.speedup or 0.0))

    def get_all_elites(self) -> list[KernelRecord]:
        """Best-per-cell deduplication across all islands."""
        best: dict[tuple[int, int, int], KernelRecord] = {}
        for island in self._islands:
            for record in island.get_all_elites():
                key = record.coords.to_tuple()
                if key not in best or record.eval_result.fitness > best[key].eval_result.fitness:
                    best[key] = record
        return list(best.values())

    def get_max_fitness(self) -> float:
        return max((island.get_max_fitness() for island in self._islands), default=0.0)

    # ------------------------------------------------------------------ lifecycle

    def advance_generation(self, gen: int) -> None:
        """Call once per generation. Triggers migration and island rotation when due."""
        if gen > 0 and gen % self._migration_freq == 0:
            self._migrate()
            self._current = (self._current + 1) % self._n_islands
            print(
                f"  [island] Migration at gen {gen}; "
                f"active island → {self._current}"
            )

    def _migrate(self) -> None:
        """Broadcast every island's elites to every other island."""
        global_elites = self.get_all_elites()
        for island in self._islands:
            for record in global_elites:
                island.insert(record)

    # ------------------------------------------------------------------ serialization

    def to_dict(self) -> dict:
        return {
            "n_islands": self._n_islands,
            "current_island": self._current,
            "islands": [island.to_dict() for island in self._islands],
        }

    def __repr__(self) -> str:
        best = self.get_best_overall()
        best_speedup = best.eval_result.speedup if best else None
        total = sum(island.size() for island in self._islands)
        if best_speedup is not None:
            return (
                f"IslandArchive(islands={self._n_islands}, current={self._current}, "
                f"total_occupied={total}, best_speedup={best_speedup:.2f}x)"
            )
        return f"IslandArchive(islands={self._n_islands}, current={self._current}, empty)"
