from __future__ import annotations

import json
from typing import Iterator

from kernel_foundry.types import BehavioralCoords, KernelRecord


class MAPElitesArchive:
    """
    64-cell (4^3) MAP-Elites archive keyed by (d_mem, d_algo, d_sync).
    Each cell holds the highest-fitness kernel discovered for that behavioral region.
    """

    def __init__(self, bins: int = 4) -> None:
        self.bins = bins
        self._grid: dict[tuple[int, int, int], KernelRecord] = {}

    def insert(self, record: KernelRecord) -> bool:
        """Insert if cell empty or record improves on incumbent. Returns True if updated."""
        key = record.coords.to_tuple()
        incumbent = self._grid.get(key)
        if incumbent is None or record.eval_result.fitness > incumbent.eval_result.fitness:
            self._grid[key] = record
            return True
        return False

    def get_elite(self, coords: BehavioralCoords) -> KernelRecord | None:
        return self._grid.get(coords.to_tuple())

    def get_all_elites(self) -> list[KernelRecord]:
        return list(self._grid.values())

    def __iter__(self) -> Iterator[KernelRecord]:
        return iter(self._grid.values())

    def get_occupied_cells(self) -> list[BehavioralCoords]:
        return [BehavioralCoords(*k) for k in self._grid]

    def get_empty_cells(self) -> list[BehavioralCoords]:
        occupied = set(self._grid.keys())
        return [
            BehavioralCoords(d_mem, d_algo, d_sync)
            for d_mem in range(self.bins)
            for d_algo in range(self.bins)
            for d_sync in range(self.bins)
            if (d_mem, d_algo, d_sync) not in occupied
        ]

    def get_best_overall(self) -> KernelRecord | None:
        if not self._grid:
            return None
        return max(self._grid.values(), key=lambda r: r.eval_result.fitness)

    def get_fitness(self, coords: BehavioralCoords) -> float:
        record = self._grid.get(coords.to_tuple())
        return record.eval_result.fitness if record else 0.0

    def get_max_fitness(self) -> float:
        if not self._grid:
            return 0.0
        return max(r.eval_result.fitness for r in self._grid.values())

    def size(self) -> int:
        return len(self._grid)

    def to_dict(self) -> dict:
        return {
            "bins": self.bins,
            "cells": {
                f"{k[0]},{k[1]},{k[2]}": {
                    "kernel_id": r.kernel_id,
                    "generation": r.generation,
                    "parent_id": r.parent_id,
                    "source_code": r.source_code,
                    "d_mem": r.coords.d_mem,
                    "d_algo": r.coords.d_algo,
                    "d_sync": r.coords.d_sync,
                    "fitness": r.eval_result.fitness,
                    "speedup": r.eval_result.speedup,
                    "compiled": r.eval_result.compiled,
                    "correct": r.eval_result.correct,
                    "kernel_time_ms": r.eval_result.kernel_time_ms,
                    "baseline_time_ms": r.eval_result.baseline_time_ms,
                    "error_log": r.eval_result.error_log,
                    "is_templated": r.is_templated,
                }
                for k, r in self._grid.items()
            },
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def __repr__(self) -> str:
        best = self.get_best_overall()
        best_speedup = best.eval_result.speedup if best else None
        return (
            f"MAPElitesArchive(occupied={self.size()}/{self.bins**3}, "
            f"best_speedup={best_speedup:.2f}x)"
            if best_speedup is not None
            else f"MAPElitesArchive(occupied={self.size()}/{self.bins**3}, empty)"
        )
