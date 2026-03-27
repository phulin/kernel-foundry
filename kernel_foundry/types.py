from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class BehavioralCoords:
    d_mem: int   # 0–3: memory access pattern
    d_algo: int  # 0–3: specialization / staging richness
    d_sync: int  # 0–3: work-partition shape

    def to_tuple(self) -> tuple[int, int, int]:
        return (self.d_mem, self.d_algo, self.d_sync)

    def __repr__(self) -> str:
        mem_labels = ["scalar", "coalesced", "tiled", "multi-level"]
        algo_labels = ["minimal", "tuned", "specialized", "staged"]
        sync_labels = ["scalar", "1d-blocked", "2d-tiled", "hierarchical"]
        return (
            f"BC(mem={self.d_mem}:{mem_labels[self.d_mem]}, "
            f"algo={self.d_algo}:{algo_labels[self.d_algo]}, "
            f"sync={self.d_sync}:{sync_labels[self.d_sync]})"
        )


@dataclass
class EvalResult:
    kernel_id: str
    compiled: bool
    correct: bool
    kernel_time_ms: float | None = None   # None if not reached
    baseline_time_ms: float | None = None
    speedup: float | None = None
    fitness: float = 0.0
    error_log: str = ""
    profiling_summary: str | None = None


@dataclass
class TransitionRecord:
    parent_coords: BehavioralCoords
    child_coords: BehavioralCoords
    delta_fitness: float
    outcome: Literal["improvement", "neutral", "regression"]
    generation: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class KernelRecord:
    kernel_id: str
    generation: int
    parent_id: str | None
    source_code: str
    coords: BehavioralCoords
    eval_result: EvalResult
    is_templated: bool = False
    template_configs: list[dict] | None = None


@dataclass
class EvolvableSections:
    optimization_philosophy: str
    optimization_strategies: str
    common_pitfalls: str
    analysis_guidance: str


@dataclass
class PromptVariant:
    variant_id: str
    sections: EvolvableSections
    best_fitness: float = 0.0
    generation_created: int = 0
    usage_count: int = 0
