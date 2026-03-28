from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from kernel_foundry.types import KernelRecord


ActionStatus = Literal["open", "closed", "pruned"]


@dataclass
class ActionIntent:
    title: str
    rationale: str


@dataclass
class FrontierAction:
    action_id: str
    parent_kernel_id: str | None
    intent: ActionIntent
    priority_score: float
    status: ActionStatus = "open"
    attempt_count: int = 0
    best_kernel_id: str | None = None
    parent_action_id: str | None = None


@dataclass
class ActionOutcome:
    action_id: str
    kernel_id: str
    compiled: bool
    correct: bool
    fitness: float
    speedup: float | None
    summary: str


@dataclass
class ActionTrajectory:
    action: FrontierAction
    candidates: list[KernelRecord] = field(default_factory=list)
    stagnation_steps: int = 0

    @property
    def best_candidate(self) -> KernelRecord | None:
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda record: record.eval_result.fitness)


@dataclass
class PlannerDirective:
    kind: Literal["insert", "update", "prune"]
    action_id: str
    title: str | None = None
    rationale: str | None = None
    parent_kernel_id: str | None = None
    priority_score: float | None = None


@dataclass
class PlannerUpdate:
    summary: str
    directives: list[PlannerDirective]


@dataclass
class ClosedActionSummary:
    action_id: str
    title: str
    rationale: str
    parent_action_id: str | None
    parent_kernel_id: str | None
    num_candidates: int
    best_fitness: float
    best_speedup: float | None
    had_correct: bool
    had_compile_error: bool
    stagnation_steps: int


@dataclass
class SearchState:
    frontier: dict[str, FrontierAction] = field(default_factory=dict)
    closed_actions: set[str] = field(default_factory=set)
    pruned_actions: set[str] = field(default_factory=set)
    history: list[ClosedActionSummary] = field(default_factory=list)

    def open_actions(self) -> list[FrontierAction]:
        actions = [action for action in self.frontier.values() if action.status == "open"]
        return sorted(actions, key=lambda action: action.priority_score, reverse=True)
