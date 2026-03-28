from __future__ import annotations

import json
import uuid

from kernel_foundry.llm.client import LLMClient
from kernel_foundry.llm.response_parser import extract_json_payload
from kernel_foundry.search.types import (
    ActionIntent,
    ActionTrajectory,
    ClosedActionSummary,
    FrontierAction,
    PlannerDirective,
    PlannerUpdate,
    SearchState,
)
from kernel_foundry.task.spec import TaskSpec
from kernel_foundry.types import KernelRecord


class WorldModelPlanner:
    """
    Maintains a frontier of optimization intents and updates it from structured
    LLM planner responses.
    """

    def __init__(self, llm: LLMClient, task: TaskSpec) -> None:
        self._llm = llm
        self._task = task

    def initialize_state(self, seed_kernel: str | None = None) -> SearchState:
        state = SearchState()
        intents = self._generate_initial_intents(seed_kernel)
        for title, rationale, score in intents:
            action = FrontierAction(
                action_id=str(uuid.uuid4())[:8],
                parent_kernel_id=None,
                intent=ActionIntent(title=title, rationale=rationale),
                priority_score=score,
                parent_action_id=None,
            )
            state.frontier[action.action_id] = action
        return state

    def _generate_initial_intents(
        self, seed_kernel: str | None
    ) -> list[tuple[str, str, float]]:
        seed_section = ""
        if seed_kernel:
            seed_section = (
                "\nA seed kernel implementation is provided. One of the actions should "
                "refine this seed as the highest-priority starting point.\n"
            )

        prompt = f"""You are initializing a search tree for GPU kernel optimization.

Task: {self._task.name}
Description: {self._task.description}

Reference implementation:
```python
{self._task.reference_code}
```
{seed_section}
Propose 3-5 high-level optimization strategies as the initial frontier of the search tree.
Each strategy should be a distinct, task-specific approach informed by the kernel's computational pattern.
Think about what an expert GPU kernel engineer would consider for this specific workload.

Return JSON only:
{{
  "actions": [
    {{
      "title": "short descriptive name for the strategy",
      "rationale": "why this strategy is promising for this specific kernel",
      "priority_score": 0.0
    }}
  ]
}}

Rules:
- Strategies must be specific to this task, not generic optimization advice.
- priority_score in [0, 1] reflecting your prior belief about which approach is most promising.
- Order from highest to lowest priority.
"""
        try:
            response = self._llm.generate_single(prompt, temperature=1.0)
            payload = json.loads(extract_json_payload(response))
            intents = []
            for item in payload.get("actions", []):
                intents.append((
                    item["title"],
                    item.get("rationale", ""),
                    float(item.get("priority_score", 0.5)),
                ))
            if intents:
                return intents
        except Exception:
            pass
        return self._fallback_intents(seed_kernel)

    @staticmethod
    def _fallback_intents(seed_kernel: str | None) -> list[tuple[str, str, float]]:
        intents = [
            (
                "Establish a correct baseline Triton kernel",
                "Start from a simple implementation that matches the reference semantics.",
                0.95,
            ),
            (
                "Improve memory access and tiling",
                "Explore coalescing, block pointers, and tiling for better bandwidth use.",
                0.75,
            ),
            (
                "Explore algorithmic restructuring",
                "Consider fused or staged formulations if the direct approach stalls.",
                0.65,
            ),
        ]
        if seed_kernel:
            intents.insert(
                0,
                (
                    "Refine the provided seed kernel",
                    "Use the seed implementation as the starting branch before broader exploration.",
                    1.0,
                ),
            )
        return intents

    def select_action(self, state: SearchState) -> FrontierAction | None:
        actions = state.open_actions()
        return actions[0] if actions else None

    def update_state(
        self,
        state: SearchState,
        action: FrontierAction,
        trajectory: ActionTrajectory,
        best_kernel: KernelRecord | None,
        budget_remaining: int,
    ) -> PlannerUpdate:
        best_candidate = trajectory.best_candidate
        update = self._request_update(
            state=state,
            action=action,
            trajectory=trajectory,
            best_kernel=best_kernel,
            budget_remaining=budget_remaining,
        )

        current = state.frontier.get(action.action_id)
        if current:
            current.status = "closed"
            if best_candidate is not None:
                current.best_kernel_id = best_candidate.kernel_id
            state.closed_actions.add(action.action_id)
            state.history.append(ClosedActionSummary(
                action_id=action.action_id,
                title=action.intent.title,
                rationale=action.intent.rationale,
                parent_action_id=action.parent_action_id,
                parent_kernel_id=action.parent_kernel_id,
                num_candidates=len(trajectory.candidates),
                best_fitness=best_candidate.eval_result.fitness if best_candidate else 0.0,
                best_speedup=best_candidate.eval_result.speedup if best_candidate else None,
                had_correct=any(c.eval_result.correct for c in trajectory.candidates),
                had_compile_error=any(not c.eval_result.compiled for c in trajectory.candidates),
                stagnation_steps=trajectory.stagnation_steps,
            ))

        for directive in update.directives:
            if directive.kind == "insert":
                inserted = FrontierAction(
                    action_id=directive.action_id or str(uuid.uuid4())[:8],
                    parent_kernel_id=directive.parent_kernel_id,
                    intent=ActionIntent(
                        title=directive.title or "Untitled intent",
                        rationale=directive.rationale or "",
                    ),
                    priority_score=directive.priority_score or 0.5,
                    parent_action_id=action.action_id,
                )
                state.frontier[inserted.action_id] = inserted
            elif directive.kind == "update":
                target = state.frontier.get(directive.action_id)
                if target:
                    if directive.title:
                        target.intent.title = directive.title
                    if directive.rationale:
                        target.intent.rationale = directive.rationale
                    if directive.priority_score is not None:
                        target.priority_score = directive.priority_score
                    if directive.parent_kernel_id is not None:
                        target.parent_kernel_id = directive.parent_kernel_id
            elif directive.kind == "prune":
                target = state.frontier.get(directive.action_id)
                if target:
                    target.status = "pruned"
                    state.pruned_actions.add(target.action_id)

        return update

    def _request_update(
        self,
        state: SearchState,
        action: FrontierAction,
        trajectory: ActionTrajectory,
        best_kernel: KernelRecord | None,
        budget_remaining: int,
    ) -> PlannerUpdate:
        self._current_state = state
        prompt = self._build_update_prompt(action, trajectory, best_kernel, budget_remaining)
        response = self._llm.generate_single(prompt, temperature=1.0)
        return parse_planner_update(response)

    def _build_update_prompt(
        self,
        action: FrontierAction,
        trajectory: ActionTrajectory,
        best_kernel: KernelRecord | None,
        budget_remaining: int,
    ) -> str:
        outcome_lines = []
        for record in trajectory.candidates:
            eval_result = record.eval_result
            speedup = f"{eval_result.speedup:.2f}x" if eval_result.speedup is not None else "N/A"
            if not eval_result.compiled:
                status = f"compile_error: {eval_result.error_log[:120]}"
            elif not eval_result.correct:
                status = f"incorrect: {eval_result.error_log[:120]}"
            else:
                status = f"correct speedup={speedup}"
            entry_lines = [
                f"- kernel_id={record.kernel_id} fitness={eval_result.fitness:.3f} {status}"
            ]
            if eval_result.profiling_summary:
                entry_lines.append(f"  profiling: {eval_result.profiling_summary}")
            if eval_result.ncu_output:
                entry_lines.append("  ncu:")
                entry_lines.extend(
                    f"    {line}" for line in eval_result.ncu_output.splitlines()
                )
            outcome_lines.append("\n".join(entry_lines))
        outcomes = "\n".join(outcome_lines) or "- no candidates evaluated"

        best_summary = "none"
        if best_kernel is not None:
            speedup = (
                f"{best_kernel.eval_result.speedup:.2f}x"
                if best_kernel.eval_result.speedup is not None
                else "N/A"
            )
            best_summary = (
                f"kernel_id={best_kernel.kernel_id} speedup={speedup} "
                f"fitness={best_kernel.eval_result.fitness:.3f}"
            )
            if best_kernel.eval_result.profiling_summary:
                best_summary += (
                    f"\n  profiling: {best_kernel.eval_result.profiling_summary}"
                )
            if best_kernel.eval_result.ncu_output:
                best_summary += "\n  ncu:"
                best_summary += "".join(
                    f"\n    {line}" for line in best_kernel.eval_result.ncu_output.splitlines()
                )

        history_lines = self._format_history(self._current_state)
        frontier_lines = self._format_frontier(self._current_state, exclude=action.action_id)

        return f"""You are maintaining the search state for GPU kernel optimization, structured as a search tree.
Your role is to act as a world model: reason over accumulated experience to guide which optimization intents to pursue next.
Return JSON only.

## Search History (closed actions, oldest first)
{history_lines}

## Current Frontier (open actions)
{frontier_lines}

## Current Action (just completed local refinement)
- action_id: {action.action_id}
- title: {action.intent.title}
- rationale: {action.intent.rationale}
- parent_action_id: {action.parent_action_id or 'root'}
- parent_kernel_id: {action.parent_kernel_id}

Trajectory outcomes:
{outcomes}

## Global State
Budget remaining: {budget_remaining}
Best kernel overall: {best_summary}

## Task
Decide how to evolve the search tree. You may Insert new child actions to deepen promising branches, \
Update priority scores of existing frontier actions based on new evidence, or Prune branches that are \
clearly redundant or unpromising given accumulated experience.

Use this schema:
{{
  "summary": "short reasoned summary of your analysis and decisions",
  "directives": [
    {{
      "kind": "insert" | "update" | "prune",
      "action_id": "string",
      "title": "required for insert, optional for update",
      "rationale": "optional",
      "parent_kernel_id": "optional, kernel_id to build on",
      "priority_score": 0.0
    }}
  ]
}}

Rules:
- Insert at most 3 new actions.
- Use kind=update to reprioritize existing frontier actions based on what you've learned.
- Use kind=prune for branches made redundant or unpromising by accumulated evidence.
- Keep priority_score in [0, 1].
- If the current action produced a promising correct kernel, deepen that branch with follow-up inserts.
- If all attempts failed, consider whether the intent itself is flawed (prune) or just needs a simpler approach (insert a simplified variant).
- Reason over the full history: strategies that failed in one context may succeed as compositions atop later progress.
"""

    @staticmethod
    def _format_history(state: SearchState) -> str:
        if not state.history:
            return "(no actions completed yet)"
        # Build depth map from frontier for indentation
        depth: dict[str, int] = {}
        all_actions = {a.action_id: a for a in state.frontier.values()}
        for entry in state.history:
            parent = entry.parent_action_id
            depth[entry.action_id] = depth.get(parent, -1) + 1 if parent else 0
        lines = []
        for entry in state.history:
            indent = "  " * depth.get(entry.action_id, 0)
            speedup = f"{entry.best_speedup:.2f}x" if entry.best_speedup is not None else "N/A"
            outcome = "correct" if entry.had_correct else ("compile errors" if entry.had_compile_error else "incorrect")
            lines.append(
                f"{indent}[{entry.action_id}] \"{entry.title}\" "
                f"(parent_action={entry.parent_action_id or 'root'}) | "
                f"{entry.num_candidates} candidates, best fitness={entry.best_fitness:.3f} speedup={speedup} | "
                f"outcome={outcome}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_frontier(state: SearchState, exclude: str) -> str:
        open_actions = [a for a in state.open_actions() if a.action_id != exclude]
        if not open_actions:
            return "(no other open actions)"
        # Compute tree depth from parent_action_id chain
        depth: dict[str, int] = {}
        all_actions = {a.action_id: a for a in state.frontier.values()}
        def get_depth(action_id: str) -> int:
            if action_id in depth:
                return depth[action_id]
            action = all_actions.get(action_id)
            if action is None or action.parent_action_id is None:
                depth[action_id] = 0
            else:
                depth[action_id] = get_depth(action.parent_action_id) + 1
            return depth[action_id]
        lines = []
        for a in open_actions:
            d = get_depth(a.action_id)
            indent = "  " * d
            lines.append(
                f"{indent}[{a.action_id}] \"{a.intent.title}\" "
                f"(parent_action={a.parent_action_id or 'root'}) | "
                f"priority={a.priority_score:.2f} attempts={a.attempt_count}"
            )
        return "\n".join(lines)


def parse_planner_update(response: str) -> PlannerUpdate:
    payload = extract_json_payload(response)
    data = json.loads(payload)
    directives = []
    for item in data.get("directives", []):
        directives.append(
            PlannerDirective(
                kind=item["kind"],
                action_id=item["action_id"],
                title=item.get("title"),
                rationale=item.get("rationale"),
                parent_kernel_id=item.get("parent_kernel_id"),
                priority_score=item.get("priority_score"),
            )
        )
    return PlannerUpdate(summary=data.get("summary", ""), directives=directives)
