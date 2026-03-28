from __future__ import annotations

import json
from pathlib import Path

from kernel_foundry.config import EvolutionConfig
from kernel_foundry.evaluation.service import CandidateEvaluator
from kernel_foundry.llm.client import LLMClient
from kernel_foundry.llm.response_parser import extract_triton_code
from kernel_foundry.prompt.evolvable_sections import DEFAULT_SECTIONS
from kernel_foundry.search.instantiator import ActionInstantiator
from kernel_foundry.search.types import ActionIntent, ActionTrajectory, FrontierAction
from kernel_foundry.search.world_model import WorldModelPlanner
from kernel_foundry.task.spec import TaskSpec
from kernel_foundry.types import EvalResult, KernelRecord


class WorldModelSearchLoop:
    """
    Experimental K-Search-style controller layered on top of the current
    compile/evaluate/archive stack.
    """

    def __init__(
        self,
        task: TaskSpec,
        config: EvolutionConfig,
        budget: int,
        stagnation_limit: int,
        refinement_population: int,
        records_path: str | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        self.task = task
        self.config = config
        self.budget = budget
        self.initial_budget = budget
        self.stagnation_limit = stagnation_limit
        self.refinement_population = refinement_population
        self.records_path = records_path
        self.checkpoint_path = checkpoint_path

        self.llm = LLMClient(config)
        self.evaluator = CandidateEvaluator(task, config)
        self.planner = WorldModelPlanner(self.llm, task)
        self.instantiator = ActionInstantiator(self.llm, task, DEFAULT_SECTIONS)
        self.state = self.planner.initialize_state(seed_kernel=task.seed_kernel)
        self.records: list[KernelRecord] = []
        self.best_kernel: KernelRecord | None = None
        self.step_count = 0
        self.actions_closed = 0
        self.compile_successes = 0
        self.correctness_successes = 0

        if self.checkpoint_path:
            self._load_checkpoint(self.checkpoint_path)

    def run(self) -> KernelRecord | None:
        print(f"\n{'='*60}")
        print(f"K-Search World Model Search: {self.task.name}")
        print(f"Baseline: {self.task.baseline_time_ms:.2f}ms | Budget: {self.budget}")
        print(f"Stagnation limit: {self.stagnation_limit} | Refinement population: {self.refinement_population}")
        print(f"Hardware: {self.task.hardware_spec}")
        print(f"{'='*60}\n")

        if self.task.seed_kernel:
            seed_record = self.evaluator.evaluate_candidate(
                self.task.seed_kernel,
                generation=0,
                parent_id=None,
            )
            self.records.append(seed_record)
            self._update_best(seed_record)
            self._update_metrics(seed_record)
            self._print_candidate_result("seed", seed_record)
            self._persist_state()

        while self.budget > 0:
            action = self.planner.select_action(self.state)
            if action is None:
                print("No open frontier actions remain.")
                break
            self.step_count += 1
            print(
                f"[Step {self.step_count}] action={action.action_id} priority={action.priority_score:.2f} "
                f"title={action.intent.title}"
            )
            trajectory = self._refine_action(action, generation=self.step_count)
            update = self.planner.update_state(
                self.state,
                action=action,
                trajectory=trajectory,
                best_kernel=self.best_kernel,
                budget_remaining=self.budget,
            )
            self.actions_closed = len(self.state.closed_actions)
            print(f"  Planner update: {update.summary}")
            self._persist_state()

        best = self.best_kernel
        if best is not None:
            speedup = f"{best.eval_result.speedup:.2f}x" if best.eval_result.speedup is not None else "N/A"
            print(f"\nBest speedup: {speedup}")
        else:
            print("\nNo correct kernel found.")

        self._print_summary()

        t = self.llm
        print(
            f"\nTokens — input: {t.tokens_input:,}  output: {t.tokens_output:,}"
            f"  cached: {t.tokens_cached:,}"
            f"  total: {t.tokens_input + t.tokens_output:,}"
        )
        return best

    def _refine_action(self, action: FrontierAction, generation: int) -> ActionTrajectory:
        trajectory = ActionTrajectory(action=action)
        parent_kernel = self._resolve_parent_kernel(action)
        best_fitness = parent_kernel.eval_result.fitness if parent_kernel is not None else 0.0

        while self.budget > 0 and trajectory.stagnation_steps < self.stagnation_limit:
            responses = self.instantiator.generate_candidates(
                action=action,
                parent_kernel=parent_kernel,
                best_kernel=self.best_kernel,
                n=self.refinement_population,
            )
            action.attempt_count += 1

            improved = False
            evaluated_any = False
            for response in responses:
                if self.budget <= 0:
                    break
                code = extract_triton_code(response)
                if code is None:
                    continue

                self.budget -= 1
                evaluated_any = True
                record = self.evaluator.evaluate_candidate(
                    code,
                    generation=generation,
                    parent_id=parent_kernel.kernel_id if parent_kernel else None,
                )
                trajectory.candidates.append(record)
                self.records.append(record)
                self._update_best(record)
                self._update_metrics(record)
                self._print_candidate_result(
                    f"{action.action_id}.{action.attempt_count}", record
                )

                if record.eval_result.fitness > best_fitness:
                    best_fitness = record.eval_result.fitness
                    action.best_kernel_id = record.kernel_id
                    parent_kernel = record
                    improved = True

            if improved:
                trajectory.stagnation_steps = 0
            else:
                trajectory.stagnation_steps += 1

        return trajectory

    def _resolve_parent_kernel(self, action: FrontierAction) -> KernelRecord | None:
        if action.parent_kernel_id:
            for record in reversed(self.records):
                if record.kernel_id == action.parent_kernel_id:
                    return record
        return self.best_kernel

    def _update_best(self, record: KernelRecord) -> None:
        if not record.eval_result.correct:
            return
        if self.best_kernel is None or (
            record.eval_result.fitness > self.best_kernel.eval_result.fitness
        ):
            self.best_kernel = record

    def _update_metrics(self, record: KernelRecord) -> None:
        if record.eval_result.compiled:
            self.compile_successes += 1
        if record.eval_result.correct:
            self.correctness_successes += 1

    def _print_candidate_result(self, idx: str, record: KernelRecord) -> None:
        result = record.eval_result
        if not result.compiled:
            status = "COMPILE ERROR"
        elif not result.correct:
            status = "INCORRECT"
        else:
            speedup = f"{result.speedup:.2f}x" if result.speedup is not None else "N/A"
            status = f"✓ speedup={speedup} fitness={result.fitness:.3f}"
        print(f"  [{idx}] {status}")

    def _flush_records(self) -> None:
        if not self.records_path:
            return
        path = Path(self.records_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for record in self.records:
                payload = self._serialize_record(record)
                handle.write(json.dumps(payload) + "\n")

    def _persist_state(self) -> None:
        self._flush_records()
        if self.checkpoint_path:
            self.save_checkpoint(self.checkpoint_path)

    def save_checkpoint(self, path: str) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "budget": self.budget,
            "initial_budget": self.initial_budget,
            "stagnation_limit": self.stagnation_limit,
            "refinement_population": self.refinement_population,
            "step_count": self.step_count,
            "metrics": {
                "actions_closed": self.actions_closed,
                "compile_successes": self.compile_successes,
                "correctness_successes": self.correctness_successes,
            },
            "state": {
                "frontier": {
                    action_id: {
                        "action_id": action.action_id,
                        "parent_kernel_id": action.parent_kernel_id,
                        "parent_action_id": action.parent_action_id,
                        "intent": {
                            "title": action.intent.title,
                            "rationale": action.intent.rationale,
                        },
                        "priority_score": action.priority_score,
                        "status": action.status,
                        "attempt_count": action.attempt_count,
                        "best_kernel_id": action.best_kernel_id,
                    }
                    for action_id, action in self.state.frontier.items()
                },
                "closed_actions": sorted(self.state.closed_actions),
                "pruned_actions": sorted(self.state.pruned_actions),
                "history": [
                    {
                        "action_id": h.action_id,
                        "title": h.title,
                        "rationale": h.rationale,
                        "parent_action_id": h.parent_action_id,
                        "parent_kernel_id": h.parent_kernel_id,
                        "num_candidates": h.num_candidates,
                        "best_fitness": h.best_fitness,
                        "best_speedup": h.best_speedup,
                        "had_correct": h.had_correct,
                        "had_compile_error": h.had_compile_error,
                        "stagnation_steps": h.stagnation_steps,
                    }
                    for h in self.state.history
                ],
            },
            "best_kernel_id": self.best_kernel.kernel_id if self.best_kernel else None,
            "records": [self._serialize_record(record) for record in self.records],
        }
        with checkpoint_path.open("w") as handle:
            json.dump(payload, handle, indent=2)

    def _load_checkpoint(self, path: str) -> None:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            return
        with checkpoint_path.open() as handle:
            payload = json.load(handle)

        self.budget = payload.get("budget", self.budget)
        self.initial_budget = payload.get("initial_budget", self.initial_budget)
        self.stagnation_limit = payload.get("stagnation_limit", self.stagnation_limit)
        self.refinement_population = payload.get(
            "refinement_population", self.refinement_population
        )
        self.step_count = payload.get("step_count", 0)
        metrics = payload.get("metrics", {})
        self.actions_closed = metrics.get("actions_closed", 0)
        self.compile_successes = metrics.get("compile_successes", 0)
        self.correctness_successes = metrics.get("correctness_successes", 0)

        state_payload = payload.get("state", {})
        self.state.frontier.clear()
        for action_id, action_data in state_payload.get("frontier", {}).items():
            self.state.frontier[action_id] = FrontierAction(
                action_id=action_data["action_id"],
                parent_kernel_id=action_data.get("parent_kernel_id"),
                intent=ActionIntent(
                    title=action_data["intent"]["title"],
                    rationale=action_data["intent"]["rationale"],
                ),
                priority_score=action_data["priority_score"],
                status=action_data.get("status", "open"),
                attempt_count=action_data.get("attempt_count", 0),
                best_kernel_id=action_data.get("best_kernel_id"),
                parent_action_id=action_data.get("parent_action_id"),
            )
        self.state.closed_actions = set(state_payload.get("closed_actions", []))
        self.state.pruned_actions = set(state_payload.get("pruned_actions", []))

        from kernel_foundry.search.types import ClosedActionSummary
        self.state.history = [
            ClosedActionSummary(
                action_id=h["action_id"],
                title=h["title"],
                rationale=h["rationale"],
                parent_action_id=h.get("parent_action_id"),
                parent_kernel_id=h.get("parent_kernel_id"),
                num_candidates=h["num_candidates"],
                best_fitness=h["best_fitness"],
                best_speedup=h.get("best_speedup"),
                had_correct=h["had_correct"],
                had_compile_error=h["had_compile_error"],
                stagnation_steps=h["stagnation_steps"],
            )
            for h in state_payload.get("history", [])
        ]

        self.records = [self._deserialize_record(record) for record in payload.get("records", [])]

        best_kernel_id = payload.get("best_kernel_id")
        self.best_kernel = None
        if best_kernel_id:
            for record in self.records:
                if record.kernel_id == best_kernel_id:
                    self.best_kernel = record
                    break

        print(f"Loaded checkpoint from {checkpoint_path}")

    def _print_summary(self) -> None:
        open_actions = self.state.open_actions()
        print("\nSummary")
        print(f"  Steps: {self.step_count}")
        print(f"  Evaluations used: {self.initial_budget - self.budget}/{self.initial_budget}")
        print(f"  Frontier: open={len(open_actions)} closed={len(self.state.closed_actions)} pruned={len(self.state.pruned_actions)}")
        print(f"  Compile successes: {self.compile_successes}")
        print(f"  Correct kernels: {self.correctness_successes}")
        if open_actions:
            print("  Top frontier actions:")
            for action in open_actions[:3]:
                print(
                    f"    - {action.action_id} score={action.priority_score:.2f} "
                    f"attempts={action.attempt_count} title={action.intent.title}"
                )

    @staticmethod
    def _serialize_record(record: KernelRecord) -> dict:
        return {
            "kernel_id": record.kernel_id,
            "generation": record.generation,
            "parent_id": record.parent_id,
            "source_code": record.source_code,
            "eval_result": {
                "kernel_id": record.eval_result.kernel_id,
                "compiled": record.eval_result.compiled,
                "correct": record.eval_result.correct,
                "kernel_time_ms": record.eval_result.kernel_time_ms,
                "baseline_time_ms": record.eval_result.baseline_time_ms,
                "speedup": record.eval_result.speedup,
                "fitness": record.eval_result.fitness,
                "error_log": record.eval_result.error_log,
                "profiling_summary": record.eval_result.profiling_summary,
                "ncu_output": record.eval_result.ncu_output,
            },
            "is_templated": record.is_templated,
            "template_configs": record.template_configs,
        }

    @staticmethod
    def _deserialize_record(payload: dict) -> KernelRecord:
        eval_payload = payload["eval_result"]
        return KernelRecord(
            kernel_id=payload["kernel_id"],
            generation=payload["generation"],
            parent_id=payload.get("parent_id"),
            source_code=payload["source_code"],
            eval_result=EvalResult(
                kernel_id=eval_payload["kernel_id"],
                compiled=eval_payload["compiled"],
                correct=eval_payload["correct"],
                kernel_time_ms=eval_payload.get("kernel_time_ms"),
                baseline_time_ms=eval_payload.get("baseline_time_ms"),
                speedup=eval_payload.get("speedup"),
                fitness=eval_payload.get("fitness", 0.0),
                error_log=eval_payload.get("error_log", ""),
                profiling_summary=eval_payload.get("profiling_summary"),
                ncu_output=eval_payload.get("ncu_output"),
            ),
            is_templated=payload.get("is_templated", False),
            template_configs=payload.get("template_configs"),
        )
