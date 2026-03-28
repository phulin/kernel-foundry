import json
from pathlib import Path
from types import SimpleNamespace

from kernel_foundry.llm.response_parser import extract_json_payload
from kernel_foundry.search.types import (
    ActionIntent,
    ActionTrajectory,
    FrontierAction,
    SearchState,
)
from kernel_foundry.search.world_model import WorldModelPlanner, parse_planner_update
from tests.conftest import make_record


def test_extract_json_payload_accepts_fenced_json():
    payload = extract_json_payload('```json\n{"summary":"ok","directives":[]}\n```')
    assert json.loads(payload) == {"summary": "ok", "directives": []}


def test_parse_planner_update_parses_directives():
    update = parse_planner_update(
        """
```json
{
  "summary": "deepen the promising branch",
  "directives": [
    {
      "kind": "insert",
      "action_id": "a2",
      "title": "Fuse reduction",
      "rationale": "the current branch is correct",
      "parent_kernel_id": "best1",
      "priority_score": 0.8
    }
  ]
}
```
"""
    )
    assert update.summary == "deepen the promising branch"
    assert len(update.directives) == 1
    assert update.directives[0].kind == "insert"
    assert update.directives[0].priority_score == 0.8


def test_world_model_planner_update_state_applies_insert_and_prune():
    planner = WorldModelPlanner(SimpleNamespace(generate_single=lambda *args, **kwargs: ""), SimpleNamespace())
    state = SearchState(
        frontier={
            "a1": FrontierAction(
                action_id="a1",
                parent_kernel_id=None,
                intent=ActionIntent(title="Baseline", rationale="start"),
                priority_score=0.9,
            ),
            "a_old": FrontierAction(
                action_id="a_old",
                parent_kernel_id=None,
                intent=ActionIntent(title="Old", rationale="stale"),
                priority_score=0.3,
            ),
        }
    )
    action = state.frontier["a1"]
    trajectory = ActionTrajectory(action=action, candidates=[make_record(1, 1, 1, kid="best1")])

    planner._request_update = lambda **kwargs: parse_planner_update(
        """
        {
          "summary": "follow the good kernel",
          "directives": [
            {
              "kind": "insert",
              "action_id": "a2",
              "title": "Tile and pipeline",
              "rationale": "build on the correct kernel",
              "parent_kernel_id": "best1",
              "priority_score": 0.85
            },
            {
              "kind": "prune",
              "action_id": "a_old"
            }
          ]
        }
        """
    )

    update = planner.update_state(
        state,
        action=action,
        trajectory=trajectory,
        best_kernel=trajectory.best_candidate,
        budget_remaining=10,
    )

    assert update.summary == "follow the good kernel"
    assert state.frontier["a1"].status == "closed"
    assert state.frontier["a2"].intent.title == "Tile and pipeline"
    assert state.frontier["a2"].parent_kernel_id == "best1"
    assert state.frontier["a2"].parent_action_id == "a1"
    assert state.frontier["a_old"].status == "pruned"
    assert len(state.history) == 1
    assert state.history[0].parent_action_id is None


def test_world_model_loop_checkpoint_round_trip(tmp_path):
    from kernel_foundry.search.world_model_loop import WorldModelSearchLoop
    from kernel_foundry.search.types import ClosedActionSummary

    loop = WorldModelSearchLoop.__new__(WorldModelSearchLoop)
    loop.budget = 7
    loop.initial_budget = 10
    loop.stagnation_limit = 2
    loop.refinement_population = 1
    loop.step_count = 3
    loop.actions_closed = 1
    loop.compile_successes = 2
    loop.correctness_successes = 1
    best_record = make_record(1, 1, 1, speedup=2.0, kid="k1")
    loop.best_kernel = best_record
    loop.state = SearchState(
        frontier={
            "a1": FrontierAction(
                action_id="a1",
                parent_kernel_id="k1",
                intent=ActionIntent(title="Baseline", rationale="start"),
                priority_score=0.9,
                parent_action_id="a0",
            )
        },
        closed_actions={"a0"},
        pruned_actions={"ax"},
        history=[
            ClosedActionSummary(
                action_id="a0",
                title="Initial baseline",
                rationale="start",
                parent_action_id=None,
                parent_kernel_id=None,
                num_candidates=3,
                best_fitness=0.9,
                best_speedup=2.0,
                had_correct=True,
                had_compile_error=False,
                stagnation_steps=2,
            )
        ],
    )
    loop.records = [best_record]

    checkpoint = tmp_path / "world_model_checkpoint.json"
    WorldModelSearchLoop.save_checkpoint(loop, str(checkpoint))

    restored = WorldModelSearchLoop.__new__(WorldModelSearchLoop)
    restored.budget = 0
    restored.initial_budget = 0
    restored.stagnation_limit = 0
    restored.refinement_population = 0
    restored.step_count = 0
    restored.actions_closed = 0
    restored.compile_successes = 0
    restored.correctness_successes = 0
    restored.best_kernel = None
    restored.state = SearchState()
    restored.records = []
    restored.planner = SimpleNamespace(
        make_action_intent=lambda title, rationale: ActionIntent(title=title, rationale=rationale)
    )

    WorldModelSearchLoop._load_checkpoint(restored, str(checkpoint))

    assert restored.budget == 7
    assert restored.initial_budget == 10
    assert restored.step_count == 3
    assert restored.state.frontier["a1"].intent.title == "Baseline"
    assert restored.state.frontier["a1"].parent_kernel_id == "k1"
    assert restored.state.frontier["a1"].parent_action_id == "a0"
    assert restored.records[0].kernel_id == "k1"
    assert restored.best_kernel is not None
    assert restored.best_kernel.kernel_id == "k1"
    assert len(restored.state.history) == 1
    assert restored.state.history[0].action_id == "a0"
    assert restored.state.history[0].parent_action_id is None
    assert restored.state.history[0].best_fitness == 0.9


def test_world_model_loop_flush_records_includes_ncu_output(tmp_path):
    from kernel_foundry.search.world_model_loop import WorldModelSearchLoop

    loop = WorldModelSearchLoop.__new__(WorldModelSearchLoop)
    record = make_record(1, 1, 1, speedup=2.0, kid="k1")
    record.source_code = "def kernel_fn(x):\n    return x\n"
    record.eval_result.error_log = "compile details"
    record.eval_result.profiling_summary = "occupancy=0.5"
    record.eval_result.ncu_output = "sm__throughput.avg.pct_of_peak_sustained_elapsed=72.1"
    loop.records = [record]
    loop.records_path = str(tmp_path / "records.jsonl")

    WorldModelSearchLoop._flush_records(loop)

    payload = json.loads(Path(loop.records_path).read_text().strip())
    assert payload["kernel_id"] == "k1"
    assert payload["source_code"] == "def kernel_fn(x):\n    return x\n"
    assert payload["eval_result"]["profiling_summary"] == "occupancy=0.5"
    assert payload["eval_result"]["ncu_output"] == "sm__throughput.avg.pct_of_peak_sustained_elapsed=72.1"


def test_world_model_update_prompt_includes_ncu_output():
    planner = WorldModelPlanner(
        SimpleNamespace(generate_single=lambda *args, **kwargs: ""),
        SimpleNamespace(),
    )
    candidate = make_record(1, 1, 1, speedup=2.0, kid="cand1")
    candidate.eval_result.profiling_summary = "occupancy=0.5"
    candidate.eval_result.ncu_output = "Kernel: fused\n  SM Throughput: 72.1 %"
    best = make_record(2, 2, 2, speedup=2.5, kid="best1")
    best.eval_result.profiling_summary = "occupancy=0.7"
    best.eval_result.ncu_output = "Kernel: best\n  Memory Throughput: 81.0 %"

    state = SearchState()
    planner._current_state = state
    prompt = planner._build_update_prompt(
        action=FrontierAction(
            action_id="a1",
            parent_kernel_id="cand1",
            intent=ActionIntent(title="Tile", rationale="test"),
            priority_score=0.8,
        ),
        trajectory=ActionTrajectory(action=state.frontier.get("a1") or FrontierAction(
            action_id="a1",
            parent_kernel_id="cand1",
            intent=ActionIntent(title="Tile", rationale="test"),
            priority_score=0.8,
        ), candidates=[candidate]),
        best_kernel=best,
        budget_remaining=7,
    )

    assert "profiling: occupancy=0.5" in prompt
    assert "Kernel: fused" in prompt
    assert "SM Throughput: 72.1 %" in prompt
    assert "profiling: occupancy=0.7" in prompt
    assert "Kernel: best" in prompt
