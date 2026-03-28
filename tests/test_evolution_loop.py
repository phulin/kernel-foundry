from types import SimpleNamespace

from kernel_foundry.evolution.loop import EvolutionLoop
from tests.conftest import make_record


class _FakeArchive:
    def __init__(self, best_record):
        self.best_record = best_record
        self.inserted = []

    def get_best_overall(self):
        return self.best_record

    def insert(self, record):
        self.inserted.append(record)
        return True


def test_insert_if_correct_rejects_incorrect_kernel():
    loop = EvolutionLoop.__new__(EvolutionLoop)
    loop.archive = _FakeArchive(make_record(0, 0, 0))

    incorrect = make_record(1, 1, 1, correct=False)

    assert loop._insert_if_correct(incorrect) is False
    assert loop.archive.inserted == []


def test_run_template_generation_phase_uses_template_prompt_and_inserts_correct(monkeypatch):
    best = make_record(1, 0, 0, speedup=2.0, kid="best")
    generated = make_record(2, 1, 0, speedup=2.5, kid="templated")

    loop = EvolutionLoop.__new__(EvolutionLoop)
    loop.config = SimpleNamespace(template_opt_iterations=1, template_opt_population=1)
    loop.current_generation = 7
    loop.archive = _FakeArchive(best)
    loop.all_records = []

    seen = {
        "prompt": None,
        "parent_id": None,
        "transitions": [],
        "results": [],
    }

    loop.constructor = SimpleNamespace(
        build_template_prompt=lambda record: seen.__setitem__("prompt", record.kernel_id) or "PROMPT"
    )
    loop.llm = SimpleNamespace(generate=lambda prompt, n: ["```python\npass\n```"])
    loop._evaluate_candidate = lambda code, generation, parent_id: (
        seen.__setitem__("parent_id", parent_id) or generated
    )
    loop._record_transition = lambda parent_coords, record, inserted, generation: seen["transitions"].append(
        (parent_coords, record.kernel_id, inserted, generation)
    )
    loop._print_candidate_result = lambda idx, record: seen["results"].append((idx, record.kernel_id))
    loop._flush_records = lambda: None

    monkeypatch.setattr("kernel_foundry.evolution.loop.extract_triton_code", lambda response: "generated_code")

    EvolutionLoop._run_template_generation_phase(loop)

    assert seen["prompt"] == "best"
    assert seen["parent_id"] == "best"
    assert loop.archive.inserted == [generated]
    assert loop.all_records == [generated]
    assert seen["transitions"] == [(best.coords, "templated", True, 7)]
    assert seen["results"] == [("tpl-0.0", "templated")]
