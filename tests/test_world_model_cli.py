from types import SimpleNamespace

import main_world_model


def test_world_model_cli_constructs_loop(monkeypatch):
    seen = {}

    fake_task = SimpleNamespace(
        name="fake_task",
        baseline_time_ms=1.0,
        build=lambda config: None,
    )

    class FakeLoop:
        def __init__(
            self,
            task,
            config,
            budget,
            stagnation_limit,
            refinement_population,
            records_path,
            checkpoint_path,
        ):
            seen["task"] = task
            seen["budget"] = budget
            seen["stagnation_limit"] = stagnation_limit
            seen["refinement_population"] = refinement_population
            seen["records_path"] = records_path
            seen["checkpoint_path"] = checkpoint_path

        def run(self):
            seen["ran"] = True
            return None

    monkeypatch.setattr(main_world_model.importlib, "import_module", lambda path: SimpleNamespace(build=lambda config: fake_task))
    monkeypatch.setattr(main_world_model, "WorldModelSearchLoop", FakeLoop)

    try:
        main_world_model.cli(
            [
                "softmax",
                "--budget",
                "12",
                "--stagnation-limit",
                "4",
                "--refinement-population",
                "3",
                "--records",
                "tmp.world_model",
                "--checkpoint",
                "tmp.world_model.checkpoint.json",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 1

    assert seen["task"] is fake_task
    assert seen["budget"] == 12
    assert seen["stagnation_limit"] == 4
    assert seen["refinement_population"] == 3
    assert seen["records_path"] == "tmp.world_model"
    assert seen["checkpoint_path"] == "tmp.world_model.checkpoint.json"
    assert seen["ran"] is True
