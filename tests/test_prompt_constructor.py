from kernel_foundry.prompt.constructor import PromptConstructor
from kernel_foundry.prompt.evolvable_sections import DEFAULT_SECTIONS
from kernel_foundry.task.spec import TaskSpec


def test_prompt_mentions_optional_sequential_multi_kernel_strategy():
    task = TaskSpec(
        name="dummy",
        description="Dummy task",
        reference_code="def reference_fn(x):\n    return x",
        reference_fn=lambda x: x,
        input_generator=lambda: tuple(),
        hardware_spec="Dummy GPU",
        baseline_time_ms=1.0,
    )

    prompt = PromptConstructor(task).build_seed_prompt(DEFAULT_SECTIONS)

    assert "one kernel or as multiple Triton kernels launched sequentially" in prompt
