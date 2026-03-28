from __future__ import annotations

from kernel_foundry.llm.client import LLMClient
from kernel_foundry.prompt.constructor import PromptConstructor
from kernel_foundry.search.types import FrontierAction
from kernel_foundry.task.spec import TaskSpec
from kernel_foundry.types import EvolvableSections, KernelRecord


class ActionInstantiator:
    """
    Converts a selected optimization intent into a code-generation prompt and
    requests concrete Triton implementations.
    """

    def __init__(
        self,
        llm: LLMClient,
        task: TaskSpec,
        sections: EvolvableSections,
    ) -> None:
        self._llm = llm
        self._task = task
        self._sections = sections
        self._constructor = PromptConstructor(task)

    def generate_candidates(
        self,
        action: FrontierAction,
        parent_kernel: KernelRecord | None,
        best_kernel: KernelRecord | None,
        n: int,
    ) -> list[str]:
        prompt = self.build_prompt(action, parent_kernel, best_kernel)
        return self._llm.generate(prompt, n=n)

    def build_prompt(
        self,
        action: FrontierAction,
        parent_kernel: KernelRecord | None,
        best_kernel: KernelRecord | None,
    ) -> str:
        base = self._constructor.build_generation_prompt(
            parent_kernel=parent_kernel,
            best_kernel=best_kernel,
            mutation_hints=[],
            sections=self._sections,
        )
        intent_block = (
            "## Selected Optimization Intent\n"
            f"Action ID: {action.action_id}\n"
            f"Intent: {action.intent.title}\n"
            f"Rationale: {action.intent.rationale}\n"
        )
        return (
            f"{base}\n\n{intent_block}\n"
            "Implement this intent on top of the chosen parent kernel when appropriate. "
            "Keep the code self-contained and respond with a single Python code block."
        )
