from __future__ import annotations

from kernel_foundry.llm.client import LLMClient
from kernel_foundry.llm.response_parser import extract_search_replace_diffs
from kernel_foundry.prompt.evolvable_sections import apply_search_replace
from kernel_foundry.types import EvolvableSections, KernelRecord


class MetaPrompter:
    """
    A separate LLM call that analyzes recent kernel generation outcomes and proposes
    targeted updates to the evolvable prompt sections.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def update_sections(
        self,
        current_sections: EvolvableSections,
        recent_records: list[KernelRecord],
        max_mutations: int = 3,
    ) -> EvolvableSections:
        """
        Analyze recent outcomes, propose prompt diffs, apply them.
        Returns updated EvolvableSections.
        """
        if not recent_records:
            return current_sections

        prompt = self._build_meta_prompt(current_sections, recent_records)
        response = self._llm.generate_single(
            prompt,
            model=self._llm._config.meta_prompter_model,
            temperature=0.5,  # slightly more creative for diagnosis
        )
        diffs = extract_search_replace_diffs(response)[:max_mutations]
        if not diffs:
            print("  [meta-prompt] No diffs found in response.")
            return current_sections

        print(f"  [meta-prompt] Applying {len(diffs)} prompt update(s).")
        return apply_search_replace(current_sections, diffs)

    def _build_meta_prompt(
        self,
        sections: EvolvableSections,
        recent_records: list[KernelRecord],
    ) -> str:
        outcome_lines = []
        for r in recent_records[-20:]:  # cap to last 20 to save tokens
            if not r.eval_result.compiled:
                status = f"COMPILE ERROR: {r.eval_result.error_log[:150]}"
            elif not r.eval_result.correct:
                status = f"INCORRECT: {r.eval_result.error_log[:150]}"
            else:
                status = f"correct, speedup={r.eval_result.speedup:.2f}x"
            outcome_lines.append(f"  - Gen {r.generation} [{r.coords}]: {status}")

        outcomes_text = "\n".join(outcome_lines)

        return f"""You are an expert prompt engineer for GPU kernel optimization.
You will analyze recent kernel generation outcomes and improve the optimization guidance.

## Current Optimization Guidance (Evolvable Sections)

### Optimization Philosophy
{sections.optimization_philosophy}

### Optimization Strategies
{sections.optimization_strategies}

### Common Pitfalls
{sections.common_pitfalls}

### Analysis Guidance
{sections.analysis_guidance}

## Recent Kernel Generation Outcomes
{outcomes_text}

## Your Task
Analyze the outcomes above. Identify patterns of failure or missed opportunities:
- Are there recurring compile errors suggesting missing guidance?
- Are there correctness failures suggesting a common mistake not covered by pitfalls?
- Are the speedups plateauing? What optimization technique might unlock further gains?

Propose up to 3 targeted text edits to the guidance sections above.
Use this EXACT format for each edit:

<<<SEARCH>>>
exact text to find (copy verbatim from the sections above)
<<<REPLACE>>>
improved replacement text
<<<END>>>

Only edit text that exists verbatim in the sections. Do not add new sections.
Focus on the most impactful change first."""
