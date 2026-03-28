from __future__ import annotations

from kernel_foundry.task.spec import TaskSpec
from kernel_foundry.types import EvolvableSections, KernelRecord


class PromptConstructor:
    """Assembles generation prompts for the kernel-writing LLM."""

    SYSTEM_PROMPT = """\
You are an expert GPU kernel engineer specializing in Triton kernels for PyTorch operations.
Your task is to write optimized Triton kernels that are faster than the PyTorch reference implementation.

REQUIREMENTS:
1. Your response must contain exactly ONE Python code block (```python ... ```)
2. The code must define a function named `kernel_fn` with the same signature as the reference
3. `kernel_fn` must return a tensor (or tuple of tensors) matching the reference output shape/dtype
4. Use `@triton.jit` for GPU kernels
5. The code must be self-contained: include all necessary imports
6. The code must be correct before being fast
7. You may implement the solution as one kernel or as multiple Triton kernels launched sequentially inside `kernel_fn` if decomposition improves correctness, simplicity, or performance

BASIC TRITON RULES:
- Every Triton device kernel must be decorated with `@triton.jit`
- Do not index Triton tensors/register blocks with dynamic Python-style expressions like `x[i]`, `x[i, j]`, `x[:, idx]`, or `x[idx, :]`
- Access memory through pointer arithmetic plus `tl.load`/`tl.store`, or through `tl.make_block_ptr` plus block loads/stores
- If you need sub-blocks or rows/cols, form pointer arrays or block pointers explicitly; do not treat Triton values like PyTorch/NumPy tensors
- Only call `tl.load` on pointers or block pointers, never on values already loaded into registers
- Prefer masked pointer loads/stores or block-pointer boundary checks for tail handling

TEMPLATE FORMAT (for parameter sweeping):
If you want to tune hardware parameters, use @triton.autotune:
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['n']
)
@triton.jit
def my_kernel(..., BLOCK_SIZE: tl.constexpr):
    ...
```"""

    def __init__(self, task: TaskSpec) -> None:
        self._task = task

    def build_seed_prompt(self, sections: EvolvableSections) -> str:
        """First-generation prompt: no parent kernel, just the reference."""
        return self._assemble(
            parent_kernel=None,
            best_kernel=None,
            mutation_hints=[],
            sections=sections,
            is_seed=True,
        )

    def build_generation_prompt(
        self,
        parent_kernel: KernelRecord | None,
        best_kernel: KernelRecord | None,
        mutation_hints: list[str],
        sections: EvolvableSections,
    ) -> str:
        return self._assemble(
            parent_kernel=parent_kernel,
            best_kernel=best_kernel,
            mutation_hints=mutation_hints,
            sections=sections,
            is_seed=False,
        )

    def build_template_prompt(self, best_kernel: KernelRecord) -> str:
        """Prompt specifically for adding @triton.autotune parameter sweep."""
        return f"""{self.SYSTEM_PROMPT}

## Task
{self._task.description}

## Current Best Kernel (speedup: {f"{best_kernel.eval_result.speedup:.2f}x" if best_kernel.eval_result.speedup is not None else "N/A"})
```python
{best_kernel.source_code}
```

## Your Task
Modify the above kernel to use `@triton.autotune` with at least 4 different
`triton.Config` entries varying BLOCK_SIZE, num_warps, and/or num_stages.
The autotuner will automatically select the best configuration for this hardware.

Keep the same algorithmic approach but tune the hardware parameters.
Make sure `kernel_fn` still has the same external interface.

Respond with a single Python code block containing the autotuned version."""

    # ------------------------------------------------------------------ private

    def _assemble(
        self,
        parent_kernel: KernelRecord | None,
        best_kernel: KernelRecord | None,
        mutation_hints: list[str],
        sections: EvolvableSections,
        is_seed: bool,
    ) -> str:
        parts = [self.SYSTEM_PROMPT, ""]

        # Task description
        parts.append(f"## Task: {self._task.name}")
        if self._task.description:
            parts.append(self._task.description)
        parts.append("")

        # Hardware
        parts.append(f"## Target Hardware\n{self._task.hardware_spec}")
        parts.append("")

        # Reference implementation
        parts.append("## Reference Implementation (PyTorch)")
        parts.append("```python")
        parts.append(self._task.reference_code)
        parts.append("```")
        parts.append("")

        # Best kernel so far (if any)
        if best_kernel and not is_seed:
            speedup_str = (
                f"{best_kernel.eval_result.speedup:.2f}x"
                if best_kernel.eval_result.speedup is not None
                else "unknown"
            )
            parts.append(f"## Current Best Kernel (speedup: {speedup_str})")
            parts.append(f"Behavioral coords: {best_kernel.coords}")
            if best_kernel.eval_result.profiling_summary:
                parts.append(f"Profiling: {best_kernel.eval_result.profiling_summary}")
            if best_kernel.template_configs:
                parts.append(self._format_template_configs(best_kernel.template_configs))
            parts.append("```python")
            parts.append(best_kernel.source_code)
            parts.append("```")
            parts.append("")

        # Parent kernel to mutate (if different from best)
        if parent_kernel and not is_seed and parent_kernel != best_kernel:
            speedup_str = (
                f"{parent_kernel.eval_result.speedup:.2f}x"
                if parent_kernel.eval_result.speedup is not None
                else "unknown"
            )
            parent_status = (
                f"speedup: {speedup_str}"
                if parent_kernel.eval_result.correct
                else f"INCORRECT — {parent_kernel.eval_result.error_log[:200]}"
            )
            parts.append(f"## Parent Kernel to Improve ({parent_status})")
            parts.append(f"Behavioral coords: {parent_kernel.coords}")
            if parent_kernel.eval_result.profiling_summary:
                parts.append(f"Profiling: {parent_kernel.eval_result.profiling_summary}")
            if parent_kernel.template_configs:
                parts.append(self._format_template_configs(parent_kernel.template_configs))
            parts.append("```python")
            parts.append(parent_kernel.source_code)
            parts.append("```")
            parts.append("")

        # Gradient mutation hints
        if mutation_hints:
            parts.append("## Suggested Optimization Directions")
            for hint in mutation_hints:
                parts.append(f"- {hint}")
            parts.append("")

        # Evolvable sections
        parts.append("## Optimization Philosophy")
        parts.append(sections.optimization_philosophy)
        parts.append("")

        parts.append("## Optimization Strategies")
        parts.append(sections.optimization_strategies)
        parts.append("")

        parts.append("## Common Pitfalls to Avoid")
        parts.append(sections.common_pitfalls)
        parts.append("")

        parts.append("## Analysis Guidance")
        parts.append(sections.analysis_guidance)
        parts.append("")

        # Final instruction
        if is_seed:
            parts.append(
                "## Your Task\n"
                "Write an optimized Triton kernel for this operation. "
                "Start with a correct implementation, then optimize for speed. "
                "Respond with a single Python code block."
            )
        else:
            parts.append(
                "## Your Task\n"
                "Write an improved Triton kernel. You may:\n"
                "- Mutate the parent kernel using the suggested directions\n"
                "- Explore a completely different optimization strategy\n"
                "- Combine ideas from the parent and best kernels\n"
                "- Split the work across multiple Triton kernels and launch them sequentially from `kernel_fn` if that is advantageous\n"
                "Correctness is required. Respond with a single Python code block."
            )

        return "\n".join(parts)

    @staticmethod
    def _format_template_configs(configs: list[dict]) -> str:
        lines = ["Config sweep results (✓=correct, ✗=incorrect):"]
        for cfg in configs[:8]:  # cap to avoid token bloat
            correct = "✓" if cfg.get("correct") else "✗"
            speedup = f"{cfg['speedup']:.2f}x" if cfg.get("speedup") else "N/A"
            raw = cfg.get("config", {})
            kwargs = raw.get("kwargs") or {}
            params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            if raw.get("num_warps") is not None:
                params += f", num_warps={raw['num_warps']}"
            if raw.get("num_stages") is not None:
                params += f", num_stages={raw['num_stages']}"
            lines.append(f"  {correct} {params} → {speedup}")
        return "\n".join(lines)
