from __future__ import annotations

from kernel_foundry.types import EvolvableSections

DEFAULT_SECTIONS = EvolvableSections(
    optimization_philosophy="""\
Prioritize memory bandwidth utilization before compute optimization.
Maximize data reuse through blocking and tiling.
Match parallelism granularity to the target hardware's warp size (32 threads for NVIDIA).
Profile before optimizing: identify whether the kernel is memory-bound or compute-bound first.""",

    optimization_strategies="""\
Memory:
- Use tl.load with BLOCK_SIZE-aligned offsets (tl.arange) for coalesced access.
- Implement tiling with tl.dot for matrix operations to exploit register file capacity.
- Use num_stages > 1 in triton.Config for software pipelining on Ampere+ GPUs.
- Set eviction_policy='evict_last' for frequently reused data.

Compute:
- Fuse elementwise operations (e.g., bias add + activation) into the same kernel pass.
- Use tl.math.exp, tl.math.log, etc. for hardware-accelerated transcendentals.
- Accumulate in fp32 even when inputs are fp16 to avoid numerical issues.

Parallelism:
- Use tl.reduce for intra-block reductions to leverage warp-level primitives.
- Reserve tl.atomic_add for global reductions that cannot be blocked.
- Tune BLOCK_SIZE to balance occupancy vs. register pressure (128–256 is common).""",

    common_pitfalls="""\
- Do not use Python-side loops over individual elements; always operate on tiles.
- Always apply a mask when the tensor size is not a multiple of BLOCK_SIZE:
    mask = offsets < n; tl.load(ptr + offsets, mask=mask, other=0.0)
- Do not mix tl.constexpr and runtime values inappropriately.
- Avoid reading the same global memory location multiple times; load once, reuse.
- Numerical overflow: when computing softmax, subtract the running max before tl.exp.
- Incorrect output shape: ensure the output tensor is allocated before calling the kernel.""",

    analysis_guidance="""\
Before writing code:
1. Identify the memory access pattern: does each thread read independent data (coalesced)
   or do threads in a block read overlapping data (tiling is beneficial)?
2. Estimate arithmetic intensity = FLOPs / bytes_loaded. If < ~10, it is memory-bound.
3. For reductions (softmax, layer norm, sum): decide between single-pass (online) or
   two-pass algorithms based on whether intermediate state fits in registers.
4. Choose BLOCK_SIZE to be a power of 2; start with 128 and tune from there.
5. Only add tl.dot if the operation has a natural matrix-multiplication structure.""",
)


def apply_search_replace(
    sections: EvolvableSections, diffs: list[tuple[str, str]]
) -> EvolvableSections:
    """
    Apply a list of (search, replace) string patches to the evolvable sections.
    Searches across all four fields. Silently skips diffs where search text is not found.
    Returns a new EvolvableSections instance.
    """
    fields = {
        "optimization_philosophy": sections.optimization_philosophy,
        "optimization_strategies": sections.optimization_strategies,
        "common_pitfalls": sections.common_pitfalls,
        "analysis_guidance": sections.analysis_guidance,
    }
    for search, replace in diffs:
        applied = False
        for field_name, text in fields.items():
            if search in text:
                fields[field_name] = text.replace(search, replace, 1)
                applied = True
                break
        if not applied:
            print(f"  [meta-prompt] SEARCH text not found, skipping diff: {search[:60]!r}...")
    return EvolvableSections(**fields)
