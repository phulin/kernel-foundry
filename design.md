# KernelFoundry: Implementation Design Document

**Source**: arXiv:2603.12440v1 — Wiedemann et al., Intel Corporation, March 2026

---

## Overview

KernelFoundry is an evolutionary framework for LLM-driven GPU kernel optimization. It addresses two failure modes of simple generate-verify-refine loops:

- **Mode collapse**: LLMs converge prematurely on variants of recent successes
- **Context degradation**: growing failure history pollutes the generation context

The system maintains a MAP-Elites archive of kernel implementations organized by three kernel-specific behavioral dimensions, co-evolves meta-prompts to improve generation quality over time, and separates algorithmic from hardware-parameter optimization via kernel templates.

### Goals

1. Generate correct, high-performance GPU kernels from PyTorch reference code, natural language specs, or existing kernels to optimize
2. Support SYCL (primary), CUDA, and Triton backends
3. Scale horizontally: LLM inference, compilation, and GPU execution are independently scalable
4. Enable cross-platform optimization for Intel, NVIDIA, and AMD accelerators

---

## Architecture

```
User Input
    │
    ▼
Task Specification Layer
    │  (PyTorch ref / NL description / existing kernel)
    ▼
Prompt Constructor
    │  (task spec + hardware guidance + parent kernels from archive + gradient hints + evolvable prompt sections)
    ▼
LLM Inference Backend ──────────────────────────────────────────► Meta-Prompter LLM
    │  (cloud API or local vLLM)                                      (every 10 iters)
    │  → 8 candidate kernels per generation                               │
    ▼                                                                     │
Compilation Workers (CPU-only, horizontally scalable)                    │
    │  (DPC++ / nvcc / triton)                                            │
    ▼                                                                     │
Execution Workers (GPU-required)                                          │
    │  (correctness test + benchmark + optional profiling)                │
    ▼                                                                     ▼
MAP-Elites Archive ◄──────────────────────────────────────── Prompt Archive
    │  (64 cells: 4³ behavioral grid, elites by fitness)     (16 prompt variants)
    │
    ▼
Gradient Estimator
    │  (∇F fitness, ∇R improvement rate, ∇E exploration)
    └──────────────────────────────────────────────────────► Prompt Constructor (next iter)
```

Four distributed worker types:
- **LLM Server**: OpenAI/Anthropic REST API or local vLLM instance
- **Compilation Workers**: stateless, GPU-agnostic; scale independently
- **Execution Workers**: one task per GPU; route by backend (Intel/NVIDIA)
- **Database Server**: persists kernels, results, evolutionary state

---

## Core Components

### 1. Task Specification Layer

Accepts three input modalities, all normalized to a common task format:

| Input Type | Description |
|---|---|
| PyTorch reference | Module implementing the operation; baseline timing and correctness oracle |
| Natural language | High-level description; correctness tests generated separately |
| Existing kernel | Starting point for optimization; preserves algorithmic structure |

**Custom task definition** uses three files:
- `config.yaml`: hyperparameters, evolution settings, target hardware
- `task.py`: pytest-compatible build function + correctness tests + performance tests
- `kernel.{sycl,cu,py}`: template with special markers for reference code, user instructions, and optional seed implementation

KernelBench tasks (250 problems across L1 single ops, L2 fusion, L3 full architectures) are natively supported.

### 2. MAP-Elites Archive

The archive is a 4×4×4 = 64-cell grid indexed by three behavioral dimensions derived via static code analysis on the generated kernel. Each cell holds at most one kernel (the elite—highest fitness for that cell).

**Behavioral descriptors** (assigned deterministically via weighted pattern matching):

| Dimension | Levels | Description |
|---|---|---|
| `d_mem` (memory access) | 0–3 | 0: scalar/strided/uncoalesced; 1: coalesced/vectorized; 2: SLM with tiling; 3: multi-level (SLM + reg blocking + prefetch) |
| `d_algo` (algorithm) | 0–3 | 0: direct PyTorch translation; 1: fused ops; 2: reformulated (flash-style); 3: novel/improved asymptotic |
| `d_sync` (parallelism) | 0–3 | 0: embarrassingly parallel; 1: work-group barriers; 2: sub-group primitives (shuffles/reductions); 3: global atomics/multi-pass |

Pattern matching must avoid double-counting (e.g., a `group_barrier` used for SLM synchronization credits `d_mem`, not `d_sync`).

**Fitness function**:
```
f(k) = 0                    if compilation fails
f(k) = 0.1                  if compiles but numerically incorrect
f(k) = 0.5 + 0.5 * s_norm  if correct

s_norm = min(1, speedup / target_speedup)   # target_speedup default: 2.0×
speedup = baseline_time / kernel_time
```

**Selection strategies** (configurable mixing ratios):
- **Uniform**: random from occupied cells (maximizes diversity)
- **Fitness-proportionate**: weighted by `f(k)` (exploits high-fitness regions)
- **Curiosity-driven**: weighted by gradient signal (prioritizes high-improvement-potential cells)
- **Island-based**: K independent sub-populations with migration every M generations

Default: curiosity-driven selection.

**Insertion**: offspring replaces incumbent elite only if `f(offspring) > f(current_elite)`, or cell is empty.

### 3. Gradient Estimator

Maintains a circular buffer of recent parent→child transitions. Each record: `(b_parent, b_child, Δf, outcome, timestamp)` where outcome ∈ {improvement, neutral, regression}.

Three gradient components per occupied cell **b**:

**Fitness gradient** ∇F — which behavioral directions improve fitness:
```
∇_d F ≈ (1/|T|) × Σ_{t∈T} Δf_t · sign(b_c^(d) − b_p^(d)) · w(t)
```
where `w(t)` = exponential time decay (recent transitions weighted higher).

**Improvement-rate gradient** ∇R — probability of improvement independent of magnitude:
```
∇_d R ≈ P(improvement | Δb_d > 0) − P(improvement | Δb_d < 0)
```

**Exploration gradient** ∇E — points toward empty cells and low-fitness regions:
```
∇_b E ∝ Σ_{c∈E} (f_max − f_c) / ‖c − b‖₁ × (c − b) / ‖c − b‖₁
```

**Combined**: `∇_b = 0.4∇F + 0.4∇R + 0.2∇E`

The gradient informs two downstream uses:
1. Parent cell sampling weights (curiosity-driven selection)
2. Natural-language mutation hints injected into the generation prompt

### 4. Prompt Constructor

Assembles the generation prompt from:
- Task specification (reference code + hardware spec)
- Sampled parent kernel(s) from archive
- Gradient-derived mutation hints (natural language)
- Four evolvable prompt sections (managed by meta-prompter)
- Optional profiling feedback from previous evaluations

### 5. LLM Inference Backend

Unified interface over:
- Cloud APIs: OpenAI (GPT-o3, GPT-4.1, GPT-5-mini), Anthropic (Claude Sonnet 4.5)
- Local: vLLM for open-source models

Generates 8 candidate kernels per generation. Default config: temperature 0.3, max\_tokens 8000, top\_p 1.0.

**Multi-LLM strategy for SYCL** (paper's setup):
- Iteration 1: Claude Sonnet 4.5 (bootstrapping to avoid early local minima)
- Iterations 2–40: equal-weight ensemble of GPT-5-mini + GPT-4.1

### 6. Compilation & Evaluation Pipeline

**Compilation workers** (GPU not required):
- SYCL: Intel oneAPI DPC++ 2025.2
- CUDA: NVIDIA CUDA Toolkit 12.8 (nvcc)
- Triton: Triton 3.5

**Execution workers** (GPU required, one task per GPU):
1. Compile output loaded and executed
2. **Correctness**: compare against reference (PyTorch eager) using relative precision `ν = |y − ŷ| / (|y| + ε)`; kernel correct if `ν < 0.01` in ≥99% of outputs
3. **Benchmarking**:
   - Warmup until ≥1s elapsed and ≥10 iterations
   - Main measurement until ≥1s elapsed and ≥10 iterations
   - Inner-loop execution for fast kernels to amortize `synchronize()` overhead
4. **Profiling** (optional): Intel unitrace 2.3 (SYCL) or NVIDIA Nsight Compute 2025.3 (CUDA); structured as natural-language summaries (bandwidth, ALU utilization, memory vs compute bound)

### 7. Template-Based Parameter Optimization

LLM can optionally produce a **templated kernel** with configurable parameters (work-group dimensions, tile sizes, unroll factors) alongside a dispatch function enumerating valid combinations.

Pipeline:
1. Detect template markers in generated code
2. Extract all parameter configurations from dispatch function
3. Evaluate each instantiation independently
4. Kernel fitness = best-performing configuration
5. All results logged for subsequent LLM iteration

Applied for 2 additional iterations after main evolution (best@8 over parameter grid).

### 8. Meta-Prompt Evolution

Every N=10 kernel generations, a dedicated **meta-prompter LLM** (separate from kernel generator) analyzes recent outcomes and updates four evolvable prompt sections:

| Section | Content |
|---|---|
| Optimization Philosophy | High-level principles (e.g., "prioritize memory bandwidth before compute") |
| Optimization Strategies | Concrete techniques with code patterns organized by category (memory / compute / parallelism) |
| Common Pitfalls | Anti-patterns to avoid (e.g., "add SLM padding to prevent bank conflicts") |
| Analysis Guidance | Pre-coding reasoning scaffold for bottleneck identification |

**Meta-prompter input**: current prompt sections + recent kernel code + evaluation metrics (correctness, speedup, error messages).

**Meta-prompter output**: targeted SEARCH/REPLACE diffs restricted to evolvable regions (max 3 mutations per update).

Evolved prompts are maintained in a separate prompt archive (capacity 16); fitness = best kernel performance achieved using that prompt variant. This co-evolutionary loop discovers task-specific optimization strategies without manual engineering.

---

## Evolutionary Loop

```
Initialize: generate 8 seed kernels, populate archive
For generation in 1..40:
    1. Sample parent(s) from archive using selection strategy
    2. Compute gradient hints from transition tracker
    3. Construct prompt (parent code + hints + evolvable sections)
    4. LLM generates 8 candidate kernels
    5. For each candidate:
       a. Compile (compilation worker)
       b. If compiles: correctness test + benchmark (execution worker)
       c. Classify behavioral coordinates (static analysis)
       d. Compute fitness
       e. Update archive if elite improves
       f. Log transition (b_parent, b_child, Δf, outcome)
    6. If generation % 10 == 0: run meta-prompter, update prompt archive
After generation 40:
    For top K kernels with templates:
        Sweep parameter grid (2 iterations of best@8)
        Update archive with best parameter configuration
Return: best kernel from archive
```

---

## Correctness & Evaluation Metrics

| Metric | Definition |
|---|---|
| Correctness rate | Fraction of tasks with compiling, numerically correct kernels |
| fast_p | Fraction of tasks with speedup > p× baseline |
| Average speedup | Mean(speedup) across tasks |
| Geometric speedup | Geometric mean of speedups |
| Hardware-speedup (hws) | t_A(k^B) / t_A(k^A) — cross-hardware evaluation |

Correctness threshold: relative precision `ν < 0.01` in ≥99% of outputs. Stricter than KernelBench's default absolute threshold (10⁻²), which admits false positives.

---

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `max_generations` | 40 | Main evolution iterations |
| `population_size` | 8 | Candidates per generation |
| `selection_strategy` | curiosity | Parent selection method |
| `archive_bins` | 4 | Bins per behavioral dimension (4³ = 64 cells) |
| `target_speedup` | 2.0× | Speedup normalization target |
| `warmup_min_time` | 1s | Min total warmup time |
| `warmup_min_iters` | 10 | Min warmup iterations |
| `benchmark_min_time` | 1s | Min main measurement time |
| `benchmark_min_iters` | 10 | Min timing iterations |
| `meta_prompt_freq` | 10 | Kernel generations between prompt updates |
| `meta_prompt_max_mutations` | 3 | Max SEARCH/REPLACE diffs per update |
| `prompt_archive_size` | 16 | Prompt variants retained |
| `llm_temperature` | 0.3 | Generation temperature |
| `llm_max_tokens` | 8000 | Max tokens per response |
| `llm_top_p` | 1.0 | Nucleus sampling parameter |
| `gradient_weights` | (0.4, 0.4, 0.2) | (∇F, ∇R, ∇E) combination weights |

---

## Open Questions / Tradeoffs

**Behavioral descriptor completeness**: The 3D grid (64 cells) may under-represent some optimization strategies. A 4th dimension (e.g., `d_compute` for vectorization / tensor core usage) would double the archive size to 256 cells. Worth prototyping on L3 full-architecture tasks where current coverage may be sparse.

**Static vs dynamic classification**: Behavioral coordinates are assigned via static pattern matching, which is fast and deterministic but can misclassify kernels with unusual patterns. An alternative: runtime profiler counters (e.g., SLM hit rate) as behavioral descriptors, though this adds execution overhead.

**Meta-prompter model choice**: The paper uses a separate LLM for meta-prompting. Whether to use the same model (cheaper, simpler) or a stronger model (better diagnosis) depends on budget. Using a stronger model for meta-prompting and a cheaper one for kernel generation may be the best tradeoff.

**Template parameter space coverage**: The dispatch function approach assumes the LLM generates a finite, enumerable parameter grid. Large grids (e.g., all tile-size combinations) are costly to sweep exhaustively. Consider Bayesian optimization or random sampling over the parameter space instead of exhaustive evaluation.

**Island migration policy**: The paper mentions island-based selection with configurable K and M but doesn't specify a migration policy (random elite, fitness-proportionate, behavioral-distance-maximizing). Behavioral-distance-maximizing migration would help cross-pollinate diverse strategies.

**Cross-hardware generalization**: The hardware-speedup metric (hws) shows kernels generalize partially across Intel GPUs. Whether kernels optimized for one vendor generalize to another (Intel → NVIDIA) is not evaluated. Given SYCL's portability goal, this would be a valuable experiment.
