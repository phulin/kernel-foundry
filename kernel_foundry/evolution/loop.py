from __future__ import annotations

import json
import traceback
import uuid
from pathlib import Path

import numpy as np

from kernel_foundry.archive.map_elites import MAPElitesArchive
from kernel_foundry.archive.prompt_archive import PromptArchive
from kernel_foundry.classifier.triton_classifier import TritonBehaviorClassifier
from kernel_foundry.config import EvolutionConfig
from kernel_foundry.evaluation.benchmarker import Benchmarker
from kernel_foundry.evaluation.compiler import TritonCompiler
from kernel_foundry.evaluation.correctness import CorrectnessChecker
from kernel_foundry.evaluation.fitness import compute_fitness
from kernel_foundry.evaluation.template_optimizer import TemplateOptimizer
from kernel_foundry.evolution.selector import make_selector
from kernel_foundry.gradient.estimator import GradientEstimator, TransitionBuffer
from kernel_foundry.llm.client import LLMClient
from kernel_foundry.llm.response_parser import extract_triton_code
from kernel_foundry.prompt.constructor import PromptConstructor
from kernel_foundry.prompt.evolvable_sections import DEFAULT_SECTIONS
from kernel_foundry.prompt.meta_prompter import MetaPrompter
from kernel_foundry.task.spec import TaskSpec
from kernel_foundry.types import (
    BehavioralCoords,
    EvalResult,
    KernelRecord,
    TransitionRecord,
)

_SENTINEL_COORDS = BehavioralCoords(0, 0, 0)  # used for seed generation parent


class EvolutionLoop:
    """
    Main orchestrator for the MAP-Elites evolutionary kernel optimization loop.

    Phases:
      1. Seed: generate `population_size` kernels from scratch to bootstrap the archive.
      2. Main: iterative MAP-Elites selection → generation → evaluation → archive update.
      3. Template: post-main autotune sweep for kernels with @triton.autotune decorators.
    """

    def __init__(self, task: TaskSpec, config: EvolutionConfig) -> None:
        self.task = task
        self.config = config
        self.rng = np.random.default_rng()

        # Core components
        self.archive = MAPElitesArchive(bins=config.archive_bins)
        self.prompt_archive = PromptArchive(capacity=config.prompt_archive_size)
        self.transition_buffer = TransitionBuffer(config.transition_buffer_size)
        self.gradient_estimator = GradientEstimator(
            self.transition_buffer,
            weights=(
                config.gradient_weight_fitness,
                config.gradient_weight_improvement,
                config.gradient_weight_exploration,
            ),
            decay=config.gradient_decay,
        )
        self.classifier = TritonBehaviorClassifier()
        self.compiler = TritonCompiler()
        self.benchmarker = Benchmarker(
            warmup_min_time=config.warmup_min_time,
            warmup_min_iters=config.warmup_min_iters,
            benchmark_min_time=config.benchmark_min_time,
            benchmark_min_iters=config.benchmark_min_iters,
            inner_loop_min_time=config.inner_loop_min_time,
        )
        self.checker = CorrectnessChecker(
            reference_fn=task.reference_fn,
            input_generator=task.input_generator,
        )
        self.llm = LLMClient(config)
        self.constructor = PromptConstructor(task)
        self.meta_prompter = MetaPrompter(self.llm)
        self.selector = make_selector(config)
        self.template_optimizer = TemplateOptimizer(
            self.compiler,
            self.checker,
            self.benchmarker,
            baseline_time_ms=task.baseline_time_ms,
            target_speedup=config.target_speedup,
        )

        # State
        self.current_generation = 0
        self.current_sections = DEFAULT_SECTIONS
        self.active_variant_id: str | None = None
        self.all_records: list[KernelRecord] = []

    def run(self) -> KernelRecord | None:
        """Run the full evolutionary loop. Returns the best kernel found."""
        print(f"\n{'='*60}")
        print(f"KernelFoundry: {self.task.name}")
        print(f"Baseline: {self.task.baseline_time_ms:.2f}ms | Target: {self.config.target_speedup}x")
        print(f"Hardware: {self.task.hardware_spec}")
        print(f"{'='*60}\n")

        # Bootstrap prompt archive with defaults
        variant = self.prompt_archive.insert(self.current_sections, generation=0)
        self.active_variant_id = variant.variant_id

        # Phase 1: Seed
        self._run_seed_phase()

        # Phase 2: Main evolution
        for gen in range(1, self.config.max_generations + 1):
            self.current_generation = gen
            self.gradient_estimator.tick(gen)
            self._run_generation(gen)

            if gen % self.config.meta_prompt_freq == 0:
                self._run_meta_prompt_update(gen)

            self._print_progress(gen)

        # Phase 3: Template optimization
        self._run_template_phase()

        best = self.archive.get_best_overall()
        if best:
            print(f"\n{'='*60}")
            speedup_str = f"{best.eval_result.speedup:.2f}x" if best.eval_result.speedup is not None else "N/A"
            print(f"Best kernel: speedup={speedup_str} | coords={best.coords}")
            print(f"{'='*60}")
        return best

    # ------------------------------------------------------------------ phases

    def _run_seed_phase(self) -> None:
        print(f"[Gen 0] Seeding with {self.config.population_size} candidates...")
        prompt = self.constructor.build_seed_prompt(self.current_sections)
        responses = self.llm.generate(prompt, n=self.config.population_size)

        for i, response in enumerate(responses):
            code = extract_triton_code(response)
            if code is None:
                print(f"  Candidate {i}: no valid Triton code extracted")
                continue
            record = self._evaluate_candidate(code, generation=0, parent_id=None)
            inserted = self.archive.insert(record)
            self._record_transition(_SENTINEL_COORDS, record, inserted, generation=0)
            self.all_records.append(record)
            self._print_candidate_result(i, record)

        print(f"  Archive: {self.archive}")

    def _run_generation(self, gen: int) -> None:
        # Select parent
        parent_coords = self.selector.select(self.archive, self.gradient_estimator, self.rng)
        parent_record = self.archive.get_elite(parent_coords) if parent_coords else None
        best_record = self.archive.get_best_overall()

        # Get mutation hints
        hints = []
        if parent_coords and self.archive.size() >= 3:
            fitnesses = {c: self.archive.get_fitness(c) for c in self.archive.get_occupied_cells()}
            hints = self.gradient_estimator.gradient_to_hints(parent_coords, fitnesses)

        # Get active prompt sections
        active_variant = self.prompt_archive.get_best_variant()
        sections = active_variant.sections if active_variant else self.current_sections

        # Build prompt and generate
        prompt = self.constructor.build_generation_prompt(
            parent_kernel=parent_record,
            best_kernel=best_record,
            mutation_hints=hints,
            sections=sections,
        )
        responses = self.llm.generate(prompt, n=self.config.population_size)

        parent_id = parent_record.kernel_id if parent_record else None
        parent_effective_coords = parent_coords or _SENTINEL_COORDS

        for i, response in enumerate(responses):
            code = extract_triton_code(response)
            if code is None:
                continue
            record = self._evaluate_candidate(code, generation=gen, parent_id=parent_id)
            inserted = self.archive.insert(record)
            self._record_transition(parent_effective_coords, record, inserted, generation=gen)
            self.all_records.append(record)

            # Update prompt variant fitness tracking
            if active_variant and record.eval_result.fitness > 0:
                self.prompt_archive.update_fitness(
                    active_variant.variant_id, record.eval_result.fitness
                )

    def _run_meta_prompt_update(self, gen: int) -> None:
        print(f"\n  [Gen {gen}] Running meta-prompt update...")
        recent = self.all_records[-(self.config.meta_prompt_freq * self.config.population_size):]
        active_variant = self.prompt_archive.get_best_variant()
        current_sections = active_variant.sections if active_variant else self.current_sections

        new_sections = self.meta_prompter.update_sections(
            current_sections,
            recent,
            max_mutations=self.config.meta_prompt_max_mutations,
        )
        new_variant = self.prompt_archive.insert(new_sections, generation=gen)
        self.active_variant_id = new_variant.variant_id
        print(f"  [Gen {gen}] New prompt variant {new_variant.variant_id} inserted.")

    def _run_template_phase(self) -> None:
        print(f"\n[Template phase] Checking {self.archive.size()} archive kernels for autotune...")
        improved = 0
        for record in list(self.archive.get_all_elites()):
            if not self.template_optimizer.is_templated(record.source_code):
                continue
            print(f"  Sweeping autotuned kernel at {record.coords}...")
            results = self.template_optimizer.sweep(
                record.source_code, kernel_id_prefix=f"tpl_{record.kernel_id}"
            )
            for tr in results:
                if tr.eval_result.fitness > record.eval_result.fitness:
                    # Update the archive cell with tuned result
                    tuned = KernelRecord(
                        kernel_id=f"tpl_{record.kernel_id}",
                        generation=self.current_generation,
                        parent_id=record.kernel_id,
                        source_code=record.source_code,
                        coords=record.coords,
                        eval_result=tr.eval_result,
                        is_templated=True,
                    )
                    if self.archive.insert(tuned):
                        improved += 1
                        print(f"    Improved: {record.eval_result.speedup:.2f}x → {tr.eval_result.speedup:.2f}x")
        print(f"  Template phase: {improved} archive cells improved.")

    # ------------------------------------------------------------------ core evaluation

    def _evaluate_candidate(
        self, source_code: str, generation: int, parent_id: str | None
    ) -> KernelRecord:
        kernel_id = str(uuid.uuid4())[:8]
        coords = BehavioralCoords(0, 0, 0)
        result = EvalResult(kernel_id=kernel_id, compiled=False, correct=False)

        # Classify behavioral coordinates (static, no GPU needed)
        try:
            coords = self.classifier.classify(source_code)
        except Exception:
            pass

        # Compile
        compile_result = self.compiler.compile(source_code, kernel_id=kernel_id)
        result.compiled = compile_result.success
        if not compile_result.success:
            result.error_log = compile_result.error_log
            result.fitness = compute_fitness(result, self.config.target_speedup)
            return KernelRecord(
                kernel_id=kernel_id,
                generation=generation,
                parent_id=parent_id,
                source_code=source_code,
                coords=coords,
                eval_result=result,
            )

        # Correctness
        correctness = self.checker.check(compile_result.module)
        result.correct = correctness.correct
        if not correctness.correct:
            result.error_log = correctness.error_log
            result.fitness = compute_fitness(result, self.config.target_speedup)
            return KernelRecord(
                kernel_id=kernel_id,
                generation=generation,
                parent_id=parent_id,
                source_code=source_code,
                coords=coords,
                eval_result=result,
            )

        # Benchmark
        try:
            kernel_fn = getattr(compile_result.module, "kernel_fn")
            inputs = self.task.input_generator()
            bench = self.benchmarker.measure(kernel_fn, inputs)
            result.kernel_time_ms = bench.mean_ms
            result.baseline_time_ms = self.task.baseline_time_ms
            result.speedup = self.task.baseline_time_ms / bench.mean_ms
        except Exception:
            result.error_log = traceback.format_exc(limit=6)
            result.fitness = compute_fitness(result, self.config.target_speedup)
            return KernelRecord(
                kernel_id=kernel_id,
                generation=generation,
                parent_id=parent_id,
                source_code=source_code,
                coords=coords,
                eval_result=result,
            )

        result.fitness = compute_fitness(result, self.config.target_speedup)
        return KernelRecord(
            kernel_id=kernel_id,
            generation=generation,
            parent_id=parent_id,
            source_code=source_code,
            coords=coords,
            eval_result=result,
        )

    def _record_transition(
        self,
        parent_coords: BehavioralCoords,
        child: KernelRecord,
        inserted: bool,
        generation: int,
    ) -> None:
        parent_fitness = self.archive.get_fitness(parent_coords)
        delta_f = child.eval_result.fitness - parent_fitness
        if inserted:
            outcome = "improvement"
        elif delta_f >= 0:
            outcome = "neutral"
        else:
            outcome = "regression"

        self.transition_buffer.add(
            TransitionRecord(
                parent_coords=parent_coords,
                child_coords=child.coords,
                delta_fitness=delta_f,
                outcome=outcome,
                generation=generation,
            )
        )

    # ------------------------------------------------------------------ logging

    def _print_candidate_result(self, idx: int, record: KernelRecord) -> None:
        r = record.eval_result
        if not r.compiled:
            status = f"COMPILE ERROR"
        elif not r.correct:
            status = f"INCORRECT"
        else:
            speedup_str = f"{r.speedup:.2f}x" if r.speedup is not None else "N/A"
            status = f"✓ speedup={speedup_str} fitness={r.fitness:.3f}"
        print(f"  [{idx}] {record.coords} → {status}")

    def _print_progress(self, gen: int) -> None:
        best = self.archive.get_best_overall()
        best_str = (
            f"best={best.eval_result.speedup:.2f}x ({best.coords})"
            if best and best.eval_result.speedup
            else "no correct kernel yet"
        )
        print(f"[Gen {gen:3d}] archive={self.archive.size()}/64 | {best_str}")

    # ------------------------------------------------------------------ checkpoint

    def checkpoint(self, path: str) -> None:
        data = {
            "generation": self.current_generation,
            "archive": self.archive.to_dict(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Checkpoint saved to {path}")
