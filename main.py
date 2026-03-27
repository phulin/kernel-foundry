#!/usr/bin/env python3
"""
KernelFoundry CLI

Usage:
  python main.py softmax
  python main.py matmul --generations 20 --population 4
  python main.py softmax --output results/softmax.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from kernel_foundry.config import EvolutionConfig
from kernel_foundry.evolution.loop import EvolutionLoop

BUILTIN_TASKS = {
    "softmax": "tasks.softmax",
    "matmul": "tasks.matmul",
    "causal_conv1d": "tasks.causal_conv1d",
}


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="KernelFoundry: evolutionary Triton kernel optimization"
    )
    parser.add_argument(
        "task",
        help=f"Task name ({', '.join(BUILTIN_TASKS)}) or Python module path",
    )
    parser.add_argument(
        "--generations", type=int, default=None, help="Override max_generations"
    )
    parser.add_argument(
        "--population", type=int, default=None, help="Override population_size"
    )
    parser.add_argument("--model", default=None, help="Override LLM model")
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    parser.add_argument(
        "--records",
        default="records",
        help="Append all evaluated kernels as JSONL to this path",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Save/load checkpoint JSON at this path",
    )
    parser.add_argument(
        "--strategy",
        choices=["uniform", "fitness", "curiosity", "island", "mixed"],
        default=None,
        help="Override selection strategy",
    )
    args = parser.parse_args()

    # Build config
    config = EvolutionConfig()
    if args.generations is not None:
        config.max_generations = args.generations
    if args.population is not None:
        config.population_size = args.population
    if args.model is not None:
        config.llm_model = args.model
    if args.strategy is not None:
        config.selection_strategy = args.strategy

    # Load task
    task_module_path = BUILTIN_TASKS.get(args.task, args.task)
    try:
        task_module = importlib.import_module(task_module_path)
    except ImportError as e:
        print(
            f"Error: could not import task module {task_module_path!r}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not hasattr(task_module, "build"):
        print(
            f"Error: task module {task_module_path!r} must define a build() function",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading task: {args.task}...")
    task = task_module.build(config)
    print(
        f"Baseline: {task.baseline_time_ms:.3f}ms | Target: {config.target_speedup}x speedup"
    )

    # Run evolution
    loop = EvolutionLoop(task, config, records_path=args.records)
    best = loop.run()

    # Save output
    if args.output and best:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "task": task.name,
            "best_speedup": best.eval_result.speedup,
            "best_fitness": best.eval_result.fitness,
            "best_kernel_time_ms": best.eval_result.kernel_time_ms,
            "baseline_time_ms": task.baseline_time_ms,
            "coords": list(best.coords.to_tuple()),
            "generation": best.generation,
            "source_code": best.source_code,
            "archive_size": loop.archive.size(),
        }
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {output_path}")

    if args.checkpoint:
        loop.checkpoint(args.checkpoint)

    if best is None:
        print("\nNo correct kernel found.")
        sys.exit(1)

    print(f"\nBest speedup: {best.eval_result.speedup:.2f}x")
    print(f"Behavioral coords: {best.coords}")
    print("\n--- Best Kernel Source ---")
    print(best.source_code)


if __name__ == "__main__":
    cli()
