#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from kernel_foundry.config import EvolutionConfig
from kernel_foundry.search.world_model_loop import WorldModelSearchLoop

BUILTIN_TASKS = {
    "softmax": "tasks.softmax",
    "matmul": "tasks.matmul",
    "causal_conv1d": "tasks.causal_conv1d",
    "solve_tril": "tasks.solve_tril",
    "fused_recurrent_gated_delta_rule": "tasks.fused_recurrent_gated_delta_rule",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="K-Search: world-model kernel optimization"
    )
    parser.add_argument(
        "task",
        help=f"Task name ({', '.join(BUILTIN_TASKS)}) or Python module path",
    )
    parser.add_argument("--budget", type=int, default=40, help="Total evaluation budget")
    parser.add_argument(
        "--stagnation-limit",
        type=int,
        default=3,
        help="Stop refining an action after this many non-improving rounds",
    )
    parser.add_argument(
        "--refinement-population",
        type=int,
        default=2,
        help="Number of concrete candidates to sample per refinement round",
    )
    parser.add_argument("--model", default=None, help="Override LLM model")
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    parser.add_argument(
        "--records",
        default="records.world_model",
        help="Write evaluated kernels as JSONL to this path",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Save and resume world-model search state from this JSON checkpoint path",
    )
    return parser


def cli(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = EvolutionConfig()
    if args.model is not None:
        config.llm_model = args.model

    task_module_path = BUILTIN_TASKS.get(args.task, args.task)
    try:
        task_module = importlib.import_module(task_module_path)
    except ImportError as exc:
        print(
            f"Error: could not import task module {task_module_path!r}: {exc}",
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

    loop = WorldModelSearchLoop(
        task=task,
        config=config,
        budget=args.budget,
        stagnation_limit=args.stagnation_limit,
        refinement_population=args.refinement_population,
        records_path=args.records,
        checkpoint_path=args.checkpoint,
    )
    best = loop.run()

    if args.output and best:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "task": task.name,
            "best_speedup": best.eval_result.speedup,
            "best_fitness": best.eval_result.fitness,
            "best_kernel_time_ms": best.eval_result.kernel_time_ms,
            "baseline_time_ms": task.baseline_time_ms,
            "generation": best.generation,
            "source_code": best.source_code,
            "total_evaluations": loop.initial_budget - loop.budget,
        }
        with output_path.open("w") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nResults saved to {output_path}")

    if best is None:
        sys.exit(1)


if __name__ == "__main__":
    cli()
