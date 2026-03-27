from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv


@dataclass
class EvolutionConfig:
    # LLM
    llm_model: str = "gpt-4o"
    meta_prompter_model: str = "gpt-4o"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 8000
    llm_top_p: float = 1.0

    # Evolution
    max_generations: int = 40
    population_size: int = 8
    selection_strategy: Literal["uniform", "fitness", "curiosity", "island"] = "curiosity"
    target_speedup: float = 2.0

    # Archive (bins per dimension → bins^3 cells total)
    archive_bins: int = 4

    # Gradient estimator
    gradient_weight_fitness: float = 0.4
    gradient_weight_improvement: float = 0.4
    gradient_weight_exploration: float = 0.2
    gradient_decay: float = 0.95
    transition_buffer_size: int = 500

    # Meta-prompting
    meta_prompt_freq: int = 10
    meta_prompt_max_mutations: int = 3
    prompt_archive_size: int = 16

    # Benchmarking
    warmup_min_time: float = 1.0
    warmup_min_iters: int = 10
    benchmark_min_time: float = 1.0
    benchmark_min_iters: int = 10
    inner_loop_min_time: float = 0.01

    # Template optimization (post-main phase)
    template_opt_iterations: int = 2
    template_opt_population: int = 8

    # Island selection
    island_count: int = 4
    island_migration_freq: int = 10

    @property
    def openai_api_key(self) -> str:
        load_dotenv()
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY not set in environment or .env file")
        return key
