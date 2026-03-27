from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class TaskSpec:
    """
    Normalized task representation fed to the evolutionary loop.

    All input modalities (PyTorch reference, NL description, existing kernel) are
    normalized to this format by the loader.
    """

    name: str
    description: str                    # NL description for the LLM
    reference_code: str                 # Source code of the PyTorch reference function
    reference_fn: Callable              # Callable reference implementation
    input_generator: Callable[[], tuple]  # Returns tuple of tensors for one trial
    hardware_spec: str                  # Human-readable GPU spec injected into prompts
    baseline_time_ms: float             # Reference runtime on the target hardware
    seed_kernel: str | None = None      # Optional starting kernel code


def detect_hardware_spec() -> str:
    """Return a human-readable description of the available GPU."""
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            sm_count = props.multi_processor_count
            mem_gb = props.total_memory / 1024**3
            return (
                f"NVIDIA {name}, {sm_count} SMs, {mem_gb:.1f} GB VRAM, "
                f"CUDA Capability {props.major}.{props.minor}"
            )
        return "CPU only (no CUDA device detected)"
    except Exception as e:
        return f"Unknown GPU ({e})"
