from __future__ import annotations

import traceback
from dataclasses import dataclass
from types import ModuleType
from typing import Callable

import torch


@dataclass
class CorrectnessResult:
    correct: bool
    max_relative_error: float
    element_pass_rate: float  # fraction of elements with ν < threshold
    error_log: str = ""


class CorrectnessChecker:
    """
    Validates a compiled kernel against a PyTorch reference using relative precision.

    Correct iff ν = |y - ŷ| / (|y| + ε) < threshold for ≥ pass_fraction of elements,
    across num_trials random input draws.

    This is stricter than KernelBench's loose absolute threshold (10^-2).
    """

    def __init__(
        self,
        reference_fn: Callable,
        input_generator: Callable[[], tuple],
        threshold: float = 0.01,
        pass_fraction: float = 0.99,
        num_trials: int = 3,
        eps: float = 1e-8,
    ) -> None:
        self._reference_fn = reference_fn
        self._input_generator = input_generator
        self._threshold = threshold
        self._pass_fraction = pass_fraction
        self._num_trials = num_trials
        self._eps = eps

    def check(self, module: ModuleType) -> CorrectnessResult:
        kernel_fn = getattr(module, "kernel_fn")
        worst_pass_rate = 1.0
        worst_rel_error = 0.0

        for trial in range(self._num_trials):
            try:
                inputs = self._input_generator()
                device = self._find_cuda_device(inputs)
                with torch.no_grad():
                    y_ref = self._reference_fn(*inputs)
                    self._synchronize_cuda(device)
                    y_kernel = kernel_fn(*inputs)
                    self._synchronize_cuda(device)

                if not isinstance(y_ref, torch.Tensor):
                    y_ref = y_ref[0] if isinstance(y_ref, (tuple, list)) else torch.tensor(y_ref)
                if not isinstance(y_kernel, torch.Tensor):
                    y_kernel = (
                        y_kernel[0]
                        if isinstance(y_kernel, (tuple, list))
                        else torch.tensor(y_kernel)
                    )

                if y_ref.shape != y_kernel.shape:
                    return CorrectnessResult(
                        False,
                        float("inf"),
                        0.0,
                        f"Shape mismatch: ref {y_ref.shape} vs kernel {y_kernel.shape}",
                    )

                y_ref_f = y_ref.float()
                y_kernel_f = y_kernel.float()

                nu = torch.abs(y_ref_f - y_kernel_f) / (torch.abs(y_ref_f) + self._eps)
                pass_rate = float((nu < self._threshold).float().mean())
                max_rel_err = float(nu.max())

                worst_pass_rate = min(worst_pass_rate, pass_rate)
                worst_rel_error = max(worst_rel_error, max_rel_err)

            except Exception:
                hint = self._cuda_failure_hint()
                return CorrectnessResult(
                    False, float("inf"), 0.0, traceback.format_exc(limit=8) + hint
                )

        correct = worst_pass_rate >= self._pass_fraction
        return CorrectnessResult(
            correct=correct,
            max_relative_error=worst_rel_error,
            element_pass_rate=worst_pass_rate,
        )

    @staticmethod
    def _find_cuda_device(inputs: tuple):
        for x in inputs:
            if isinstance(x, torch.Tensor) and x.is_cuda:
                return x.device
        return None

    @staticmethod
    def _synchronize_cuda(device) -> None:
        if device is not None:
            torch.cuda.synchronize(device)

    @staticmethod
    def _cuda_failure_hint() -> str:
        return (
            "\n\n[KernelFoundry note] CUDA work is synchronized immediately after the "
            "reference and candidate kernel calls. If this error mentions illegal memory "
            "access, it likely came from the candidate kernel that just ran rather than "
            "from the subsequent input generation step."
        )
