from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict


def find_ncu() -> str | None:
    """Find the ``ncu`` binary on PATH or in /usr/local/cuda/bin."""
    ncu = shutil.which("ncu")
    if ncu:
        return ncu
    candidate = "/usr/local/cuda/bin/ncu"
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None


def profile_kernel(
    source_code: str,
    inputs: tuple,
    ncu_path: str,
    timeout_s: float = 120.0,
) -> str | None:
    """Run NCU on a single kernel invocation and return a formatted summary.

    Writes the kernel source and serialised inputs to a temp directory,
    launches a small driver script under ``ncu --profile-from-start off``
    so only the measured call is captured, and parses the CSV output into
    a human-readable string suitable for an LLM prompt.

    Returns ``None`` on any failure (missing permissions, timeout, etc.).
    """
    import torch

    try:
        with tempfile.TemporaryDirectory(prefix="kf_ncu_") as tmpdir:
            source_path = os.path.join(tmpdir, "kernel_source.py")
            inputs_path = os.path.join(tmpdir, "inputs.pt")
            runner_path = os.path.join(tmpdir, "runner.py")

            with open(source_path, "w") as f:
                f.write(source_code)

            torch.save(list(inputs), inputs_path)

            with open(runner_path, "w") as f:
                f.write(
                    "import torch, importlib.util\n"
                    f"spec = importlib.util.spec_from_file_location('kmod', {source_path!r})\n"
                    "mod = importlib.util.module_from_spec(spec)\n"
                    "spec.loader.exec_module(mod)\n"
                    f"inputs = torch.load({inputs_path!r}, weights_only=False)\n"
                    "mod.kernel_fn(*inputs)\n"
                    "torch.cuda.synchronize()\n"
                    "torch.cuda.cudart().cudaProfilerStart()\n"
                    "mod.kernel_fn(*inputs)\n"
                    "torch.cuda.synchronize()\n"
                    "torch.cuda.cudart().cudaProfilerStop()\n"
                )

            result = subprocess.run(
                [
                    ncu_path,
                    "--profile-from-start", "off",
                    "--set", "basic",
                    "--csv",
                    sys.executable,
                    runner_path,
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )

            if result.returncode != 0 or not result.stdout.strip():
                return None

            return _format_ncu_csv(result.stdout)
    except Exception:
        return None


# -- Pretty-printing helpers -------------------------------------------------

_METRIC_LABELS: list[tuple[str, str]] = [
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "SM Throughput"),
    ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "Memory Throughput"),
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "Achieved Occupancy"),
    ("launch__registers_per_thread", "Registers/Thread"),
    ("launch__shared_mem_per_block_static", "Static Shared Mem"),
    ("launch__shared_mem_per_block_dynamic", "Dynamic Shared Mem"),
    ("launch__grid_size", "Grid Size"),
    ("launch__block_size", "Block Size"),
    ("gpu__time_duration.sum", "Duration"),
    ("launch__waves_per_multiprocessor", "Waves/SM"),
    ("sm__maximum_warps_per_active_cycle_pct", "Theoretical Occupancy"),
]

_KNOWN_KEYS = {k for k, _ in _METRIC_LABELS}


def _format_ncu_csv(raw_csv: str) -> str | None:
    """Parse NCU ``--csv`` output into a concise per-kernel summary."""
    try:
        lines = [l for l in raw_csv.splitlines() if not l.startswith("==")]
        if not lines:
            return None

        reader = csv.DictReader(io.StringIO("\n".join(lines)))

        kernels: dict[str, dict[str, str]] = defaultdict(dict)
        for row in reader:
            name = row.get("Kernel Name", "unknown")
            metric = row.get("Metric Name", "")
            value = row.get("Metric Value", "")
            unit = row.get("Metric Unit", "")
            if metric:
                display_val = f"{value} {unit}".strip() if unit else value
                kernels[name][metric] = display_val

        if not kernels:
            return None

        parts: list[str] = []
        for kernel_name, metrics in kernels.items():
            short = kernel_name[:80] + "…" if len(kernel_name) > 80 else kernel_name
            section = [f"Kernel: {short}"]

            for metric_key, label in _METRIC_LABELS:
                if metric_key in metrics:
                    section.append(f"  {label}: {metrics[metric_key]}")

            for metric_key, value in metrics.items():
                if metric_key not in _KNOWN_KEYS:
                    short_key = metric_key.rsplit(".", 1)[-1] if "." in metric_key else metric_key
                    section.append(f"  {short_key}: {value}")

            parts.append("\n".join(section))

        return "\n".join(parts)
    except Exception:
        return None
