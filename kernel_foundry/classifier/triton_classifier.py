from __future__ import annotations

import ast
import re

from kernel_foundry.types import BehavioralCoords


class TritonBehaviorClassifier:
    """
    Assigns MAP-Elites behavioral coordinates to a Triton kernel via static AST analysis.

    d_mem (0-3): memory access sophistication
      0 - scalar/uncoalesced: basic tl.load/tl.store without alignment hints
      1 - coalesced/vectorized: tl.arange-based contiguous access, alignment hints
      2 - tiled: tl.dot with K-loop accumulation (shared-memory simulation via registers)
      3 - multi-level: software pipelining (num_stages>1), prefetch, or multi-level blocking

    d_algo (0-3): algorithmic structure
      0 - direct: single elementwise op per element
      1 - fused: multiple distinct ops fused into one kernel
      2 - reformulated: flash-attention-style online algorithm (running max/sum with tl.exp)
      3 - novel: asymptotically different (heuristic; hard to detect statically)

    d_sync (0-3): parallelism coordination
      0 - embarrassingly parallel: no cross-work-item coordination
      1 - barrier: tl.debug_barrier or sequential tile accumulation
      2 - sub-group: tl.reduce, tl.associative_scan
      3 - global: tl.atomic_add, tl.atomic_max, tl.atomic_cas, multi-pass
    """

    def classify(self, source_code: str) -> BehavioralCoords:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return BehavioralCoords(0, 0, 0)

        tl_calls = self._collect_tl_calls(tree)
        has_tiling = self._has_tiling(tree, tl_calls)
        has_pipeline = self._has_pipeline(source_code)

        d_mem = self._classify_d_mem(tl_calls, has_tiling, has_pipeline)
        d_algo = self._classify_d_algo(tree, tl_calls, source_code)
        d_sync = self._classify_d_sync(tl_calls, has_tiling)

        return BehavioralCoords(
            d_mem=min(d_mem, 3),
            d_algo=min(d_algo, 3),
            d_sync=min(d_sync, 3),
        )

    # ------------------------------------------------------------------ helpers

    # Sugar reductions: tl.sum/max/min/cumsum are wrappers around tl.reduce
    _SUGAR_REDUCTIONS = frozenset({"sum", "max", "min", "cumsum"})
    # Normalise base-2 variants to their natural-base counterparts
    _TRANSCENDENTAL_ALIASES = {"exp2": "exp", "log2": "log"}

    def _collect_tl_calls(self, tree: ast.AST) -> dict[str, int]:
        """Return {tl_func_name: count} for all tl.X(...) calls."""
        counts: dict[str, int] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if not (isinstance(func.value, ast.Name) and func.value.id == "tl"):
                continue
            attr = func.attr
            if attr in self._SUGAR_REDUCTIONS:
                key = "reduce"
            else:
                key = self._TRANSCENDENTAL_ALIASES.get(attr, attr)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _has_tiling(self, tree: ast.AST, tl_calls: dict[str, int]) -> bool:
        """True if code has tl.dot inside a for loop — the canonical tiling pattern."""
        if tl_calls.get("dot", 0) == 0:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func = child.func
                        if (
                            isinstance(func, ast.Attribute)
                            and isinstance(func.value, ast.Name)
                            and func.value.id == "tl"
                            and func.attr == "dot"
                        ):
                            return True
        return False

    def _has_pipeline(self, source_code: str) -> bool:
        """Detect num_stages > 1 in triton.Config or @triton.autotune."""
        match = re.search(r"num_stages\s*=\s*([0-9]+)", source_code)
        if match and int(match.group(1)) > 1:
            return True
        return False

    # ------------------------------------------------------------------ d_mem

    def _classify_d_mem(
        self, tl_calls: dict[str, int], has_tiling: bool, has_pipeline: bool
    ) -> int:
        # Level 3: multi-level (pipeline + register blocking)
        if has_pipeline and has_tiling:
            return 3
        # Level 2: tiled (tl.dot in K-loop)
        if has_tiling:
            return 2
        # Level 1: coalesced/vectorized (any tl.load with likely contiguous access)
        if tl_calls.get("load", 0) > 0 or tl_calls.get("store", 0) > 0:
            return 1
        # Level 0: scalar or no explicit memory ops
        return 0

    # ------------------------------------------------------------------ d_algo

    def _classify_d_algo(
        self, tree: ast.AST, tl_calls: dict[str, int], source_code: str
    ) -> int:
        # Level 2: flash-attention-style online softmax
        # Signature: running max (m_i) + running sum (l_i) + tl.exp
        if tl_calls.get("exp", 0) > 0 and tl_calls.get("maximum", 0) > 0:
            return 2
        # Also detect "online" normalization by variable name heuristic
        if re.search(r"\bm_i\b|\bl_i\b|\bm_prev\b|\bl_prev\b|\brunning_max\b", source_code):
            return 2

        # Level 1: fused ops — count distinct operation "categories" in the kernel
        op_categories = self._count_op_categories(tl_calls)
        if op_categories >= 2 or (tl_calls.get("dot", 0) > 0 and tl_calls.get("exp", 0) > 0):
            return 1
        # tl.dot alone (matmul) counts as level 1 since it's a non-trivial reduction
        if tl_calls.get("dot", 0) > 0:
            return 1

        # Level 0: single direct elementwise translation
        return 0

    def _count_op_categories(self, tl_calls: dict[str, int]) -> int:
        """Count how many distinct computation categories appear."""
        categories = 0
        if any(tl_calls.get(f, 0) > 0 for f in ("exp", "log", "sqrt", "rsqrt", "sigmoid")):
            categories += 1
        if tl_calls.get("dot", 0) > 0:
            categories += 1
        if tl_calls.get("reduce", 0) > 0 or tl_calls.get("associative_scan", 0) > 0:
            categories += 1
        basic_arith = any(tl_calls.get(f, 0) > 0 for f in ("load", "store"))
        if basic_arith:
            categories += 1
        return categories

    # ------------------------------------------------------------------ d_sync

    def _classify_d_sync(self, tl_calls: dict[str, int], has_tiling: bool) -> int:
        # Level 3: global atomics or multi-pass
        if any(
            tl_calls.get(f, 0) > 0
            for f in ("atomic_add", "atomic_max", "atomic_min", "atomic_cas", "atomic_xchg")
        ):
            return 3

        # Level 2: sub-group primitives (tl.reduce, tl.associative_scan)
        # Anti-double-count: if has_tiling, a tl.reduce used to accumulate the dot-product
        # tile already credits d_mem. Only count d_sync=2 if reduce is the *primary*
        # coordination mechanism (i.e., no tiling).
        if not has_tiling and (
            tl_calls.get("reduce", 0) > 0 or tl_calls.get("associative_scan", 0) > 0
        ):
            return 2
        # If has_tiling AND reduce/scan, could still be level 2 if there's also a cross-block
        # reduction (e.g., a global sum after the tiled matmul). Conservative: return 2
        # only if reduce count is suspicious for being non-tile-local.
        if has_tiling and tl_calls.get("reduce", 0) > 1:
            return 2

        # Level 1: explicit barrier
        if tl_calls.get("debug_barrier", 0) > 0:
            return 1

        # Level 0: embarrassingly parallel
        return 0
