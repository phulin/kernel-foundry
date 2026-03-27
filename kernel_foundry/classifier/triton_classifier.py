from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from kernel_foundry.types import BehavioralCoords


@dataclass(frozen=True)
class KernelFeatures:
    tl_calls: dict[str, int]
    jit_fn_count: int
    kernel_launch_count: int
    for_loop_count: int
    load_count: int
    store_count: int
    has_arange_indexing: bool
    has_alignment_hint: bool
    has_block_ptr: bool
    has_block_ptr_advance: bool
    has_tiled_dot_loop: bool
    has_pipeline: bool
    has_multi_dim_accumulator: bool
    has_online_state_names: bool
    has_online_state_updates: bool
    has_serial_accumulation_loop: bool


class TritonBehaviorClassifier:
    """
    Assign MAP-Elites behavioral coordinates to a Triton kernel via deterministic
    static analysis.

    The classifier is intentionally feature-driven:
      1. extract a compact, backend-specific feature vector from the AST/source
      2. score those features into the three behavioral dimensions

    This keeps the mechanism general while making the heuristics explicit and testable.
    """

    _SUGAR_REDUCTIONS = frozenset({"sum", "max", "min", "cumsum"})
    _TRANSCENDENTAL_ALIASES = {"exp2": "exp", "log2": "log"}
    _TRANSCENDENTALS = frozenset({"exp", "log", "sqrt", "rsqrt", "sigmoid"})
    _ATOMICS = frozenset({"atomic_add", "atomic_max", "atomic_min", "atomic_cas", "atomic_xchg"})

    def classify(self, source_code: str) -> BehavioralCoords:
        features = self.extract_features(source_code)
        return BehavioralCoords(
            d_mem=self._classify_d_mem(features),
            d_algo=self._classify_d_algo(features),
            d_sync=self._classify_d_sync(features),
        )

    def extract_features(self, source_code: str) -> KernelFeatures:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return KernelFeatures(
                tl_calls={},
                jit_fn_count=0,
                kernel_launch_count=0,
                for_loop_count=0,
                load_count=0,
                store_count=0,
                has_arange_indexing=False,
                has_alignment_hint=False,
                has_block_ptr=False,
                has_block_ptr_advance=False,
                has_tiled_dot_loop=False,
                has_pipeline=False,
                has_multi_dim_accumulator=False,
                has_online_state_names=False,
                has_online_state_updates=False,
                has_serial_accumulation_loop=False,
            )

        tl_calls = self._collect_tl_calls(tree)
        return KernelFeatures(
            tl_calls=tl_calls,
            jit_fn_count=self._count_jit_functions(tree),
            kernel_launch_count=self._count_kernel_launches(tree),
            for_loop_count=sum(1 for node in ast.walk(tree) if isinstance(node, ast.For)),
            load_count=tl_calls.get("load", 0),
            store_count=tl_calls.get("store", 0),
            has_arange_indexing=self._has_arange_indexing(tl_calls),
            has_alignment_hint=self._has_alignment_hint(tl_calls),
            has_block_ptr=self._has_block_ptr(tl_calls),
            has_block_ptr_advance=tl_calls.get("advance", 0) > 0,
            has_tiled_dot_loop=self._has_tiled_dot_loop(tree, tl_calls),
            has_pipeline=self._has_pipeline(source_code),
            has_multi_dim_accumulator=self._has_multi_dim_accumulator(tree),
            has_online_state_names=self._has_online_state_names(source_code),
            has_online_state_updates=self._has_online_state_updates(source_code),
            has_serial_accumulation_loop=self._has_serial_accumulation_loop(tree, tl_calls),
        )

    # ------------------------------------------------------------------ feature extraction

    def _collect_tl_calls(self, tree: ast.AST) -> dict[str, int]:
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
            key = "reduce" if attr in self._SUGAR_REDUCTIONS else self._TRANSCENDENTAL_ALIASES.get(attr, attr)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _has_tiled_dot_loop(self, tree: ast.AST, tl_calls: dict[str, int]) -> bool:
        if tl_calls.get("dot", 0) == 0:
            return False
        for node in ast.walk(tree):
            if not isinstance(node, ast.For):
                continue
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
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
        match = re.search(r"num_stages\s*=\s*([0-9]+)", source_code)
        return bool(match and int(match.group(1)) > 1)

    def _has_arange_indexing(self, tl_calls: dict[str, int]) -> bool:
        return tl_calls.get("arange", 0) > 0

    def _has_alignment_hint(self, tl_calls: dict[str, int]) -> bool:
        return tl_calls.get("multiple_of", 0) > 0 or tl_calls.get("max_contiguous", 0) > 0

    def _has_block_ptr(self, tl_calls: dict[str, int]) -> bool:
        return tl_calls.get("make_block_ptr", 0) > 0 or tl_calls.get("advance", 0) > 0

    def _count_jit_functions(self, tree: ast.AST) -> int:
        count = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for deco in node.decorator_list:
                if (
                    isinstance(deco, ast.Attribute)
                    and isinstance(deco.value, ast.Name)
                    and deco.value.id == "triton"
                    and deco.attr == "jit"
                ):
                    count += 1
                elif (
                    isinstance(deco, ast.Call)
                    and isinstance(deco.func, ast.Attribute)
                    and isinstance(deco.func.value, ast.Name)
                    and deco.func.value.id == "triton"
                    and deco.func.attr == "jit"
                ):
                    count += 1
        return count

    def _count_kernel_launches(self, tree: ast.AST) -> int:
        launches = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Subscript):
                continue
            callee = func.value
            if isinstance(callee, (ast.Name, ast.Attribute)):
                launches += 1
        return launches

    def _has_multi_dim_accumulator(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "tl"
                and func.attr == "zeros"
            ):
                continue
            if not node.args:
                continue
            shape = node.args[0]
            if isinstance(shape, (ast.Tuple, ast.List)) and len(shape.elts) >= 2:
                return True
        return False

    def _has_online_state_names(self, source_code: str) -> bool:
        return bool(re.search(r"\bm_i\b|\bl_i\b|\bm_prev\b|\bl_prev\b|\brunning_(max|sum)\b", source_code))

    def _has_online_state_updates(self, source_code: str) -> bool:
        patterns = [
            r"\bm_i\s*=",
            r"\bl_i\s*=",
            r"\bm_prev\s*=",
            r"\bl_prev\s*=",
            r"running_max\s*=",
            r"running_sum\s*=",
        ]
        update_count = sum(1 for pattern in patterns if re.search(pattern, source_code))
        return update_count >= 2

    def _has_serial_accumulation_loop(self, tree: ast.AST, tl_calls: dict[str, int]) -> bool:
        if tl_calls.get("reduce", 0) > 0 or tl_calls.get("associative_scan", 0) > 0:
            return False
        for node in ast.walk(tree):
            if not isinstance(node, ast.For):
                continue
            saw_augassign = False
            saw_load = False
            for child in ast.walk(node):
                if isinstance(child, ast.AugAssign):
                    saw_augassign = True
                elif isinstance(child, ast.Call):
                    func = child.func
                    if (
                        isinstance(func, ast.Attribute)
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "tl"
                        and func.attr == "load"
                    ):
                        saw_load = True
            if saw_augassign and saw_load:
                return True
        return False

    # ------------------------------------------------------------------ scoring

    def _classify_d_mem(self, features: KernelFeatures) -> int:
        tiled_evidence = sum(
            [
                1 if features.has_tiled_dot_loop else 0,
                1 if features.has_block_ptr else 0,
                1 if features.has_multi_dim_accumulator else 0,
            ]
        )
        multilevel_evidence = sum(
            [
                1 if features.has_pipeline else 0,
                1 if features.has_block_ptr_advance else 0,
                1 if features.has_alignment_hint else 0,
            ]
        )
        coalesced_evidence = sum(
            [
                1 if features.has_arange_indexing else 0,
                1 if features.has_alignment_hint else 0,
                1 if features.load_count > 0 and features.store_count > 0 else 0,
            ]
        )

        if tiled_evidence >= 2 and multilevel_evidence >= 1:
            return 3
        if tiled_evidence >= 1:
            return 2
        if coalesced_evidence >= 2:
            return 1
        return 0

    def _classify_d_algo(self, features: KernelFeatures) -> int:
        if features.jit_fn_count >= 2 and features.kernel_launch_count >= 2:
            return 3

        reformulation_evidence = sum(
            [
                1 if features.tl_calls.get("exp", 0) > 0 else 0,
                1 if features.tl_calls.get("maximum", 0) > 0 else 0,
                1 if features.has_online_state_names else 0,
                1 if features.has_online_state_updates else 0,
            ]
        )
        if reformulation_evidence >= 3:
            return 2

        if self._count_op_categories(features.tl_calls) >= 2:
            return 1
        if features.tl_calls.get("dot", 0) > 0 or features.tl_calls.get("reduce", 0) > 0:
            return 1
        return 0

    def _classify_d_sync(self, features: KernelFeatures) -> int:
        if any(features.tl_calls.get(name, 0) > 0 for name in self._ATOMICS):
            return 3
        if features.kernel_launch_count >= 2:
            return 3

        if features.tl_calls.get("associative_scan", 0) > 0:
            return 2
        if not features.has_tiled_dot_loop and features.tl_calls.get("reduce", 0) > 0:
            return 2
        if features.has_tiled_dot_loop and features.tl_calls.get("reduce", 0) > 1:
            return 2

        if features.tl_calls.get("debug_barrier", 0) > 0 or features.has_serial_accumulation_loop:
            return 1
        return 0

    def _count_op_categories(self, tl_calls: dict[str, int]) -> int:
        categories = 0
        if any(tl_calls.get(name, 0) > 0 for name in self._TRANSCENDENTALS):
            categories += 1
        if tl_calls.get("dot", 0) > 0:
            categories += 1
        if tl_calls.get("reduce", 0) > 0 or tl_calls.get("associative_scan", 0) > 0:
            categories += 1
        if tl_calls.get("load", 0) > 0 or tl_calls.get("store", 0) > 0:
            categories += 1
        return categories
