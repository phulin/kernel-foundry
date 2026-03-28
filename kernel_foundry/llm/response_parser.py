from __future__ import annotations

import json
import re


def extract_triton_code(response: str) -> str | None:
    """
    Extract a Python code block from an LLM response.

    Looks for ```python ... ``` or ``` ... ``` fenced blocks.
    Validates that the extracted code contains at least one @triton.jit decorated function
    and a kernel_fn entry point.

    Returns None if no valid Triton code is found.
    """
    # Try ```python ... ``` first, then plain ``` ... ```
    patterns = [
        r"```python\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if _looks_like_triton_kernel(code):
                return code

    # Fallback: if response itself looks like raw code (no markdown fencing)
    if _looks_like_triton_kernel(response):
        return response.strip()

    return None


def extract_search_replace_diffs(response: str) -> list[tuple[str, str]]:
    """
    Parse SEARCH/REPLACE diffs from a meta-prompter response.

    Expected format:
        <<<SEARCH>>>
        text to find
        <<<REPLACE>>>
        replacement text
        <<<END>>>

    Returns list of (search, replace) pairs, up to 3.
    """
    pattern = r"<<<SEARCH>>>\s*(.*?)<<<REPLACE>>>\s*(.*?)<<<END>>>"
    matches = re.findall(pattern, response, re.DOTALL)
    return [(s.strip(), r.strip()) for s, r in matches[:3]]


def extract_json_payload(response: str) -> str:
    """
    Extract a JSON payload from an LLM response.

    Accepts a raw JSON object/array or a fenced ```json block.
    """
    patterns = [
        r"```json\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            json.loads(candidate)
            return candidate

    response = response.strip()
    json.loads(response)
    return response


def _looks_like_triton_kernel(code: str) -> bool:
    return "@triton.jit" in code and "kernel_fn" in code
