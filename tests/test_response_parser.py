import pytest
from kernel_foundry.llm.response_parser import extract_triton_code, extract_search_replace_diffs

VALID_KERNEL = """\
import triton
import triton.language as tl
import torch

@triton.jit
def my_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n)
    tl.store(out_ptr + offs, x * 2, mask=offs < n)

def kernel_fn(x):
    out = torch.empty_like(x)
    my_kernel[(x.numel() // 128,)](x, out, x.numel(), BLOCK=128)
    return out"""


class TestExtractTritonCode:
    def test_extracts_python_fenced_block(self):
        response = f"Here is the kernel:\n```python\n{VALID_KERNEL}\n```\n"
        code = extract_triton_code(response)
        assert code is not None
        assert "@triton.jit" in code
        assert "kernel_fn" in code

    def test_extracts_plain_fenced_block(self):
        response = f"```\n{VALID_KERNEL}\n```"
        code = extract_triton_code(response)
        assert code is not None

    def test_prefers_python_fenced_over_plain(self):
        response = f"```\nno triton here\n```\n```python\n{VALID_KERNEL}\n```"
        code = extract_triton_code(response)
        assert "@triton.jit" in code

    def test_no_code_block_returns_none(self):
        assert extract_triton_code("just prose, no code") is None

    def test_missing_triton_jit_returns_none(self):
        response = "```python\ndef kernel_fn(x):\n    return x\n```"
        assert extract_triton_code(response) is None

    def test_missing_kernel_fn_returns_none(self):
        response = "```python\n@triton.jit\ndef my_kernel(): pass\n```"
        assert extract_triton_code(response) is None

    def test_raw_code_without_fencing(self):
        code = extract_triton_code(VALID_KERNEL)
        assert code is not None

    def test_strips_whitespace(self):
        response = f"```python\n\n{VALID_KERNEL}\n\n```"
        code = extract_triton_code(response)
        assert code == VALID_KERNEL.strip()


class TestExtractSearchReplaceDiffs:
    def test_single_diff(self):
        text = "<<<SEARCH>>>\nold text\n<<<REPLACE>>>\nnew text\n<<<END>>>"
        diffs = extract_search_replace_diffs(text)
        assert diffs == [("old text", "new text")]

    def test_multiple_diffs(self):
        block = "<<<SEARCH>>>\na\n<<<REPLACE>>>\nb\n<<<END>>>\n"
        diffs = extract_search_replace_diffs(block * 2)
        assert len(diffs) == 2
        assert diffs[0] == ("a", "b")

    def test_capped_at_three(self):
        block = "<<<SEARCH>>>\na\n<<<REPLACE>>>\nb\n<<<END>>>\n"
        diffs = extract_search_replace_diffs(block * 6)
        assert len(diffs) == 3

    def test_no_diffs_returns_empty(self):
        assert extract_search_replace_diffs("no diffs here") == []

    def test_multiline_search_and_replace(self):
        text = "<<<SEARCH>>>\nline 1\nline 2\n<<<REPLACE>>>\nnew line 1\nnew line 2\n<<<END>>>"
        diffs = extract_search_replace_diffs(text)
        assert diffs == [("line 1\nline 2", "new line 1\nnew line 2")]

    def test_strips_surrounding_whitespace(self):
        text = "<<<SEARCH>>>  \n  old  \n  <<<REPLACE>>>  \n  new  \n  <<<END>>>"
        diffs = extract_search_replace_diffs(text)
        assert diffs[0] == ("old", "new")


class TestApplySearchReplace:
    def test_applies_to_correct_section(self):
        from kernel_foundry.prompt.evolvable_sections import DEFAULT_SECTIONS, apply_search_replace
        diffs = [("Prioritize memory bandwidth", "ALWAYS prioritize memory bandwidth")]
        new = apply_search_replace(DEFAULT_SECTIONS, diffs)
        assert "ALWAYS" in new.optimization_philosophy
        assert new.optimization_strategies == DEFAULT_SECTIONS.optimization_strategies

    def test_missing_search_text_is_skipped(self, capsys):
        from kernel_foundry.prompt.evolvable_sections import DEFAULT_SECTIONS, apply_search_replace
        diffs = [("THIS TEXT DOES NOT EXIST XYZ123", "replacement")]
        new = apply_search_replace(DEFAULT_SECTIONS, diffs)
        # Sections unchanged
        assert new.optimization_philosophy == DEFAULT_SECTIONS.optimization_philosophy
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or "skipping" in captured.out.lower()

    def test_multiple_diffs_applied_in_order(self):
        from kernel_foundry.prompt.evolvable_sections import DEFAULT_SECTIONS, apply_search_replace
        diffs = [
            ("Prioritize memory bandwidth", "A: memory bandwidth"),
            ("Maximize data reuse", "B: data reuse"),
        ]
        new = apply_search_replace(DEFAULT_SECTIONS, diffs)
        assert "A: memory bandwidth" in new.optimization_philosophy
        assert "B: data reuse" in new.optimization_philosophy
