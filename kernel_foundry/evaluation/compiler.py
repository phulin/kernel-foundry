from __future__ import annotations

import importlib.util
import sys
import tempfile
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


@dataclass
class CompileResult:
    success: bool
    module: ModuleType | None
    error_log: str
    kernel_fn_name: str | None = None  # name of the @triton.jit function found


class TritonCompiler:
    """
    "Compiles" a Triton kernel by importing its Python source into an isolated module.

    Triton's actual JIT compilation happens at first call. This step catches:
    - Python syntax errors
    - Import errors (missing packages)
    - Module-level exceptions

    The module is imported with a unique name to avoid collisions between candidates.
    """

    def compile(self, source_code: str, kernel_id: str | None = None) -> CompileResult:
        kid = kernel_id or str(uuid.uuid4())[:8]

        # Write to a temp file (needed for importlib to give the module a proper __file__)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix=f"kf_{kid}_",
            delete=False,
        ) as f:
            f.write(source_code)
            tmp_path = f.name

        try:
            module_name = f"kf_kernel_{kid}"
            spec = importlib.util.spec_from_file_location(module_name, tmp_path)
            if spec is None or spec.loader is None:
                return CompileResult(False, None, "Failed to create module spec")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[attr-defined]

            # Verify kernel_fn exists
            if not hasattr(module, "kernel_fn"):
                return CompileResult(
                    False, None, "kernel_fn not defined in generated code"
                )

            # Find @triton.jit function name for informational purposes
            jit_fn_name = self._find_jit_fn(module)
            return CompileResult(True, module, "", kernel_fn_name=jit_fn_name)

        except SyntaxError as e:
            return CompileResult(False, None, f"SyntaxError: {e}")
        except ImportError as e:
            return CompileResult(False, None, f"ImportError: {e}")
        except Exception:
            return CompileResult(False, None, traceback.format_exc(limit=10))
        finally:
            # Clean up temp file (module is already loaded into memory)
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass

    @staticmethod
    def _find_jit_fn(module: ModuleType) -> str | None:
        """Return the name of the first @triton.jit decorated function, if any."""
        try:
            import triton

            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, triton.JITFunction):
                    return name
        except ImportError:
            pass
        return None
