"""Recursive Language Models (RLM) scaffold.

This package implements a minimal, practical adaptation of the "Recursive Language Models"
idea for codebase research. The core trick is that the (potentially huge) input prompt is
stored as a Python variable inside a persistent REPL, and the LLM interacts by writing
Python code to explore and analyze it, optionally delegating to recursive sub-calls.

Public API is intentionally small; most users will use the CLI:

    rlm ask --repo ./path --question "..."

"""

from .engine import RLMConfig, RLMEngine, LiteLLMClient
from .codebase import CodebasePrompt, load_codebase_from_dir, load_codebase_from_repomix

__all__ = [
    "CodebasePrompt",
    "LiteLLMClient",
    "RLMConfig",
    "RLMEngine",
    "load_codebase_from_dir",
    "load_codebase_from_repomix",
]
