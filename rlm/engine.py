from __future__ import annotations

import datetime as _dt
import io
import json
import os
import re
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .prompts import ROOT_SYSTEM_PROMPT, SUB_SYSTEM_PROMPT


class LLMClient:
    """Minimal interface for chat-style LLM calls."""

    def complete(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        raise NotImplementedError


class LiteLLMClient(LLMClient):
    """LLM client backed by `litellm`.

    This is a thin wrapper so the rest of the codebase doesn't depend on a specific SDK.
    """

    def complete(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        try:
            from litellm import completion  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "litellm is required. Install dependencies (e.g., `uv pip install -e .`) and retry."
            ) from e

        resp = completion(
            model=model,
            messages=messages,
            temperature=temperature,
        )

        # LiteLLM typically returns an object with `.choices[0].message.content`, but we
        # handle dict-like returns too.
        try:
            return resp.choices[0].message.content or ""
        except Exception:
            try:
                return resp["choices"][0]["message"]["content"] or ""
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"Unexpected litellm response shape: {type(resp)}") from e


def _infer_sub_model(root_model: str, explicit_sub: str | None = None) -> str:
    """Infer a sub-model from the root model, preserving provider prefix.

    If the user explicitly sets RLM_SUB_MODEL, use it as-is.
    Otherwise, derive a cheaper model from the same provider.
    """
    if explicit_sub:
        return explicit_sub

    # Known provider prefixes (litellm convention: "provider/model-name").
    # If root has a prefix like "openrouter/anthropic/...", keep it.
    parts = root_model.split("/")
    if len(parts) >= 2:
        # e.g. "openrouter/anthropic/claude-sonnet-4-5" → prefix = "openrouter/"
        # e.g. "anthropic/claude-sonnet-4-5" → prefix = "anthropic/"
        prefix = "/".join(parts[:-1]) + "/"
        base = parts[-1]
    else:
        prefix = ""
        base = root_model

    # Map known model families to cheaper variants.
    SUB_DEFAULTS = {
        "gpt-5": "gpt-4.1-mini",
        "gpt-5.2": "gpt-4.1-mini",
        "gpt-4o": "gpt-4o-mini",
        "claude-sonnet-4-5": "claude-haiku-3-5",
        "claude-opus-4-6": "claude-sonnet-4-5",
    }

    # Check if base matches or starts with a known family.
    sub_base = SUB_DEFAULTS.get(base, "gpt-4.1-mini")
    return prefix + sub_base


@dataclass(frozen=True)
class RLMConfig:
    root_model: str = os.environ.get("RLM_MODEL", "openrouter/gpt-5")
    sub_model: str = ""  # Empty = auto-infer from root_model

    temperature: float = float(os.environ.get("RLM_TEMPERATURE", "0.0"))

    max_iterations: int = int(os.environ.get("RLM_MAX_ITERATIONS", "24"))
    sub_max_iterations: int = int(os.environ.get("RLM_SUB_MAX_ITERATIONS", "12"))
    max_depth: int = int(os.environ.get("RLM_MAX_DEPTH", "3"))

    # Metadata limits (what the LLM sees).
    prompt_prefix_chars: int = int(os.environ.get("RLM_PROMPT_PREFIX_CHARS", "600"))
    stdout_preview_chars: int = int(os.environ.get("RLM_STDOUT_PREVIEW_CHARS", "1200"))
    stderr_preview_chars: int = int(os.environ.get("RLM_STDERR_PREVIEW_CHARS", "800"))

    # Persistent debug logs.
    log_dir: str = os.environ.get("RLM_LOG_DIR", ".rlm_logs")


class _JSONLLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, entry: Dict[str, Any]) -> None:
        # Avoid non-serializable objects.
        def _default(o: Any) -> str:
            return f"<{type(o).__name__}>"

        line = json.dumps(entry, default=_default, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


class RLMEngine:
    """Recursive Language Model engine.

    This engine runs a loop:
      - LLM produces Python code
      - Code executes in a persistent sandboxed REPL
      - Only metadata about stdout/state is fed back to the LLM
      - Session terminates when `Final` is set in the REPL state

    The key design goal is that the large `prompt` is never inserted directly into the
    model context.
    """

    def __init__(
        self,
        *,
        client: LLMClient,
        config: RLMConfig | None = None,
        root_system_prompt: str = ROOT_SYSTEM_PROMPT,
        sub_system_prompt: str = SUB_SYSTEM_PROMPT,
    ):
        self.client = client
        self.config = config or RLMConfig()
        self.root_system_prompt = root_system_prompt
        self.sub_system_prompt = sub_system_prompt

    def run(
        self,
        *,
        prompt: str,
        repo_index: Optional[Dict[str, Any]] = None,
        repo_meta: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> str:
        """Run a root RLM session and return `Final` as a string."""

        session_id = uuid.uuid4().hex
        log_path = Path(self.config.log_dir) / f"rlm_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{session_id}.jsonl"
        logger = _JSONLLogger(log_path)

        final = self._run_internal(
            prompt=prompt,
            repo_prompt=prompt,
            repo_index=repo_index or {},
            repo_meta=repo_meta or {},
            depth=0,
            session_id=session_id,
            logger=logger,
            verbose=verbose,
        )

        logger.write(
            {
                "ts": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
                "type": "final",
                "session_id": session_id,
                "final": final,
            }
        )

        if verbose:
            print(f"[rlm] log: {log_path}")

        return final

    # ------------------------- Internal implementation -------------------------

    def _run_internal(
        self,
        *,
        prompt: str,
        repo_prompt: str,
        repo_index: Dict[str, Any],
        repo_meta: Dict[str, Any],
        depth: int,
        session_id: str,
        logger: _JSONLLogger,
        verbose: bool,
    ) -> str:
        if depth > self.config.max_depth:
            raise RuntimeError(
                f"Max recursion depth exceeded: depth={depth} max_depth={self.config.max_depth}"
            )

        env = self._init_env(
            prompt=prompt,
            repo_prompt=repo_prompt,
            repo_index=repo_index,
            repo_meta=repo_meta,
            depth=depth,
            session_id=session_id,
            logger=logger,
            verbose=verbose,
        )

        system_prompt = self.root_system_prompt if depth == 0 else self.sub_system_prompt
        resolved_sub = self.config.sub_model or _infer_sub_model(self.config.root_model, os.environ.get("RLM_SUB_MODEL") or None)
        model = self.config.root_model if depth == 0 else resolved_sub
        max_iters = self.config.max_iterations if depth == 0 else self.config.sub_max_iterations

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._metadata(env=env, depth=depth, iteration=0, last_exec=None)},
        ]

        for iteration in range(1, max_iters + 1):
            raw = self.client.complete(
                model=model,
                messages=messages,
                temperature=self.config.temperature,
            ) or ""
            code = _extract_python_code(raw)

            messages.append({"role": "assistant", "content": code})

            last_exec = self._exec(code=code, env=env)

            # Log everything (for debugging) even if the model doesn't see it.
            logger.write(
                {
                    "ts": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
                    "type": "step",
                    "session_id": session_id,
                    "depth": depth,
                    "iteration": iteration,
                    "model": model,
                    "code": code,
                    "stdout": _truncate_for_log(last_exec.stdout),
                    "stderr": _truncate_for_log(last_exec.stderr),
                    "exception": last_exec.exception,
                    "final_set": "Final" in env and env.get("Final") not in (None, ""),
                }
            )

            if verbose:
                print(f"\n[rlm] depth={depth} iter={iteration} model={model}")
                print("[rlm] code:\n" + (code or "(empty)"))
                if last_exec.stdout.strip():
                    print("[rlm] stdout:\n" + last_exec.stdout)
                if last_exec.stderr.strip():
                    print("[rlm] stderr:\n" + last_exec.stderr)
                if last_exec.exception:
                    print("[rlm] exception:\n" + last_exec.exception)

            # Stop condition.
            if "Final" in env and env.get("Final") not in (None, ""):
                return str(env["Final"])

            messages.append(
                {
                    "role": "user",
                    "content": self._metadata(
                        env=env,
                        depth=depth,
                        iteration=iteration,
                        last_exec=last_exec,
                    ),
                }
            )

        raise RuntimeError(
            f"Max iterations reached without setting Final (depth={depth}, iters={max_iters})."
        )

    def _init_env(
        self,
        *,
        prompt: str,
        repo_prompt: str,
        repo_index: Dict[str, Any],
        repo_meta: Dict[str, Any],
        depth: int,
        session_id: str,
        logger: _JSONLLogger,
        verbose: bool,
    ) -> Dict[str, Any]:
        safe_builtins = _safe_builtins()

        env: Dict[str, Any] = {
            "__builtins__": safe_builtins,
            "prompt": prompt,
            "repo_prompt": repo_prompt,
            "Final": None,
            "repo_index": repo_index,
            "repo_meta": repo_meta,
            "_depth": depth,
            "_session_id": session_id,
        }

        # Cache prompt invariants (avoid recomputation each step).
        env["_prompt_chars"] = len(prompt)
        env["_prompt_sha256"] = _sha256(prompt)
        env["_prompt_prefix"] = prompt[: self.config.prompt_prefix_chars]

        # Cache repo prompt invariants separately (repo_index offsets refer to repo_prompt).
        env["_repo_prompt_chars"] = len(repo_prompt)
        env["_repo_prompt_sha256"] = _sha256(repo_prompt)
        env["_repo_prompt_prefix"] = repo_prompt[: self.config.prompt_prefix_chars]

        # Prompt helper functions.
        env["prompt_len"] = lambda: env["_prompt_chars"]
        env["prompt_peek"] = lambda n=400, start=0: env["prompt"][start : start + n]
        env["prompt_slice"] = lambda start, end: env["prompt"][start:end]
        env["prompt_find"] = lambda pattern, flags=0, max_matches=20: _prompt_find(
            env["prompt"], pattern, flags=flags, max_matches=max_matches
        )

        # Codebase helper functions.
        env["list_files"] = lambda: sorted(list(env.get("repo_index", {}).keys()))
        env["file_meta"] = lambda path: _file_meta(env, path)
        env["file_text"] = lambda path: _file_text(env, path)
        env["file_peek"] = lambda path, start_line=1, n_lines=60: _file_peek(
            env, path, start_line=start_line, n_lines=n_lines
        )
        env["search_files"] = lambda pattern, flags=0, max_results=200: _search_files(
            env, pattern, flags=flags, max_results=max_results
        )
        env["ripgrep"] = lambda pattern, flags=0, max_results=200, context=0: _ripgrep(
            env, pattern, flags=flags, max_results=max_results, context=context
        )
        env["grep_file"] = lambda path, pattern, flags=0, max_results=200, context=0: _grep_file(
            env, path, pattern, flags=flags, max_results=max_results, context=context
        )

        # Convenience.
        env["set_final"] = lambda s: env.__setitem__("Final", s)

        # Recursive call support.
        def _sub_rlm(subprompt: str) -> str:
            return self._run_internal(
                prompt=subprompt,
                repo_prompt=repo_prompt,
                repo_index=repo_index,
                repo_meta=repo_meta,
                depth=depth + 1,
                session_id=session_id,
                logger=logger,
                verbose=verbose,
            )

        env["sub_rlm"] = _sub_rlm

        return env

    def _exec(self, *, code: str, env: Dict[str, Any]) -> "_ExecResult":
        stdout_io = io.StringIO()
        stderr_io = io.StringIO()
        exc_text: str | None = None

        if not code or not code.strip():
            return _ExecResult(stdout="", stderr="(no code to execute)", exception=None)

        try:
            # Redirect print() and stderr writes.
            # We override print in builtins for this exec only by binding it in globals.
            def _print(*args: Any, **kwargs: Any) -> None:
                sep = kwargs.get("sep", " ")
                end = kwargs.get("end", "\n")
                text = sep.join(str(a) for a in args) + end
                stdout_io.write(text)

            # Ensure essential sandbox elements are present even if the model overwrote them.
            safe_builtins = _safe_builtins()
            safe_builtins["print"] = _print
            env["__builtins__"] = safe_builtins

            # Provide print as a global name too.
            env["print"] = _print

            exec(code, env, env)
        except Exception:
            exc_text = traceback.format_exc(limit=20)
            stderr_io.write(exc_text)

        stdout = stdout_io.getvalue()
        stderr = stderr_io.getvalue()

        return _ExecResult(stdout=stdout, stderr=stderr, exception=exc_text)

    def _metadata(
        self,
        *,
        env: Dict[str, Any],
        depth: int,
        iteration: int,
        last_exec: Optional["_ExecResult"],
    ) -> str:
        cfg = self.config

        lines: list[str] = []
        lines.append("# REPL METADATA")
        lines.append(f"depth: {depth} / max_depth: {cfg.max_depth}")
        lines.append(f"iteration: {iteration}")

        # Prompt metadata.
        lines.append("\n## prompt")
        lines.append(f"chars: {env.get('_prompt_chars', '?')}")
        lines.append(f"sha256: {env.get('_prompt_sha256', '?')}")
        prefix = env.get("_prompt_prefix", "")
        lines.append(f"prefix_{cfg.prompt_prefix_chars}: {repr(prefix)}")

        # Repo prompt metadata (when subcalls keep access to the full codebase backing store).
        if env.get("repo_prompt") is not None and env.get("repo_prompt") != env.get("prompt"):
            lines.append("\n## repo_prompt (backing store for repo_index)")
            lines.append(f"chars: {env.get('_repo_prompt_chars', '?')}")
            lines.append(f"sha256: {env.get('_repo_prompt_sha256', '?')}")
            repo_prefix = env.get("_repo_prompt_prefix", "")
            lines.append(f"prefix_{cfg.prompt_prefix_chars}: {repr(repo_prefix)}")
            lines.append("note: file_text/grep/ripgrep use repo_prompt + repo_index offsets")

        # Repo meta (if present).
        if env.get("repo_meta"):
            lines.append("\n## repo_meta (summary)")
            # Keep this compact. Show a few stable keys.
            rm = env["repo_meta"]
            for key in [
                "root",
                "repomix_path",
                "file_count",
                "total_bytes",
                "included_bytes",
                "max_file_bytes",
                "max_total_bytes",
                "tree_lines",
            ]:
                if key in rm:
                    lines.append(f"{key}: {rm[key]}")

            # Extension histogram and largest files are often very helpful.
            if "ext_counts" in rm and isinstance(rm["ext_counts"], dict):
                # Show top 12.
                items = list(rm["ext_counts"].items())[:12]
                lines.append(f"ext_counts_top: {items}")
            if "largest_files" in rm:
                lines.append(f"largest_files: {rm['largest_files']}")

        # State variables snapshot.
        lines.append("\n## state")
        lines.append(_summarize_state(env))

        # Last execution results.
        if last_exec is not None:
            lines.append("\n## last_stdout")
            lines.append(_preview_text(last_exec.stdout, cfg.stdout_preview_chars))

            lines.append("\n## last_stderr")
            lines.append(_preview_text(last_exec.stderr, cfg.stderr_preview_chars))

            if last_exec.exception:
                lines.append("\n## last_exception")
                lines.append(_preview_text(last_exec.exception, cfg.stderr_preview_chars))

        return "\n".join(lines).strip() + "\n"


@dataclass(frozen=True)
class _ExecResult:
    stdout: str
    stderr: str
    exception: str | None


# ------------------------- Helpers (metadata) -------------------------


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _preview_text(text: str, limit: int) -> str:
    if not text:
        return "(empty)"
    if len(text) <= limit:
        return text
    head = text[:limit]
    return head + f"\n...[truncated; chars={len(text)} limit={limit}]"


def _truncate_for_log(text: str, limit: int = 200_000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated for log; chars={len(text)} limit={limit}]"


def _summarize_state(env: Dict[str, Any]) -> str:
    # Avoid dumping huge values.
    keys = [k for k in env.keys() if not k.startswith("_") and k not in {"__builtins__", "prompt"}]
    keys.sort()

    parts: list[str] = []
    parts.append(f"keys: {keys}")

    # Provide lightweight type/value summaries.
    def summarize_value(v: Any) -> str:
        if v is None:
            return "None"
        if isinstance(v, (int, float, bool)):
            return repr(v)
        if isinstance(v, str):
            return f"str(len={len(v)})"
        if isinstance(v, list):
            return f"list(len={len(v)})"
        if isinstance(v, dict):
            return f"dict(len={len(v)})"
        return f"<{type(v).__name__}>"

    sample_keys = keys[:30]
    parts.append("samples:")
    for k in sample_keys:
        try:
            parts.append(f"- {k}: {summarize_value(env.get(k))}")
        except Exception:
            parts.append(f"- {k}: <unavailable>")

    if len(keys) > len(sample_keys):
        parts.append(f"... ({len(keys) - len(sample_keys)} more keys)")

    # Show whether Final is set.
    parts.append(f"Final_set: {env.get('Final') not in (None, '')}")

    return "\n".join(parts)


# ------------------------- Helpers (sandbox) -------------------------


def _safe_builtins() -> Dict[str, Any]:
    # Restrictive by default; add capabilities intentionally.
    allowed: Dict[str, Any] = {
        # Types / constructors
        "bool": bool,
        "int": int,
        "float": float,
        "str": str,
        "dict": dict,
        "list": list,
        "set": set,
        "tuple": tuple,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "min": min,
        "max": max,
        "sum": sum,
        "sorted": sorted,
        "reversed": reversed,
        "abs": abs,
        "all": all,
        "any": any,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "type": type,
        "repr": repr,
        "round": round,
        # Exceptions
        "Exception": Exception,
        "ValueError": ValueError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "TypeError": TypeError,
        "RuntimeError": RuntimeError,
        # Basic helpers
        "print": print,
        "__import__": _safe_import,
    }
    return allowed


_ALLOWED_IMPORTS = {
    "re",
    "json",
    "math",
    "statistics",
    "hashlib",
    "textwrap",
    "collections",
    "itertools",
    "functools",
    "dataclasses",
    "ast",
    "typing",
}


def _safe_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0):
    root = name.split(".", 1)[0]
    if root not in _ALLOWED_IMPORTS:
        raise ImportError(f"Import not allowed: {name}")
    return __import__(name, globals, locals, fromlist, level)


# ------------------------- Helpers (prompt & codebase search) -------------------------


def _prompt_find(text: str, pattern: str, *, flags: int = 0, max_matches: int = 20) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    rx = re.compile(pattern, flags)

    for m in rx.finditer(text):
        out.append(
            {
                "match": m.group(0)[:200],
                "start": m.start(),
                "end": m.end(),
            }
        )
        if len(out) >= max_matches:
            break

    return out


def _file_meta(env: Dict[str, Any], path: str) -> Dict[str, Any]:
    idx = env.get("repo_index") or {}
    if path not in idx:
        raise KeyError(f"Unknown file path: {path}")
    e = idx[path]
    # FileIndexEntry dataclass or dict-like.
    if hasattr(e, "__dict__"):
        d = dict(e.__dict__)
    else:
        d = dict(e)
    # Avoid leaking offsets unless needed.
    return {k: v for k, v in d.items() if k not in {"content_start", "content_end"}}


def _file_text(env: Dict[str, Any], path: str) -> str:
    idx = env.get("repo_index") or {}
    if path not in idx:
        raise KeyError(f"Unknown file path: {path}")
    e = idx[path]
    start = getattr(e, "content_start", None) if not isinstance(e, dict) else e.get("content_start")
    end = getattr(e, "content_end", None) if not isinstance(e, dict) else e.get("content_end")
    if start is None or end is None:
        raise RuntimeError("repo_index entry missing content offsets")
    return env["repo_prompt"][int(start) : int(end)]


def _file_peek(env: Dict[str, Any], path: str, *, start_line: int = 1, n_lines: int = 60) -> str:
    text = _file_text(env, path)
    lines = text.splitlines()
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), start_idx + n_lines)
    out = []
    for i in range(start_idx, end_idx):
        out.append(f"{i+1:>6}: {lines[i]}")
    return "\n".join(out)


def _search_files(env: Dict[str, Any], pattern: str, *, flags: int = 0, max_results: int = 200) -> List[str]:
    idx = env.get("repo_index") or {}
    rx = re.compile(pattern, flags)
    out: List[str] = []
    for p in sorted(idx.keys()):
        if rx.search(p):
            out.append(p)
        if len(out) >= max_results:
            break
    return out


def _ripgrep(
    env: Dict[str, Any],
    pattern: str,
    *,
    flags: int = 0,
    max_results: int = 200,
    context: int = 0,
) -> List[Dict[str, Any]]:
    idx = env.get("repo_index") or {}
    rx = re.compile(pattern, flags)

    results: List[Dict[str, Any]] = []

    for path in sorted(idx.keys()):
        try:
            text = _file_text(env, path)
        except Exception:
            continue

        lines = text.splitlines()
        for i, line in enumerate(lines, start=1):
            if rx.search(line):
                item: Dict[str, Any] = {
                    "file": path,
                    "path": path,
                    "line_no": i, "line_number": i,
                    "line": line[:500],
                }
                if context and context > 0:
                    lo = max(1, i - context)
                    hi = min(len(lines), i + context)
                    ctx = []
                    for j in range(lo, hi + 1):
                        ctx.append({"line_no": j, "line": lines[j - 1][:500]})
                    item["context"] = ctx
                results.append(item)

                if len(results) >= max_results:
                    return results

    return results


def _grep_file(
    env: Dict[str, Any],
    path: str,
    pattern: str,
    *,
    flags: int = 0,
    max_results: int = 200,
    context: int = 0,
) -> List[Dict[str, Any]]:
    rx = re.compile(pattern, flags)
    text = _file_text(env, path)
    lines = text.splitlines()

    results: List[Dict[str, Any]] = []
    for i, line in enumerate(lines, start=1):
        if rx.search(line):
            item: Dict[str, Any] = {
                "file": path,
                "path": path,
                "line_no": i, "line_number": i,
                "line": line[:500],
            }
            if context and context > 0:
                lo = max(1, i - context)
                hi = min(len(lines), i + context)
                ctx = []
                for j in range(lo, hi + 1):
                    ctx.append({"line_no": j, "line": lines[j - 1][:500]})
                item["context"] = ctx
            results.append(item)
            if len(results) >= max_results:
                break

    return results


# ------------------------- Helpers (LLM output sanitization) -------------------------


def _extract_python_code(text: str) -> str:
    """Extract raw python code from an LLM response.

    - If response contains a fenced code block, prefer its contents.
    - Otherwise return the whole response.

    This is defensive: even with strict system prompts, models sometimes emit fences.
    """

    if not text:
        return ""

    # Prefer fenced code blocks.
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if fenced:
        return "\n\n".join(fenced).strip()

    # Strip markdown-style separators (---) and comment-only preambles.
    # Some models emit "--- " between code blocks.
    lines = text.split("\n")
    code_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == "---":
            continue
        code_lines.append(line)

    result = "\n".join(code_lines).strip()
    return result if result else "" 