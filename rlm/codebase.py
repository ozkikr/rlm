from __future__ import annotations

import dataclasses
import hashlib
import io
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    "node_modules",
    "target",
}

# A conservative set of binary-ish file suffixes to skip by default.
DEFAULT_EXCLUDE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
    ".jar",
    ".war",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".obj",
    ".o",
    ".a",
    ".class",
    ".pyc",
    ".pyo",
    ".pyd",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",
    ".mp3",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
}


def estimate_tokens(text: str, *, model: str | None = None) -> int:
    """Best-effort token estimate.

    This is intentionally lightweight and dependency-free.

    If `tiktoken` is installed and a compatible model name is provided, it will be used.
    Otherwise, we fall back to a rough heuristic (chars / 4).
    """

    # Optional accurate counting for OpenAI-style models.
    if model:
        try:
            import tiktoken  # type: ignore

            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            pass

    # Heuristic: for English/code, 3-4 chars/token is a common ballpark.
    # We bias slightly upward to avoid under-estimating.
    return max(1, (len(text) + 3) // 4)


def _is_probably_binary(sample: bytes) -> bool:
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    # Heuristic: lots of non-text bytes.
    text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x7F)))
    nontext = sum(b not in text_chars for b in sample)
    return (nontext / max(1, len(sample))) > 0.30


@dataclass(frozen=True)
class FileIndexEntry:
    path: str
    size_bytes: int
    line_count: int
    sha256: str
    truncated: bool
    content_start: int
    content_end: int


@dataclass(frozen=True)
class CodebasePrompt:
    """A structured prompt and an index that points into it."""

    text: str
    index: dict[str, FileIndexEntry]
    meta: dict[str, Any]

    def with_prefix(self, prefix: str) -> "CodebasePrompt":
        """Return a new CodebasePrompt with `prefix` prepended, shifting index offsets."""
        if not prefix:
            return self
        shifted: dict[str, FileIndexEntry] = {}
        shift = len(prefix)
        for p, e in self.index.items():
            shifted[p] = dataclasses.replace(
                e,
                content_start=e.content_start + shift,
                content_end=e.content_end + shift,
            )
        meta = dict(self.meta)
        meta["prefixed_by_chars"] = shift
        return CodebasePrompt(text=prefix + self.text, index=shifted, meta=meta)


def load_codebase_from_dir(
    repo_path: str | Path,
    *,
    exclude_dirs: set[str] | None = None,
    exclude_suffixes: set[str] | None = None,
    max_file_bytes: int = 512_000,
    max_total_bytes: int | None = None,
) -> CodebasePrompt:
    """Load a directory into a single prompt string plus an index.

    The prompt includes:
      - A tree/manifest (paths, sizes, line counts)
      - Each file wrapped in BEGIN/END delimiters

    For very large files, content is truncated to `max_file_bytes`.

    `max_total_bytes` can be set to cap total included bytes; additional files are omitted
    after the cap is reached (but still listed in the tree).
    """

    root = Path(repo_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Repo path is not a directory: {root}")

    exclude_dirs = set(exclude_dirs or DEFAULT_EXCLUDE_DIRS)
    exclude_suffixes = set(exclude_suffixes or DEFAULT_EXCLUDE_SUFFIXES)

    # Collect candidate files.
    file_paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Mutate dirnames in-place to prune traversal.
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs and not d.startswith(".")]

        for fn in filenames:
            if fn.startswith("."):
                continue
            p = Path(dirpath) / fn
            if not p.is_file():
                continue
            if p.suffix.lower() in exclude_suffixes:
                continue
            file_paths.append(p)

    file_paths.sort()

    # Read files and compute per-file metadata.
    manifest: list[tuple[str, int, int, str, bool, str]] = []
    # tuple: (rel_path, size_bytes, line_count, sha256, truncated, content)

    total_included_bytes = 0
    omitted_due_to_total_cap: list[str] = []

    for p in file_paths:
        rel = p.relative_to(root).as_posix()
        size_bytes = p.stat().st_size

        if max_total_bytes is not None and total_included_bytes >= max_total_bytes:
            omitted_due_to_total_cap.append(rel)
            # Still include in manifest with empty content and truncated flag.
            manifest.append((rel, size_bytes, 0, "", True, ""))
            continue

        with p.open("rb") as f:
            sample = f.read(4096)
            if _is_probably_binary(sample):
                # Skip binary.
                continue
            f.seek(0)
            raw = f.read()

        sha256 = hashlib.sha256(raw).hexdigest()

        truncated = False
        if len(raw) > max_file_bytes:
            truncated = True
            raw = raw[:max_file_bytes]

        try:
            content = raw.decode("utf-8", errors="replace")
        except Exception:
            # Extremely rare with errors="replace", but keep safe.
            content = raw.decode("latin-1", errors="replace")

        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

        total_included_bytes += len(raw)
        manifest.append((rel, size_bytes, line_count, sha256, truncated, content))

    # Build tree/manifest text.
    ext_counts: Counter[str] = Counter()
    total_files = 0
    total_bytes = 0
    largest: list[tuple[int, str]] = []
    for rel, size_bytes, line_count, sha256, truncated, content in manifest:
        total_files += 1
        total_bytes += size_bytes
        ext_counts[Path(rel).suffix.lower() or "<none>"] += 1
        largest.append((size_bytes, rel))

    largest.sort(reverse=True)
    largest_preview = [
        {"path": rel, "size_bytes": sz}
        for (sz, rel) in largest[: min(10, len(largest))]
    ]

    tree_text = _render_tree([(rel, size, line_count) for (rel, size, line_count, *_rest) in manifest])

    # Build prompt with stable delimiters and record content offsets.
    out = io.StringIO()
    out.write("<CODEBASE>\n")
    out.write(f"ROOT: {root.as_posix()}\n")
    out.write("\n<TREE>\n")
    out.write(tree_text)
    out.write("\n</TREE>\n")

    if omitted_due_to_total_cap:
        out.write("\n<OMITTED_FILES reason=\"max_total_bytes\">\n")
        for rel in omitted_due_to_total_cap[:200]:
            out.write(rel + "\n")
        if len(omitted_due_to_total_cap) > 200:
            out.write(f"... ({len(omitted_due_to_total_cap) - 200} more)\n")
        out.write("</OMITTED_FILES>\n")

    out.write("\n<FILES>\n")

    index: dict[str, FileIndexEntry] = {}

    for rel, size_bytes, line_count, sha256, truncated, content in manifest:
        out.write(f"===== BEGIN FILE: {rel} =====\n")
        content_start = out.tell()

        if content:
            out.write(content)
            if not content.endswith("\n"):
                out.write("\n")
        else:
            if sha256 == "":
                out.write("[OMITTED]\n")
            else:
                out.write("[EMPTY FILE]\n")

        if truncated:
            out.write("[TRUNCATED]\n")

        content_end = out.tell()
        out.write(f"===== END FILE: {rel} =====\n\n")

        index[rel] = FileIndexEntry(
            path=rel,
            size_bytes=size_bytes,
            line_count=line_count,
            sha256=sha256,
            truncated=truncated,
            content_start=content_start,
            content_end=content_end,
        )

    out.write("</FILES>\n")
    out.write("</CODEBASE>\n")

    text = out.getvalue()
    meta: dict[str, Any] = {
        "root": root.as_posix(),
        "file_count": total_files,
        "total_bytes": total_bytes,
        "included_bytes": total_included_bytes,
        "max_file_bytes": max_file_bytes,
        "max_total_bytes": max_total_bytes,
        "ext_counts": dict(ext_counts.most_common()),
        "largest_files": largest_preview,
        "tree_lines": tree_text.count("\n") + 1,
        "prompt_chars": len(text),
        "approx_prompt_tokens": estimate_tokens(text),
    }

    return CodebasePrompt(text=text, index=index, meta=meta)


def load_codebase_from_repomix(path: str | Path) -> CodebasePrompt:
    """Load repomix-packed text as a prompt.

    Repomix formats vary; this loader does not attempt to fully parse file boundaries.
    It provides basic metadata and leaves deeper parsing to the RLM REPL helpers (regex).
    """

    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Repomix path is not a file: {p}")

    text = p.read_text(encoding="utf-8", errors="replace")

    # Very light structure hints.
    file_marker_matches = len(re.findall(r"(?im)^\s*(?:file|path)\s*:\s*.+$", text))
    begin_markers = len(re.findall(r"(?im)^=+\s*begin\s+file\s*:\s*.+?=+\s*$", text))

    meta: dict[str, Any] = {
        "repomix_path": p.as_posix(),
        "prompt_chars": len(text),
        "approx_prompt_tokens": estimate_tokens(text),
        "heuristic_file_marker_lines": file_marker_matches,
        "heuristic_begin_file_markers": begin_markers,
    }

    return CodebasePrompt(text=text, index={}, meta=meta)


def _render_tree(entries: Iterable[tuple[str, int, int]]) -> str:
    """Render a simple directory tree.

    entries: (rel_path, size_bytes, line_count)
    """

    # Build nested mapping.
    root: dict[str, Any] = {}

    for rel, size, lines in entries:
        parts = rel.split("/")
        node = root
        for part in parts[:-1]:
            key = part + "/"
            node = node.setdefault(key, {})
        node[parts[-1]] = {"__file__": True, "size": size, "lines": lines}

    out_lines: list[str] = []

    def walk(node: dict[str, Any], indent: str) -> None:
        for name in sorted(node.keys()):
            val = node[name]
            if isinstance(val, dict) and val.get("__file__"):
                out_lines.append(f"{indent}- {name} ({val['size']} B, {val['lines']} L)")
            elif isinstance(val, dict):
                out_lines.append(f"{indent}{name}")
                walk(val, indent + "  ")

    walk(root, "")
    return "\n".join(out_lines)
