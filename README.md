# RLM — Recursive Language Models for Codebase Research

A scaffold implementing the core idea from the [RLM paper](https://arxiv.org/abs/2512.24601v2): treat the prompt as an external environment variable that the LLM explores via a Python REPL, rather than stuffing it all into context.

## How it works

1. Your codebase is loaded into a string (`prompt`) with a file index
2. The LLM receives only **metadata** about the prompt (length, hash, prefix) — never the full text
3. Each turn, the LLM writes Python code that runs in a sandboxed REPL
4. Helper functions (`file_text()`, `ripgrep()`, `search_files()`, etc.) let the LLM navigate the codebase
5. The loop continues until the LLM sets `Final` with its answer
6. Sub-calls (`sub_rlm()`) enable recursive delegation for complex analyses

This means even multi-million-token codebases can be analyzed without blowing up context windows.

## Install

```bash
# Clone
git clone https://github.com/ozkikr/rlm.git
cd rlm

# Install with uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -e .

# Or pip
pip install -e .
```

## Usage

```bash
# Ask a question about a local repo
rlm ask --repo /path/to/your/project --question "How does the authentication system work?"

# Use a specific model (litellm format)
rlm ask --repo ./my-app --question "Find all SQL injection risks" --model openrouter/anthropic/claude-sonnet-4-5

# Verbose mode (shows REPL iterations)
rlm ask --repo ./my-app --question "Map the data flow from API to database" --verbose

# Use a repomix-packed file instead of a directory
rlm ask --repomix ./packed-repo.txt --question "Summarize the architecture"

# Control iteration limits
rlm ask --repo ./my-app --question "..." --max-iterations 12 --max-depth 2
```

## Configuration

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `RLM_MODEL` | `openrouter/gpt-5` | Root model |
| `RLM_SUB_MODEL` | *(auto-inferred)* | Sub-model for recursive calls |
| `RLM_TEMPERATURE` | `0.0` | Sampling temperature |
| `RLM_MAX_ITERATIONS` | `24` | Max REPL turns (root) |
| `RLM_SUB_MAX_ITERATIONS` | `12` | Max REPL turns (sub-calls) |
| `RLM_MAX_DEPTH` | `3` | Max recursion depth |
| `RLM_LOG_DIR` | `.rlm_logs` | JSONL log directory |

The sub-model is automatically inferred from the root model's provider prefix. For example, if you use `openrouter/anthropic/claude-sonnet-4-5`, the sub-model will be `openrouter/anthropic/claude-haiku-3-5`.

## REPL helpers available to the LLM

**Prompt inspection:**
- `prompt_len()` — total chars
- `prompt_peek(n, start)` — read a window
- `prompt_slice(start, end)` — exact slice
- `prompt_find(pattern)` — regex search with offsets

**Codebase navigation:**
- `list_files()` — all file paths
- `file_text(path)` — full file content
- `file_peek(path, start_line, n_lines)` — numbered line window
- `search_files(pattern)` — regex match on file paths
- `ripgrep(pattern)` — grep across all files
- `grep_file(path, pattern)` — grep within one file
- `file_meta(path)` — size, lines, sha256

**Recursion:**
- `sub_rlm(subprompt)` — delegate a sub-analysis

## How it's different from RAG

RAG retrieves chunks and hopes they're relevant. RLM lets the model **decide what to read**, iteratively, like a human developer navigating a codebase. It can:
- Start broad (`list_files()`, `search_files()`)
- Drill into specific files (`file_text()`, `file_peek()`)
- Search for patterns (`ripgrep()`)
- Delegate sub-tasks (`sub_rlm()`)
- Build up understanding across multiple turns

## Logs

Every session writes a JSONL log to `.rlm_logs/` with full code, stdout, and metadata for each iteration. Useful for debugging and understanding how the model explored the codebase.

## License

MIT
