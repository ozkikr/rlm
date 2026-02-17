from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .codebase import load_codebase_from_dir, load_codebase_from_repomix
from .engine import LiteLLMClient, RLMConfig, RLMEngine


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rlm", description="Recursive Language Models (RLM) scaffold")

    sub = p.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="Ask a question about a codebase")

    src = ask.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", type=str, help="Path to a repository directory")
    src.add_argument("--repomix", type=str, help="Path to repomix-packed text file")

    ask.add_argument("--question", type=str, required=True, help="Question to answer")

    ask.add_argument("--model", type=str, default=RLMConfig().root_model, help="Root model name")
    ask.add_argument("--sub-model", type=str, default="", help="Sub-model for recursive calls (default: auto-infer from root model)")
    ask.add_argument(
        "--max-iterations",
        type=int,
        default=RLMConfig().max_iterations,
        help="Maximum REPL iterations (root call)",
    )
    ask.add_argument(
        "--max-depth",
        type=int,
        default=RLMConfig().max_depth,
        help="Maximum recursion depth",
    )
    ask.add_argument("--verbose", action="store_true", help="Print REPL interactions and log path")

    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd != "ask":
        parser.print_help()
        return 2

    # Load codebase into prompt text.
    if args.repo:
        cb = load_codebase_from_dir(Path(args.repo))
    else:
        cb = load_codebase_from_repomix(Path(args.repomix))

    # Put question at the beginning so it appears in the prompt prefix metadata.
    prefix = f"QUESTION:\n{args.question}\n\n"
    cb = cb.with_prefix(prefix)

    cfg = RLMConfig(
        root_model=args.model,
        sub_model=args.sub_model,
        max_iterations=args.max_iterations,
        max_depth=args.max_depth,
    )

    engine = RLMEngine(client=LiteLLMClient(), config=cfg)

    try:
        answer = engine.run(prompt=cb.text, repo_index=cb.index, repo_meta=cb.meta, verbose=args.verbose)
    except Exception as e:
        print(f"[rlm] error: {e}", file=sys.stderr)
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
