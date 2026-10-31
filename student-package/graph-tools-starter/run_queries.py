"""Run natural-language graph queries through the LangGraph orchestrator."""
import argparse
import json
import sys
from pathlib import Path

from orchestrator import Orchestrator, load_queries

ROOT = Path(__file__).resolve().parent
TERMINAL_STATUSES = {"answered", "clarification_needed", "unsupported"}


def _load_existing(path):
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Existing answers file must contain a JSON list.")
    return payload


def _write_atomic(path, answers):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(answers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, help="Text, JSON, or JSONL query file.")
    parser.add_argument("--out", required=True, help="Destination answers JSON file.")
    parser.add_argument("--trace-dir", default="traces", help="Directory for per-query JSONL traces.")
    parser.add_argument("--state-dir", default=".orchestrator", help="Checkpoint and cache directory.")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted queries and skip terminal answers.")
    parser.add_argument("--retry-errors", action="store_true", help="With --resume, rerun previous error results.")
    parser.add_argument("--no-cache", action="store_true", help="Disable the Gemini response cache.")
    parser.add_argument("--allow-incomplete-registry", action="store_true", help="Warn and continue with missing or duplicate curated tools.")
    args = parser.parse_args()
    output = Path(args.out)
    try:
        queries = load_queries(args.queries)
        existing = _load_existing(output) if args.resume else []
        orchestrator = Orchestrator(
            ROOT, state_dir=ROOT / args.state_dir, cache_enabled=not args.no_cache,
            allow_incomplete_registry=args.allow_incomplete_registry,
        )
    except Exception as exc:
        print(f"Setup error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    trace_dir = Path(args.trace_dir)
    if not trace_dir.is_absolute():
        trace_dir = ROOT / trace_dir
    previous = {item.get("query_id"): item for item in existing if isinstance(item, dict)}
    answers = []
    try:
        for query in queries:
            old = previous.get(query["query_id"])
            skip = args.resume and old and (old.get("status") in TERMINAL_STATUSES or
                                              old.get("status") == "error" and not args.retry_errors)
            if skip:
                answer = old
            else:
                safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in query["query_id"]).strip("._") or "query"
                resume_workflow = args.resume and not (old and old.get("status") == "error" and args.retry_errors)
                answer = orchestrator.answer(
                    query["query_id"], query["question"], trace_dir / f"{safe_id}.jsonl",
                    resume=resume_workflow,
                )
            answers.append(answer)
            _write_atomic(output, answers)
    finally:
        orchestrator.close()
    print(f"Wrote {len(answers)} answer(s) to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
