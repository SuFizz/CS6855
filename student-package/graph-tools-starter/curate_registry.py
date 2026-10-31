"""Evaluate Registry.zip implementations and materialize one selected implementation per tool."""
import argparse
from pathlib import Path

from orchestrator.curation import evaluate_registry, materialize_registry, write_evaluation
from graphkit.validate import validate
from orchestrator.registry import discover_project_tools

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--source", type=Path, required=True)
    evaluate.add_argument("--report", type=Path, default=ROOT / ".orchestrator" / "curation_report.json")
    evaluate.add_argument("--selection", type=Path, default=ROOT / ".orchestrator" / "registry_selection.json")
    evaluate.add_argument("--repeats", type=int, default=2)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--source", type=Path, required=True)
    materialize.add_argument("--selection", type=Path, required=True)
    materialize.add_argument("--dest", type=Path, default=ROOT / "registry" / "tools")
    materialize.add_argument("--replace", action="store_true")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--project-root", type=Path, default=ROOT)
    verify.add_argument("--allow-incomplete-registry", action="store_true")

    args = parser.parse_args()
    if args.command == "evaluate":
        results, selection = evaluate_registry(args.source.resolve(), ROOT / "data", repeats=args.repeats)
        write_evaluation(results, selection, args.report, args.selection)
        print(f"Evaluated {len(results)} implementations; selected {len(selection)} of 33 tools.")
        print(f"Report: {args.report}\nSelection: {args.selection}")
        return 0 if len(selection) == 33 else 1
    if args.command == "materialize":
        destination = materialize_registry(
            args.source.resolve(), args.selection, args.dest, replace=args.replace, data_dir=ROOT / "data",
        )
        print(f"Materialized curated registry at {destination}")
        return 0
    tools = discover_project_tools(args.project_root, allow_incomplete=args.allow_incomplete_registry)
    failures = {name: validate(tool.path).errors for name, tool in tools.items() if validate(tool.path).errors}
    if failures:
        for name, errors in failures.items():
            print(f"{name}: {' | '.join(errors)}")
        return 1
    print(f"Loaded and validated {len(tools)} curated analytical and infrastructure tools.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
