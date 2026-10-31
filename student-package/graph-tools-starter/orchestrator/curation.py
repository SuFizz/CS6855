"""Offline evaluation, selection and byte-preserving registry materialization."""
from __future__ import annotations

import json
import shutil
import statistics
import time
from pathlib import Path

from graphkit import GraphStore, load_tool
from graphkit.menu import MENU
from graphkit.validate import validate

from .curation_cases import adapted_arguments, semantic_check
from .schema import validate_tool_output


def candidate_folders(source):
    return sorted(path.parent for path in Path(source).rglob("tool.json") if "template" not in path.parts)


def evaluate_candidate(folder, data_dir, repeats=2):
    report = validate(folder)
    if report.errors:
        return {"status": "rejected", "path": str(folder), "errors": report.errors}
    try:
        tool = load_tool(folder)
        arguments = adapted_arguments(tool)
    except Exception as exc:
        return {"status": "rejected", "path": str(folder), "errors": [f"load/setup: {type(exc).__name__}: {exc}"]}
    outputs, timings = [], []
    semantic_messages = []
    for _ in range(repeats):
        store = GraphStore(data_dir)
        graph_id = arguments.get("graph_id")
        if graph_id:
            store.get(graph_id)
        started = time.perf_counter()
        output = tool.call(store, **arguments)
        timings.append((time.perf_counter() - started) * 1000)
        issue = validate_tool_output(output, store)
        if issue:
            return {"status": "rejected", "tool": tool.name, "implementation_id": tool.implementation_id,
                    "path": str(folder), "arguments": arguments, "errors": [issue]}
        if output["status"] != "ok":
            return {"status": "rejected", "tool": tool.name, "implementation_id": tool.implementation_id,
                    "path": str(folder), "arguments": arguments, "errors": [output["error"]]}
        passed, message = semantic_check(tool, output, store, arguments)
        semantic_messages.append(message)
        if not passed:
            return {"status": "rejected", "tool": tool.name, "implementation_id": tool.implementation_id,
                    "path": str(folder), "arguments": arguments, "errors": [message]}
        outputs.append(json.dumps(output, sort_keys=True, allow_nan=False))
    if len(set(outputs)) != 1:
        return {"status": "rejected", "tool": tool.name, "implementation_id": tool.implementation_id,
                "path": str(folder), "arguments": arguments, "errors": ["canonical case is nondeterministic"]}
    returns_subgraph = MENU.get(tool.name, (None, None, False))[2]
    return {"status": "eligible", "tool": tool.name, "implementation_id": tool.implementation_id,
            "path": str(folder), "arguments": arguments, "returns_subgraph": returns_subgraph,
            "median_ms": round(statistics.median(timings), 3), "warnings": report.warnings,
            "semantic_check": semantic_messages[0] if semantic_messages else "passed"}


def evaluate_registry(source, data_dir, repeats=2):
    results = [evaluate_candidate(folder, data_dir, repeats=repeats) for folder in candidate_folders(source)]
    grouped = {}
    for result in results:
        if result.get("tool"):
            grouped.setdefault(result["tool"], []).append(result)
    selection = {}
    for name in sorted(MENU):
        eligible = [row for row in grouped.get(name, []) if row["status"] == "eligible"]
        if eligible:
            winner = min(eligible, key=lambda row: (row["median_ms"], row["implementation_id"]))
            selection[name] = winner["implementation_id"]
    return results, selection


def write_evaluation(results, selection, report_path, selection_path):
    report_path, selection_path = Path(report_path), Path(selection_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"results": results, "selection": selection}, indent=2) + "\n", encoding="utf-8")
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_selection(path):
    selection = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(selection, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in selection.items()):
        raise ValueError("Selection must be an object mapping tool names to implementation IDs.")
    missing = sorted(set(MENU) - set(selection))
    extra = sorted(set(selection) - set(MENU))
    if missing or extra:
        raise ValueError(f"Selection must contain exactly the 33 menu tools; missing={missing}, extra={extra}.")
    return selection


def materialize_registry(source, selection_path, destination, replace=False, data_dir=None, verify_candidates=True):
    source, destination = Path(source), Path(destination)
    selection = load_selection(selection_path)
    indexed = {}
    for folder in candidate_folders(source):
        manifest = json.loads((folder / "tool.json").read_text(encoding="utf-8"))
        indexed[manifest.get("implementation_id")] = folder
    absent = sorted(implementation_id for implementation_id in selection.values() if implementation_id not in indexed)
    if absent:
        raise ValueError(f"Selected implementations are absent from source: {', '.join(absent)}")
    if verify_candidates:
        candidate_data = Path(data_dir) if data_dir else destination.parents[1] / "data"
        failures = []
        for tool_name, implementation_id in sorted(selection.items()):
            result = evaluate_candidate(indexed[implementation_id], candidate_data, repeats=2)
            if result["status"] != "eligible":
                failures.append(f"{tool_name}={implementation_id}: {' | '.join(result.get('errors', []))}")
        if failures:
            raise ValueError("Selected implementations failed verification: " + " ; ".join(failures))
    existing = [path for path in destination.iterdir() if path.name != ".gitkeep"] if destination.is_dir() else []
    if existing and not replace:
        raise FileExistsError(f"{destination} is not empty; use --replace to rebuild it.")
    if replace and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for tool_name, implementation_id in sorted(selection.items()):
        target = destination / tool_name / indexed[implementation_id].name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(indexed[implementation_id], target)
    return destination
