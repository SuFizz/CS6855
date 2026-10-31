"""Deterministic execution, binding and GraphStore rehydration for planned steps."""
from __future__ import annotations

import json

from graphkit import GraphStore
from graphkit.contract import error

from .schema import resolve_pointer, validate_arguments, validate_tool_output


class PlanExecutor:
    def __init__(self, tools, data_dir, trace):
        self.tools = tools
        self.data_dir = data_dir
        self.trace = trace
        self.store = GraphStore(data_dir)

    @staticmethod
    def ready_step(plan, pending, evidence):
        completed = set(evidence)
        for step in plan.get("steps", []):
            if step["id"] in pending and set(step.get("depends_on", [])) <= completed:
                return step
        return None

    def resolve_arguments(self, step, evidence):
        arguments = dict(step.get("arguments", {}))
        for binding in step.get("bindings", []):
            source = evidence[binding["step_id"]]["output"]
            arguments[binding["parameter"]] = resolve_pointer(source, binding["pointer"])
        return arguments

    def execute(self, step, evidence, *, replay=False):
        tool = self.tools[step["tool"]]
        try:
            arguments = self.resolve_arguments(step, evidence)
        except (KeyError, TypeError, ValueError) as exc:
            return None, f"Cannot resolve step {step['id']!r} bindings: {exc}"
        issue = validate_arguments(tool.manifest, arguments)
        if issue:
            return None, issue
        output = tool.call(self.store, **arguments)
        issue = validate_tool_output(output, self.store)
        if issue:
            output = error(issue)
        record = {
            "step_id": step["id"],
            "tool": tool.name,
            "implementation_id": tool.implementation_id,
            "arguments": arguments,
            "output": output,
        }
        self.trace.event("tool_replay" if replay else "tool_call", **record)
        if output.get("status") != "ok":
            return record, output.get("error") or "Tool failed."
        return record, None

    def rehydrate(self, evidence):
        original = list(evidence.values())
        self.store = GraphStore(self.data_dir)
        rebuilt = {}
        for expected in original:
            step = {
                "id": expected["step_id"], "tool": expected["tool"], "purpose": "replay",
                "arguments": expected["arguments"], "bindings": [], "depends_on": [],
            }
            actual, issue = self.execute(step, rebuilt, replay=True)
            if issue:
                raise RuntimeError(f"Cannot rehydrate {step['id']}: {issue}")
            expected_json = json.dumps(expected["output"], sort_keys=True, allow_nan=False)
            actual_json = json.dumps(actual["output"], sort_keys=True, allow_nan=False)
            if expected_json != actual_json:
                raise RuntimeError(f"Replayed output changed for step {step['id']!r}.")
            rebuilt[step["id"]] = actual
        return rebuilt
