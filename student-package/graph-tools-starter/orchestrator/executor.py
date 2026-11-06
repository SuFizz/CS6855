"""Deterministic execution, binding and GraphStore rehydration for planned steps."""
from __future__ import annotations

import json

from graphkit import GraphStore
from graphkit.contract import error
from graphkit.llm import tool_schema

from .schema import resolve_pointer, validate_arguments, validate_tool_output
from .graph_catalogue import find_entity_hints


class PlanExecutor:
    def __init__(self, tools, data_dir, trace, provider=None, question=""):
        self.tools = tools
        self.data_dir = data_dir
        self.trace = trace
        self.provider = provider
        self.question = question
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

    def prepare_arguments(self, step, evidence, arguments):
        tool = self.tools[step["tool"]]
        required = tool.manifest.get("parameters", {}).get("required", [])
        missing = [name for name in required if name not in arguments]
        if "node" in missing and arguments.get("graph_id"):
            prior_nodes = {
                record.get("arguments", {}).get("node")
                for record in evidence.values()
                if record.get("tool") == tool.name
                and record.get("arguments", {}).get("graph_id") == arguments["graph_id"]
                and isinstance(record.get("arguments", {}).get("node"), str)
            }
            if len(prior_nodes) == 1:
                arguments["node"] = next(iter(prior_nodes))
                missing.remove("node")
        if not missing:
            return arguments, 0, None
        if self.provider is None:
            return arguments, 0, f"Missing required parameter(s) for {tool.name}: {', '.join(missing)}."
        dependency_outputs = {step_id: evidence[step_id]["output"] for step_id in step.get("depends_on", []) if step_id in evidence}
        entity_hints = find_entity_hints(self.data_dir, self.question)
        payload = {"question": self.question, "step": {"id": step["id"], "purpose": step["purpose"]},
                   "known_arguments": arguments, "missing_required": missing,
                   "available_graph_ids": [item["graph_id"] for item in self.store.list_graphs()],
                   "entity_hints": entity_hints,
                   "dependency_outputs": dependency_outputs}
        messages = [
            {"role": "system", "content": "Call the supplied graph tool exactly once. Fill missing required arguments from the original question, exact entity_hints IDs, and dependency_outputs. Preserve every known argument exactly. Include user-requested optional values such as k, seed and top_k when stated. For graph_id use only an exact available_graph_ids value or a dependency-produced result_graph_id. For node identifiers, copy an exact ID from entity_hints or dependency_outputs; never strip prefixes, convert IDs to numbers, or invent an ID. Do not invent, rename or transform graph or node IDs. For lookup_node, if a phrase contains a short uppercase/alphanumeric code token, use that exact token as name instead of including surrounding city, location or generic words."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            message = self.provider.complete(
                messages=messages, tools=[tool_schema(tool.manifest)],
                tool_choice={"type": "function", "function": {"name": tool.name}},
                purpose=f"arguments:{step['id']}", trace=self.trace,
            )
            calls = message.get("tool_calls") or []
            selected = next(call for call in calls if call.get("function", {}).get("name") == tool.name)
            raw = selected["function"].get("arguments", "{}")
            generated = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(generated, dict):
                raise ValueError("tool arguments must be an object")
            graph_id = generated.get("graph_id", arguments.get("graph_id"))
            node_id = generated.get("node", arguments.get("node"))
            if node_id is not None and graph_id and self.store.has_graph(graph_id):
                graph = self.store.get(graph_id)
                if not graph.has_node(node_id):
                    hinted_ids = {
                        match["id"]
                        for hint in entity_hints if hint["graph_id"] == graph_id
                        for match in hint["matches"]
                    }
                    exact_hint = next(
                        (node for node in hinted_ids if node.casefold() == str(node_id).casefold()),
                        None,
                    )
                    if exact_hint is None and len(hinted_ids) == 1:
                        exact_hint = next(iter(hinted_ids))
                    if exact_hint is None:
                        raise ValueError(
                            f"Generated node ID {node_id!r} is not valid in graph {graph_id!r}."
                        )
                    generated["node"] = exact_hint
        except Exception as exc:
            return arguments, 1, f"Could not prepare arguments for {tool.name}: {type(exc).__name__}: {exc}"
        generated.update(arguments)
        return generated, 1, None

    def execute(self, step, evidence, *, replay=False):
        tool = self.tools[step["tool"]]
        try:
            arguments = self.resolve_arguments(step, evidence)
        except (KeyError, TypeError, ValueError) as exc:
            return None, f"Cannot resolve step {step['id']!r} bindings: {exc}", 0
        llm_calls = 0
        if not replay:
            arguments, llm_calls, issue = self.prepare_arguments(step, evidence, arguments)
            if issue:
                return None, issue, llm_calls
        properties = tool.manifest.get("parameters", {}).get("properties", {})
        graph_id = arguments.get("graph_id")
        if graph_id is not None and not self.store.has_graph(graph_id):
            available = [item["graph_id"] for item in self.store.list_graphs()]
            return None, f"Unknown graph_id {graph_id!r}; valid graph IDs are {available}.", llm_calls
        if graph_id and tool.manifest.get("category") == "centrality":
            graph = self.store.get(graph_id)
            distance_weights = graph.weighted and any(token in graph.description.lower() for token in ("distance", " km", "kilomet"))
            if distance_weights:
                for parameter in ("weighted", "use_weights"):
                    if parameter in properties and parameter not in arguments:
                        arguments[parameter] = False
        if tool.name == "personalized_pagerank" and any(phrase in self.question.lower() for phrase in ("other than", "apart from", "excluding")):
            if "exclude_seeds" in properties and "exclude_seeds" not in arguments:
                arguments["exclude_seeds"] = True
            if "include_source" in properties and "include_source" not in arguments:
                arguments["include_source"] = False
        issue = validate_arguments(tool.manifest, arguments)
        if issue:
            return None, issue, llm_calls
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
            return record, output.get("error") or "Tool failed.", llm_calls
        return record, None, llm_calls

    def rehydrate(self, evidence):
        original = list(evidence.values())
        self.store = GraphStore(self.data_dir)
        rebuilt = {}
        for expected in original:
            step = {
                "id": expected["step_id"], "tool": expected["tool"], "purpose": "replay",
                "arguments": expected["arguments"], "bindings": [], "depends_on": [],
            }
            actual, issue, _ = self.execute(step, rebuilt, replay=True)
            if issue:
                raise RuntimeError(f"Cannot rehydrate {step['id']}: {issue}")
            expected_json = json.dumps(expected["output"], sort_keys=True, allow_nan=False)
            actual_json = json.dumps(actual["output"], sort_keys=True, allow_nan=False)
            if expected_json != actual_json:
                raise RuntimeError(f"Replayed output changed for step {step['id']!r}.")
            rebuilt[step["id"]] = actual
        return rebuilt
