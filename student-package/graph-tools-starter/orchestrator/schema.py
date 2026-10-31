"""Schema, JSON Pointer, plan, output and final-answer validation."""
from __future__ import annotations

import json
from typing import Any

from graphkit.contract import MAX_OUTPUT_BYTES, OUTPUT_KEYS

ANSWER_TYPES = ("number", "boolean", "node", "node_list", "path", "text")
STATUSES = ("answered", "clarification_needed", "unsupported", "error")


def _type_matches(value, expected):
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return False


def validate_value(value, schema, path="value"):
    expected = schema.get("type")
    if expected and not _type_matches(value, expected):
        return f"{path} must have type {expected}."
    if "enum" in schema and value not in schema["enum"]:
        return f"{path} must be one of {schema['enum']}."
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path} must be at least {schema['minimum']}."
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path} must be at most {schema['maximum']}."
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            issue = validate_value(item, schema["items"], f"{path}[{index}]")
            if issue:
                return issue
    if isinstance(value, dict) and expected == "object":
        properties = schema.get("properties", {})
        missing = [name for name in schema.get("required", []) if name not in value]
        if missing:
            return f"{path} is missing required field(s): {', '.join(missing)}."
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                return f"{path} has unknown field(s): {', '.join(unknown)}."
        for name, item in value.items():
            if name in properties:
                issue = validate_value(item, properties[name], f"{path}.{name}")
                if issue:
                    return issue
    return None


def validate_arguments(manifest, arguments, bound_parameters=()):
    if not isinstance(arguments, dict):
        return "Tool arguments must be an object."
    schema = manifest.get("parameters", {})
    properties = schema.get("properties", {})
    bound = set(bound_parameters)
    unknown = sorted(set(arguments) - set(properties))
    if unknown:
        return f"Unknown parameter(s) for {manifest.get('name')}: {', '.join(unknown)}."
    missing = sorted(name for name in schema.get("required", []) if name not in arguments and name not in bound)
    if missing:
        return f"Missing required parameter(s) for {manifest.get('name')}: {', '.join(missing)}."
    for name, value in arguments.items():
        issue = validate_value(value, properties.get(name, {}), name)
        if issue:
            return issue
    return None


def resolve_pointer(document, pointer):
    if pointer == "":
        return document
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("JSON Pointer must be empty or start with '/'.")
    value = document
    for encoded in pointer.split("/")[1:]:
        token = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            try:
                value = value[int(token)]
            except (ValueError, IndexError) as exc:
                raise KeyError(pointer) from exc
        elif isinstance(value, dict) and token in value:
            value = value[token]
        else:
            raise KeyError(pointer)
    return value


def _walk_dicts(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_dicts(item)


def validate_tool_output(output, store):
    if not isinstance(output, dict) or set(output) != set(OUTPUT_KEYS):
        return "Tool output does not follow the five-key contract."
    try:
        encoded = json.dumps(output, allow_nan=False)
    except (TypeError, ValueError) as exc:
        return f"Tool output is not valid JSON: {exc}."
    if len(encoded.encode("utf-8")) > MAX_OUTPUT_BYTES:
        return f"Tool output exceeds {MAX_OUTPUT_BYTES} bytes."
    status = output.get("status")
    if status not in ("ok", "error"):
        return "Tool output status must be ok or error."
    if status == "ok":
        if output.get("error") is not None or not isinstance(output.get("summary"), str) or not output["summary"].strip():
            return "Tool success output has inconsistent error or summary fields."
    elif not isinstance(output.get("error"), str) or not output["error"].strip():
        return "Tool error output needs a non-empty error message."
    for item in _walk_dicts(output.get("result")):
        if "id" in item and not isinstance(item.get("label"), str):
            return f"Node {item.get('id')!r} has no label."
    graph_id = output.get("result_graph_id")
    if graph_id is not None and (not isinstance(graph_id, str) or not store.has_graph(graph_id)):
        return "Tool output has an unregistered result_graph_id."
    return None


def validate_plan(plan, tools, max_steps=5, completed_ids=()):
    if not isinstance(plan, dict):
        return ["Plan must be an object."]
    mode = plan.get("mode")
    if mode not in ("execute", "clarification_needed", "unsupported"):
        return ["Plan mode is invalid."]
    if plan.get("answer_type") not in ANSWER_TYPES:
        return ["Plan answer_type is invalid."]
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return ["Plan steps must be a list."]
    if mode != "execute" and steps:
        return ["Terminal clarification/unsupported plans cannot contain steps."]
    if len(steps) > max_steps:
        return [f"Plan has {len(steps)} steps; maximum is {max_steps}."]
    errors = []
    ids = [step.get("id") for step in steps if isinstance(step, dict)]
    if len(ids) != len(steps) or any(not isinstance(step_id, str) or not step_id for step_id in ids):
        errors.append("Every step needs a non-empty string id.")
    if len(ids) != len(set(ids)):
        errors.append("Plan step ids must be unique.")
    known = set(ids) | set(completed_ids)
    graph = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = step.get("id")
        tool_name = step.get("tool")
        if tool_name not in tools:
            errors.append(f"Step {step_id!r} names unknown tool {tool_name!r}.")
            continue
        dependencies = step.get("depends_on", [])
        bindings = step.get("bindings", [])
        if not isinstance(dependencies, list) or any(dep not in known or dep == step_id for dep in dependencies):
            errors.append(f"Step {step_id!r} has invalid dependencies.")
            dependencies = []
        graph[step_id] = set(dependencies)
        if not isinstance(bindings, list):
            errors.append(f"Step {step_id!r} bindings must be a list.")
            bindings = []
        bound = []
        for binding in bindings:
            if not isinstance(binding, dict):
                errors.append(f"Step {step_id!r} has a non-object binding.")
                continue
            parameter = binding.get("parameter")
            source = binding.get("step_id")
            pointer = binding.get("pointer")
            if source not in dependencies:
                errors.append(f"Step {step_id!r} binding source {source!r} must be a dependency.")
            try:
                resolve_pointer({}, pointer)
            except KeyError:
                pass
            except (TypeError, ValueError) as exc:
                errors.append(f"Step {step_id!r} binding for {parameter!r}: {exc}")
            bound.append(parameter)
        issue = validate_arguments(tools[tool_name].manifest, step.get("arguments", {}), bound)
        if issue:
            errors.append(f"Step {step_id!r}: {issue}")
    visiting, visited = set(), set()
    def visit(node):
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(dep) for dep in graph.get(node, ())):
            return True
        visiting.remove(node)
        visited.add(node)
        return False
    if any(visit(node) for node in graph):
        errors.append("Plan dependency graph contains a cycle.")
    return errors


def parse_answer(arguments, evidence):
    status = arguments.get("status")
    text = arguments.get("answer_text")
    answer_type = arguments.get("answer_type")
    raw = arguments.get("answer_value_json")
    references = arguments.get("evidence", [])
    if status not in STATUSES:
        return None, "Final status is invalid."
    if not isinstance(text, str) or not text.strip():
        return None, "answer_text must be non-empty."
    if answer_type not in ANSWER_TYPES:
        return None, "answer_type is invalid."
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as exc:
        return None, f"answer_value_json is invalid: {exc}."
    valid = {
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "node": isinstance(value, str),
        "node_list": isinstance(value, list) and all(isinstance(item, str) for item in value),
        "path": isinstance(value, list) and all(isinstance(item, str) for item in value),
        "text": isinstance(value, str),
    }[answer_type]
    if not valid:
        return None, f"Answer value does not match {answer_type}."
    if status == "answered" and not references:
        return None, "Answered results require evidence references."
    for reference in references:
        if not isinstance(reference, dict) or reference.get("step_id") not in evidence:
            return None, "Final answer references unknown evidence."
        pointers = reference.get("pointers", [])
        if not isinstance(pointers, list) or not pointers:
            return None, "Evidence references need at least one JSON Pointer."
        for pointer in pointers:
            try:
                resolve_pointer(evidence[reference["step_id"]]["output"], pointer)
            except (KeyError, TypeError, ValueError):
                return None, f"Evidence pointer {pointer!r} does not exist."
    return {"status": status, "answer_text": text.strip(), "answer": {"type": answer_type, "value": value}}, None
