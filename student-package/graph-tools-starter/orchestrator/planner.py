"""Gemini prompts and structured schemas for planning, replanning and finalization."""
from __future__ import annotations

import json

from .registry import manifest_catalogue
from .schema import ANSWER_TYPES, STATUSES, parse_answer, validate_plan


def _plan_schema():
    binding = {
        "type": "object",
        "properties": {
            "parameter": {"type": "string", "description": "Target tool parameter."},
            "step_id": {"type": "string", "description": "Completed dependency step id."},
            "pointer": {"type": "string", "description": "JSON Pointer into that step's complete five-key output. A derived graph id is always /result_graph_id, not nested under /result."},
        },
        "required": ["parameter", "step_id", "pointer"],
    }
    step = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Unique short step id."},
            "tool": {"type": "string", "description": "Exact discovered manifest name."},
            "purpose": {"type": "string", "description": "Why this call is necessary."},
            "arguments": {"type": "object", "description": "Literal arguments known before dependencies execute."},
            "bindings": {"type": "array", "items": binding, "description": "Arguments read from dependency outputs."},
            "depends_on": {"type": "array", "items": {"type": "string"}, "description": "Step ids that must finish first."},
        },
        "required": ["id", "tool", "purpose", "arguments", "bindings", "depends_on"],
    }
    return {
        "type": "function",
        "function": {
            "name": "submit_plan",
            "description": "Submit a validated graph-tool plan or a terminal clarification/unsupported decision.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["execute", "clarification_needed", "unsupported"]},
                    "explanation": {"type": "string", "description": "Plan rationale or terminal user-facing explanation."},
                    "missing_information": {"type": "array", "items": {"type": "string"}, "description": "Essential user choices or inputs still needed. Non-empty only for clarification_needed."},
                    "unsupported_capability": {"type": "string", "description": "Requested capability absent from every manifest. Non-empty only for unsupported."},
                    "answer_type": {"type": "string", "enum": list(ANSWER_TYPES)},
                    "steps": {"type": "array", "items": step},
                },
                "required": ["mode", "explanation", "missing_information", "unsupported_capability", "answer_type", "steps"],
            },
        },
    }


def _answer_schema():
    evidence = {
        "type": "object",
        "properties": {
            "step_id": {"type": "string"},
            "pointers": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["step_id", "pointers"],
    }
    return {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final structured answer with exact evidence references.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": list(STATUSES)},
                    "answer_text": {"type": "string"},
                    "answer_type": {"type": "string", "enum": list(ANSWER_TYPES)},
                    "answer_value_json": {"type": "string", "description": "JSON encoding of the structured value. For answer_type text this must be a JSON string, not an object; normally encode answer_text itself."},
                    "evidence": {"type": "array", "items": evidence},
                    "derivation": {"type": "string", "description": "How cited values support the answer."},
                },
                "required": ["status", "answer_text", "answer_type", "answer_value_json", "evidence"],
            },
        },
    }


def _arguments(message, expected):
    calls = message.get("tool_calls") or []
    for call in calls:
        function = call.get("function", {})
        if function.get("name") == expected:
            raw = function.get("arguments", "{}")
            value = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(value, dict):
                raise ValueError(f"{expected} arguments must be an object.")
            return value
    raise ValueError(f"Model did not call {expected}.")


def request_plan(provider, question, tools, trace, *, evidence=None, failures=None,
                 max_steps=5, correction_attempts=3):
    catalogue = manifest_catalogue(tools)
    context = {
        "question": question,
        "available_tools": catalogue,
        "completed_evidence": evidence or {},
        "failures": failures or [],
        "maximum_tool_steps": max_steps,
    }
    system = """Plan graph analysis using only the supplied manifest catalogue. Every graph fact must come from a tool result. Keep the plan minimal, acyclic and within the step limit.

Argument rules:
- Copy exact graph IDs, node IDs, numbers, seeds and top_k values already present in the user's question into the step's literal arguments.
- A first step has no dependencies: its bindings must be empty and every required parameter must be literal.
- Use a binding only when its value will be produced by a different, completed dependency step. A step must never bind from itself.
- A binding source must appear in depends_on.
- The tool contract has five top-level keys. To chain a derived subgraph, bind graph_id with pointer /result_graph_id exactly; never use /result/0/result_graph_id.
- Use /result/0/id only when a prior lookup tool actually returns a result list of node references.
- Use lookup_node only for human labels, never for an exact node ID.

Status rules:
- execute: manifests support the request and enough information is available. Set missing_information to [] and unsupported_capability to "".
- clarification_needed: one or more manifests could answer, but essential user input or intent is missing or ambiguous. Set missing_information to specific missing choices, unsupported_capability to "", and steps to []. Examples include no graph, missing endpoints, or "most important" without choosing among available centrality definitions.
- unsupported: no available manifest provides the requested capability. Set unsupported_capability to that capability, missing_information to [], and steps to []. Do not use unsupported merely because user input is missing.
- A supported operation that later fails because an explicit graph, node, parameter or execution is invalid is an error handled by the workflow, not unsupported.

Before choosing unsupported, inspect the manifest catalogue for related tools. If multiple related tools exist but the user's intent does not select one, choose clarification_needed."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
    schema = _plan_schema()
    last_errors = []
    for attempt in range(correction_attempts):
        message = provider.complete(messages=messages, tools=[schema],
                                    tool_choice={"type": "function", "function": {"name": "submit_plan"}},
                                    purpose="replan" if failures else "plan", trace=trace)
        try:
            plan = _arguments(message, "submit_plan")
            errors = validate_plan(plan, tools, max_steps=max_steps, completed_ids=(evidence or {}).keys())
            collisions = sorted({step.get("id") for step in plan.get("steps", [])} & set((evidence or {}).keys()))
            if collisions:
                errors.append("Replanned step ids must not reuse completed evidence ids: " + ", ".join(collisions))
        except (ValueError, json.JSONDecodeError) as exc:
            plan, errors = None, [str(exc)]
        if not errors:
            return plan, attempt + 1
        last_errors = errors
        trace.event("plan_validation_error", attempt=attempt + 1, errors=errors, rejected_plan=plan)
        correction = ("Correct the plan using the original question and manifest catalogue. "
                      "Do not bind a step to itself; put user-supplied exact values in literal arguments; "
                      "use /result_graph_id for subgraph chaining. Validation errors: " + " | ".join(errors))
        messages.extend([message, {"role": "user", "content": correction}])
    raise ValueError("Planner could not produce a valid plan: " + " | ".join(last_errors))


def request_answer(provider, question, evidence, answer_type, trace, correction_attempts=2):
    system = """Answer only from the supplied successful tool evidence. Cite exact step ids and JSON Pointers. Each evidence item already identifies its step, so every pointer is relative to that step's five-key output: use /result/..., /summary or /result_graph_id and never prefix a pointer with /<step_id>. Do not add uncited graph facts. Prefer scalar count fields over counting truncated preview lists. Preserve ranked order for ranked node lists. For arithmetic or comparisons, explain the derivation from cited values."""
    payload = {"question": question, "expected_answer_type": answer_type, "evidence": evidence}
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    schema = _answer_schema()
    last_issue = ""
    for attempt in range(correction_attempts):
        message = provider.complete(messages=messages, tools=[schema],
                                    tool_choice={"type": "function", "function": {"name": "submit_answer"}},
                                    purpose="finalize", trace=trace)
        try:
            arguments = _arguments(message, "submit_answer")
            final, issue = parse_answer(arguments, evidence)
        except (ValueError, json.JSONDecodeError) as exc:
            final, issue = None, str(exc)
        if not issue:
            return final, attempt + 1
        last_issue = issue
        messages.extend([message, {"role": "user", "content": "Correct the final answer. Validation error: " + issue}])
    raise ValueError("Finalizer could not produce a valid answer: " + last_issue)
