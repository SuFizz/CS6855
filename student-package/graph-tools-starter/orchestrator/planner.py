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
            "pointer": {"type": "string", "description": "JSON Pointer into that step's complete five-key output."},
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
                    "answer_type": {"type": "string", "enum": list(ANSWER_TYPES)},
                    "steps": {"type": "array", "items": step},
                },
                "required": ["mode", "explanation", "answer_type", "steps"],
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
                    "answer_value_json": {"type": "string"},
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
                 max_steps=5, correction_attempts=2):
    catalogue = manifest_catalogue(tools)
    context = {
        "question": question,
        "available_tools": catalogue,
        "completed_evidence": evidence or {},
        "failures": failures or [],
        "maximum_tool_steps": max_steps,
    }
    system = """Plan graph analysis using only the supplied manifest catalogue. Every graph fact must come from a tool result. Use lookup_node when a user gives a label instead of an exact id. Use bindings for values produced by earlier steps, especially /result/0/id and /result_graph_id. Keep the plan minimal, acyclic and within the step limit. If the request is essentially ambiguous, ask one focused clarification. If no manifest supports it, mark unsupported."""
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
        messages.extend([message, {"role": "user", "content": "Correct the plan. Validation errors: " + " | ".join(errors)}])
    raise ValueError("Planner could not produce a valid plan: " + " | ".join(last_errors))


def request_answer(provider, question, evidence, answer_type, trace, correction_attempts=2):
    system = """Answer only from the supplied successful tool evidence. Cite exact step ids and JSON Pointers. Do not add uncited graph facts. Preserve ranked order for ranked node lists. For arithmetic or comparisons, explain the derivation from cited values."""
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
