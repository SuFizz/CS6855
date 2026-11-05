"""Gemini prompts and structured schemas for planning, replanning and finalization."""
from __future__ import annotations

import json

from .registry import manifest_catalogue
from .schema import ANSWER_TYPES, STATUSES, parse_answer, validate_plan, validate_query_contract


class ExplicitInputError(ValueError):
    pass


def _contract_schema():
    return {
        "type": "function",
        "function": {
            "name": "submit_query_contract",
            "description": "Interpret the requested structured answer and information requirements without planning tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer_type": {"type": "string", "enum": list(ANSWER_TYPES), "description": "number for one scalar, boolean for true/false, node for one node ID, node_list for multiple node IDs, path for an ordered traversed node-ID sequence, text only when no other type can represent the value."},
                    "cardinality": {"type": "string", "enum": ["one", "many"]},
                    "ordered": {"type": "boolean", "description": "True only when order is part of correctness, such as rankings or paths."},
                    "value_description": {"type": "string", "description": "Semantic description of answer.value, including IDs rather than labels for graph nodes."},
                    "required_outputs": {"type": "array", "items": {"type": "string"}, "description": "Every fact, metric, ranking, comparison or path needed to answer fully."},
                    "needs_clarification": {"type": "boolean"},
                    "missing_information": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["answer_type", "cardinality", "ordered", "value_description", "required_outputs", "needs_clarification", "missing_information"],
            },
        },
    }


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
                    "steps": {"type": "array", "items": step},
                },
                "required": ["mode", "explanation", "missing_information", "unsupported_capability", "steps"],
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


def request_contract(provider, question, trace, graph_catalogue=None, entity_hints=(), correction_attempts=2):
    system = """Interpret the user's requested output semantically. Do not choose tools and do not answer the question. Select the narrowest structured type that can represent the graded value: number is one numeric scalar; boolean is true/false; node is exactly one graph node ID; node_list is multiple node IDs and ordered only for rankings or requested sequences; path is an ordered traversed node-ID route; text is only for a genuinely composite value that cannot fit another type. Labels, scores, units and explanations belong in answer_text later, not in node or node_list values. Identify every output needed for a complete answer. Request clarification only when essential user intent or inputs are missing, not when a capability may be unavailable. Use available_graphs as authoritative context: resolve natural references to a dataset when one description clearly matches, and do not ask for a timeframe or scope already fixed by that dataset's description. entity_hints lists bounded node-label matches from the available graphs for routing only; if the referenced entities appear together in one graph, do not guess another graph or claim the dataset is absent. Hints are not tool results and cannot support final graph facts. Graph edges represent relationships between nodes: paths sharing no edge/relationship are edge-disjoint, while paths sharing no intermediate node/person are vertex-disjoint. Never substitute one meaning for the other."""
    payload = {"question": question, "available_graphs": graph_catalogue or [],
               "entity_hints": list(entity_hints)}
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    schema = _contract_schema()
    last_errors = []
    for attempt in range(correction_attempts):
        message = provider.complete(
            messages=messages, tools=[schema],
            tool_choice={"type": "function", "function": {"name": "submit_query_contract"}},
            purpose="interpret_query", trace=trace,
        )
        try:
            contract = _arguments(message, "submit_query_contract")
            errors = validate_query_contract(contract)
        except (ValueError, json.JSONDecodeError) as exc:
            contract, errors = None, [str(exc)]
        if not errors:
            return contract, attempt + 1
        trace.event("contract_validation_error", attempt=attempt + 1, errors=errors, rejected_contract=contract)
        last_errors = errors
        messages.extend([message, {"role": "user", "content": "Correct the query contract. Validation errors: " + " | ".join(errors)}])
    raise ValueError("Query interpretation failed: " + " | ".join(last_errors))


def request_plan(provider, question, tools, trace, *, query_contract, graph_catalogue=None, entity_hints=(), evidence=None, failures=None,
                 max_steps=5, correction_attempts=3):
    catalogue = manifest_catalogue(tools)
    context = {
        "question": question,
        "available_graphs": graph_catalogue or [],
        "entity_hints": list(entity_hints),
        "available_tools": catalogue,
        "query_contract": query_contract,
        "completed_evidence": evidence or {},
        "failures": failures or [],
        "maximum_tool_steps": max_steps,
    }
    system = """Plan graph analysis using only the supplied graph and manifest catalogues. Every final graph fact must come from a tool result. Keep the plan minimal, acyclic and within the step limit.

Grounding and completeness rules:
- available_graphs is authoritative. Use only those exact graph_id values. Resolve graph references from descriptions and question-relevant entity_hints, not from guesses or outside knowledge. When multiple named entities match the same graph, use that graph and resolve exact node IDs with lookup_node. Hints are routing aids, not evidence for the final answer. Never invent or transform a graph_id.
- An edge is a relationship; paths sharing no edge/relationship are edge-disjoint (max flow with unit capacities), whereas paths sharing no intermediate node/person are vertex-disjoint. Follow the user's distinction.
- Do not call list_graphs when the graph is already identifiable from available_graphs.
- Treat query_contract as immutable. Build enough analytical steps to compute every query_contract.required_outputs item; do not reinterpret its answer type.
- list_graphs, lookup_node, node_attributes and filter_nodes provide infrastructure evidence, not analytical metrics. Do not stop after them when the contract requires centrality, paths, communities, matching, cliques, motifs or another analysis.
- For centrality on a graph whose weights are distances, use an available weighted/use_weights parameter set to false because distance is not link strength.
- For multi-step patterns, use filter_nodes -> analysis for attribute-restricted questions, lookup_node -> ego/subgraph -> analysis for named-node neighborhoods, and community/core tools -> result_graph_id -> ranking tools for derived groups.

Argument rules:
- Copy exact graph IDs, node IDs, numbers, seeds and top_k values already present in the user's question into the step's literal arguments.
- A first step has no dependencies: its bindings must be empty and every required parameter must be literal.
- Use a binding only when its value will be produced by a different, completed dependency step. A step must never bind from itself.
- A binding source must appear in depends_on.
- The tool contract has five top-level keys. To chain a derived subgraph, bind graph_id with pointer /result_graph_id exactly; never use /result/0/result_graph_id.
- Use /result/0/id only when a prior lookup tool actually returns a result list of node references.
- Use lookup_node only for human labels, never for an exact node ID.
- For a named entity containing a short uppercase/alphanumeric code token, prefer that exact token as the lookup name rather than passing surrounding location or generic words.
- If a full-phrase lookup returned no matches, retry at most once with the most identifying segment: first an exact code token, otherwise a distinctive proper-name segment with possessives and generic words removed. Never repeat the same lookup arguments or blindly try every word.
- If lookup returns multiple similarly ranked matches, request clarification instead of selecting one silently.

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
            valid_graph_ids = {item["graph_id"] for item in (graph_catalogue or [])}
            valid_graph_ids.update(record["output"].get("result_graph_id") for record in (evidence or {}).values()
                                   if record.get("output", {}).get("result_graph_id"))
            errors = validate_plan(
                plan, tools, max_steps=max_steps, completed_ids=(evidence or {}).keys(),
                valid_graph_ids=valid_graph_ids,
            )
            invalid_graphs = {step.get("arguments", {}).get("graph_id") for step in plan.get("steps", [])
                              if isinstance(step, dict) and isinstance(step.get("arguments"), dict)
                              and step["arguments"].get("graph_id") not in valid_graph_ids}
            explicit_invalid = sorted(graph_id for graph_id in invalid_graphs
                                      if isinstance(graph_id, str) and graph_id.lower() in question.lower())
            if explicit_invalid:
                raise ExplicitInputError(f"Explicit graph_id value(s) do not exist: {', '.join(explicit_invalid)}. Valid graph IDs are {sorted(valid_graph_ids)}.")
            collisions = sorted({step.get("id") for step in plan.get("steps", [])} & set((evidence or {}).keys()))
            if collisions:
                errors.append("Replanned step ids must not reuse completed evidence ids: " + ", ".join(collisions))
            failed_lookups = {json.dumps(failure.get("arguments", {}), sort_keys=True)
                              for failure in (failures or []) if failure.get("kind") == "resolution"}
            for step in plan.get("steps", []):
                if step.get("tool") == "lookup_node" and json.dumps(step.get("arguments", {}), sort_keys=True) in failed_lookups:
                    errors.append(f"Step {step.get('id')!r} repeats lookup_node arguments that already returned no matches.")
        except ExplicitInputError:
            raise
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


def request_assessment(provider, question, required_outputs, evidence, tools, trace):
    schema = {
        "type": "function",
        "function": {
            "name": "submit_assessment",
            "description": "Decide whether successful tool evidence supports every requested output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sufficient": {"type": "boolean"},
                    "missing_outputs": {"type": "array", "items": {"type": "string"}},
                    "unsupported_capability": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["sufficient", "missing_outputs", "unsupported_capability", "explanation"],
            },
        },
    }
    payload = {
        "question": question,
        "required_outputs": required_outputs,
        "executed_tools": [{"step_id": step_id, "tool": record["tool"],
                            "manifest_description": tools[record["tool"]].manifest.get("description", ""),
                            "output": record["output"]} for step_id, record in evidence.items()],
    }
    messages = [
        {"role": "system", "content": "Judge evidence strictly. Sufficient is true only when tool outputs directly support every required output and requested capability. Metadata or lookup alone cannot support analytics. Do not use outside knowledge. If no available executed tool supports a requested capability, name it in unsupported_capability."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    message = provider.complete(
        messages=messages, tools=[schema],
        tool_choice={"type": "function", "function": {"name": "submit_assessment"}},
        purpose="assess_evidence", trace=trace,
    )
    assessment = _arguments(message, "submit_assessment")
    if not isinstance(assessment.get("sufficient"), bool):
        raise ValueError("Evidence assessment needs a boolean sufficient field.")
    if not isinstance(assessment.get("missing_outputs"), list):
        raise ValueError("Evidence assessment needs missing_outputs.")
    return assessment


def request_answer(provider, question, evidence, query_contract, trace, correction_attempts=2):
    system = """Answer only from the supplied successful tool evidence. Cite exact step ids and JSON Pointers. Each evidence item already identifies its step, so every pointer is relative to that step's five-key output: use /result/..., /summary or /result_graph_id and never prefix a pointer with /<step_id>. Do not add uncited graph facts. Prefer scalar count fields over counting truncated preview lists. For answer_type node return the exact node id string, never its label. For node_list return only node id strings and preserve ranked order. For path return ordered node id strings. Put labels, scores, units, differences and explanations in answer_text. For arithmetic or comparisons, explain the derivation from cited values."""
    payload = {"question": question, "answer_contract": query_contract, "evidence": evidence}
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
            if not issue and final["answer"]["type"] != query_contract["answer_type"]:
                issue = f"Answer type must be {query_contract['answer_type']!r}."
        except (ValueError, json.JSONDecodeError) as exc:
            final, issue = None, str(exc)
        if not issue:
            return final, attempt + 1
        last_issue = issue
        messages.extend([message, {"role": "user", "content": "Correct the final answer. Validation error: " + issue}])
    raise ValueError("Finalizer could not produce a valid answer: " + last_issue)
