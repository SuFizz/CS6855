from types import SimpleNamespace

from orchestrator.schema import parse_answer, resolve_pointer, validate_arguments, validate_plan


def tool(name, required=None):
    manifest = {"name": name, "parameters": {"type": "object", "properties": {
        "graph_id": {"type": "string"},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
        "method": {"type": "string", "enum": ["a", "b"]},
    }, "required": required or ["graph_id"]}}
    return SimpleNamespace(manifest=manifest)


def test_argument_validation_enforces_ranges_and_enum():
    manifest = tool("x").manifest
    assert validate_arguments(manifest, {"graph_id": "g", "top_k": 0})
    assert validate_arguments(manifest, {"graph_id": "g", "method": "c"})
    assert validate_arguments(manifest, {"graph_id": "g", "top_k": 5, "method": "a"}) is None


def test_json_pointer_handles_lists_and_escaping():
    document = {"result": [{"a/b": {"~key": 7}}]}
    assert resolve_pointer(document, "/result/0/a~1b/~0key") == 7


def test_plan_validation_rejects_cycles_unknown_tools_and_too_many_steps():
    tools = {"summary": tool("summary")}
    cyclic = {"mode": "execute", "explanation": "x", "answer_type": "number", "steps": [
        {"id": "a", "tool": "summary", "purpose": "x", "arguments": {"graph_id": "g"}, "bindings": [], "depends_on": ["b"]},
        {"id": "b", "tool": "summary", "purpose": "x", "arguments": {"graph_id": "g"}, "bindings": [], "depends_on": ["a"]},
    ]}
    assert any("cycle" in error for error in validate_plan(cyclic, tools))
    unknown = {"mode": "execute", "explanation": "x", "answer_type": "text", "steps": [
        {"id": "a", "tool": "missing", "purpose": "x", "arguments": {}, "bindings": [], "depends_on": []}
    ]}
    assert any("unknown tool" in error for error in validate_plan(unknown, tools))


def test_replan_can_bind_to_completed_evidence_ids():
    tools = {"summary": tool("summary")}
    plan = {"mode": "execute", "explanation": "continue", "answer_type": "number", "steps": [
        {"id": "new", "tool": "summary", "purpose": "use subgraph", "arguments": {},
         "bindings": [{"parameter": "graph_id", "step_id": "old", "pointer": "/result_graph_id"}],
         "depends_on": ["old"]}
    ]}
    assert validate_plan(plan, tools, completed_ids={"old"}) == []


def test_final_answer_requires_existing_evidence_paths():
    evidence = {"s": {"output": {"result": {"value": 3}}}}
    arguments = {"status": "answered", "answer_text": "Three.", "answer_type": "number",
                 "answer_value_json": "3", "evidence": [{"step_id": "s", "pointers": ["/result/value"]}]}
    final, issue = parse_answer(arguments, evidence)
    assert issue is None and final["answer"]["value"] == 3
    arguments["evidence"][0]["pointers"] = ["/missing"]
    assert parse_answer(arguments, evidence)[1]
