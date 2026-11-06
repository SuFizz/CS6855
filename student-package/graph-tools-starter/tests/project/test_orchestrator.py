import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from graphkit.validate import validate
from orchestrator.agent import Orchestrator
from orchestrator.planner import _plan_schema
from orchestrator.query_io import load_queries
from orchestrator.registry import discover_project_tools

ROOT = Path(__file__).resolve().parents[2]


class Message:
    def __init__(self, calls=None, content=None):
        self.calls = calls or []
        self.content = content

    def model_dump(self, exclude_none=True):
        value = {"role": "assistant", "tool_calls": self.calls}
        if self.content is not None:
            value["content"] = self.content
        return value


class FakeClient:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.calls = 0
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        self.requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=next(self.messages))])


def function_call(call_id, name, arguments):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


def contract_message(answer_type="number", required_outputs=None, *, cardinality="one", ordered=False,
                     needs_clarification=False, missing_information=None):
    if required_outputs is None:
        required_outputs = ["requested answer"]
    return Message([function_call("contract", "submit_query_contract", {
        "answer_type": answer_type, "cardinality": cardinality, "ordered": ordered,
        "value_description": "Structured value requested by the user.",
        "required_outputs": required_outputs,
        "needs_clarification": needs_clarification,
        "missing_information": missing_information or [],
    })])


def plan_message(steps, mode="execute", explanation="Use the required graph tools.", answer_type="number",
                 missing_information=None, unsupported_capability="", required_outputs=None):
    if mode == "clarification_needed" and missing_information is None:
        missing_information = ["graph_id"]
    if mode == "unsupported" and not unsupported_capability:
        unsupported_capability = "requested graph capability"
    return Message([function_call("plan", "submit_plan", {
        "mode": mode, "explanation": explanation, "missing_information": missing_information or [],
        "unsupported_capability": unsupported_capability, "steps": steps,
    })])


def assessment_message(sufficient=True, missing_outputs=None, unsupported_capability=""):
    return Message([function_call("assessment", "submit_assessment", {
        "sufficient": sufficient, "missing_outputs": missing_outputs or [],
        "unsupported_capability": unsupported_capability,
        "explanation": "Evidence is sufficient." if sufficient else "More analysis is required.",
    })])


def answer_message(value="34", text="The karate graph has 34 nodes.", answer_type="number", evidence=None):
    return Message([function_call("answer", "submit_answer", {
        "status": "answered", "answer_text": text, "answer_type": answer_type,
        "answer_value_json": value,
        "evidence": evidence or [{"step_id": "summary", "pointers": ["/summary"]}],
    })])


def test_discovers_and_validates_curated_registry():
    tools = discover_project_tools(ROOT)
    assert len(tools) == 37
    assert set(tools) >= {"graph_summary", "node_similarity", "lookup_node", "filter_nodes"}
    failures = {name: validate(tool.path).errors for name, tool in tools.items() if validate(tool.path).errors}
    assert not failures


def test_plan_schema_uses_discovered_names_without_embedded_arguments():
    tools = discover_project_tools(ROOT)
    tool_property = _plan_schema(tools)["function"]["parameters"]["properties"]["steps"]["items"]["properties"]["tool"]
    assert tool_property["enum"] == sorted(tools)
    assert "{" not in "".join(tool_property["enum"])


def test_single_tool_langgraph_answer_and_trace(tmp_path):
    client = FakeClient([
        contract_message("number", ["node count"]),
        plan_message([{"id": "summary", "tool": "graph_summary", "purpose": "Count nodes",
                       "arguments": {"graph_id": "karate"}, "bindings": [], "depends_on": []}]),
        assessment_message(),
        answer_message(),
    ])
    trace = tmp_path / "q1.jsonl"
    with Orchestrator(ROOT, client=client, model="test-model", state_dir=tmp_path / "state", cache_enabled=False) as agent:
        result = agent.answer("q1", "How many nodes are in the karate graph?", trace)
    assert result["status"] == "answered"
    assert result["answer"] == {"type": "number", "value": 34}
    interpretation_payload = json.loads(client.requests[0]["messages"][1]["content"])
    assert {item["graph_id"] for item in interpretation_payload["available_graphs"]} == {
        "airports", "karate", "lesmis", "courses", "projects",
    }
    expected = discover_project_tools(ROOT)["graph_summary"].implementation_id
    assert result["tools_used"] == [expected]
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [event["type"] for event in events] == [
        "question", "llm_call", "llm_call", "tool_call", "llm_call",
        "evidence_assessment", "llm_call", "final_answer",
    ]


def test_interpretation_uses_question_relevant_node_labels(tmp_path):
    client = FakeClient([contract_message("number", ["edge-disjoint path count"]),
                         plan_message([], mode="unsupported", answer_type="text")])
    with Orchestrator(ROOT, client=client, model="test-model", state_dir=tmp_path / "state", cache_enabled=False) as agent:
        agent.answer("roles", "How many independent friendship chains connect the instructor and administrator?",
                     tmp_path / "roles.jsonl")
    context = json.loads(client.requests[0]["messages"][1]["content"])
    assert {entry["id"] for hint in context["entity_hints"] if hint["graph_id"] == "karate"
            for entry in hint["matches"]} >= {"m1", "m34"}
    assert "edge-disjoint" in client.requests[0]["messages"][0]["content"]
    plan_context = json.loads(client.requests[1]["messages"][1]["content"])
    assert any(hint["graph_id"] == "karate" for hint in plan_context["entity_hints"])


def test_finalizer_corrects_answer_type_to_query_contract(tmp_path):
    plan = [{"id": "node", "tool": "lookup_node", "purpose": "Resolve airport",
             "arguments": {"graph_id": "airports", "name": "FRA"}, "bindings": [], "depends_on": []}]
    evidence = [{"step_id": "node", "pointers": ["/result/0/id"]}]
    client = FakeClient([
        contract_message("node", ["airport ID"]),
        plan_message(plan), assessment_message(),
        answer_message(value='"Frankfurt"', text="Frankfurt", answer_type="text", evidence=evidence),
        answer_message(value='"FRA"', text="Frankfurt (FRA)", answer_type="node", evidence=evidence),
    ])
    with Orchestrator(ROOT, client=client, model="test-model", state_dir=tmp_path / "state", cache_enabled=False) as agent:
        result = agent.answer("type", "Identify the airport with code FRA.", tmp_path / "type.jsonl")
    assert result["status"] == "answered"
    assert result["answer"] == {"type": "node", "value": "FRA"}


def test_successful_evidence_can_finalize_after_more_than_eight_llm_calls(tmp_path):
    invalid_contract = contract_message("number", ["node count"], needs_clarification=True)
    invalid_plan = plan_message([{"id": "summary", "tool": "graph_summary{}", "purpose": "Count nodes",
                                 "arguments": {}, "bindings": [], "depends_on": []}])
    corrected = plan_message([{"id": "summary", "tool": "graph_summary", "purpose": "Count nodes",
                               "arguments": {}, "bindings": [], "depends_on": []}])
    client = FakeClient([
        invalid_contract, contract_message("number", ["node count"]),
        invalid_plan, invalid_plan, corrected,
        Message([function_call("args", "graph_summary", {"graph_id": "karate"})]),
        assessment_message(), answer_message(value='"thirty-four"', answer_type="text"), answer_message(),
    ])
    trace = tmp_path / "many_llm_calls.jsonl"
    with Orchestrator(ROOT, client=client, model="test-model", state_dir=tmp_path / "state", cache_enabled=False) as agent:
        result = agent.answer("many_llm_calls", "Return the karate node count.", trace)
    assert client.calls == 9
    assert result["status"] == "answered"
    assert result["answer"] == {"type": "number", "value": 34}


def test_missing_plan_arguments_are_filled_using_exact_tool_schema(tmp_path):
    incomplete = [{"id": "summary", "tool": "graph_summary", "purpose": "Count karate nodes",
                   "arguments": {}, "bindings": [], "depends_on": []}]
    argument_call = Message([function_call("args", "graph_summary", {"graph_id": "karate"})])
    client = FakeClient([contract_message("number", ["node count"]), plan_message(incomplete), argument_call, assessment_message(), answer_message()])
    trace = tmp_path / "arguments.jsonl"
    with Orchestrator(ROOT, client=client, model="test-model", state_dir=tmp_path / "state", cache_enabled=False) as agent:
        result = agent.answer("arguments", "How many nodes are in exact graph karate?", trace)
    assert result["status"] == "answered"
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    tool_event = next(event for event in events if event["type"] == "tool_call")
    assert tool_event["arguments"] == {"graph_id": "karate"}
    assert any(event.get("purpose") == "arguments:summary" for event in events if event["type"] == "llm_call")


def test_result_graph_id_binding_is_chained(tmp_path):
    steps = [
        {"id": "filter", "tool": "filter_nodes", "purpose": "Select India airports",
         "arguments": {"graph_id": "airports", "attribute": "country", "value": "India"},
         "bindings": [], "depends_on": []},
        {"id": "summary", "tool": "graph_summary", "purpose": "Summarize filtered graph",
         "arguments": {}, "bindings": [{"parameter": "graph_id", "step_id": "filter", "pointer": "/result_graph_id"}],
         "depends_on": ["filter"]},
    ]
    client = FakeClient([
        contract_message("text", ["filtered graph summary"]),
        plan_message(steps, answer_type="text"),
        assessment_message(),
        answer_message(value='"Filtered graph summarized."', text="Filtered graph summarized.",
                       answer_type="text", evidence=[{"step_id": "summary", "pointers": ["/summary"]}]),
    ])
    trace = tmp_path / "chain.jsonl"
    with Orchestrator(ROOT, client=client, model="test-model", state_dir=tmp_path / "state", cache_enabled=False) as agent:
        result = agent.answer("chain", "Summarize airports in India.", trace)
    assert result["status"] == "answered"
    expected_summary = discover_project_tools(ROOT)["graph_summary"].implementation_id
    assert result["tools_used"] == ["filter_nodes.instructors", expected_summary]
    tool_events = [json.loads(line) for line in trace.read_text().splitlines() if '"type": "tool_call"' in line]
    assert tool_events[1]["arguments"]["graph_id"] == "airports~1"
    assessment_payload = json.loads(client.requests[2]["messages"][1]["content"])
    assert any(item["output"]["result_graph_id"] == "airports~1"
               for item in assessment_payload["executed_tools"])
    assert "derived subgraph" in client.requests[2]["messages"][0]["content"]


def test_interpreter_can_return_clarification_without_tools(tmp_path):
    client = FakeClient([contract_message("text", [], needs_clarification=True,
                                          missing_information=["graph to analyze", "importance definition"])])
    with Orchestrator(ROOT, client=client, model="test-model", state_dir=tmp_path / "state", cache_enabled=False) as agent:
        result = agent.answer("ambiguous", "Which node is most important?", tmp_path / "ambiguous.jsonl")
    assert result["status"] == "clarification_needed"
    assert result["tools_used"] == []


def test_planner_can_return_unsupported_without_tools(tmp_path):
    client = FakeClient([
        contract_message("text", ["stock forecast"]),
        plan_message([], mode="unsupported", explanation="No graph tool forecasts stocks.", answer_type="text"),
    ])
    with Orchestrator(ROOT, client=client, model="test-model", state_dir=tmp_path / "state", cache_enabled=False) as agent:
        result = agent.answer("unsupported", "Forecast a stock price.", tmp_path / "unsupported.jsonl")
    assert result["status"] == "unsupported"
    assert "unavailable data" in client.requests[0]["messages"][0]["content"]
    assert "dataset attribute" in client.requests[1]["messages"][0]["content"]


def test_invented_plan_graph_id_is_corrected_before_execution(tmp_path):
    wrong = [{"id": "summary", "tool": "graph_summary", "purpose": "Count airport nodes",
              "arguments": {"graph_id": "airport_graph"}, "bindings": [], "depends_on": []}]
    corrected = [{"id": "summary", "tool": "graph_summary", "purpose": "Count airport nodes",
                  "arguments": {"graph_id": "airports"}, "bindings": [], "depends_on": []}]
    client = FakeClient([
        contract_message("number", ["airport node count"]),
        plan_message(wrong), plan_message(corrected), assessment_message(), answer_message(value="500", text="There are 500 airports."),
    ])
    trace = tmp_path / "graph_id.jsonl"
    with Orchestrator(ROOT, client=client, model="test-model", state_dir=tmp_path / "state", cache_enabled=False) as agent:
        result = agent.answer("graph_id", "How many nodes are in the airport network?", trace)
    assert result["status"] == "answered"
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert sum(event["type"] == "tool_call" for event in events) == 1
    assert next(event for event in events if event["type"] == "tool_call")["arguments"]["graph_id"] == "airports"


def test_argument_preparation_unknown_graph_becomes_recoverable_error(tmp_path):
    incomplete = [{"id": "rank", "tool": "pagerank", "purpose": "Rank airports",
                   "arguments": {}, "bindings": [], "depends_on": []}]
    invalid_arguments = Message([function_call("args", "pagerank", {"graph_id": "airport_graph", "top_k": 5})])
    client = FakeClient([
        contract_message("node_list", ["ranked airport IDs"], cardinality="many", ordered=True),
        plan_message(incomplete), invalid_arguments,
        plan_message([], mode="unsupported", explanation="The requested graph is invalid.",
                     unsupported_capability="requested graph instance"),
    ])
    with Orchestrator(ROOT, client=client, model="test-model", state_dir=tmp_path / "state", cache_enabled=False) as agent:
        result = agent.answer("argument_graph", "Rank the airport graph.", tmp_path / "argument_graph.jsonl")
    assert result["status"] == "error"
    assert "airport_graph" in result["answer_text"]


def test_invented_graph_id_is_corrected_without_wasting_a_tool_call(tmp_path):
    bad = [{"id": "bad", "tool": "graph_summary", "purpose": "Try missing graph",
            "arguments": {"graph_id": "no_such_graph"}, "bindings": [], "depends_on": []}]
    good = [{"id": "summary", "tool": "graph_summary", "purpose": "Count nodes",
             "arguments": {"graph_id": "karate"}, "bindings": [], "depends_on": []}]
    client = FakeClient([contract_message("number", ["node count"]), plan_message(bad), plan_message(good), assessment_message(), answer_message()])
    trace = tmp_path / "replan.jsonl"
    with Orchestrator(ROOT, client=client, model="test-model", state_dir=tmp_path / "state", cache_enabled=False) as agent:
        result = agent.answer("replan", "How many karate nodes?", trace)
    assert result["status"] == "answered"
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert any(event["type"] == "plan_validation_error" for event in events)
    assert sum(event["type"] == "tool_call" for event in events) == 1


def test_missing_graph_failure_remains_error_when_replanner_says_unsupported(tmp_path):
    bad = [{"id": "summary", "tool": "graph_summary", "purpose": "Summarize exact graph",
            "arguments": {"graph_id": "no_such_graph"}, "bindings": [], "depends_on": []}]
    client = FakeClient([
        contract_message("text", ["graph summary"]),
        plan_message(bad, answer_type="text"),
        plan_message([], mode="unsupported", explanation="The graph does not exist.",                     answer_type="text", unsupported_capability="requested graph instance"),
    ])
    with Orchestrator(ROOT, client=client, model="test-model", state_dir=tmp_path / "state", cache_enabled=False) as agent:
        result = agent.answer("missing_graph", "Summarize exact graph no_such_graph.", tmp_path / "missing_graph.jsonl")
    assert result["status"] == "error"
    assert "no_such_graph" in result["answer_text"]


def test_missing_node_failure_remains_error_when_replanner_says_unsupported(tmp_path):
    bad = [{"id": "neighbors", "tool": "k_hop_neighborhood", "purpose": "Use exact node",
            "arguments": {"graph_id": "karate", "node": "m999", "k": 2},
            "bindings": [], "depends_on": []}]
    client = FakeClient([
        contract_message("text", ["two-hop neighborhood"]),
        plan_message(bad, answer_type="text"),
        plan_message([], mode="unsupported", explanation="The node does not exist.",                     answer_type="text", unsupported_capability="requested node instance"),
    ])
    with Orchestrator(ROOT, client=client, model="test-model", state_dir=tmp_path / "state", cache_enabled=False) as agent:
        result = agent.answer("missing_node", "Use exact node m999 in karate.", tmp_path / "missing_node.jsonl")
    assert result["status"] == "error"
    assert "m999" in result["answer_text"]


def test_insufficient_metadata_evidence_triggers_analytical_replan(tmp_path):
    metadata_plan = [{"id": "graphs", "tool": "list_graphs", "purpose": "Inspect graph metadata",
                      "arguments": {}, "bindings": [], "depends_on": []}]
    analytical_plan = [{"id": "components", "tool": "strongly_connected_components",
                        "purpose": "Test mutual reachability", "arguments": {"graph_id": "airports"},
                        "bindings": [], "depends_on": []}]
    client = FakeClient([
        contract_message("boolean", ["strong connectivity boolean"]),
        plan_message(metadata_plan, answer_type="boolean", required_outputs=["strong connectivity boolean"]),
        assessment_message(False, ["strong connectivity boolean"]),
        plan_message(analytical_plan, answer_type="boolean", required_outputs=["strong connectivity boolean"]),
        assessment_message(),
        answer_message(value="true", text="Every airport is mutually reachable.", answer_type="boolean",
                       evidence=[{"step_id": "components", "pointers": ["/summary"]}]),
    ])
    with Orchestrator(ROOT, client=client, model="test-model", state_dir=tmp_path / "state", cache_enabled=False) as agent:
        result = agent.answer("sufficiency", "Can every airport reach every other airport?", tmp_path / "sufficiency.jsonl")
    assert result["status"] == "answered"
    assert result["answer"] == {"type": "boolean", "value": True}
    assert len(result["tools_used"]) == 2


def test_empty_full_phrase_lookup_replans_with_exact_code_token(tmp_path):
    first = [{"id": "lookup_full", "tool": "lookup_node", "purpose": "Resolve destination",
              "arguments": {"graph_id": "airports", "name": "New York JFK"},
              "bindings": [], "depends_on": []}]
    retry = [{"id": "lookup_code", "tool": "lookup_node", "purpose": "Resolve exact code token",
              "arguments": {"graph_id": "airports", "name": "JFK"},
              "bindings": [], "depends_on": []}]
    client = FakeClient([
        contract_message("node", ["destination airport ID"]),
        plan_message(first), plan_message(retry), assessment_message(),
        answer_message(value='"JFK"', text="The destination is JFK.", answer_type="node",
                       evidence=[{"step_id": "lookup_code", "pointers": ["/result/0/id"]}]),
    ])
    with Orchestrator(ROOT, client=client, model="test-model", state_dir=tmp_path / "state", cache_enabled=False) as agent:
        result = agent.answer("jfk", "Find New York JFK.", tmp_path / "jfk.jsonl")
    assert result["status"] == "answered"
    assert result["answer"] == {"type": "node", "value": "JFK"}
    events = [json.loads(line) for line in (tmp_path / "jfk.jsonl").read_text().splitlines()]
    lookup_calls = [event for event in events if event["type"] == "tool_call"]
    assert [event["arguments"]["name"] for event in lookup_calls] == ["New York JFK", "JFK"]


def test_ambiguous_lookup_routes_to_clarification(tmp_path):
    lookup = [{"id": "paris", "tool": "lookup_node", "purpose": "Resolve Paris airport",
               "arguments": {"graph_id": "airports", "name": "Paris"}, "bindings": [], "depends_on": []}]
    client = FakeClient([
        contract_message("path", ["source airport id", "destination airport id", "route"], ordered=True),
        plan_message(lookup, answer_type="path", required_outputs=["source airport id"]),
        plan_message([], mode="clarification_needed", explanation="Which Paris airport?",
                     answer_type="text", missing_information=["Paris airport"]),
    ])
    with Orchestrator(ROOT, client=client, model="test-model", state_dir=tmp_path / "state", cache_enabled=False) as agent:
        result = agent.answer("paris", "Find a route from Paris.", tmp_path / "paris.jsonl")
    assert result["status"] == "clarification_needed"
    assert result["tools_used"] == ["lookup_node.instructors"]


def test_completed_query_resumes_without_another_provider_call(tmp_path):
    first_client = FakeClient([
        contract_message("number", ["node count"]),
        plan_message([{"id": "summary", "tool": "graph_summary", "purpose": "Count nodes",
                       "arguments": {"graph_id": "karate"}, "bindings": [], "depends_on": []}]),
        assessment_message(),
        answer_message(),
    ])
    trace = tmp_path / "resume.jsonl"
    state_dir = tmp_path / "state"
    with Orchestrator(ROOT, client=first_client, model="test-model", state_dir=state_dir, cache_enabled=False) as agent:
        first = agent.answer("resume", "How many nodes are in karate?", trace)
    second_client = FakeClient([])
    with Orchestrator(ROOT, client=second_client, model="test-model", state_dir=state_dir, cache_enabled=False) as agent:
        second = agent.answer("resume", "How many nodes are in karate?", trace, resume=True)
    assert second == first
    assert second_client.calls == 0
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert sum(event["type"] == "final_answer" for event in events) == 1
    assert any(event["type"] == "tool_replay" for event in events)


def test_registry_query_set_covers_every_selected_registry_tool():
    # The 41-question development set is not committed; skip without it.
    if not (ROOT / "queries_registry_all.json").is_file():
        pytest.skip("queries_registry_all.json is not in the repository")
    queries = load_queries(ROOT / "queries_registry_all.json")
    covered = {item["query_id"].removeprefix("tool_") for item in queries if item["query_id"].startswith("tool_")}
    registry = {name for name, tool in discover_project_tools(ROOT).items() if tool.manifest.get("tier") != "infra"}
    assert covered == registry


def test_query_loader_supports_text_json_and_jsonl(tmp_path):
    text_file = tmp_path / "queries.txt"
    text_file.write_text("# comment\nFirst question?\nSecond question?\n")
    assert [item["query_id"] for item in load_queries(text_file)] == ["q1", "q2"]
    json_file = tmp_path / "queries.json"
    json_file.write_text(json.dumps([{"query_id": "custom", "question": "A question?"}]))
    assert load_queries(json_file) == [{"query_id": "custom", "question": "A question?"}]
    jsonl_file = tmp_path / "queries.jsonl"
    jsonl_file.write_text('{"id":"a","query":"One?"}\n{"id":"b","text":"Two?"}\n')
    assert load_queries(jsonl_file) == [
        {"query_id": "a", "question": "One?"}, {"query_id": "b", "question": "Two?"},
    ]
