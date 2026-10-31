import json
from pathlib import Path
from types import SimpleNamespace

from graphkit.validate import validate
from orchestrator.agent import Orchestrator
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
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=next(self.messages))])


def function_call(call_id, name, arguments):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


def plan_message(steps, mode="execute", explanation="Use the required graph tools.", answer_type="number"):
    return Message([function_call("plan", "submit_plan", {
        "mode": mode, "explanation": explanation, "answer_type": answer_type, "steps": steps,
    })])


def answer_message(value="34", text="The karate graph has 34 nodes.", answer_type="number", evidence=None):
    return Message([function_call("answer", "submit_answer", {
        "status": "answered", "answer_text": text, "answer_type": answer_type,
        "answer_value_json": value,
        "evidence": evidence or [{"step_id": "summary", "pointers": ["/result/nodes"]}],
    })])


def test_discovers_and_validates_curated_registry():
    tools = discover_project_tools(ROOT)
    assert len(tools) == 37
    assert set(tools) >= {"graph_summary", "node_similarity", "lookup_node", "filter_nodes"}
    failures = {name: validate(tool.path).errors for name, tool in tools.items() if validate(tool.path).errors}
    assert not failures


def test_single_tool_langgraph_answer_and_trace(tmp_path):
    client = FakeClient([
        plan_message([{"id": "summary", "tool": "graph_summary", "purpose": "Count nodes",
                       "arguments": {"graph_id": "karate"}, "bindings": [], "depends_on": []}]),
        answer_message(),
    ])
    trace = tmp_path / "q1.jsonl"
    with Orchestrator(ROOT, client=client, model="test-model", state_dir=tmp_path / "state", cache_enabled=False) as agent:
        result = agent.answer("q1", "How many nodes are in the karate graph?", trace)
    assert result["status"] == "answered"
    assert result["answer"] == {"type": "number", "value": 34}
    assert result["tools_used"] == ["graph_summary.2f7893e5"]
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [event["type"] for event in events] == ["question", "llm_call", "tool_call", "llm_call", "final_answer"]


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
        plan_message(steps, answer_type="text"),
        answer_message(value='"Filtered graph summarized."', text="Filtered graph summarized.",
                       answer_type="text", evidence=[{"step_id": "summary", "pointers": ["/result/nodes"]}]),
    ])
    trace = tmp_path / "chain.jsonl"
    with Orchestrator(ROOT, client=client, model="test-model", state_dir=tmp_path / "state", cache_enabled=False) as agent:
        result = agent.answer("chain", "Summarize airports in India.", trace)
    assert result["status"] == "answered"
    assert result["tools_used"] == ["filter_nodes.instructors", "graph_summary.2f7893e5"]
    tool_events = [json.loads(line) for line in trace.read_text().splitlines() if '"type": "tool_call"' in line]
    assert tool_events[1]["arguments"]["graph_id"] == "airports~1"


def test_planner_can_return_clarification_without_tools(tmp_path):
    client = FakeClient([plan_message([], mode="clarification_needed",
                                      explanation="Which graph should I analyze?", answer_type="text")])
    with Orchestrator(ROOT, client=client, model="test-model", state_dir=tmp_path / "state", cache_enabled=False) as agent:
        result = agent.answer("ambiguous", "Which node is most important?", tmp_path / "ambiguous.jsonl")
    assert result["status"] == "clarification_needed"
    assert result["tools_used"] == []


def test_planner_can_return_unsupported_without_tools(tmp_path):
    client = FakeClient([plan_message([], mode="unsupported",
                                      explanation="No graph tool forecasts stocks.", answer_type="text")])
    with Orchestrator(ROOT, client=client, model="test-model", state_dir=tmp_path / "state", cache_enabled=False) as agent:
        result = agent.answer("unsupported", "Forecast a stock price.", tmp_path / "unsupported.jsonl")
    assert result["status"] == "unsupported"


def test_failed_tool_triggers_bounded_replan(tmp_path):
    bad = [{"id": "bad", "tool": "graph_summary", "purpose": "Try missing graph",
            "arguments": {"graph_id": "no_such_graph"}, "bindings": [], "depends_on": []}]
    good = [{"id": "summary", "tool": "graph_summary", "purpose": "Count nodes",
             "arguments": {"graph_id": "karate"}, "bindings": [], "depends_on": []}]
    client = FakeClient([plan_message(bad), plan_message(good), answer_message()])
    trace = tmp_path / "replan.jsonl"
    with Orchestrator(ROOT, client=client, model="test-model", state_dir=tmp_path / "state", cache_enabled=False) as agent:
        result = agent.answer("replan", "How many karate nodes?", trace)
    assert result["status"] == "answered"
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert any(event["type"] == "replan" for event in events)
    assert sum(event["type"] == "tool_call" for event in events) == 2


def test_completed_query_resumes_without_another_provider_call(tmp_path):
    first_client = FakeClient([
        plan_message([{"id": "summary", "tool": "graph_summary", "purpose": "Count nodes",
                       "arguments": {"graph_id": "karate"}, "bindings": [], "depends_on": []}]),
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
