"""Part A: answer graph questions through LLM function calling."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from graphkit import GraphStore, load_tool
from graphkit.llm import make_client, model_name, tool_schema
from graphkit.trace import Trace

MAX_STEPS = 8


def system_prompt(store: GraphStore) -> str:
    graphs = "\n".join(f"- {g['graph_id']}: {g['description']}" for g in store.list_graphs())
    return ("You answer questions about the available graph datasets. Use the provided tools "
            "to look up node ids and compute degrees; do not guess graph facts. Explain the "
            "calculation briefly and answer the user's exact question.\n\nAvailable graphs:\n" + graphs)


def answer(question: str, client, tools: dict, store: GraphStore, trace: Trace) -> str:
    messages = [
        {"role": "system", "content": system_prompt(store)},
        {"role": "user", "content": question},
    ]
    schemas = [tool_schema(t.manifest) for t in tools.values()]
    trace.question(question)
    for _ in range(MAX_STEPS):
        response = client.chat.completions.create(model=model_name(), messages=messages,
                                                  tools=schemas, tool_choice="auto")
        message = response.choices[0].message
        trace.llm_call(model_name(), message)
        tool_calls = getattr(message, "tool_calls", None) or []
        content = getattr(message, "content", None)
        if not tool_calls:
            text = content or "The model did not return an answer."
            trace.final_answer(text)
            return text
        messages.append(message.model_dump(exclude_none=True) if hasattr(message, "model_dump") else message)
        for call in tool_calls:
            name = getattr(getattr(call, "function", None), "name", "")
            raw_args = getattr(getattr(call, "function", None), "arguments", "{}")
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be a JSON object")
                output = tools[name].call(store, **arguments) if name in tools else {
                    "status": "error", "result": None, "summary": f"Unknown tool '{name}'.",
                    "result_graph_id": None, "error": f"Unknown tool '{name}'."
                }
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                output = {"status": "error", "result": None, "summary": f"Malformed tool arguments: {exc}",
                          "result_graph_id": None, "error": f"Malformed tool arguments: {exc}"}
                arguments = {"raw": raw_args}
            trace.tool_call(name, arguments, output)
            messages.append({"role": "tool", "tool_call_id": getattr(call, "id", ""),
                             "name": name, "content": json.dumps(output)})
    text = "I could not complete the analysis within the tool-call limit."
    trace.final_answer(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", type=int, help="answer only question number N")
    args = parser.parse_args()
    store = GraphStore()
    tools = {t.name: t for t in (load_tool(ROOT / "bootcamp" / "node_degree"),
                                 load_tool(ROOT / "infra_tools" / "lookup_node"))}
    client = make_client()
    lines = (ROOT / "bootcamp" / "questions.txt").read_text(encoding="utf-8").splitlines()
    questions = [q.strip() for q in lines if q.strip() and not q.startswith("#")]
    for i, question in enumerate(questions, 1):
        if args.only and i != args.only:
            continue
        trace = Trace(ROOT / "bootcamp" / "traces" / f"q{i}.jsonl")
        try:
            text = answer(question, client, tools, store, trace)
        except Exception as exc:
            trace.error(f"{type(exc).__name__}: {exc}")
            text = f"ERROR: {type(exc).__name__}: {exc}"
        print(f"Q{i}: {question}\nA{i}: {text}\n")


if __name__ == "__main__":
    main()
