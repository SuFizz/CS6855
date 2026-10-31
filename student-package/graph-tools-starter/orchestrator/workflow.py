"""LangGraph plan, execute, replan and evidence-bound finalization workflow."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .planner import request_answer, request_plan
from .state import OrchestratorState


def build_workflow(*, provider, tools, executor, trace):
    def plan_node(state):
        try:
            plan, calls = request_plan(provider, state["question"], tools, trace,
                                       max_steps=state["max_tool_calls"])
            route = "execute" if plan["mode"] == "execute" else "terminal"
            return {"plan": plan, "plan_revision": 1, "pending_step_ids": [s["id"] for s in plan["steps"]],
                    "llm_call_count": state.get("llm_call_count", 0) + calls, "route": route}
        except Exception as exc:
            return {"error": f"Planning failed: {type(exc).__name__}: {exc}", "route": "error"}

    def terminal_node(state):
        plan = state["plan"]
        status = plan["mode"]
        text = plan.get("explanation") or ("Clarification is needed." if status == "clarification_needed" else "No available tool supports this request.")
        return {"final": {"status": status, "answer_text": text,
                          "answer": {"type": "text", "value": text}}, "route": "done"}

    def execute_node(state):
        pending = list(state.get("pending_step_ids", []))
        evidence = dict(state.get("evidence", {}))
        if not pending:
            return {"route": "finalize"}
        if state.get("tool_call_count", 0) >= state["max_tool_calls"]:
            return {"error": "Graph-tool call limit reached.", "route": "error"}
        step = executor.ready_step(state["plan"], pending, evidence)
        if step is None:
            return {"failures": state.get("failures", []) + [{"error": "No pending plan step is ready."}], "route": "replan"}
        record, issue = executor.execute(step, evidence)
        calls = state.get("tool_call_count", 0) + 1
        pending.remove(step["id"])
        if issue:
            failure = {"step_id": step["id"], "tool": step["tool"], "error": issue,
                       "output": record["output"] if record else None}
            return {"pending_step_ids": pending, "tool_call_count": calls,
                    "failures": state.get("failures", []) + [failure], "route": "replan"}
        evidence[step["id"]] = record
        return {"pending_step_ids": pending, "evidence": evidence, "tool_call_count": calls,
                "route": "execute" if pending else "finalize"}

    def replan_node(state):
        if state.get("replan_count", 0) >= state["max_replans"]:
            return {"error": "Replan limit reached after tool failure.", "route": "error"}
        if state.get("llm_call_count", 0) >= state["max_llm_calls"]:
            return {"error": "LLM call limit reached before replanning.", "route": "error"}
        try:
            plan, calls = request_plan(provider, state["question"], tools, trace,
                                       evidence=state.get("evidence", {}), failures=state.get("failures", []),
                                       max_steps=state["max_tool_calls"] - state.get("tool_call_count", 0))
            route = "execute" if plan["mode"] == "execute" else "terminal"
            trace.event("replan", revision=state.get("plan_revision", 1) + 1, plan=plan)
            return {"plan": plan, "plan_revision": state.get("plan_revision", 1) + 1,
                    "pending_step_ids": [s["id"] for s in plan["steps"]],
                    "replan_count": state.get("replan_count", 0) + 1,
                    "llm_call_count": state.get("llm_call_count", 0) + calls, "route": route}
        except Exception as exc:
            return {"error": f"Replanning failed: {type(exc).__name__}: {exc}", "route": "error"}

    def finalize_node(state):
        if state.get("llm_call_count", 0) >= state["max_llm_calls"]:
            return {"error": "LLM call limit reached before finalization.", "route": "error"}
        try:
            final, calls = request_answer(provider, state["question"], state.get("evidence", {}),
                                          state["plan"]["answer_type"], trace)
            return {"final": final, "llm_call_count": state.get("llm_call_count", 0) + calls, "route": "done"}
        except Exception as exc:
            return {"error": f"Finalization failed: {type(exc).__name__}: {exc}", "route": "error"}

    def error_node(state):
        text = state.get("error") or "The orchestrator could not complete this query."
        return {"final": {"status": "error", "answer_text": text,
                          "answer": {"type": "text", "value": text}}, "route": "done"}

    graph = StateGraph(OrchestratorState)
    graph.add_node("plan", plan_node)
    graph.add_node("terminal", terminal_node)
    graph.add_node("execute", execute_node)
    graph.add_node("replan", replan_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("error", error_node)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", lambda state: state["route"],
                                {"execute": "execute", "terminal": "terminal", "error": "error"})
    graph.add_conditional_edges("execute", lambda state: state["route"],
                                {"execute": "execute", "replan": "replan", "finalize": "finalize", "error": "error"})
    graph.add_conditional_edges("replan", lambda state: state["route"],
                                {"execute": "execute", "terminal": "terminal", "error": "error"})
    graph.add_edge("terminal", END)
    graph.add_edge("finalize", END)
    graph.add_edge("error", END)
    return graph
