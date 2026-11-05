"""LangGraph plan, execute, replan and evidence-bound finalization workflow."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .planner import request_answer, request_assessment, request_contract, request_plan
from .state import OrchestratorState


def build_workflow(*, provider, tools, executor, trace, graph_catalogue, entity_hints=()):
    def interpret_node(state):
        try:
            contract, calls = request_contract(provider, state["question"], trace,
                                               graph_catalogue=graph_catalogue, entity_hints=entity_hints)
            route = "contract_terminal" if contract["needs_clarification"] else "plan"
            return {"query_contract": contract, "llm_call_count": state.get("llm_call_count", 0) + calls,
                    "route": route}
        except Exception as exc:
            return {"error": f"Query interpretation failed: {type(exc).__name__}: {exc}", "route": "error"}

    def contract_terminal_node(state):
        contract = state["query_contract"]
        text = "Please clarify: " + "; ".join(contract["missing_information"])
        return {"final": {"status": "clarification_needed", "answer_text": text,
                          "answer": {"type": "text", "value": text}}, "route": "done"}

    def plan_node(state):
        try:
            plan, calls = request_plan(provider, state["question"], tools, trace,
                                       query_contract=state["query_contract"], graph_catalogue=graph_catalogue,
                                       entity_hints=entity_hints, max_steps=state["max_tool_calls"])
            route = "execute" if plan["mode"] == "execute" else "terminal"
            return {"plan": plan, "plan_revision": 1, "pending_step_ids": [s["id"] for s in plan["steps"]],
                    "llm_call_count": state.get("llm_call_count", 0) + calls, "route": route}
        except Exception as exc:
            return {"error": f"Planning failed: {type(exc).__name__}: {exc}", "route": "error"}

    def terminal_node(state):
        plan = state["plan"]
        failures = state.get("failures", [])
        execution_failures = [failure for failure in failures if failure.get("kind") == "execution"]
        if execution_failures:
            status = "error"
            text = execution_failures[-1].get("error") or "A supported graph-tool operation failed."
        else:
            status = plan["mode"]
            text = plan.get("explanation") or ("Clarification is needed." if status == "clarification_needed" else "No available tool supports this request.")
        return {"final": {"status": status, "answer_text": text,
                          "answer": {"type": "text", "value": text}}, "route": "done"}

    def execute_node(state):
        pending = list(state.get("pending_step_ids", []))
        evidence = dict(state.get("evidence", {}))
        if not pending:
            return {"route": "assess"}
        if state.get("tool_call_count", 0) >= state["max_tool_calls"]:
            return {"error": "Graph-tool call limit reached.", "route": "error"}
        step = executor.ready_step(state["plan"], pending, evidence)
        if step is None:
            return {"failures": state.get("failures", []) + [{"kind": "execution", "error": "No pending plan step is ready."}], "route": "replan"}
        record, issue, argument_llm_calls = executor.execute(step, evidence)
        calls = state.get("tool_call_count", 0) + 1
        llm_calls = state.get("llm_call_count", 0) + argument_llm_calls
        pending.remove(step["id"])
        if issue:
            failure = {"kind": "execution", "step_id": step["id"], "tool": step["tool"], "error": issue,
                       "output": record["output"] if record else None}
            return {"pending_step_ids": pending, "tool_call_count": calls, "llm_call_count": llm_calls,
                    "failures": state.get("failures", []) + [failure], "route": "replan"}
        evidence[step["id"]] = record
        if step["tool"] == "lookup_node":
            matches = record["output"].get("result")
            if matches == []:
                failure = {"kind": "resolution", "step_id": step["id"], "tool": step["tool"],
                           "error": "Node lookup returned no matches; retry once with an exact code token or a more identifying name segment.",
                           "arguments": record["arguments"], "output": record["output"]}
                return {"pending_step_ids": pending, "evidence": evidence, "tool_call_count": calls,
                        "llm_call_count": llm_calls, "failures": state.get("failures", []) + [failure],
                        "route": "replan"}
            if isinstance(matches, list) and len(matches) > 1:
                first = matches[0].get("score", 0) if isinstance(matches[0], dict) else 0
                second = matches[1].get("score", 0) if isinstance(matches[1], dict) else 0
                if abs(first - second) <= 0.05:
                    failure = {"kind": "ambiguity", "step_id": step["id"], "tool": step["tool"],
                               "error": "Node lookup returned multiple similarly ranked matches; user clarification is required.",
                               "output": record["output"]}
                    return {"pending_step_ids": pending, "evidence": evidence, "tool_call_count": calls,
                            "llm_call_count": llm_calls, "failures": state.get("failures", []) + [failure],
                            "route": "replan"}
        return {"pending_step_ids": pending, "evidence": evidence, "tool_call_count": calls,
                "llm_call_count": llm_calls, "route": "execute" if pending else "assess"}

    def assess_node(state):
        if state.get("llm_call_count", 0) >= state["max_llm_calls"]:
            return {"error": "LLM call limit reached before evidence assessment.", "route": "error"}
        try:
            assessment = request_assessment(
                provider, state["question"], state["query_contract"]["required_outputs"],
                state.get("evidence", {}), tools, trace,
            )
        except Exception as exc:
            return {"error": f"Evidence assessment failed: {type(exc).__name__}: {exc}", "route": "error"}
        trace.event("evidence_assessment", assessment=assessment)
        llm_calls = state.get("llm_call_count", 0) + 1
        if assessment["sufficient"]:
            return {"llm_call_count": llm_calls, "route": "finalize"}
        failure = {"kind": "evidence", "error": assessment.get("explanation") or "Evidence is insufficient.",
                   "missing_outputs": assessment.get("missing_outputs", []),
                   "unsupported_capability": assessment.get("unsupported_capability", "")}
        return {"llm_call_count": llm_calls, "failures": state.get("failures", []) + [failure],
                "route": "replan"}

    def replan_node(state):
        if state.get("replan_count", 0) >= state["max_replans"]:
            return {"error": "Replan limit reached after tool failure.", "route": "error"}
        if state.get("llm_call_count", 0) >= state["max_llm_calls"]:
            return {"error": "LLM call limit reached before replanning.", "route": "error"}
        try:
            plan, calls = request_plan(provider, state["question"], tools, trace,
                                       query_contract=state["query_contract"], graph_catalogue=graph_catalogue,
                                       entity_hints=entity_hints,
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
                                          state["query_contract"], trace)
            return {"final": final, "llm_call_count": state.get("llm_call_count", 0) + calls, "route": "done"}
        except Exception as exc:
            return {"error": f"Finalization failed: {type(exc).__name__}: {exc}", "route": "error"}

    def error_node(state):
        text = state.get("error") or "The orchestrator could not complete this query."
        return {"final": {"status": "error", "answer_text": text,
                          "answer": {"type": "text", "value": text}}, "route": "done"}

    graph = StateGraph(OrchestratorState)
    graph.add_node("interpret", interpret_node)
    graph.add_node("contract_terminal", contract_terminal_node)
    graph.add_node("plan", plan_node)
    graph.add_node("terminal", terminal_node)
    graph.add_node("execute", execute_node)
    graph.add_node("assess", assess_node)
    graph.add_node("replan", replan_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("error", error_node)
    graph.add_edge(START, "interpret")
    graph.add_conditional_edges("interpret", lambda state: state["route"],
                                {"plan": "plan", "contract_terminal": "contract_terminal", "error": "error"})
    graph.add_conditional_edges("plan", lambda state: state["route"],
                                {"execute": "execute", "terminal": "terminal", "error": "error"})
    graph.add_conditional_edges("execute", lambda state: state["route"],
                                {"execute": "execute", "replan": "replan", "assess": "assess", "error": "error"})
    graph.add_conditional_edges("assess", lambda state: state["route"],
                                {"replan": "replan", "finalize": "finalize", "error": "error"})
    graph.add_conditional_edges("replan", lambda state: state["route"],
                                {"execute": "execute", "terminal": "terminal", "error": "error"})
    graph.add_edge("contract_terminal", END)
    graph.add_edge("terminal", END)
    graph.add_edge("finalize", END)
    graph.add_edge("error", END)
    return graph
