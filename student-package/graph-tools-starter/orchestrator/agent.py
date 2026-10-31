"""Public facade for the manifest-driven LangGraph graph analytics orchestrator."""
from __future__ import annotations

import hashlib
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from .executor import PlanExecutor
from .provider import GeminiProvider
from .registry import discover_project_tools
from .tracing import ProjectTrace
from .workflow import build_workflow


class Orchestrator:
    def __init__(self, project_root, *, client=None, model=None, state_dir=None,
                 cache_enabled=True, allow_incomplete_registry=False,
                 max_tool_calls=5, max_llm_calls=8, max_replans=2):
        self.project_root = Path(project_root).resolve()
        self.state_dir = Path(state_dir) if state_dir else self.project_root / ".orchestrator"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.tools = discover_project_tools(
            self.project_root, allow_incomplete=allow_incomplete_registry,
        )
        self.provider = GeminiProvider(
            client=client, model=model, cache_path=self.state_dir / "llm_cache.json",
            cache_enabled=cache_enabled,
        )
        self.max_tool_calls = max_tool_calls
        self.max_llm_calls = max_llm_calls
        self.max_replans = max_replans
        self._saver_context = SqliteSaver.from_conn_string(str(self.state_dir / "checkpoints.sqlite"))
        self._saver = self._saver_context.__enter__()

    @staticmethod
    def _thread_id(query_id, question):
        digest = hashlib.sha256(f"{query_id}\0{question}".encode("utf-8")).hexdigest()[:16]
        return f"{query_id}-{digest}"

    def answer(self, query_id, question, trace_path, *, resume=False):
        trace = ProjectTrace(trace_path, resume=resume)
        if trace.sequence == 0:
            trace.event("question", query_id=str(query_id), text=question)
        executor = PlanExecutor(self.tools, self.project_root / "data", trace)
        graph = build_workflow(provider=self.provider, tools=self.tools, executor=executor, trace=trace)
        app = graph.compile(checkpointer=self._saver)
        thread_id = self._thread_id(query_id, question)
        config = {"configurable": {"thread_id": thread_id}}
        if not resume:
            self._saver.delete_thread(thread_id)
        snapshot = app.get_state(config) if resume else None
        if snapshot and snapshot.values:
            executor.rehydrate(snapshot.values.get("evidence", {}))
            state = snapshot.values if not snapshot.next else app.invoke(None, config=config)
        else:
            initial = {
                "query_id": str(query_id),
                "query_hash": thread_id,
                "question": question,
                "trace_path": str(trace_path),
                "plan_revision": 0,
                "pending_step_ids": [],
                "evidence": {},
                "failures": [],
                "tool_call_count": 0,
                "llm_call_count": 0,
                "replan_count": 0,
                "max_tool_calls": self.max_tool_calls,
                "max_llm_calls": self.max_llm_calls,
                "max_replans": self.max_replans,
                "final": None,
                "error": None,
            }
            state = app.invoke(initial, config=config)
        final = state.get("final") or {
            "status": "error",
            "answer_text": state.get("error") or "The orchestrator ended without a final answer.",
            "answer": {"type": "text", "value": state.get("error") or "The orchestrator ended without a final answer."},
        }
        evidence = state.get("evidence", {})
        tools_used = []
        for record in evidence.values():
            implementation_id = record["implementation_id"]
            if implementation_id not in tools_used:
                tools_used.append(implementation_id)
        try:
            relative_trace = Path(trace_path).resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            relative_trace = str(trace_path)
        answer = {
            "query_id": str(query_id),
            "status": final["status"],
            "answer_text": final["answer_text"],
            "answer": final["answer"],
            "tools_used": tools_used,
            "trace_file": relative_trace,
        }
        if not trace.has_terminal():
            trace.event("final_answer", answer=answer)
        return answer

    def close(self):
        if self._saver_context is not None:
            self._saver_context.__exit__(None, None, None)
            self._saver_context = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


__all__ = ["Orchestrator", "discover_project_tools"]
