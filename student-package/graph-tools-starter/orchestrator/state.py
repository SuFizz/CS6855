"""Serializable state and records for the LangGraph orchestrator."""
from __future__ import annotations

from typing import Any, Literal, TypedDict


class Binding(TypedDict):
    parameter: str
    step_id: str
    pointer: str


class PlanStep(TypedDict):
    id: str
    tool: str
    purpose: str
    arguments: dict[str, Any]
    bindings: list[Binding]
    depends_on: list[str]


class Plan(TypedDict):
    mode: Literal["execute", "clarification_needed", "unsupported"]
    explanation: str
    answer_type: Literal["number", "boolean", "node", "node_list", "path", "text"]
    steps: list[PlanStep]


class EvidenceRecord(TypedDict):
    step_id: str
    tool: str
    implementation_id: str
    arguments: dict[str, Any]
    output: dict[str, Any]


class OrchestratorState(TypedDict, total=False):
    query_id: str
    query_hash: str
    question: str
    trace_path: str
    plan: Plan
    plan_revision: int
    pending_step_ids: list[str]
    evidence: dict[str, EvidenceRecord]
    failures: list[dict[str, Any]]
    tool_call_count: int
    llm_call_count: int
    replan_count: int
    max_tool_calls: int
    max_llm_calls: int
    max_replans: int
    registry_warning: str | None
    route: str
    final: dict[str, Any] | None
    error: str | None
