"""Read project query sets in text, JSON, or JSONL form."""
from __future__ import annotations

import json
from pathlib import Path


def _normalise(item, index):
    if isinstance(item, str):
        question = item
        query_id = f"q{index}"
    elif isinstance(item, dict):
        question = item.get("question") or item.get("query") or item.get("text")
        query_id = item.get("query_id") or item.get("id") or f"q{index}"
    else:
        raise ValueError(f"Query {index} must be a string or object.")
    if not isinstance(query_id, str) or not query_id.strip():
        raise ValueError(f"Query {index} has an invalid id.")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Query {query_id!r} has no question text.")
    return {"query_id": query_id.strip(), "question": question.strip()}


def load_queries(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("queries")
        if not isinstance(payload, list):
            raise ValueError("A JSON query file must contain a list or a 'queries' list.")
        items = payload
    elif path.suffix.lower() == ".jsonl":
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        items = [line.strip() for line in text.splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
    queries = [_normalise(item, index) for index, item in enumerate(items, 1)]
    ids = [item["query_id"] for item in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("Query ids must be unique.")
    return queries
