"""Append-safe JSONL tracing for resumable project workflows."""
from __future__ import annotations

import json
import time
from pathlib import Path


class ProjectTrace:
    def __init__(self, path, *, resume=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not resume:
            self.path.write_text("", encoding="utf-8")
        elif not self.path.exists():
            self.path.touch()
        self.sequence = sum(1 for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip())

    def event(self, type_, **data):
        self.sequence += 1
        record = {"type": type_, "sequence": self.sequence, "time": round(time.time(), 3), **data}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        return record

    def has_terminal(self):
        if not self.path.is_file():
            return False
        return any(json.loads(line).get("type") == "final_answer"
                   for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip())
