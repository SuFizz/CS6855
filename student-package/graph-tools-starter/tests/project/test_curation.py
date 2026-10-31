import json
from pathlib import Path

import pytest

from orchestrator.curation import load_selection, materialize_registry


def test_selection_requires_all_menu_tools(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="exactly the 33"):
        load_selection(path)


def test_materialization_preserves_selected_files(tmp_path, monkeypatch):
    source = tmp_path / "source"
    folder = source / "graph_summary" / "abc"
    folder.mkdir(parents=True)
    (folder / "tool.json").write_text(json.dumps({"name": "graph_summary", "implementation_id": "graph_summary.abc"}))
    (folder / "tool.py").write_bytes(b"print('unchanged')\r\n")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"graph_summary": "graph_summary.abc"}))
    monkeypatch.setattr("orchestrator.curation.load_selection", lambda _: {"graph_summary": "graph_summary.abc"})
    destination = tmp_path / "registry" / "tools"
    materialize_registry(source, selection, destination, verify_candidates=False)
    assert (destination / "graph_summary" / "abc" / "tool.py").read_bytes() == b"print('unchanged')\r\n"
