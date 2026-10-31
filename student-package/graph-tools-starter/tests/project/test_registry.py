import json
from pathlib import Path

import pytest

from graphkit.menu import INFRA_TOOLS, MENU
from orchestrator.registry import RegistryError, discover_project_tools, scan_manifests


def write_tool(folder, name, implementation_id):
    folder.mkdir(parents=True)
    (folder / "tool.json").write_text(json.dumps({
        "name": name, "implementation_id": implementation_id, "description": "A sufficiently detailed tool description.",
        "tier": "infra" if name in INFRA_TOOLS else MENU[name][0],
        "category": "infrastructure" if name in INFRA_TOOLS else MENU[name][1],
        "parameters": {"type": "object", "properties": {}, "required": []}, "examples": [{}], "author": "test",
    }))
    (folder / "tool.py").write_text("from graphkit.contract import ok\ndef run(store, **params):\n    return ok({}, 'ok')\n")


def make_complete_project(tmp_path):
    for name in INFRA_TOOLS:
        write_tool(tmp_path / "infra_tools" / name, name, f"{name}.instructors")
    for name in MENU:
        write_tool(tmp_path / "registry" / "tools" / name / "one", name, f"{name}.one")
    return tmp_path


def test_manifest_scan_does_not_import_tool_code(tmp_path):
    folder = tmp_path / "registry" / "tools" / "graph_summary" / "bad"
    folder.mkdir(parents=True)
    (folder / "tool.json").write_text(json.dumps({
        "name": "graph_summary", "implementation_id": "graph_summary.bad", "description": "description",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }))
    (folder / "tool.py").write_text("raise RuntimeError('must not import during scan')\n")
    records = scan_manifests(tmp_path)
    assert records[0].implementation_id == "graph_summary.bad"


def test_strict_registry_rejects_missing_and_duplicates(tmp_path):
    project = make_complete_project(tmp_path)
    duplicate = project / "registry" / "tools" / "graph_summary" / "two"
    write_tool(duplicate, "graph_summary", "graph_summary.two")
    with pytest.raises(RegistryError, match="duplicate"):
        discover_project_tools(project)
    tools = discover_project_tools(project, allow_incomplete=True)
    assert tools["graph_summary"].implementation_id == "graph_summary.one"
