"""Manifest-first discovery for a physically curated project registry."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

from graphkit.loader import Tool, load_tool
from graphkit.menu import INFRA_TOOLS, MENU


class RegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class ManifestRecord:
    name: str
    implementation_id: str
    manifest: dict
    path: Path
    infrastructure: bool


def _read_record(path, infrastructure):
    try:
        manifest = json.loads((path / "tool.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"Cannot read {path / 'tool.json'}: {exc}") from exc
    for field in ("name", "implementation_id", "description", "parameters"):
        if field not in manifest:
            raise RegistryError(f"{path / 'tool.json'} is missing {field!r}.")
    if not isinstance(manifest["name"], str) or not manifest["name"]:
        raise RegistryError(f"{path / 'tool.json'} has an invalid name.")
    parameters = manifest["parameters"]
    if not isinstance(parameters, dict) or parameters.get("type") != "object" or not isinstance(parameters.get("properties"), dict):
        raise RegistryError(f"{path / 'tool.json'} has an invalid parameters schema.")
    if not (path / "tool.py").is_file():
        raise RegistryError(f"{path} has no tool.py.")
    return ManifestRecord(manifest["name"], manifest["implementation_id"], manifest, path, infrastructure)


def scan_manifests(project_root):
    root = Path(project_root).resolve()
    records = []
    locations = ((root / "infra_tools", True), (root / "registry" / "tools", False))
    for location, infrastructure in locations:
        if not location.is_dir():
            continue
        for manifest_path in sorted(location.rglob("tool.json")):
            records.append(_read_record(manifest_path.parent, infrastructure))
    return records


def select_records(records, allow_incomplete=False):
    grouped = {}
    for record in records:
        grouped.setdefault(record.name, []).append(record)
    expected = set(MENU) | set(INFRA_TOOLS)
    missing = sorted(expected - set(grouped))
    duplicates = {name: rows for name, rows in grouped.items() if len(rows) > 1}
    unexpected = sorted(set(grouped) - expected)
    messages = []
    if missing:
        messages.append(f"missing tools: {', '.join(missing)}")
    if duplicates:
        detail = "; ".join(f"{name}=[{', '.join(row.implementation_id for row in rows)}]" for name, rows in sorted(duplicates.items()))
        messages.append(f"duplicate tools: {detail}")
    if unexpected:
        messages.append(f"unexpected tools: {', '.join(unexpected)}")
    if messages and not allow_incomplete:
        raise RegistryError("Curated registry validation failed: " + " | ".join(messages))
    if messages:
        warnings.warn("Curated registry warning: " + " | ".join(messages), RuntimeWarning, stacklevel=2)
    selected = {}
    for name, rows in grouped.items():
        if name not in expected:
            continue
        selected[name] = min(rows, key=lambda row: (row.implementation_id, str(row.path)))
    return selected


def discover_project_tools(project_root, allow_incomplete=False):
    selected = select_records(scan_manifests(project_root), allow_incomplete=allow_incomplete)
    tools: dict[str, Tool] = {}
    failures = []
    for name, record in sorted(selected.items()):
        try:
            tools[name] = load_tool(record.path)
        except Exception as exc:
            failures.append(f"{record.path}: {type(exc).__name__}: {exc}")
    if failures and not allow_incomplete:
        raise RegistryError("Selected tools failed to load: " + " | ".join(failures))
    if failures:
        warnings.warn("Selected tools skipped: " + " | ".join(failures), RuntimeWarning, stacklevel=2)
    return tools


def manifest_catalogue(tools):
    catalogue = []
    for name, tool in sorted(tools.items()):
        manifest = tool.manifest
        parameters = manifest.get("parameters", {})
        catalogue.append({
            "name": name,
            "implementation_id": tool.implementation_id,
            "category": manifest.get("category"),
            "description": manifest.get("description", ""),
            "required": parameters.get("required", []),
            "parameters": parameters.get("properties", {}),
        })
    return catalogue
