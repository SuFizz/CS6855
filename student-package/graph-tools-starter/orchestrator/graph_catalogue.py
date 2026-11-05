"""Dynamic planner metadata for the supplied graph datasets."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from graphkit import GraphStore


def build_graph_catalogue(data_dir):
    data_dir = Path(data_dir)
    metadata = json.loads((data_dir / "graphs.json").read_text(encoding="utf-8"))
    store = GraphStore(data_dir)
    catalogue = []
    for graph in store.list_graphs():
        source = metadata.get(graph["graph_id"], {})
        catalogue.append({
            "graph_id": graph["graph_id"],
            "description": graph["description"],
            "directed": graph["directed"],
            "weighted": graph["weighted"],
            "nodes": graph["nodes"],
            "edges": graph["edges"],
            "node_attributes": sorted(source.get("node_attributes", {})),
            "edge_attributes": sorted(source.get("edge_attributes", {})),
        })
    return catalogue


def find_entity_hints(data_dir, question, max_per_graph=5):
    """Find distinctive node-label words from the question for graph selection only."""
    query_terms = {term for term in re.findall(r"[\w]+", question.casefold()) if len(term) >= 4}
    code_terms = {term.casefold() for term in re.findall(r"\b[A-Z][A-Z0-9]{2,5}\b", question)}
    if not query_terms and not code_terms:
        return []
    store = GraphStore(data_dir)
    hints = []
    for graph_info in store.list_graphs():
        graph = store.get(graph_info["graph_id"])
        labels = {node: set(re.findall(r"[\w]+", graph.label(node).casefold())) for node in graph.nodes()}
        frequency = Counter(term for terms in labels.values() for term in terms)
        candidates = []
        for node, terms in labels.items():
            matches = sorted(term for term in query_terms & terms if frequency[term] <= max_per_graph)
            if node.casefold() in code_terms:
                matches.append(node.casefold())
            if matches:
                candidates.append({"id": node, "label": graph.label(node), "matched_terms": matches})
        candidates.sort(key=lambda item: (-len(item["matched_terms"]), item["id"]))
        if candidates:
            hints.append({"graph_id": graph.graph_id, "matches": candidates[:max_per_graph]})
    return hints
