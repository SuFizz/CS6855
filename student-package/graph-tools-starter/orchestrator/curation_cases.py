"""Independent canonical workloads and semantic checks for registry curation."""
from __future__ import annotations

import math

from graphkit.testing import to_networkx

CASES = {
    "articulation_points_bridges": {"graph_id": "karate", "top_k": 100},
    "betweenness_centrality": {"graph_id": "karate", "top_k": 100},
    "bipartite_check": {"graph_id": "projects", "top_k": 100},
    "bipartite_matching": {"graph_id": "projects", "top_k": 100},
    "closeness_centrality": {"graph_id": "karate", "node": "m1", "top_k": 100},
    "clustering_coefficient": {"graph_id": "karate", "node": "m1", "top_k": 100},
    "connected_components": {"graph_id": "projects", "top_k": 100},
    "degree_centrality": {"graph_id": "karate", "top_k": 100},
    "degree_distribution": {"graph_id": "karate"},
    "diameter_eccentricity": {"graph_id": "karate", "top_k": 100},
    "ego_network": {"graph_id": "karate", "node": "m1", "radius": 1, "level": 1, "top_k": 100},
    "girvan_newman": {"graph_id": "karate", "num_communities": 2, "communities": 2, "top_k": 100},
    "graph_summary": {"graph_id": "karate"},
    "greedy_coloring": {"graph_id": "karate", "top_k": 100},
    "hits": {"graph_id": "courses", "top_k": 100, "rank_by": "authority", "score": "authority", "score_type": "authority", "sort_by": "authority"},
    "k_core": {"graph_id": "karate", "top_k": 100},
    "k_hop_neighborhood": {"graph_id": "karate", "node": "m1", "k": 2, "top_k": 100},
    "label_propagation": {"graph_id": "karate", "seed": 0, "top_k": 100},
    "link_prediction": {"graph_id": "karate", "method": "common_neighbors", "top_k": 100},
    "louvain": {"graph_id": "karate", "seed": 0, "top_k": 100},
    "max_flow_min_cut": {"graph_id": "courses", "source": "CS101", "sink": "CS401", "top_k": 100},
    "maximal_cliques": {"graph_id": "karate", "top_k": 100},
    "minimum_spanning_tree": {"graph_id": "lesmis", "top_k": 100},
    "motif_count": {"graph_id": "karate", "top_k": 100},
    "node_embedding_similarity": {"graph_id": "karate", "node": "m1", "top_k": 10, "seed": 0},
    "node_similarity": {"graph_id": "karate", "node": "m1", "other": "m2", "node1": "m1", "node2": "m2", "node_a": "m1", "node_b": "m2", "target_node": "m2"},
    "pagerank": {"graph_id": "karate", "damping": 0.85, "top_k": 100},
    "personalized_pagerank": {"graph_id": "karate", "node": "m1", "seed_node": "m1", "seeds": ["m1"], "top_k": 100},
    "reachability": {"graph_id": "courses", "source": "CS101", "target": "CS405"},
    "shortest_path_unweighted": {"graph_id": "karate", "source": "m1", "target": "m34", "src": "m1", "dest": "m34"},
    "shortest_path_weighted": {"graph_id": "airports", "source": "MAA", "target": "DEL"},
    "strongly_connected_components": {"graph_id": "courses", "top_k": 100},
    "topological_sort": {"graph_id": "courses", "top_k": 100},
}


def adapted_arguments(tool):
    properties = tool.manifest.get("parameters", {}).get("properties", {})
    intended = CASES[tool.name]
    arguments = {name: value for name, value in intended.items() if name in properties}
    for required in tool.manifest.get("parameters", {}).get("required", []):
        if required not in arguments:
            examples = tool.manifest.get("examples") or []
            if examples and required in examples[0]:
                arguments[required] = examples[0][required]
    return arguments


def _find(result, names):
    if isinstance(result, dict):
        for name in names:
            if name in result:
                return result[name]
        for value in result.values():
            found = _find(value, names)
            if found is not None:
                return found
    return None


def _node_ids(value):
    if isinstance(value, list):
        ids = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                ids.append(item["id"])
            elif isinstance(item, str):
                ids.append(item)
        return ids
    return []


def _ranking(result):
    if isinstance(result, list):
        return _node_ids(result)
    for key in ("ranking", "nodes", "authorities", "top_items", "similar"):
        value = _find(result, (key,))
        ids = _node_ids(value)
        if ids:
            return ids
    return []


def semantic_check(tool, output, store, arguments):
    import networkx as nx

    graph = store.get(arguments["graph_id"])
    network = to_networkx(graph)
    result = output["result"]
    name = tool.name
    if result is None:
        return False, "successful output has null result"
    if name == "graph_summary":
        nodes = _find(result, ("nodes", "node_count", "number_of_nodes"))
        edges = _find(result, ("edges", "edge_count", "number_of_edges"))
        return nodes == network.number_of_nodes() and edges == network.number_of_edges(), "node/edge counts differ"
    if name in ("degree_centrality", "betweenness_centrality", "pagerank", "closeness_centrality"):
        ids = _ranking(result)
        return bool(ids) and len(ids) == len(set(ids)), "ranking is empty or repeats nodes"
    if name == "reachability":
        expected = nx.has_path(network, arguments["source"], arguments["target"])
        actual = _find(result, ("reachable",))
        return actual == expected, "reachability differs from NetworkX"
    if name in ("shortest_path_unweighted", "shortest_path_weighted"):
        path = _find(result, ("path",))
        ids = (_node_ids(path) or path) if isinstance(path, list) else []
        if not ids or ids[0] not in (arguments.get("source"), arguments.get("src")) or ids[-1] not in (arguments.get("target"), arguments.get("dest")):
            return False, "path endpoints are wrong"
        return all(network.has_edge(u, v) for u, v in zip(ids, ids[1:])), "path uses a missing edge"
    if name == "node_similarity":
        left = set(graph.undirected_neighbors("m1")); right = set(graph.undirected_neighbors("m2"))
        expected = len(left & right) / len(left | right)
        actual = _find(result, ("jaccard", "jaccard_score", "jaccard_similarity"))
        return isinstance(actual, (int, float)) and abs(actual - expected) < 1e-6, "Jaccard score is wrong"
    if name == "diameter_eccentricity":
        actual = _find(result, ("diameter",))
        return isinstance(actual, (int, float)) and abs(actual - nx.diameter(network)) < 1e-6, "diameter is wrong"
    if name == "bipartite_check":
        return _find(result, ("is_bipartite",)) is True, "projects graph was not recognized as bipartite"
    if name == "bipartite_matching":
        expected = len(nx.algorithms.bipartite.maximum_matching(network, top_nodes={n for n, d in network.nodes(data=True) if d.get("side") == "student"})) // 2
        actual = _find(result, ("matching_size",))
        return actual == expected, "matching size differs from NetworkX"
    if name == "connected_components":
        expected = nx.number_connected_components(network)
        actual = _find(result, ("num_components", "number_of_components", "component_count"))
        if actual is None and isinstance(result, list):
            actual = len(result)
        return actual == expected, "component count differs from NetworkX"
    if name == "strongly_connected_components":
        expected = nx.number_strongly_connected_components(network)
        actual = _find(result, ("num_components", "number_of_components", "component_count"))
        return actual == expected, "SCC count differs from NetworkX"
    if name == "topological_sort":
        order = _node_ids(result) if isinstance(result, list) else _node_ids(_find(result, ("order",)))
        positions = {node: index for index, node in enumerate(order)}
        valid = len(order) == len(network) and all(positions[u] < positions[v] for u, v in network.edges())
        return valid, "topological order is incomplete or invalid"
    if name == "minimum_spanning_tree":
        expected = nx.minimum_spanning_tree(network, weight="weight").size(weight="weight")
        actual = _find(result, ("total_weight",))
        return isinstance(actual, (int, float)) and abs(actual - expected) < 1e-6, "MST total weight differs"
    if name == "max_flow_min_cut":
        expected = nx.maximum_flow_value(network, arguments["source"], arguments["sink"], capacity="weight")
        actual = _find(result, ("max_flow", "max_flow_value", "flow_value"))
        return isinstance(actual, (int, float)) and abs(actual - expected) < 1e-6, "max-flow value differs"
    if name == "articulation_points_bridges":
        expected_points = set(nx.articulation_points(network)); actual_points = set(_node_ids(_find(result, ("articulation_points",))))
        return expected_points == actual_points, "articulation points differ"
    if name == "maximal_cliques":
        expected = max(map(len, nx.find_cliques(network)))
        cliques = _find(result, ("cliques", "largest_cliques")) or result if isinstance(result, list) else []
        sizes = [item.get("size", len(item.get("nodes", []))) if isinstance(item, dict) else len(item) for item in cliques]
        return bool(sizes) and max(sizes) == expected, "maximum clique size differs"
    if name == "greedy_coloring":
        valid = _find(result, ("valid", "is_proper_coloring"))
        if isinstance(valid, bool):
            return valid, "tool reports invalid coloring"
    if name == "motif_count":
        expected = sum(nx.triangles(network).values()) // 3
        actual = _find(result, ("count", "triangle_count", "triangles"))
        if isinstance(actual, list):
            actual = len(actual)
        return actual == expected, "triangle count differs"
    if name in ("ego_network", "k_hop_neighborhood", "k_core", "label_propagation", "louvain", "girvan_newman", "bipartite_check", "minimum_spanning_tree", "maximal_cliques"):
        if output.get("result_graph_id") is None:
            return False, "subgraph-producing tool did not register result_graph_id"
    def finite(value):
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, list):
            return all(finite(item) for item in value)
        return True
    return finite(result), "result contains a non-finite number"
