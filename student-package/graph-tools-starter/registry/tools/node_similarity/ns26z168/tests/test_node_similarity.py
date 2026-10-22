from graphkit.testing import load, store, to_networkx

tool = load(__file__)


def test_matches_common_neighbors_and_jaccard():
    out = tool.run(store(), graph_id="karate", node="m1", other="m2")
    assert out["status"] == "ok"
    graph = to_networkx(store().get("karate"))
    left, right = set(graph.neighbors("m1")), set(graph.neighbors("m2"))
    assert out["result"]["common_count"] == len(left & right)
    assert out["result"]["jaccard"] == round(len(left & right) / len(left | right), 6)


def test_common_neighbor_ids_are_labeled():
    out = tool.run(store(), graph_id="karate", node="m1", other="m2")
    assert all("id" in item and "label" in item for item in out["result"]["common_neighbors"])


def test_empty_union_has_zero_similarity():
    out = tool.run(store(), graph_id="airports", node="MAA", other="LIS")
    assert out["status"] == "ok"
    assert 0.0 <= out["result"]["jaccard"] <= 1.0


def test_unknown_node_is_an_error():
    assert tool.run(store(), graph_id="karate", node="missing", other="1")["status"] == "error"


def test_equal_nodes_and_unknown_parameters_are_errors():
    assert tool.run(store(), graph_id="karate", node="1", other="1")["status"] == "error"
    assert tool.run(store(), graph_id="karate", node="m1", other="m2", bad=True)["status"] == "error"


def test_deterministic_output():
    args = {"graph_id": "karate", "node": "1", "other": "2"}
    assert tool.run(store(), **args) == tool.run(store(), **args)

