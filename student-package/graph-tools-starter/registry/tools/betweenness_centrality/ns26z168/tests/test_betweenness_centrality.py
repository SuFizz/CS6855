import networkx as nx

from graphkit.testing import load, store, to_networkx

tool = load(__file__)


def test_matches_networkx_on_karate():
    out = tool.run(store(), graph_id="karate", top_k=100)
    expected = nx.betweenness_centrality(to_networkx(store().get("karate")))
    actual = {item["id"]: item["score"] for item in out["result"]}
    for node, score in expected.items():
        assert abs(actual[node] - score) < 1e-6


def test_top_k_and_ordering():
    out = tool.run(store(), graph_id="lesmis", top_k=5)
    assert len(out["result"]) == 5
    assert all(a["score"] >= b["score"] for a, b in zip(out["result"], out["result"][1:]))


def test_scores_are_normalized():
    out = tool.run(store(), graph_id="karate", top_k=100)
    assert all(0.0 <= item["score"] <= 1.0 for item in out["result"])


def test_small_graph_behavior_is_non_crashing():
    out = tool.run(store(), graph_id="courses", top_k=1)
    assert out["status"] == "ok"


def test_invalid_top_k_and_unknown_graph_are_errors():
    assert tool.run(store(), graph_id="karate", top_k=0)["status"] == "error"
    assert tool.run(store(), graph_id="no_such_graph")["status"] == "error"


def test_deterministic_output():
    args = {"graph_id": "karate", "top_k": 10}
    assert tool.run(store(), **args) == tool.run(store(), **args)

