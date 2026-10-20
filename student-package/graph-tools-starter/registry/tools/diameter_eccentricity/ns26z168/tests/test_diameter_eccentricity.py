import networkx as nx

from graphkit.testing import load, store, to_networkx

tool = load(__file__)


def test_diameter_matches_networkx():
    out = tool.run(store(), graph_id="karate")
    graph = to_networkx(store().get("karate"))
    assert out["status"] == "ok"
    assert out["result"]["diameter"] == nx.diameter(graph)


def test_eccentricity_matches_networkx():
    out = tool.run(store(), graph_id="karate", top_k=100)
    expected = nx.eccentricity(to_networkx(store().get("karate")))
    actual = {item["id"]: item["score"] for item in out["result"]["eccentricity"]}
    assert actual == {node: float(score) for node, score in expected.items()}


def test_center_and_periphery_match_networkx():
    out = tool.run(store(), graph_id="karate", top_k=100)
    graph = to_networkx(store().get("karate"))
    assert {x["id"] for x in out["result"]["center"]} == set(nx.center(graph))
    assert {x["id"] for x in out["result"]["periphery"]} == set(nx.periphery(graph))


def test_top_k_bounds_lists():
    out = tool.run(store(), graph_id="karate", top_k=2)
    assert len(out["result"]["eccentricity"]) == 2
    assert len(out["result"]["center"]) <= 2
    assert len(out["result"]["periphery"]) <= 2


def test_disconnected_graph_is_an_error():
    assert tool.run(store(), graph_id="projects")["status"] == "error"


def test_invalid_top_k_and_unknown_graph_are_errors():
    assert tool.run(store(), graph_id="karate", top_k=0)["status"] == "error"
    assert tool.run(store(), graph_id="no_such_graph")["status"] == "error"

