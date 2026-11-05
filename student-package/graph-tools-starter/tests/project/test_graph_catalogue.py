from pathlib import Path

from orchestrator.graph_catalogue import build_graph_catalogue, find_entity_hints

ROOT = Path(__file__).resolve().parents[2]


def test_graph_catalogue_is_dynamic_and_contains_schema_metadata():
    catalogue = build_graph_catalogue(ROOT / "data")
    by_id = {item["graph_id"]: item for item in catalogue}
    assert set(by_id) == {"airports", "karate", "lesmis", "courses", "projects"}
    assert by_id["airports"]["directed"] is True
    assert by_id["airports"]["weighted"] is True
    assert by_id["airports"]["node_attributes"] == ["city", "country", "lat", "lon"]
    assert by_id["airports"]["edge_attributes"] == ["airlines"]
    assert by_id["karate"]["node_attributes"] == ["club"]


def test_question_entity_hints_find_both_named_roles_in_same_graph():
    hints = find_entity_hints(ROOT / "data", "How many independent friendship chains connect the instructor and administrator?")
    karate = next(item for item in hints if item["graph_id"] == "karate")
    assert {item["id"] for item in karate["matches"]} >= {"m1", "m34"}
    assert all(len(item["matches"]) <= 5 for item in hints)
    assert not any(item["graph_id"] == "courses" for item in hints)


def test_short_exact_node_code_remains_a_hint():
    hints = find_entity_hints(ROOT / "data", "Find a route to JFK")
    airports = next(item for item in hints if item["graph_id"] == "airports")
    assert any(item["id"] == "JFK" for item in airports["matches"])
