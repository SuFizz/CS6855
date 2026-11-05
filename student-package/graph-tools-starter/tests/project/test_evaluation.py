from evaluation.score_public import compare, topological_valid


def test_public_scorer_checks_status_type_and_numeric_tolerance():
    expected = {"status": "answered", "type": "number", "value": 100.0}
    assert compare(expected, {"status": "answered", "answer": {"type": "number", "value": 100.09}})[0]
    assert not compare(expected, {"status": "unsupported", "answer": {"type": "text", "value": "x"}})[0]
    assert not compare(expected, {"status": "answered", "answer": {"type": "text", "value": "100"}})[0]


def test_public_scorer_validates_topological_order():
    valid = ["CS101", "MA101", "MA102", "CS102", "CS202", "MA201", "MA202", "CS203",
             "CS201", "CS302", "CS301", "CS303", "CS304", "CS305", "CS306", "CS307",
             "CS401", "CS402", "CS405", "CS403", "CS404"]
    assert topological_valid(valid)
    assert not topological_valid(list(reversed(valid)))
