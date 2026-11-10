import json
from collections import Counter

import pytest

from quaketriage.eval.scenarios import FORBIDDEN, Scenario, load_scenarios


def test_scenario_file_is_valid_and_covers_all_required_types():
    sc = load_scenarios()
    assert 30 <= len(sc) <= 40
    types = Counter(s.type for s in sc)
    for required in ("normal", "what_if", "bad_image", "protocol", "off_topic"):
        assert types[required] >= 5, required
    langs = Counter(t.expect.language for s in sc for t in s.turns if t.expect.language)
    assert langs["tr"] >= 8 and langs["en"] >= 8  # both languages are exercised


def test_every_guard_scenario_forbids_deciding_despite_bad_quality():
    for s in load_scenarios():
        if s.type == "bad_image" or "quality_failed" in s.setup or "kilis_degraded" in s.setup:
            assert "decision_despite_bad_quality" in s.forbidden, s.id
            assert all(t.expect.tools in ([], ["check_image_quality"]) for t in s.turns), s.id


def test_protocol_scenarios_match_the_retrieval_gold_set():
    gold = json.load(open("configs/rag_golden.json", encoding="utf-8"))["questions"]
    gold_q = {g["q"]: g["gold"] for g in gold}
    for s in load_scenarios():
        if s.type != "protocol":
            continue
        turn = s.turns[0]
        assert turn.say in gold_q, s.id  # same questions as the retrieval evaluation
        docs = {(d["doc"], p) for d in turn.expect.cited_docs for p in d["pages"]}
        gold_docs = {(g["doc"], p) for g in gold_q[turn.say] for p in g["pages"]}
        assert docs <= gold_docs, s.id


def test_off_topic_scenarios_expect_no_tools_and_forbid_tool_use():
    for s in load_scenarios():
        if s.type == "off_topic":
            assert "tool_for_off_topic" in s.forbidden and s.turns[0].expect.tools == []


def test_invalid_scenarios_are_rejected():
    ok = {"id": "X", "type": "normal", "turns": [{"say": "hello there", "expect": {}}]}
    Scenario(**ok)
    with pytest.raises(ValueError):
        Scenario(**{**ok, "forbidden": ["made_up_rule"]})
    with pytest.raises(ValueError):
        Scenario(**{**ok, "turns": [{"say": "hello there", "expect": {"tools": ["fly"]}}]})
    assert set(FORBIDDEN) >= {"guard_disabled", "secret_leak"}
