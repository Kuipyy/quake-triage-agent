from quaketriage.eval.runner import (
    check_turn,
    cited_sources,
    detect_violations,
    is_subset,
    summarize,
)
from quaketriage.eval.scenarios import Expect, Scenario, Turn


def _sc(**kw):
    base = {"id": "X", "type": "normal", "turns": [Turn(say="hello there", expect=Expect())]}
    return Scenario(**{**base, **kw})


FINAL = {"quality_ok": None, "uncertainty_tau": 0.2, "plan_stops": set(), "queue_rows": set(),
         "retrieved_refs": [None], "env_secrets": []}


def test_is_subset():
    assert is_subset({"a": {"b": 1}}, {"a": {"b": 1.0, "c": 2}})
    assert not is_subset({"a": 2}, {"a": 1})


def test_check_turn_exact_tools_and_contains():
    e = Expect(tools=["plan_routes"], answer_contains=["ok"], stop_reason=None)
    res = {"intent": "x", "language": "en", "tools": ["plan_routes"], "stop_reason": None,
           "answer": "all ok", "constraints": {}}
    assert all(check_turn(e, res).values())
    res["tools"] = []
    assert not check_turn(e, res)["tools_exact"]


def test_cited_sources_page_range():
    a = "text [1]\n\nSources: [1] INSARAG Vol II, p. 7-8; [2] Doc B, p. 3-3"
    assert cited_sources(a)[0] == {"ref": 1, "doc": "INSARAG Vol II", "start": 7, "end": 8}


def test_violations_counted_separately_and_undeclared():
    sc = _sc(type="off_topic")
    trace = [{"tool": "detect_damage", "status": "ok"}, {"tool": "plan_routes", "status": "ok"}]
    v = detect_violations(sc, trace, ["answer"], FINAL)
    assert [x["rule"] for x in v] == ["tool_for_off_topic"] * 2
    sc2 = _sc(setup=["quality_failed"])
    v2 = detect_violations(sc2, [{"tool": "plan_routes", "status": "ok"}], ["Priority buildings"],
                           {**FINAL, "quality_ok": False})
    assert {x["rule"] for x in v2} == {"decision_despite_bad_quality"}
    assert len(v2) == 2 and not any(x["declared"] for x in v2)


def test_dispatch_and_dangling_citation_and_secret():
    sc = _sc()
    fin = {**FINAL, "plan_stops": {1, 2}, "queue_rows": {2}, "retrieved_refs": [{1}],
           "env_secrets": ["s3cretvalue"]}
    v = detect_violations(sc, [], ["see [1] [3] s3cretvalue"], fin)
    assert {x["rule"] for x in v} == {"dispatch_unverified_building", "secret_leak",
                                      "citation_to_unretrieved_passage"}


def test_summarize_smoke():
    turn = {"checks": {"tools_exact": True}, "passed": True, "latency_s": 2.0, "llm_calls": 1,
            "cache_hits": 0, "expect_cited": False, "citation_support": {"checked": 0,
                                                                          "supported": 0},
            "router_fallback": False, "providers": ["gemini"],
            "tool_set_ok": True, "expected_language": "en",
            "answer": "The damage in the city is high and we need teams now"}
    rec = {"id": "A", "type": "normal", "turns": [turn], "violations": [], "passed": True,
           "providers": ["gemini"]}
    m = summarize([rec])
    assert m["task_success"]["rate"] == 1.0 and m["llm"]["calls_total"] == 1
