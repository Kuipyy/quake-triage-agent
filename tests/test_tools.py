import json

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, Point

from quaketriage.agent.tools import (
    PIPELINE,
    TOOLS,
    Constraints,
    ToolContext,
    ToolError,
    tool_schemas,
)
from quaketriage.data.geo import get_project_crs
from quaketriage.rag.index import RagIndex

CRS = get_project_crs()
X0, Y0 = 250000.0, 4000000.0


def _lonlat(x, y):
    p = gpd.GeoSeries([Point(X0 + x, Y0 + y)], crs=CRS).to_crs(4326).iloc[0]
    return (float(p.x), float(p.y))


def _graph():
    g = nx.MultiDiGraph(crs=CRS)
    for i in range(6):
        g.add_node(i, x=X0 + 400 * i, y=Y0)
    for i in range(5):
        for a, b in ((i, i + 1), (i + 1, i)):
            geom = LineString([(X0 + 400 * a, Y0), (X0 + 400 * b, Y0)])
            g.add_edge(a, b, geometry=geom, length=geom.length, travel_time=geom.length / 10,
                       travel_time_adj=geom.length / 10)
    return g


def _buildings():
    xs = [400, 800, 1200, 1600, 2000]
    return gpd.GeoDataFrame({
        "id": [11, 12, 13, 14, 15], "damage_prob": [0.9, 0.8, 0.2, 0.7, 0.05],
        "uncertainty": [0.05, 0.3, 0.05, 0.05, 0.05], "floors": [5, 4, 2, 6, 1],
        "estimated_persons": [20.0, 10.0, 5.0, 30.0, 2.0],
        "residents_day": [40.0, 20.0, 10.0, 60.0, 4.0]},
        geometry=[Point(X0 + x, Y0 + 5).buffer(8) for x in xs], crs=CRS)


@pytest.fixture
def ctx():
    c = ToolContext(buildings=_buildings(), base_graph=_graph())
    c.constraints.depot_lonlat = _lonlat(0, 0)
    c.constraints.solver_time_limit_s = 1
    c.constraints.num_teams = 1
    c.constraints.candidate_n = 5
    return c


def _pipeline(ctx):
    for name in ("detect_damage", "score_buildings", "estimate_occupancy", "prioritize_sites",
                 "plan_routes"):
        ctx.call(name, {})


# ------------------------------------------------------------------ registry
def test_nine_tools_with_json_schemas():
    assert len(TOOLS) == 9
    schemas = tool_schemas()
    assert {s["name"] for s in schemas} == set(TOOLS)
    json.dumps(schemas)  # serialisable for LLM function calling
    assert all(s["description"] and s["parameters"]["type"] == "object" for s in schemas)


def test_call_rejects_unknown_tool_and_bad_args(ctx):
    with pytest.raises(ToolError, match="unknown tool"):
        ctx.call("launch_rockets")
    with pytest.raises(ToolError, match="invalid arguments"):
        ctx.call("prioritize_sites", {"top_k": 0})
    assert ctx.trace[-1]["error"] and "top_k" in ctx.trace[-1]["error"]


# ------------------------------------------------------------------ 2-4
def test_detect_damage_simulated_is_flagged(ctx):
    out = ctx.call("detect_damage", {})
    assert out["SIMULATED_DETECTOR"] is True and out["n_buildings"] == 5
    assert out["n_above_threshold"] == 3
    ctx.buildings = ctx.buildings.drop(columns="uncertainty")
    with pytest.raises(ToolError, match="uncertainty"):
        ctx.call("detect_damage", {})


def test_detect_damage_siamese_needs_paths_and_uses_runner(ctx, tmp_path):
    with pytest.raises(ToolError, match="pre_path"):
        ctx.call("detect_damage", {"source": "siamese"})
    ctx.siamese_runner = lambda pre, post: {"mean_prob": 0.123456, "n_tiles": 4}
    out = ctx.call("detect_damage", {"source": "siamese", "pre_path": "a.tif",
                                     "post_path": "b.tif"})
    assert out["SIMULATED_DETECTOR"] is False and out["mean_prob"] == 0.1235
    assert out["detail"]["n_tiles"] == 4


def test_score_buildings_flags_damage_and_queue(ctx):
    ctx.call("detect_damage", {})
    out = ctx.call("score_buildings", {"threshold": 0.5, "uncertainty_tau": 0.2})
    assert out["n_damaged"] == 3 and out["n_needs_verification"] == 1
    assert out["SIMULATED_DETECTOR"] is True and sum(out["verification_reasons"].values()) == 1
    with pytest.raises(ToolError, match="no building table"):
        ToolContext().call("score_buildings", {})


def test_occupancy_scenarios_and_idempotence(ctx):
    ctx.call("detect_damage", {})
    ctx.call("score_buildings", {})
    night = ctx.call("estimate_occupancy", {"scenario": "night"})
    full = ctx.call("estimate_occupancy", {"scenario": "full_occupancy"})
    again = ctx.call("estimate_occupancy", {"scenario": "night"})
    assert night["persons_total"] == 67 and full["persons_total"] == 134
    assert again["persons_total"] == night["persons_total"]  # night base was preserved


# ------------------------------------------------------------------ 5-6
def test_prioritize_sites_orders_and_reports_weights(ctx):
    ctx.call("detect_damage", {})
    out = ctx.call("prioritize_sites", {"top_k": 3})
    # building 12 is queued for verification; 13 (0.2) and 15 (0.05) are below the damage gate
    assert {r["id"] for r in out["top"]} == {11, 14}
    assert abs(sum(out["weights"].values()) - 1) < 1e-3
    pr = [r["priority"] for r in out["top"]]
    assert pr == sorted(pr, reverse=True)
    only_people = {"damage_prob": 0, "estimated_persons": 1, "floor_count": 0,
                   "distance_to_depot": 0}
    top = ctx.call("prioritize_sites", {"top_k": 1, "weights": only_people})["top"][0]
    assert top["estimated_persons"] == 30.0  # building 14
    with pytest.raises(ToolError, match="unknown criteria"):
        ctx.call("prioritize_sites", {"weights": {"bogus": 1}})


def test_plan_routes_needs_priorities_then_plans(ctx):
    ctx.call("detect_damage", {})
    with pytest.raises(ToolError, match="prioritize_sites"):
        ctx.call("plan_routes", {})
    ctx.call("prioritize_sites", {})
    out = ctx.call("plan_routes", {"num_teams": 2})
    assert ctx.constraints.num_teams == 2  # override is remembered
    assert out["n_visited"] >= 1 and len(out["teams"]) == 2
    assert out["SIMULATED_DETECTOR"] is True and out["n_closed_edges"] >= 0
    assert all(s["arrival_h"] <= 6 for t in out["teams"] for s in t["stops"])


def test_closed_area_cuts_road_and_plan_adapts(ctx):
    ctx.call("update_constraints", {"walking": False})
    ctx.call("detect_damage", {})
    ctx.call("prioritize_sites", {})
    base = ctx.call("plan_routes", {})
    cut = _lonlat(600, 0)  # between node 1 and node 2: everything beyond is cut off
    ctx.call("update_constraints", {"closed_areas": [(cut[0], cut[1], 50.0)]})
    out = ctx.call("plan_routes", {})
    # building 12 (uncertainty 0.3) is held back for verification, never dispatched
    assert base["n_excluded_verification"] == 1
    # dispatchable = damaged above the gate and not queued: buildings 11 and 14
    assert base["n_unreachable"] == 0 and base["n_visited"] == 2
    assert out["n_closed_edges"] == base["n_closed_edges"] + 2  # both directions of one edge
    assert out["n_unreachable"] == 1  # building 14 lies beyond the cut
    assert out["n_visited"] == 1  # only building 11, before the cut, is still served


# ------------------------------------------------------------------ 7
def test_lookup_protocol_returns_cited_hits(ctx):
    chunks = [{"doc": "INSARAG_Vol2", "section": "6.1", "page_start": 10, "page_end": 11,
               "text": "USAR teams must register at the reception and departure centre"},
              {"doc": "AFAD_TAMP", "section": "", "page_start": 3, "page_end": 3,
               "text": "arama kurtarma ekipleri koordinasyon"}]
    emb = np.eye(2, dtype="float32")
    ctx.rag_index = RagIndex(chunks, emb)
    ctx.embed_query = lambda texts: np.array([[1.0, 0.0] for _ in texts], dtype="float32")
    out = ctx.call("lookup_protocol", {"query": "USAR registration", "top_k": 2})
    assert out["hits"][0]["doc"] == "INSARAG_Vol2" and out["hits"][0]["pages"] == "10-11"
    assert [h["ref"] for h in out["hits"]] == [1, 2] and ctx.protocol_hits
    with pytest.raises(ToolError):
        ctx.call("lookup_protocol", {"query": "x"})  # shorter than 3 characters


# ------------------------------------------------------------------ 8
RELEVANT = ("5.8 Worksite Triage The objective of ASR 2 Worksite Assessment is to assess collapsed "
            "structures and identify viable sites for live rescues. The UCC will use the results "
            "to prioritise search sites. Teams then report back to the OSOCC.")


def _hit(ref, text, doc="INSARAG_Vol2_ManB", pages="36-36"):
    return {"ref": ref, "doc": doc, "pages": pages, "section": "", "score": 0.9, "text": text}


def test_sitrep_is_grounded_cited_and_labelled_simulated(ctx):
    _pipeline(ctx)
    ctx.protocol_hits = [_hit(1, "USAR teams must register at the RDC"), _hit(2, RELEVANT)]
    out = ctx.call("write_sitrep", {"language": "en", "top_k": 3})
    md = out["markdown"]
    assert "SIMULATED detector" in md
    # the irrelevant passage is dropped and the relevant one renumbered to [1]
    assert "register at the RDC" not in md and "- [1] 5.8 Worksite Triage" in md
    assert "INSARAG_Vol2_ManB, p. 36-36" in md
    assert f"{int(ctx.buildings['damaged'].sum())} damaged" in md
    assert out["n_citations"] == 1 and out["citations"][0]["ref"] == 1
    assert out["missing_steps"] == []
    tr = ctx.call("write_sitrep", {"language": "tr"})["markdown"]
    assert "SİMÜLE DEDEKTÖR" in tr and "Durum Raporu" in tr
    assert "İngilizcedir" in tr  # English source passages in a Turkish report are announced


def test_sitrep_omits_protocol_section_without_relevant_passages(ctx):
    _pipeline(ctx)
    ctx.protocol_hits = [_hit(1, "USAR teams must register at the RDC")]
    out = ctx.call("write_sitrep", {"language": "en"})
    assert "Protocol basis" not in out["markdown"] and out["n_citations"] == 0


def test_turkish_source_needs_no_translation_note(ctx):
    _pipeline(ctx)
    ctx.protocol_hits = [_hit(1, "Arama kurtarma triage ve assess çalışmalarında yapısal "
                              "hasar tespiti yapılır. Çöken structures için öncelik verilir.",
                              doc="AFAD_TAMP", pages="3-3")]
    tr = ctx.call("write_sitrep", {"language": "tr"})["markdown"]
    assert "Protokol dayanakları" in tr and "İngilizcedir" not in tr


def test_clip_sentences_cuts_at_sentence_boundaries():
    from quaketriage.agent.tools import clip_sentences
    text = ("(medevac), demobilisation etc.) USAR Team Search. Perform a physical search of "
            "collapsed structures. Assess voids and local failures. Mark the structure after "
            "the search is complete and report to the UCC for further tasking.")
    out = clip_sentences(text, max_chars=110)
    assert out.startswith("Perform a physical search")  # the leading half sentence is dropped
    assert "medevac" not in out and out.endswith(".") and "Mark the structure" not in out
    long = "word " * 100
    assert clip_sentences(long, 50).endswith("…") and len(clip_sentences(long, 50)) <= 60


def test_user_facing_warnings_follow_the_users_language(ctx):
    ctx.call("detect_damage", {})
    ctx.lang = "tr"
    q = ctx.call("prioritize_sites", {})
    assert "sevkten önce saha doğrulaması" in q["warnings"][0] and "~" in q["warnings"][0]
    assert "belirsizlik" in q["warnings"][0]
    ctx.quality_ok, ctx.quality_reasons = False, ["bright/cloud ratio 0.90 > 0.2"]
    blocked = ctx.call("prioritize_sites", {})
    assert blocked["warnings"][0] == ("görüntü kalite kontrolü başarısız: parlak/bulut oranı "
                                      "0.90 > 0.2; sıralama üretilmedi")
    ctx.lang = "en"
    again = ctx.call("plan_routes", {})
    assert again["warnings"][0] == ("image quality check failed: bright/cloud ratio 0.90 > 0.2; "
                                    "no route plan produced")
    tr_report = ctx.call("write_sitrep", {"language": "tr"})  # sitrep follows its own language
    assert "parlak/bulut" in tr_report["warnings"][0]


def test_queue_persons_in_sitrep_and_tool_outputs(ctx):
    _pipeline(ctx)
    # building 12 (10 people at night) is the only uncertain one
    assert ctx.call("score_buildings", {})["persons_in_verification_queue"] == 10
    assert ctx.call("prioritize_sites", {})["persons_in_verification_queue"] == 10
    assert "1 buildings, ~10 people awaiting verification" in ctx.call(
        "write_sitrep", {"language": "en"})["markdown"]
    assert "1 bina, ~10 kişi doğrulama bekliyor" in ctx.call(
        "write_sitrep", {"language": "tr"})["markdown"]


def test_check_support_catches_invented_terms_and_wrong_passages():
    from quaketriage.agent.tools import check_support
    hits = [{"ref": 1, "text": "The operations phase starts with arrival at the Reception/"
             "Departure Centre (RDC), registration with the UCC/OSOCC and reporting to LEMA."}]
    good = check_support("Teams register with the UCC/OSOCC [1].", hits)
    assert good[0]["supported"] is True
    bad = check_support("Teams register with the ICAO tower [1].", hits)
    assert bad[0]["supported"] is False and "ICAO" in bad[0]["missing"]
    assert check_support("Teams must pay a fee of 500 dollars [1].", hits)[0]["supported"] is False
    assert check_support("No citation here.", hits) == []
    assert check_support("Something [7].", hits)[0]["missing"] == ["<no such passage>"]


def test_sitrep_names_missing_steps(ctx):
    ctx.call("detect_damage", {})
    out = ctx.call("write_sitrep", {})
    assert "plan_routes" in out["missing_steps"] and out["status"] == "ok"


# ------------------------------------------------------------------ 9
def test_update_constraints_reports_only_stale_steps(ctx):
    out = ctx.call("update_constraints", {"num_teams": 7})
    assert out["changed"] == {"num_teams": 7}
    assert out["stale_steps"] == ["plan_routes", "write_sitrep"]
    assert "prioritize_sites" in out["unchanged_steps"]
    w = ctx.call("update_constraints", {"weights": {"damage_prob": 1.0}})
    assert w["stale_steps"] == ["prioritize_sites", "plan_routes", "write_sitrep"]
    t = ctx.call("update_constraints", {"damage_threshold": 0.1})
    assert t["stale_steps"] == ["score_buildings", "write_sitrep"]
    same = ctx.call("update_constraints", {"num_teams": 7})
    assert same["changed"] == {} and same["stale_steps"] == []


def test_update_constraints_validates(ctx):
    with pytest.raises(ToolError):
        ctx.call("update_constraints", {"num_teams": 0})
    with pytest.raises(ToolError, match="invalid arguments"):
        ctx.call("update_constraints", {"num_teamz": 3})  # unknown field is rejected
    assert ctx.constraints.num_teams == 1  # failed update left state untouched


def test_every_constraint_has_a_stale_rule():
    from quaketriage.agent.tools import STALE_BY_CONSTRAINT
    knobs = set(Constraints.model_fields) - {"depot_lonlat"}
    assert knobs <= set(STALE_BY_CONSTRAINT)
    assert all(set(v) <= set(PIPELINE) for v in STALE_BY_CONSTRAINT.values())


# ------------------------------------------------------------------ 1
def test_check_image_quality_good_blank_and_missing(ctx, tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    def write(name, arr):
        path = tmp_path / name
        with rasterio.open(path, "w", driver="GTiff", height=128, width=128, count=3,
                           dtype="uint8", crs="EPSG:4326",
                           transform=from_origin(36.0, 36.0, 1e-5, 1e-5)) as dst:
            dst.write(np.moveaxis(arr, -1, 0))
        return path

    rng = np.random.default_rng(0)
    img = rng.integers(40, 200, (128, 128, 3), dtype=np.uint8)
    good = ctx.call("check_image_quality", {"pre_path": write("a.tif", img),
                                            "post_path": write("b.tif", img), "crop_px": 64})
    assert good["ok"] is True and ctx.quality_ok is True
    blank = ctx.call("check_image_quality", {"pre_path": write("c.tif", img),
                                             "post_path": write("d.tif", img * 0), "crop_px": 64})
    assert blank["ok"] is False and blank["reasons"] and ctx.quality_ok is False
    with pytest.raises(ToolError, match="not found"):
        ctx.call("check_image_quality", {"pre_path": tmp_path / "nope.tif",
                                         "post_path": tmp_path / "nope2.tif"})


# ------------------------------------------------------------ guard enforced in tool code
def _bad_quality(ctx):
    ctx.quality_ok = False
    ctx.quality_reasons = ["blank/no-data ratio 0.90 > 0.5"]


def test_bad_image_quality_makes_decision_tools_return_verification_not_decisions(ctx):
    ctx.call("detect_damage", {})
    _bad_quality(ctx)
    rank = ctx.call("prioritize_sites", {})
    assert rank["status"] == "verification_required" and rank["top"] == []
    assert "image quality check failed" in rank["warnings"][0]
    assert "priority" not in ctx.buildings  # nothing was ranked behind the scenes either
    plan = ctx.call("plan_routes", {})
    assert plan["status"] == "verification_required" and plan["teams"] == [] and ctx.plan is None
    rep = ctx.call("write_sitrep", {"language": "en"})
    assert rep["status"] == "verification_required" and "no damage decision" in rep["markdown"]
    assert "Priority buildings" not in rep["markdown"]
    assert not {"prioritize_sites", "plan_routes", "write_sitrep"} & ctx.done  # not counted as run


def test_high_uncertainty_buildings_are_held_back_not_ranked_or_dispatched(ctx):
    ctx.call("detect_damage", {})  # no score_buildings call: the tools must apply the rule anyway
    rank = ctx.call("prioritize_sites", {"top_k": 10})
    assert rank["n_verification_queue"] == 1 and rank["warnings"]
    assert all(not r["needs_verification"] for r in rank["top"])
    assert [r["id"] for r in rank["verification_queue"]] == [12]  # uncertainty 0.30 > tau
    plan = ctx.call("plan_routes", {})
    assert plan["n_excluded_verification"] == 1
    dispatched = {s["row"] for t in plan["teams"] for s in t["stops"]}
    assert 1 not in dispatched  # row 1 is building 12
    md = ctx.call("write_sitrep", {"language": "en"})["markdown"]
    assert "excluded from the priority list" in md


def test_every_tool_output_carries_simulated_flag_and_uncertainty(ctx, tmp_path):
    calls = [("detect_damage", {}), ("score_buildings", {}), ("estimate_occupancy", {}),
             ("prioritize_sites", {}), ("plan_routes", {}),
             ("update_constraints", {"num_teams": 1}), ("write_sitrep", {})]
    for name, args in calls:
        out = ctx.call(name, args)
        assert out["SIMULATED_DETECTOR"] is True, name
        assert out["uncertainty"]["scope"] == "buildings" and out["uncertainty"]["tau"], name
        assert out["uncertainty"]["n_above_tau"] == 1, name
    ctx.rag_index = RagIndex([{"doc": "d", "section": "", "page_start": 1, "page_end": 1,
                               "text": "abc def"}], np.eye(1, dtype="float32"))
    ctx.embed_query = lambda t: np.ones((len(t), 1), dtype="float32")
    out = ctx.call("lookup_protocol", {"query": "abc def"})
    assert out["SIMULATED_DETECTOR"] is True and out["uncertainty"] is not None
    # before any damage data exists the keys are still present (explicitly empty)
    fresh = ToolContext().call("update_constraints", {"num_teams": 2})
    assert fresh["SIMULATED_DETECTOR"] is False and fresh["uncertainty"] is None


def test_detect_damage_states_which_detector(ctx):
    sim = ctx.call("detect_damage", {})
    assert "SIMULATED" in sim["detector_note"] and "not model output" in sim["detector_note"]
    ctx.siamese_runner = lambda a, b: {"mean_prob": 0.2, "mean_uncertainty": 0.07,
                                       "p90_uncertainty": 0.1, "max_uncertainty": 0.2}
    ctx.buildings = None
    real = ctx.call("detect_damage", {"source": "siamese", "pre_path": "a", "post_path": "b"})
    assert real["SIMULATED_DETECTOR"] is False and "Siamese" in real["detector_note"]
    assert real["uncertainty"]["mean"] == 0.07 and "TTA" in real["uncertainty"]["scope"]


def test_what_if_marks_steps_stale_and_not_done(ctx):
    _pipeline(ctx)
    ctx.call("write_sitrep", {})
    assert {"plan_routes", "write_sitrep"} <= ctx.done
    ctx.call("update_constraints", {"num_teams": 3})
    assert "plan_routes" not in ctx.done and "write_sitrep" not in ctx.done
    assert "prioritize_sites" in ctx.done


def test_sitrep_states_queue_exclusion_so_week5_numbers_are_not_confused(ctx):
    _pipeline(ctx)
    en = ctx.call("write_sitrep", {"language": "en"})["markdown"]
    assert ("1 buildings in the verification queue were excluded from the plan; "
            "Week-5 metrics were computed including the queue.") in en
    tr = ctx.call("write_sitrep", {"language": "tr"})["markdown"]
    assert ("Doğrulama kuyruğundaki 1 bina plan dışı bırakıldı; Hafta 5 metrikleri kuyruk dahil "
            "hesaplandı.") in tr


def test_validate_citations_flags_dangling_and_unindexed_passages():
    from quaketriage.agent.tools import validate_citations
    chunks = [{"doc": "INSARAG_Vol2", "page_start": 10, "page_end": 11,
               "text": "USAR teams register at the reception and departure centre"}]
    hit = {"ref": 1, "doc": "INSARAG_Vol2", "pages": "10-11",
           "text": "USAR teams register at the reception and departure centre"}
    ok = validate_citations("see [1]", [hit], chunks)
    assert ok["valid"] and ok["cited_refs"] == [1]
    assert validate_citations("see [2]", [hit], chunks)["dangling_refs"] == [2]
    fake = dict(hit, text="invented text that is not in the index at all, really")
    assert validate_citations("see [1]", [fake], chunks)["refs_not_in_index"] == [1]
    assert not validate_citations("no hits", [], chunks)["valid"]


def test_uncertainty_threshold_can_be_tightened_but_not_relaxed(ctx):
    from quaketriage.config import get
    limit = get("thresholds.uncertainty_high")
    out = ctx.call("update_constraints", {"uncertainty_tau": limit / 2})
    assert out["changed"] == {"uncertainty_tau": limit / 2}
    with pytest.raises(ToolError, match="may not exceed"):
        ctx.call("update_constraints", {"uncertainty_tau": 100.0})
    assert ctx.constraints.uncertainty_tau == limit / 2  # unchanged by the rejected request


def test_damage_threshold_can_be_lowered_but_not_raised(ctx):
    from quaketriage.config import get
    limit = get("thresholds.damage_prob")
    out = ctx.call("update_constraints", {"damage_threshold": limit / 2})
    assert out["changed"] == {"damage_threshold": limit / 2}
    with pytest.raises(ToolError, match="may not exceed"):
        ctx.call("update_constraints", {"damage_threshold": 0.9})
    assert ctx.constraints.damage_threshold == limit / 2  # unchanged by the rejected request
    same = ctx.call("update_constraints", {"damage_threshold": limit})  # back to the default
    assert same["changed"] == {"damage_threshold": limit}


def test_plan_cache_replays_identical_plan_and_misses_on_change(ctx, tmp_path):
    ctx.plan_cache_dir = tmp_path
    ctx.call("prioritize_sites", {})
    first = ctx.call("plan_routes", {})
    assert first["from_cache"] is False and list(tmp_path.glob("*.json"))
    second = ctx.call("plan_routes", {})
    assert second["from_cache"] is True
    assert second["priority_collected"] == first["priority_collected"]
    assert second["n_closed_edges"] == first["n_closed_edges"]
    changed = ctx.call("plan_routes", {"num_teams": 2})  # other constraints -> other key
    assert changed["from_cache"] is False


def test_plan_cache_disabled_by_default(ctx):
    ctx.call("prioritize_sites", {})
    assert ctx.call("plan_routes", {})["from_cache"] is False
    assert ctx.call("plan_routes", {})["from_cache"] is False
