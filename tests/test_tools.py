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
    assert len(out["top"]) == 3 and abs(sum(out["weights"].values()) - 1) < 1e-3
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
    assert base["n_unreachable"] == 0 and base["n_visited"] == 3
    assert out["n_closed_edges"] == base["n_closed_edges"] + 2  # both directions of one edge
    assert out["n_unreachable"] == 3  # the three dispatchable buildings beyond the cut
    assert out["n_visited"] == 1  # only the building before the cut is still served


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
def test_sitrep_is_grounded_cited_and_labelled_simulated(ctx):
    _pipeline(ctx)
    ctx.protocol_hits = [{"ref": 1, "doc": "INSARAG_Vol2", "pages": "10-11",
                          "text": "USAR teams must register", "section": "", "score": 0.9}]
    out = ctx.call("write_sitrep", {"language": "en", "top_k": 3})
    md = out["markdown"]
    assert "SIMULATED detector" in md and "[1]" in md and "INSARAG_Vol2, p. 10-11" in md
    assert f"{int(ctx.buildings['damaged'].sum())} damaged" in md
    assert out["n_citations"] == 1 and out["missing_steps"] == []
    tr = ctx.call("write_sitrep", {"language": "tr"})["markdown"]
    assert "SİMÜLE DEDEKTÖR" in tr and "Durum Raporu" in tr


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
    t = ctx.call("update_constraints", {"damage_threshold": 0.6})
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
