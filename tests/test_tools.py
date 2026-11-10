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
    assert out["simulated"] is True and out["n_buildings"] == 5 and out["n_above_threshold"] == 3
    ctx.buildings = ctx.buildings.drop(columns="uncertainty")
    with pytest.raises(ToolError, match="uncertainty"):
        ctx.call("detect_damage", {})


def test_detect_damage_siamese_needs_paths_and_uses_runner(ctx, tmp_path):
    with pytest.raises(ToolError, match="pre_path"):
        ctx.call("detect_damage", {"source": "siamese"})
    ctx.siamese_runner = lambda pre, post: {"mean_prob": 0.123456, "n_tiles": 4}
    out = ctx.call("detect_damage", {"source": "siamese", "pre_path": "a.tif",
                                     "post_path": "b.tif"})
    assert out["simulated"] is False and out["mean_prob"] == 0.1235
    assert out["detail"]["n_tiles"] == 4


def test_score_buildings_flags_damage_and_queue(ctx):
    ctx.call("detect_damage", {})
    out = ctx.call("score_buildings", {"threshold": 0.5, "uncertainty_tau": 0.2})
    assert out["n_damaged"] == 3 and out["n_needs_verification"] == 1
    assert out["simulated"] is True and sum(out["verification_reasons"].values()) == 1
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
    assert out["simulated"] is True and out["n_closed_edges"] >= 0
    assert all(s["arrival_h"] <= 6 for t in out["teams"] for s in t["stops"])


def test_closed_area_cuts_road_and_plan_adapts(ctx):
    ctx.call("update_constraints", {"walking": False})
    ctx.call("detect_damage", {})
    ctx.call("prioritize_sites", {})
    base = ctx.call("plan_routes", {})
    cut = _lonlat(600, 0)  # between node 1 and node 2: everything beyond is cut off
    ctx.call("update_constraints", {"closed_areas": [(cut[0], cut[1], 50.0)]})
    out = ctx.call("plan_routes", {})
    assert base["n_unreachable"] == 0 and base["n_visited"] == 4
    assert out["n_closed_edges"] == base["n_closed_edges"] + 2  # both directions of one edge
    assert out["n_unreachable"] == 4  # the four buildings beyond the cut
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


def test_sitrep_names_missing_steps_and_quality_warning(ctx):
    ctx.call("detect_damage", {})
    ctx.quality_ok = False
    out = ctx.call("write_sitrep", {})
    assert "plan_routes" in out["missing_steps"] and "Görüntü kalite" in out["markdown"]


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
