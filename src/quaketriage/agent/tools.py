"""The nine agent tools: plain functions ``fn(inp: PydanticModel, ctx: ToolContext) -> BaseModel``.

Heavy data (building table, road graph, RAG index) lives in the ``ToolContext``, never in the
LLM prompt: tools return short summaries. Everything that depends on the damage detector is
flagged ``SIMULATED_DETECTOR`` when the Antakya demo detector (HOTOSM labels + noise) is the source.
Constraint changes go through ``update_constraints``, which reports which steps became stale so
the graph re-runs only those.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from quaketriage.buildings.queue import flag_verification
from quaketriage.config import get
from quaketriage.planning.debris import DebrisInput, apply_debris_risk, close_edges_near
from quaketriage.planning.topsis import TopsisInput, prioritise
from quaketriage.planning.vrp import VrpInput, plan_routes
from quaketriage.rag.index import SearchInput

ROOT = Path(__file__).parents[3]
PIPELINE = ("detect_damage", "score_buildings", "estimate_occupancy", "prioritize_sites",
            "plan_routes", "write_sitrep")


_M = {
    "en": {
        "quality_failed": "image quality check failed: {reasons}",
        "no_reason": "see guard",
        "no_ranking": "no ranking produced",
        "no_plan": "no route plan produced",
        "guard": "image quality guard failed: no damage decision may be taken from this pair; "
                 "request field verification",
        "queue": "{n} buildings (~{p:.0f} people) exceed the uncertainty threshold or lack valid "
                 "imagery: field verification required before dispatch",
        "held": "{n} buildings held back for field verification",
    },
    "tr": {
        "quality_failed": "görüntü kalite kontrolü başarısız: {reasons}",
        "no_reason": "bkz. guard",
        "no_ranking": "sıralama üretilmedi",
        "no_plan": "rota planı üretilmedi",
        "guard": "görüntü kalite koruması başarısız: bu görüntü çiftinden hasar kararı "
                 "verilemez; saha doğrulaması isteyin",
        "queue": "{n} bina (~{p:.0f} kişi) belirsizlik eşiğini aşıyor veya geçerli görüntüsü "
                 "yok: sevkten önce saha doğrulaması gerekli",
        "held": "{n} bina saha doğrulaması için ayrıldı",
    },
}
_REASON_TR = (("blank/no-data ratio", "boş/veri-yok oranı"),
              ("bright/cloud ratio", "parlak/bulut oranı"),
              ("alignment SSIM", "hizalama SSIM"))


def msg(lang: str, key: str, **kw) -> str:
    return _M[lang][key].format(**kw)


def localize_reason(reason: str, lang: str) -> str:
    """Guard reasons come from ``data/quality.py`` in English; translate the known phrases."""
    if lang != "tr":
        return reason
    for en, tr in _REASON_TR:
        reason = reason.replace(en, tr)
    return reason


class UncertaintySummary(BaseModel):
    """Uncertainty of the damage estimates a result is based on (None = nothing to summarise)."""

    scope: str
    mean: float
    p90: float
    max: float
    n_above_tau: int | None = None
    tau: float | None = None


class ToolOutput(BaseModel):
    """Fields every tool result carries; ``ToolContext.call`` fills them in, tools cannot omit them.

    ``status='verification_required'`` means the tool refused to produce a decision (bad imagery);
    ``SIMULATED_DETECTOR`` is True whenever the damage source was the Antakya demo simulator.
    """

    SIMULATED_DETECTOR: bool = False
    status: Literal["ok", "verification_required"] = "ok"
    warnings: list[str] = []
    uncertainty: UncertaintySummary | None = None


class ToolError(ValueError):
    """A tool refused its input or lacks a prerequisite; the message is safe to show the LLM."""


# ---------------------------------------------------------------- constraints and context
class Constraints(BaseModel):
    """What-if knobs. Defaults come from ``configs/default.yaml``."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    num_teams: int = Field(default_factory=lambda: get("routing.num_teams"), ge=1, le=50)
    mission_hours: float = Field(default_factory=lambda: get("routing.mission_duration_hours"),
                                 gt=0, le=72)
    intervention_minutes: float = Field(
        default_factory=lambda: get("routing.intervention_time_minutes"), ge=0, le=600)
    candidate_n: int = Field(default_factory=lambda: get("routing.candidate_n"), ge=1, le=2000)
    walking: bool = True
    solver_time_limit_s: int = Field(default_factory=lambda: get("routing.solver_time_limit_s"),
                                     ge=1, le=600)
    closed_areas: list[tuple[float, float, float]] = Field(
        default_factory=list, description="(lon, lat, radius_m) circles where roads are closed")
    weights: dict[str, float] | None = None
    depot_lonlat: tuple[float, float] = Field(
        default_factory=lambda: tuple(get("routing.depots")[0]))
    damage_threshold: float = Field(default_factory=lambda: get("thresholds.damage_prob"),
                                    ge=0, le=1)
    uncertainty_tau: float = Field(default_factory=lambda: get("thresholds.uncertainty_high"),
                                   ge=0)
    scenario: Literal["night", "full_occupancy"] = "night"


# which pipeline steps a changed constraint makes stale (lookup_protocol never depends on them)
STALE_BY_CONSTRAINT: dict[str, tuple[str, ...]] = {
    "damage_threshold": ("score_buildings", "write_sitrep"),
    "uncertainty_tau": ("score_buildings", "write_sitrep"),
    "scenario": ("estimate_occupancy", "prioritize_sites", "plan_routes", "write_sitrep"),
    "weights": ("prioritize_sites", "plan_routes", "write_sitrep"),
    "depot_lonlat": ("prioritize_sites", "plan_routes", "write_sitrep"),
    "num_teams": ("plan_routes", "write_sitrep"),
    "mission_hours": ("plan_routes", "write_sitrep"),
    "intervention_minutes": ("plan_routes", "write_sitrep"),
    "candidate_n": ("plan_routes", "write_sitrep"),
    "walking": ("plan_routes", "write_sitrep"),
    "solver_time_limit_s": ("plan_routes", "write_sitrep"),
    "closed_areas": ("plan_routes", "write_sitrep"),
}


@dataclass
class ToolContext:
    """Mutable pipeline state shared by the tools (injectable pieces make tests offline)."""

    constraints: Constraints = field(default_factory=Constraints)
    buildings: gpd.GeoDataFrame | None = None
    simulated: bool = False
    quality_ok: bool | None = None
    plan: dict | None = None
    protocol_hits: list[dict] = field(default_factory=list)
    done: set[str] = field(default_factory=set)
    trace: list[dict] = field(default_factory=list)
    base_graph: Any = None  # projected road graph (lazy)
    rag_index: Any = None
    embed_query: Callable | None = None
    siamese_runner: Callable[[Path, Path], dict] | None = None
    image_uncertainty: UncertaintySummary | None = None  # from the Siamese/TTA path
    quality_reasons: list[str] = field(default_factory=list)
    lang: str = "en"  # language of user-facing warnings (set per turn by the graph)

    def quality_block(self, lang: str | None = None) -> str | None:
        """Why decisions are blocked (failed image-quality guard), enforced inside the tools."""
        if self.quality_ok is False:
            lang = lang or self.lang
            reasons = "; ".join(localize_reason(r, lang) for r in self.quality_reasons) \
                or msg(lang, "no_reason")
            return msg(lang, "quality_failed", reasons=reasons)
        return None

    def ensure_scored(self) -> gpd.GeoDataFrame:
        """Building table with the verification-queue flags (computed here if score_buildings
        has not run, so a decision tool can never skip the uncertainty rule)."""
        b = self.need_buildings()
        if "needs_verification" not in b:
            b = flag_verification(b, self.constraints.uncertainty_tau)
            b["damaged"] = b["damage_prob"] > self.constraints.damage_threshold
            self.buildings = b
        return b

    def uncertainty_summary(self) -> UncertaintySummary | None:
        b = self.buildings
        if b is not None and "uncertainty" in b:
            u = b["uncertainty"].to_numpy(float)
            u = u[np.isfinite(u)]
            if u.size:
                tau = self.constraints.uncertainty_tau
                return UncertaintySummary(
                    scope="buildings", mean=round(float(u.mean()), 4),
                    p90=round(float(np.quantile(u, 0.9)), 4), max=round(float(u.max()), 4),
                    n_above_tau=int((u > tau).sum()), tau=tau)
        return self.image_uncertainty

    # -- lazy resources
    def graph(self):
        if self.base_graph is None:
            from quaketriage.planning.roads import load_road_graph
            self.base_graph = load_road_graph()
        return self.base_graph

    def rag(self):
        if self.rag_index is None:
            from quaketriage.rag.index import RagIndex, config_embedders
            path = ROOT / get("paths.data_processed") / "rag"
            if not path.exists():
                raise ToolError("RAG index missing; run scripts/build_rag_index.py")
            self.rag_index = RagIndex.load(path)
            self.embed_query = config_embedders()[1]
        return self.rag_index, self.embed_query

    def need_buildings(self) -> gpd.GeoDataFrame:
        if self.buildings is None:
            raise ToolError("no building table loaded; run detect_damage first")
        return self.buildings

    def call(self, name: str, args: dict | None = None) -> dict:
        """Validate ``args`` against the tool's schema, run it, record the trace entry."""
        if name not in TOOLS:
            raise ToolError(f"unknown tool {name!r}; available: {sorted(TOOLS)}")
        spec = TOOLS[name]
        t0 = time.time()
        try:
            inp = spec.input_model(**(args or {}))
        except ValidationError as exc:
            msg = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            self.trace.append({"tool": name, "args": args, "error": msg})
            raise ToolError(f"invalid arguments for {name}: {msg}") from exc
        try:
            res = spec.fn(inp, self)
        except ToolError as exc:
            self.trace.append({"tool": name, "args": args, "error": str(exc)})
            raise
        unc = res.uncertainty or self.uncertainty_summary()
        out = res.model_dump()
        out["SIMULATED_DETECTOR"] = self.simulated
        out["uncertainty"] = unc.model_dump() if unc else None
        if name in PIPELINE and out["status"] == "ok":
            self.done.add(name)
        self.trace.append({"tool": name, "args": inp.model_dump(mode="json"),
                           "seconds": round(time.time() - t0, 2), "status": out["status"],
                           "summary": _brief(out)})
        return out


def _brief(out: dict, limit: int = 300) -> str:
    return ", ".join(f"{k}={v}" for k, v in out.items()
                     if isinstance(v, (int, float, str, bool)))[:limit]


# ------------------------------------------------------------------ 1. check_image_quality
class CheckImageQualityInput(BaseModel):
    pre_path: Path
    post_path: Path
    crop_px: int = Field(default=1024, ge=64, le=4096, description="centre crop that is checked")


class CheckImageQualityOutput(ToolOutput):
    ok: bool
    blank_ratio: float
    bright_ratio: float
    ssim: float | None
    reasons: list[str]


def _centre_rgb(path: Path, crop: int) -> np.ndarray:
    import rasterio
    from rasterio.windows import Window

    if not Path(path).exists():
        raise ToolError(f"image not found: {path}")
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        win = Window(max(0, (w - crop) // 2), max(0, (h - crop) // 2), min(crop, w), min(crop, h))
        arr = src.read([1, 2, 3], window=win)
    return np.moveaxis(arr, 0, -1).astype(np.uint8)


def check_image_quality(inp: CheckImageQualityInput, ctx: ToolContext) -> CheckImageQualityOutput:
    """Cloud / blank / misalignment check of a pre/post image pair (centre crop)."""
    from quaketriage.data.quality import QualityInput
    from quaketriage.data.quality import check_image_quality as run_guard

    pre, post = _centre_rgb(inp.pre_path, inp.crop_px), _centre_rgb(inp.post_path, inp.crop_px)
    if pre.shape != post.shape:
        raise ToolError(f"pre/post crops differ in shape: {pre.shape} vs {post.shape}")
    rep = run_guard(QualityInput(pre=pre, post=post))
    ctx.quality_ok = rep.ok
    ctx.quality_reasons = list(rep.reasons)
    return CheckImageQualityOutput(
        **{**rep.model_dump(),
           "reasons": [localize_reason(r, ctx.lang) for r in rep.reasons]},
        status="ok" if rep.ok else "verification_required",
        warnings=[] if rep.ok else [msg(ctx.lang, "guard")])


# ------------------------------------------------------------------ 2. detect_damage
class DetectDamageInput(BaseModel):
    source: Literal["simulated", "siamese"] = Field(
        default="simulated",
        description="'simulated' = Antakya demo table (SIMULATED detector); "
                    "'siamese' = run the trained model on a GeoTIFF pair")
    pre_path: Path | None = None
    post_path: Path | None = None


class DetectDamageOutput(ToolOutput):
    source: str
    detector_note: str
    n_buildings: int | None = None
    n_above_threshold: int | None = None
    mean_prob: float
    detail: dict = {}


def detect_damage(inp: DetectDamageInput, ctx: ToolContext) -> DetectDamageOutput:
    thr = ctx.constraints.damage_threshold
    if inp.source == "simulated":
        if ctx.buildings is None:
            path = ROOT / get("paths.data_processed") / "buildings.gpkg"
            if not path.exists():
                raise ToolError("building table missing; run scripts/build_buildings.py")
            ctx.buildings = gpd.read_file(path)
        b = ctx.buildings
        missing = {"damage_prob", "uncertainty"} - set(b.columns)
        if missing:
            raise ToolError(f"building table lacks {sorted(missing)}")
        ctx.simulated = True
        return DetectDamageOutput(
            source="simulated", detector_note="SIMULATED detector: HOTOSM destroyed-building "
            "labels plus injected noise, not model output; no imagery was analysed",
            n_buildings=len(b),
            n_above_threshold=int((b["damage_prob"] > thr).sum()),
            mean_prob=round(float(b["damage_prob"].mean()), 4))
    if inp.pre_path is None or inp.post_path is None:
        raise ToolError("source='siamese' needs pre_path and post_path")
    runner = ctx.siamese_runner or _run_siamese
    stats = runner(inp.pre_path, inp.post_path)
    ctx.simulated = False
    mean_unc = stats.get("mean_uncertainty")
    if mean_unc is not None and np.isfinite(mean_unc):
        ctx.image_uncertainty = UncertaintySummary(
            scope="image pixels (TTA std)", mean=round(float(mean_unc), 4),
            p90=round(float(stats.get("p90_uncertainty", mean_unc)), 4),
            max=round(float(stats.get("max_uncertainty", mean_unc)), 4))
    return DetectDamageOutput(
        source="siamese", detector_note="Siamese U-Net + 8-view TTA on the given GeoTIFF pair "
        "(pipeline demo; accuracy is reported on KATE-CD only)",
        mean_prob=round(float(stats.pop("mean_prob")), 4), detail=stats)


def _unc_stats(unc: np.ndarray | None) -> dict:
    if unc is None:
        return {}
    v = unc[np.isfinite(unc)]
    if not v.size:
        return {}
    return {"mean_uncertainty": float(v.mean()), "p90_uncertainty": float(np.quantile(v, 0.9)),
            "max_uncertainty": float(v.max())}


def _run_siamese(pre_path: Path, post_path: Path) -> dict:
    """Siamese + TTA over the whole GeoTIFF pair; summary only (rasters: scripts/infer_area.py)."""
    import rasterio
    import torch

    from quaketriage.models.infer import infer_area, predict_tile
    from quaketriage.models.loading import load_model

    for p in (pre_path, post_path):
        if not Path(p).exists():
            raise ToolError(f"image not found: {p}")
    ckpt = ROOT / "models" / "siamese_best.pt"
    if not ckpt.exists():
        raise ToolError("models/siamese_best.pt missing")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, _ = load_model("siamese", ckpt, get("model.encoder"), device)
    arrays = []
    for p in (pre_path, post_path):
        with rasterio.open(p) as src:
            arrays.append(np.moveaxis(src.read([1, 2, 3]), 0, -1).astype(np.uint8))
    if arrays[0].shape != arrays[1].shape:
        raise ToolError(f"pre/post grids differ: {arrays[0].shape} vs {arrays[1].shape}")
    res = infer_area(arrays[0], arrays[1],
                     lambda x: predict_tile(model, x.to(device), tta=True),  # TTA std = uncertainty
                     tile=get("inference.tile_size"), overlap=get("inference.overlap"))
    valid = res.prob[~np.isnan(res.prob)]
    return {"mean_prob": float(valid.mean()) if valid.size else float("nan"),
            **_unc_stats(res.uncertainty),
            "n_tiles": res.n_tiles, "n_rejected_tiles": len(res.rejected),
            "frac_pixels_above_threshold": round(
                float((valid > get("thresholds.damage_prob")).mean()) if valid.size else 0.0, 4)}


# ------------------------------------------------------------------ 3. score_buildings
class ScoreBuildingsInput(BaseModel):
    threshold: float | None = Field(default=None, ge=0, le=1)
    uncertainty_tau: float | None = Field(default=None, ge=0)


class ScoreBuildingsOutput(ToolOutput):
    n_damaged: int
    n_needs_verification: int
    persons_in_verification_queue: float = 0.0
    verification_reasons: dict[str, int]


def _queue_persons(b: gpd.GeoDataFrame) -> float:
    """Estimated people in buildings that wait for field verification."""
    if "needs_verification" not in b or "estimated_persons" not in b:
        return 0.0
    return round(float(b.loc[b["needs_verification"], "estimated_persons"].sum()), 0)


def score_buildings(inp: ScoreBuildingsInput, ctx: ToolContext) -> ScoreBuildingsOutput:
    b = ctx.need_buildings()
    thr = ctx.constraints.damage_threshold if inp.threshold is None else inp.threshold
    tau = ctx.constraints.uncertainty_tau if inp.uncertainty_tau is None else inp.uncertainty_tau
    b = flag_verification(b, tau)
    b["damaged"] = b["damage_prob"] > thr
    ctx.buildings = b
    reasons = b.loc[b["needs_verification"], "verification_reason"].value_counts().to_dict()
    return ScoreBuildingsOutput(n_damaged=int(b["damaged"].sum()),
                                n_needs_verification=int(b["needs_verification"].sum()),
                                persons_in_verification_queue=_queue_persons(b),
                                verification_reasons={str(k): int(v) for k, v in reasons.items()})


# ------------------------------------------------------------------ 4. estimate_occupancy
class EstimateOccupancyInput(BaseModel):
    scenario: Literal["night", "full_occupancy"] | None = Field(
        default=None, description="night = 04:17 factors (homes occupied, others mostly empty); "
                                  "full_occupancy = all residents at home, no time-of-day factor")


class EstimateOccupancyOutput(ToolOutput):
    scenario: str
    persons_total: float
    persons_in_damaged: float
    note: str


def estimate_occupancy(inp: EstimateOccupancyInput, ctx: ToolContext) -> EstimateOccupancyOutput:
    b = ctx.need_buildings()
    scenario = inp.scenario or ctx.constraints.scenario
    if "estimated_persons_night" not in b:
        b["estimated_persons_night"] = b["estimated_persons"]
    b["estimated_persons"] = (b["estimated_persons_night"] if scenario == "night"
                              else b["residents_day"])
    ctx.buildings = b
    dmg = b["damaged"] if "damaged" in b else b["damage_prob"] > ctx.constraints.damage_threshold
    return EstimateOccupancyOutput(
        scenario=scenario, persons_total=round(float(b["estimated_persons"].sum()), 0),
        persons_in_damaged=round(float(b.loc[dmg, "estimated_persons"].sum()), 0),
        note="WorldPop 100 m, pre-quake; time-of-day factors are assumptions (config population.*)")


# ------------------------------------------------------------------ 5. prioritize_sites
class PrioritizeSitesInput(BaseModel):
    top_k: int = Field(default=10, ge=1, le=100)
    weights: dict[str, float] | None = None
    ahp_pairwise: list[list[float]] | None = None


class SiteRow(BaseModel):
    row: int
    id: int | None
    priority: float
    damage_prob: float
    estimated_persons: float
    floors: float
    distance_to_depot_m: float
    needs_verification: bool


class PrioritizeSitesOutput(ToolOutput):
    weights: dict[str, float] = {}
    top: list[SiteRow] = []
    n_verification_queue: int = 0
    persons_in_verification_queue: float = 0.0
    verification_queue: list[SiteRow] = Field(
        default=[], description="highest-priority buildings held back for field verification")


def _site_rows(frame: gpd.GeoDataFrame) -> list[SiteRow]:
    return [SiteRow(row=int(i), id=int(r["id"]) if "id" in frame and np.isfinite(r["id"]) else None,
                    priority=round(float(r["priority"]), 4),
                    damage_prob=round(float(r["damage_prob"]), 3),
                    estimated_persons=round(float(r["estimated_persons"]), 1),
                    floors=float(r["floors"]),
                    distance_to_depot_m=round(float(r["distance_to_depot_m"]), 0),
                    needs_verification=bool(r["needs_verification"]))
            for i, r in frame.iterrows()]


def prioritize_sites(inp: PrioritizeSitesInput, ctx: ToolContext) -> PrioritizeSitesOutput:
    """Rank buildings. Enforced here, not in any prompt: a failed image-quality guard returns a
    verification request instead of a ranking, and buildings above the uncertainty threshold are
    held back in ``verification_queue`` instead of appearing in ``top``."""
    reason = ctx.quality_block()
    if reason:
        return PrioritizeSitesOutput(
            status="verification_required",
            warnings=[f"{reason}; {msg(ctx.lang, 'no_ranking')}"])
    ctx.ensure_scored()
    if inp.weights is not None:
        ctx.constraints.weights = inp.weights
    try:
        out = prioritise(ctx.need_buildings(), TopsisInput(
            depots=[ctx.constraints.depot_lonlat], weights=ctx.constraints.weights,
            ahp_pairwise=inp.ahp_pairwise))
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    ctx.buildings = out
    queued = out["needs_verification"].to_numpy(bool)
    warn = []
    if queued.any():
        warn.append(msg(ctx.lang, "queue", n=int(queued.sum()), p=_queue_persons(out)))
    return PrioritizeSitesOutput(
        weights=out.attrs["topsis_weights"], warnings=warn,
        top=_site_rows(out[~queued].nlargest(inp.top_k, "priority")),
        n_verification_queue=int(queued.sum()),
        persons_in_verification_queue=_queue_persons(out),
        verification_queue=_site_rows(out[queued].nlargest(inp.top_k, "priority")))


# ------------------------------------------------------------------ 6. plan_routes
class PlanRoutesInput(BaseModel):
    """Optional overrides are stored in the constraints (so later calls keep them)."""

    num_teams: int | None = Field(default=None, ge=1, le=50)
    mission_hours: float | None = Field(default=None, gt=0, le=72)
    candidate_n: int | None = Field(default=None, ge=1, le=2000)


class StopRow(BaseModel):
    row: int
    building_id: int | None
    arrival_h: float
    priority: float
    estimated_persons: float
    walk_m: float


class TeamRoute(BaseModel):
    team: int
    return_h: float
    stops: list[StopRow]


class PlanRoutesOutput(ToolOutput):
    n_visited: int = 0
    priority_collected: float = 0.0
    persons_reached: float = 0.0
    n_unreachable: int = 0
    n_need_walking: int = 0
    n_closed_edges: int = 0
    n_excluded_verification: int = 0
    teams: list[TeamRoute] = []


def plan_routes_tool(inp: PlanRoutesInput, ctx: ToolContext) -> PlanRoutesOutput:
    reason = ctx.quality_block()
    if reason:
        return PlanRoutesOutput(
            status="verification_required",
            warnings=[f"{reason}; {msg(ctx.lang, 'no_plan')}"])
    b = ctx.ensure_scored()
    if "priority" not in b:
        raise ToolError("no priorities yet; run prioritize_sites first")
    c = ctx.constraints
    for k in ("num_teams", "mission_hours", "candidate_n"):
        if getattr(inp, k) is not None:
            setattr(c, k, getattr(inp, k))
    g = apply_debris_risk(ctx.graph(), b, DebrisInput())
    if c.closed_areas:
        by_radius: dict[float, list[tuple[float, float]]] = {}
        for lon, lat, r in c.closed_areas:
            by_radius.setdefault(r, []).append((lon, lat))
        for r, pts in by_radius.items():
            g = close_edges_near(g, pts, r)
    params = VrpInput(candidate_n=c.candidate_n, num_teams=c.num_teams,
                      mission_hours=c.mission_hours, intervention_minutes=c.intervention_minutes,
                      time_limit_s=c.solver_time_limit_s, walking=c.walking)
    held = b["needs_verification"].to_numpy(bool)  # never dispatched before verification
    plan = plan_routes(g, b[~held], c.depot_lonlat, params)
    ctx.plan = plan
    closed = sum(bool(d.get("blocked")) for *_, d in g.edges(data=True))
    teams = [TeamRoute(team=r["team"] + 1, return_h=round(r["return_s"] / 3600, 2),
                       stops=[StopRow(row=s["row"], building_id=s["building_id"],
                                      arrival_h=round(s["arrival_s"] / 3600, 2),
                                      priority=s["priority"],
                                      estimated_persons=s["estimated_persons"],
                                      walk_m=s["walk_m"]) for s in r["stops"]])
             for r in plan["routes"]]
    return PlanRoutesOutput(
        n_excluded_verification=int(held.sum()),
        warnings=([msg(ctx.lang, "held", n=int(held.sum()))] if held.any() else []),
        n_visited=plan["n_visited"],
        priority_collected=plan["priority_collected"], persons_reached=plan["persons_reached"],
        n_unreachable=plan["n_unreachable"], n_need_walking=plan["n_need_walking"],
        n_closed_edges=closed, teams=teams)


# ------------------------------------------------------------------ 7. lookup_protocol
class LookupProtocolInput(SearchInput):
    pass


class ProtocolHit(BaseModel):
    ref: int
    doc: str
    pages: str
    section: str
    score: float
    text: str


class LookupProtocolOutput(ToolOutput):
    query: str
    hits: list[ProtocolHit]


def lookup_protocol(inp: LookupProtocolInput, ctx: ToolContext) -> LookupProtocolOutput:
    """Search AFAD TAMP / INSARAG text; every hit carries document and page for citation."""
    index, embed = ctx.rag()
    qv = embed([inp.query])[0]
    found = (index.search_hybrid(inp.query, qv, inp.top_k) if get("rag.hybrid")
             else index.search(qv, inp.top_k))
    hits = [ProtocolHit(ref=i, doc=c["doc"], pages=f"{c['page_start']}-{c['page_end']}",
                        section=c.get("section", ""), score=round(float(c["score"]), 4),
                        text=c["text"][:600]) for i, c in enumerate(found, 1)]
    ctx.protocol_hits = [h.model_dump() for h in hits]
    return LookupProtocolOutput(query=inp.query, hits=hits)


# ------------------------------------------------------------------ 8. write_sitrep
class WriteSitrepInput(BaseModel):
    language: Literal["tr", "en"] = "tr"
    top_k: int = Field(default=5, ge=1, le=20)


class WriteSitrepOutput(ToolOutput):
    markdown: str
    n_citations: int = 0
    citations: list[dict] = Field(default=[], description="passages shown, numbered as in the text")
    missing_steps: list[str] = []


_TXT = {
    "tr": {"title": "Durum Raporu (SITREP)", "banner": "SİMÜLE DEDEKTÖR — demo; operasyonel karar "
           "için kullanılamaz", "sit": "Durum", "prio": "Öncelikli binalar", "plan": "Rota planı",
           "caveats": "Varsayımlar ve sınırlar", "proto": "Protokol dayanakları",
           "none": "Çalıştırılmadı",
           "head": "Sıra | Bina | Öncelik | Hasar olasılığı | Kişi (tahmini)"},
    "en": {"title": "Situation Report (SITREP)", "banner": "SIMULATED detector — demo, not for "
           "operational decisions", "sit": "Situation", "prio": "Priority buildings",
           "plan": "Route plan", "caveats": "Assumptions and limits", "proto": "Protocol basis",
           "none": "Not run", "head": "Rank | Building | Priority | Damage prob. | Persons (est.)"},
}


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def _doc_lang(doc: str) -> str:
    """AFAD documents are Turkish, INSARAG documents English."""
    return "tr" if doc.upper().startswith("AFAD") else "en"


def clip_sentences(text: str, max_chars: int = 220) -> str:
    """Cut at a sentence boundary: drop a leading half-sentence, then add whole sentences while
    they fit. A single over-long sentence is cut at a word boundary and marked with an ellipsis."""
    text = " ".join(text.split())
    parts = _SENTENCE.split(text)
    if len(parts) > 1 and not parts[0][:1].isupper() and not parts[0][:1].isdigit():
        parts = parts[1:]  # chunk started mid-sentence
    out = ""
    for sent in parts:
        if out and len(out) + 1 + len(sent) > max_chars:
            break
        out = f"{out} {sent}".strip()
        if len(out) >= 0.6 * max_chars:
            break
    if len(out) > max_chars * 1.5:
        out = out[:max_chars].rsplit(" ", 1)[0] + "…"
    elif out and out[-1] not in ".!?":
        out += "…"
    return out


def relevant_hits(hits: list[dict]) -> list[dict]:
    """Passages that really concern damage assessment / search triage (keyword rule from the
    config), renumbered 1..k. Empty if nothing qualifies, then the sitrep omits the section."""
    keys = [k.lower() for k in get("agent.sitrep_protocol_keywords")]
    need = get("agent.sitrep_protocol_min_keywords")
    top = get("agent.sitrep_protocol_max_hits")
    kept = [h for h in hits if sum(k in h["text"].lower() for k in keys) >= need][:top]
    return [{**h, "ref": i} for i, h in enumerate(kept, 1)]


def write_sitrep(inp: WriteSitrepInput, ctx: ToolContext) -> WriteSitrepOutput:
    """Deterministic Markdown from the pipeline state: no number is produced by the LLM."""
    t = _TXT[inp.language]
    tr = inp.language == "tr"
    reason = ctx.quality_block(inp.language)
    if reason:
        headline = ("Görüntü kalite kontrolü başarısız; hasar kararı verilmedi, saha doğrulaması "
                    "gerekli." if tr else "Image quality check failed; no damage decision was "
                    "made, field verification is required.")
        return WriteSitrepOutput(
            status="verification_required", warnings=[reason],
            markdown=f"# {t['title']}\n\n⚠ **{headline}**\n\n- {reason}")
    b = ctx.buildings
    missing = [s for s in PIPELINE[:-1] if s not in ctx.done]
    L = [f"# {t['title']}"]
    if ctx.simulated:
        L.append(f"> **{t['banner']}**")
    L.append(f"\n## {t['sit']}")
    if b is not None and "damaged" in b:
        persons = float(b.loc[b["damaged"], "estimated_persons"].sum())
        L.append((f"- {len(b)} bina değerlendirildi; {int(b['damaged'].sum())} bina hasarlı "
                  f"(olasılık > {ctx.constraints.damage_threshold}); hasarlı binalarda tahmini "
                  f"{persons:.0f} kişi.") if tr else
                 (f"- {len(b)} buildings assessed; {int(b['damaged'].sum())} damaged "
                  f"(prob > {ctx.constraints.damage_threshold}); about {persons:.0f} people in "
                  "damaged buildings."))
        nv = int(b["needs_verification"].sum()) if "needs_verification" in b else 0
        pq = _queue_persons(b)
        L.append((f"- {nv} bina, ~{pq:.0f} kişi doğrulama bekliyor (belirsizlik/görüntü kalitesi); "
                  "bu binalar öncelik listesine ve rota planına alınmadı.") if tr else
                 (f"- {nv} buildings, ~{pq:.0f} people awaiting verification (uncertainty/"
                  "imagery); they are excluded from the priority list and the route plan."))
    else:
        L.append(f"- {t['none']}")
    if ctx.quality_ok is False:
        L.append("- ⚠ " + ("Görüntü kalite kontrolü başarısız; sonuçlara güvenmeyin."
                            if tr else "Image quality check failed; do not rely on results."))
    L.append(f"\n## {t['prio']}")
    if b is not None and "priority" in b:
        L.append(t["head"])
        L.append("---|---|---|---|---")
        shown = b[~b["needs_verification"]] if "needs_verification" in b else b
        for rank, (i, r) in enumerate(shown.nlargest(inp.top_k, "priority").iterrows(), 1):
            bid = int(r["id"]) if "id" in b and np.isfinite(r["id"]) else f"#{i}"
            L.append(f"{rank} | {bid} | {r['priority']:.3f} | {r['damage_prob']:.2f} | "
                     f"{r['estimated_persons']:.0f}")
    else:
        L.append(f"- {t['none']}")
    L.append(f"\n## {t['plan']}")
    if ctx.plan:
        p, c = ctx.plan, ctx.constraints
        L.append((f"- {c.num_teams} ekip, {c.mission_hours:g} saat, bina başına "
                  f"{c.intervention_minutes:g} dk; {p['n_visited']} bina ziyaret edilecek, "
                  f"toplanan öncelik {p['priority_collected']:.2f}, ulaşılan tahmini kişi "
                  f"{p['persons_reached']:.0f}.") if tr else
                 (f"- {c.num_teams} teams, {c.mission_hours:g} h, "
                  f"{c.intervention_minutes:g} min per building; {p['n_visited']} buildings "
                  f"visited, priority collected {p['priority_collected']:.2f}, estimated persons "
                  "reached "
                  f"{p['persons_reached']:.0f}."))
        n_held = int(b["needs_verification"].sum()) if b is not None and \
            "needs_verification" in b else 0
        L.append((f"- Doğrulama kuyruğundaki {n_held} bina plan dışı bırakıldı; Hafta 5 metrikleri "
                  "kuyruk dahil hesaplandı.") if tr else
                 (f"- {n_held} buildings in the verification queue were excluded from the plan; "
                  "Week-5 metrics were computed including the queue."))
        if p["n_unreachable"]:
            L.append(f"- {p['n_unreachable']} " + ("aday bina ulaşılamaz." if tr
                                                    else "candidate buildings unreachable."))
        if c.closed_areas:
            L.append("- " + (f"{len(c.closed_areas)} kapalı bölge varsayıldı." if tr
                             else f"{len(c.closed_areas)} closed areas assumed."))
    else:
        L.append(f"- {t['none']}")
    L.append(f"\n## {t['caveats']}")
    L += (["- Hasar olasılıkları simüle dedektörden (HOTOSM etiketleri + gürültü); model doğruluğu "
           "KATE-CD'de raporlanır, bu rapor için kanıt değildir.",
           "- Depo, enkaz, yürüme ve gece nüfus katsayıları senaryo varsayımıdır."] if tr else
          ["- Damage probabilities come from a simulated detector (HOTOSM labels + noise); model "
           "accuracy is reported on KATE-CD only and is no evidence for this report.",
           "- Depot, debris, walking and night-occupancy factors are scenario assumptions."])
    shown = relevant_hits(ctx.protocol_hits)
    if shown:
        L.append(f"\n## {t['proto']}")
        if tr and any(_doc_lang(h["doc"]) == "en" for h in shown):
            L.append("_(Kaynak parçaları İngilizcedir (INSARAG); çevrilmedi.)_")
        for h in shown:
            L.append(f"- [{h['ref']}] {clip_sentences(h['text'])} ({h['doc']}, p. {h['pages']})")
    return WriteSitrepOutput(markdown="\n".join(L), n_citations=len(shown), citations=shown,
                             missing_steps=missing)


# ------------------------------------------------------------------ 9. update_constraints
class UpdateConstraintsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_teams: int | None = None
    mission_hours: float | None = None
    intervention_minutes: float | None = None
    candidate_n: int | None = None
    walking: bool | None = None
    closed_areas: list[tuple[float, float, float]] | None = Field(
        default=None, description="REPLACES the list of (lon, lat, radius_m) closure circles")
    weights: dict[str, float] | None = None
    damage_threshold: float | None = None
    uncertainty_tau: float | None = None
    scenario: Literal["night", "full_occupancy"] | None = None


class UpdateConstraintsOutput(ToolOutput):
    changed: dict[str, Any]
    stale_steps: list[str]
    unchanged_steps: list[str]


def update_constraints(inp: UpdateConstraintsInput, ctx: ToolContext) -> UpdateConstraintsOutput:
    """Apply what-if changes and say which steps must be re-run (only those)."""
    changed: dict[str, Any] = {}
    for key, value in inp.model_dump(exclude_none=True).items():
        if getattr(ctx.constraints, key) != value:
            try:
                setattr(ctx.constraints, key, value)
            except ValidationError as exc:
                raise ToolError(f"{key}: {exc.errors()[0]['msg']}") from exc
            changed[key] = value
    stale = {s for k in changed for s in STALE_BY_CONSTRAINT[k]}
    ordered = [s for s in PIPELINE if s in stale]
    ctx.done -= set(ordered)  # they must be re-run before their results are trusted again
    return UpdateConstraintsOutput(changed=changed, stale_steps=ordered,
                                   unchanged_steps=[s for s in PIPELINE if s not in stale])


# ------------------------------------------------------------------ registry
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[[Any, ToolContext], BaseModel]


def _spec(name: str, model: type[BaseModel], fn, description: str) -> tuple[str, ToolSpec]:
    return name, ToolSpec(name, description, model, fn)


TOOLS: dict[str, ToolSpec] = dict([
    _spec("check_image_quality", CheckImageQualityInput, check_image_quality,
          "Check a pre/post image pair for clouds, blank areas and misalignment."),
    _spec("detect_damage", DetectDamageInput, detect_damage,
          "Get damage probabilities: the simulated Antakya detector table, or the Siamese model "
          "on a GeoTIFF pair."),
    _spec("score_buildings", ScoreBuildingsInput, score_buildings,
          "Flag damaged buildings and the field-verification queue (uncertain / bad imagery)."),
    _spec("estimate_occupancy", EstimateOccupancyInput, estimate_occupancy,
          "Estimate people per building (night 04:17 or full occupancy) and total at risk."),
    _spec("prioritize_sites", PrioritizeSitesInput, prioritize_sites,
          "Rank buildings with TOPSIS (adjustable weights or AHP matrix); returns the top k."),
    _spec("plan_routes", PlanRoutesInput, plan_routes_tool,
          "Plan rescue-team routes (OR-Tools) with debris-closed roads and a walking leg."),
    _spec("lookup_protocol", LookupProtocolInput, lookup_protocol,
          "Search AFAD TAMP / INSARAG guidelines; returns cited passages (document, pages)."),
    _spec("write_sitrep", WriteSitrepInput, write_sitrep,
          "Write the Markdown situation report from the current state, with citations."),
    _spec("update_constraints", UpdateConstraintsInput, update_constraints,
          "What-if: change teams, hours, closed areas, weights, thresholds; lists stale steps."),
])


def tool_schemas() -> list[dict]:
    """JSON-schema descriptions of all tools, for LLM function calling."""
    return [{"name": s.name, "description": s.description,
             "parameters": s.input_model.model_json_schema()} for s in TOOLS.values()]


_CITE = __import__("re").compile(r"\[(\d+)\]")


def validate_citations(markdown: str, hits: list[dict], chunks: list[dict]) -> dict:
    """Check that every ``[n]`` in ``markdown`` points to a retrieved passage and that each
    retrieved passage really exists in the index (same document, page range and text prefix).
    Whether a passage *supports* a claim is not decidable here (manual check in Week 7)."""
    refs = sorted({int(m) for m in _CITE.findall(markdown)})
    known = {h["ref"] for h in hits}
    index = {(c["doc"], f"{c['page_start']}-{c['page_end']}"): c["text"] for c in chunks}
    missing_in_index = []
    for h in hits:
        text = index.get((h["doc"], h["pages"]))
        probe = " ".join(h["text"].split())[:60]
        found = text is not None and any(
            " ".join(c["text"].split()).startswith(probe) or probe in " ".join(c["text"].split())
            for c in chunks if c["doc"] == h["doc"])
        if not found:
            missing_in_index.append(h["ref"])
    return {"cited_refs": refs, "dangling_refs": [r for r in refs if r not in known],
            "refs_not_in_index": missing_in_index, "n_hits": len(hits),
            "valid": not [r for r in refs if r not in known] and not missing_in_index
            and bool(hits)}


_STOP = {"the", "and", "that", "with", "from", "this", "they", "their", "have", "also", "when",
         "which", "will", "such", "into", "than", "then", "there", "these", "those", "been",
         "being", "are", "was", "were", "for", "not", "but", "its", "can", "may", "must", "should"}


def check_support(answer: str, hits: list[dict]) -> list[dict]:
    """ROUGH support check for each cited sentence of ``answer``.

    A sentence with ``[n]`` counts as supported if the cited passage contains all its acronyms /
    numbers and at least half of its other content words (matched by 5-letter prefix, so
    'register' ~ 'registration'). This catches invented terms and wrong passages, but it cannot
    judge meaning, negation or omissions; it is a smoke test, not a fact check.
    """
    by_ref = {h["ref"]: " ".join(h["text"].split()) for h in hits}
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", answer):
        refs = [int(r) for r in _CITE.findall(sent)]
        if not refs:
            continue
        body = _CITE.sub("", sent)
        keys = set(re.findall(r"\b[A-Z]{2,}[A-Za-z/]*\b|\b\d+(?:\.\d+)?\b", body))
        words = {w[:5] for w in re.findall(r"[A-Za-z]{4,}", body.lower()) if w not in _STOP}
        for ref in refs:
            passage = by_ref.get(ref)
            if passage is None:
                out.append({"ref": ref, "sentence": sent.strip(), "supported": False,
                            "missing": ["<no such passage>"]})
                continue
            low = passage.lower()
            miss_keys = sorted(k for k in keys
                               if k not in passage and k.split("/")[0] not in passage)
            ptoks = {w[:5] for w in re.findall(r"[A-Za-z]{4,}", low)}
            miss_words = sorted(words - ptoks - {k.lower()[:5] for k in keys})
            frac = 1 - len(miss_words) / max(1, len(words))
            out.append({"ref": ref, "sentence": sent.strip(),
                        "supported": not miss_keys and frac >= 0.5,
                        "missing": miss_keys + miss_words[:5], "word_overlap": round(frac, 2)})
    return out
