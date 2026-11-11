"""Pure view builders for the Gradio app (no Gradio import, so they are unit-testable).

Every view carries the SIMULATED-detector notice; the Antakya building table comes from a simulated
detector (HOTOSM labels + noise), the Kilis crop is the only place the real Siamese model runs.
"""
from __future__ import annotations

import html
import io
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd

from quaketriage.agent.tools import ToolContext, build_plan_graph
from quaketriage.config import get
from quaketriage.planning.roads import snap_to_nodes
from quaketriage.ui.preview_map import build_preview_map
from quaketriage.ui.route_map import add_plan_layers

ROOT = Path(__file__).parents[3]
NOTICE = "SIMULATED detector, demo only — not for operational decisions."
TOP_MARKERS = 20
NO_IMAGERY = ("KATE-CD'yi `scripts/download_kate_cd.py` ile indirin; görseller lisans nedeniyle "
              "repoda yok. (Download KATE-CD with `scripts/download_kate_cd.py`; the images are "
              "not in the repository for licence reasons.)")


def notice_md(extra: str = "") -> str:
    return f"**⚠ {NOTICE}** {extra}".strip()


def iframe(map_html: str, height: int = 640) -> str:
    """Embed a full HTML document (a folium map) in an iframe (works inside gr.HTML)."""
    return (f'<iframe srcdoc="{html.escape(map_html, quote=True)}" '
            f'style="width:100%;height:{height}px;border:1px solid #999;'
            f'border-radius:6px"></iframe>')


def _priority_layer(m: folium.Map, b: gpd.GeoDataFrame) -> None:
    cand = b[b["candidate"].to_numpy(bool) & ~b["needs_verification"].to_numpy(bool)]
    top = cand.nlargest(TOP_MARKERS, "priority")
    ll = top.geometry.centroid.to_crs(4326)
    grp = folium.FeatureGroup(f"top {TOP_MARKERS} priority", show=True)
    for rank, (idx, pt) in enumerate(zip(top.index, ll, strict=True), 1):
        row = top.loc[idx]
        tip = (f"#{rank}: priority {row['priority']:.2f}, damage {row['damage_prob']:.2f}, "
               f"~{row['estimated_persons']:.0f} persons")
        folium.Marker((pt.y, pt.x), tooltip=tip, icon=folium.DivIcon(
            html=f"<div style='font:bold 11px sans-serif;background:#fff;border:2px solid #b30000;"
                 f"border-radius:11px;width:22px;height:22px;line-height:18px;text-align:center'>"
                 f"{rank}</div>")).add_to(grp)
    grp.add_to(m)


def _fit_to_plan(m: folium.Map, b: gpd.GeoDataFrame, plan: dict,
                 depot: tuple[float, float]) -> None:
    """Zoom to the dispatched buildings and the depot (the plan, not the whole table)."""
    rows = [s["row"] for r in plan["routes"] for s in r["stops"]]
    if not rows:
        return
    ll = b.loc[rows].geometry.centroid.to_crs(4326)
    lons, lats = [*ll.x, depot[0]], [*ll.y, depot[1]]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(30, 30))


def map_html(ctx: ToolContext) -> str:
    """Damage layer + priority markers + team routes (if a plan exists), as an iframe."""
    if ctx.buildings is None:
        return ("<p>No building table yet. Ask the agent to assess the area "
                "(e.g. “Assess Antakya and plan routes”).</p>")
    b = ctx.buildings
    m = build_preview_map(b, layer_control=False)
    if "priority" in b and "candidate" in b:
        _priority_layer(m, b)
    if ctx.plan:
        g = build_plan_graph(ctx, b)
        depot = tuple(ctx.constraints.depot_lonlat)
        depot_node = int(snap_to_nodes(g, [depot])[0][0])
        add_plan_layers(m, g, ctx.plan, b, depot_node, depot)
        _fit_to_plan(m, b, ctx.plan, depot)
    folium.LayerControl().add_to(m)
    return iframe(m.get_root().render())


# ------------------------------------------------------------------ tables
PRIORITY_COLUMNS = ["rank", "building", "priority", "damage_prob", "uncertainty",
                    "est_persons", "floors"]


def priority_table(ctx: ToolContext, limit: int = 50) -> pd.DataFrame:
    b = ctx.buildings
    if b is None or "priority" not in b:
        return pd.DataFrame(columns=PRIORITY_COLUMNS)
    queue = b["needs_verification"].to_numpy(bool)
    cand = b["candidate"].to_numpy(bool)
    t = b[~queue & cand].nlargest(limit, "priority")  # queue / below-gate rows are never listed
    # same label as the sitrep: OSM id, else the table row ("#5777")
    ids = pd.Series([f"#{int(r['id'])}" if "id" in t and pd.notna(r["id"]) else f"#{i}"
                     for i, r in t.iterrows()], index=t.index)
    return pd.DataFrame({
        "rank": range(1, len(t) + 1),
        "building": ids.to_numpy(),
        "priority": t["priority"].round(3).to_numpy(),
        "damage_prob": t["damage_prob"].round(3).to_numpy(),
        "uncertainty": t["uncertainty"].round(3).to_numpy() if "uncertainty" in t else np.nan,
        "est_persons": t["estimated_persons"].round(1).to_numpy(),
        "floors": t["floors"].to_numpy() if "floors" in t else np.nan})


def queue_summary(ctx: ToolContext) -> str:
    b = ctx.buildings
    if b is None or "needs_verification" not in b:
        return ""
    q = b["needs_verification"].to_numpy(bool)
    gate = ~b["candidate"].to_numpy(bool) if "candidate" in b else np.zeros(len(b), bool)
    return (f"{int(q.sum())} buildings are in the field-verification queue (never dispatched), "
            f"{int(gate.sum())} are below the damage gate (not candidates).")


def trace_table(ctx: ToolContext) -> pd.DataFrame:
    rows = [{"#": i, "tool": e["tool"], "status": e.get("status", "error"),
             "seconds": e.get("seconds", ""),
             "summary": (e.get("summary") or e.get("error", ""))[:200]}
            for i, e in enumerate(ctx.trace, 1)]
    return pd.DataFrame(rows, columns=["#", "tool", "status", "seconds", "summary"])


def sitrep_markdown(ctx: ToolContext) -> str | None:
    out = ctx.outputs.get("write_sitrep")
    return out["markdown"] if out and out.get("markdown") else None


def sitrep_file(ctx: ToolContext, folder: Path | None = None) -> Path | None:
    md = sitrep_markdown(ctx)
    if not md:
        return None
    folder = folder or ROOT / "outputs" / "ui"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "sitrep.md"
    path.write_text(md)
    return path


def model_card_md() -> str:
    path = ROOT / "docs" / "model_card.md"
    return path.read_text() if path.exists() else "Model card not found."


# ------------------------------------------------------------------ before / after
def _shrink(arr: np.ndarray, max_px: int) -> np.ndarray:
    from PIL import Image
    img = Image.fromarray(arr)
    img.thumbnail((max_px, max_px))
    return np.asarray(img)


def kate_cd_pair(index: int = 0, split: str | None = None,
                 max_px: int | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """Pre/post tiles of a KATE-CD sample and a one-line caption (label = damaged pixel share)."""
    from quaketriage.data.kate_cd import SPLITS, _decode_mask, _decode_rgb
    split = split or get("ui.kate_cd_split")
    max_px = max_px or get("ui.max_image_px")
    path = ROOT / get("paths.data_raw") / "kate_cd" / SPLITS[split]
    df = pd.read_parquet(path)
    index = int(index) % len(df)
    row = df.iloc[index]
    share = float((_decode_mask(row["label"]) > 127).mean())
    cap = f"KATE-CD {split} sample {index} — labelled damaged pixels: {share:.1%}"
    return _shrink(_decode_rgb(row["pre_image"]), max_px), \
        _shrink(_decode_rgb(row["post_image"]), max_px), cap


def kilis_pair(max_px: int | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """Pre/post of the Kilis Vantor crop (CC BY-NC 4.0; shown locally, never committed)."""
    import rasterio
    max_px = max_px or get("ui.max_image_px")
    cfg = get("regions.kilis")
    imgs = []
    for key in ("pre", "post"):
        with rasterio.open(ROOT / cfg[key]) as src:
            a = src.read([1, 2, 3]).transpose(1, 2, 0)
        imgs.append(_shrink(a.astype(np.uint8), max_px))
    return imgs[0], imgs[1], "Kilis crop — imagery © Vantor/Maxar Open Data (CC BY-NC 4.0)"


def png_bytes(arr: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return buf.getvalue()
