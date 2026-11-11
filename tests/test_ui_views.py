import html

import pytest

from quaketriage.ui import views
from tests.test_tools import ctx  # noqa: F401  (fixture: small buildings + road graph)


def _full(ctx):  # noqa: F811
    ctx.call("detect_damage", {})
    ctx.call("prioritize_sites", {})
    ctx.call("plan_routes", {})
    ctx.call("write_sitrep", {"language": "en"})
    return ctx


def test_notice_is_on_every_view_helper():
    assert "SIMULATED detector" in views.notice_md()
    assert "demo only" in views.notice_md("extra")


def test_priority_table_lists_only_dispatchable_candidates(ctx):  # noqa: F811
    _full(ctx)
    t = views.priority_table(ctx)
    assert list(t.columns) == views.PRIORITY_COLUMNS
    b = ctx.buildings
    held = set(b.index[b["needs_verification"]])
    gated = set(b.index[~b["candidate"]])
    shown = set(t["building"])
    assert shown and not any(f"#{int(b.loc[i, 'id'])}" in shown for i in held | gated)
    assert t["priority"].is_monotonic_decreasing
    assert "verification queue" in views.queue_summary(ctx)


def test_empty_views_before_any_tool_ran():
    from quaketriage.agent.tools import ToolContext
    c = ToolContext()
    assert views.priority_table(c).empty and views.trace_table(c).empty
    assert "No building table" in views.map_html(c)
    assert views.sitrep_markdown(c) is None and views.sitrep_file(c) is None


def test_trace_and_sitrep_file(ctx, tmp_path):  # noqa: F811
    _full(ctx)
    tr = views.trace_table(ctx)
    assert list(tr["tool"])[-1] == "write_sitrep" and (tr["status"] == "ok").all()
    path = views.sitrep_file(ctx, tmp_path)
    assert path.read_text().startswith("#") and "SIMULATED" in path.read_text()


def test_map_has_routes_markers_and_single_layer_control(ctx):  # noqa: F811
    _full(ctx)
    doc = html.unescape(views.map_html(ctx))
    assert "<iframe" in doc and "SIMULATED detector" in doc
    assert "team 1" in doc and "top 20 priority" in doc and "fitBounds" in doc
    assert doc.count("L.control.layers") == 1  # two controls broke the map script


def test_before_after_without_local_data_gives_the_download_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(views, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        views.kate_cd_pair(0)
    assert "download_kate_cd.py" in views.NO_IMAGERY and "lisans" in views.NO_IMAGERY


def test_app_builds_and_every_tab_carries_the_notice():
    from quaketriage.ui.app import build_app
    cfg = build_app().get_config_file()
    texts = [c.get("props", {}).get("value", "") for c in cfg["components"]]
    assert sum("SIMULATED detector" in str(t) for t in texts) >= 6
