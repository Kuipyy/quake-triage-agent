"""Results-table generator tests."""
import json

import pytest

from quaketriage.reporting import (
    BEGIN,
    END,
    collect_metrics,
    flatten,
    render_table,
    update_markdown,
)


def test_flatten_nested_and_skips_provenance():
    data = {"train": {"f1": 0.5, "n": 3}, "note": "ok", "ids": [1, 2], "provenance": {"seed": 1}}
    assert flatten(data) == {"train.f1": 0.5, "train.n": 3, "note": "ok"}


def test_collect_and_render(tmp_path):
    (tmp_path / "metrics_a.json").write_text(json.dumps({"val": {"f1": 0.123456}}))
    (tmp_path / "list.json").write_text("[1, 2]")
    rows = collect_metrics(tmp_path)
    assert rows == [("metrics_a", "val.f1", 0.123456)]
    assert "| `metrics_a` | `val.f1` | 0.1235 |" in render_table(rows)


def test_render_empty_placeholder():
    assert "No results yet" in render_table([])


def test_update_markdown_replaces_between_markers(tmp_path):
    md = tmp_path / "README.md"
    md.write_text(f"intro\n{BEGIN}\nold\n{END}\noutro\n")
    assert update_markdown(md, "NEW") is True
    assert md.read_text() == f"intro\n{BEGIN}\nNEW\n{END}\noutro\n"
    assert update_markdown(md, "NEW") is False


def test_update_markdown_requires_markers(tmp_path):
    md = tmp_path / "README.md"
    md.write_text("no markers")
    with pytest.raises(ValueError):
        update_markdown(md, "x")


def test_readme_planning_claims_match_metrics():
    """README prose must not drift from reports/metrics_planning.json."""
    import json
    from pathlib import Path

    root = Path(__file__).parents[1]
    path = root / "reports" / "metrics_planning.json"
    if not path.exists():
        return
    b = json.loads(path.read_text())["baselines_closures_with_walking"]
    readme = (root / "README.md").read_text()
    gain = round((b["vrp"]["priority_collected"] / b["priority_first"]["priority_collected"] - 1)
                 * 100)
    assert f"{gain} % more priority than the highest-priority-first rule" in readme
    for key in ("vrp", "priority_first", "greedy_nearest"):
        assert f"{b[key]['priority_collected']:.2f}" in readme
