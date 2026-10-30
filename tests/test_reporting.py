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
