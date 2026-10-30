"""Turn `reports/*.json` metric files into a Markdown table and keep README.md in sync.

Convention: a report is a JSON object whose leaves are numbers/strings; nested objects are
flattened to dotted keys. A top-level "provenance" key (see provenance.py) is ignored.
"""
import json
from pathlib import Path

BEGIN = "<!-- results:begin -->"
END = "<!-- results:end -->"
SKIP_KEYS = {"provenance"}


def flatten(data: dict, prefix: str = "") -> dict[str, object]:
    """Flatten nested dicts to dotted keys, keeping scalar leaves only."""
    flat: dict[str, object] = {}
    for key, value in data.items():
        if not prefix and key in SKIP_KEYS:
            continue
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten(value, name))
        elif isinstance(value, (int, float, str, bool)):
            flat[name] = value
    return flat


def collect_metrics(reports_dir: str | Path) -> list[tuple[str, str, object]]:
    """(report name, metric key, value) rows for every *.json directly under `reports_dir`."""
    rows = []
    for path in sorted(Path(reports_dir).glob("*.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            continue
        rows.extend((path.stem, key, value) for key, value in flatten(data).items())
    return rows


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_table(rows: list[tuple[str, str, object]]) -> str:
    """Markdown table; a placeholder line when there are no reports yet."""
    if not rows:
        return "_No results yet — run the pipeline, then `make report`._"
    lines = ["| Report | Metric | Value |", "|---|---|---|"]
    lines += [f"| `{report}` | `{key}` | {_fmt(value)} |" for report, key, value in rows]
    return "\n".join(lines)


def update_markdown(path: str | Path, table: str) -> bool:
    """Replace the text between the results markers in `path`. Returns True if the file changed."""
    path = Path(path)
    text = path.read_text()
    start, end = text.find(BEGIN), text.find(END)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"{path} must contain {BEGIN} ... {END}")
    new = text[: start + len(BEGIN)] + "\n" + table + "\n" + text[end:]
    if new == text:
        return False
    path.write_text(new)
    return True
