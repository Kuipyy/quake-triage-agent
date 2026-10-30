"""Refresh the results table in README.md from reports/*.json.

Usage:
    uv run python scripts/make_report.py           # rewrite README.md if it is out of date
    uv run python scripts/make_report.py --check   # exit 1 if README.md is out of date (for CI)
"""
import argparse
import sys
from pathlib import Path

from quaketriage.reporting import BEGIN, END, collect_metrics, render_table, update_markdown

ROOT = Path(__file__).parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--check", action="store_true", help="do not write; fail if out of date")
    args = parser.parse_args()

    table = render_table(collect_metrics(args.reports))
    if args.check:
        text = args.readme.read_text()
        current = text[text.find(BEGIN) + len(BEGIN) : text.find(END)].strip()
        if current != table:
            print("README results table is out of date — run `make report`.")
            return 1
        print("README results table is up to date.")
        return 0
    changed = update_markdown(args.readme, table)
    print("README results table updated." if changed else "README table already up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
