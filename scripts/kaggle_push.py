"""Push a training notebook to Kaggle with YOUR username (the committed metadata has a placeholder).

kaggle/kernel-metadata.json and kaggle/siamese/kernel-metadata.json carry ``YOUR_KAGGLE_USERNAME``.
This script copies the kernel folder to a temp directory, substitutes ``KAGGLE_USERNAME`` (from the
environment or .env) and runs ``kaggle kernels push`` there, so no personal value is ever written to
the repository. Needs the Kaggle CLI (``pip install kaggle``) and ``KAGGLE_USERNAME`` /
``KAGGLE_KEY``.

Usage: uv run python scripts/kaggle_push.py [baseline|siamese]
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parents[1]
FOLDERS = {"baseline": ROOT / "kaggle", "siamese": ROOT / "kaggle" / "siamese"}


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "siamese"
    if which not in FOLDERS:
        print(f"usage: kaggle_push.py [{'|'.join(FOLDERS)}]")
        return 2
    load_dotenv(ROOT / ".env")
    user = os.environ.get("KAGGLE_USERNAME", "")
    if not user or user.startswith("your_"):
        print("Set KAGGLE_USERNAME (and KAGGLE_KEY) in .env first.")
        return 2
    if shutil.which("kaggle") is None:
        print("Kaggle CLI not found: pip install kaggle")
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        for f in FOLDERS[which].iterdir():
            if f.is_file() and not f.name.endswith(".local.json"):
                shutil.copy(f, tmp)
        meta = Path(tmp) / "kernel-metadata.json"
        meta.write_text(meta.read_text().replace("YOUR_KAGGLE_USERNAME", user))
        return subprocess.run(["kaggle", "kernels", "push", "-p", tmp]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
