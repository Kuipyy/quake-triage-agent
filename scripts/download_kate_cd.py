"""Download KATE-CD dataset from HuggingFace Hub to data/raw/kate_cd/."""
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

DEST = Path("data/raw/kate_cd")
REPO_ID = "CSCRS/kate-cd"

def main():
    token = os.getenv("HF_TOKEN")
    print(f"Downloading {REPO_ID} → {DEST} ...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(DEST),
        token=token,
    )
    files = list(DEST.rglob("*"))
    total_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1e6
    print(f"Done. {len(files)} files, {total_mb:.1f} MB total.")

if __name__ == "__main__":
    main()
