"""Download AFAD TAMP and INSARAG guidelines PDFs for RAG."""
from pathlib import Path

import requests

OUT_DIR = Path("data/raw/docs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOCS = [
    {
        "name": "AFAD_TAMP.pdf",
        "urls": [
            "https://www.afad.gov.tr/kurumlar/afad.gov.tr/e_Kutuphane/Planlar/TAMP.pdf",
            "https://mebakub.meb.gov.tr/akub/dosyalar/TAMP.pdf",
        ],
        "description": "AFAD Türkiye Afet Müdahale Planı (TAMP)",
    },
    {
        "name": "INSARAG_Vol2_ManB_Operations.pdf",
        "urls": [
            "https://www.insarag.org/wp-content/uploads/2021/01/INSARAG_Guidelines_2020_Vol_II_Manual_B.pdf",
        ],
        "description": "INSARAG Guidelines 2020 Vol II Manual B — Operations",
    },
    {
        "name": "INSARAG_Vol3_FieldGuide.pdf",
        "urls": [
            "https://www.insarag.org/wp-content/uploads/2021/01/INSARAG_Guidelines_2020_Vol_III.pdf",
        ],
        "description": "INSARAG Guidelines 2020 Vol III — Field Guide",
    },
]


def download(name: str, urls: list[str], description: str):
    out = OUT_DIR / name
    if out.exists():
        print(f"  Already exists: {name} ({out.stat().st_size/1e3:.0f} KB)")
        return True
    for url in urls:
        try:
            print(f"  Trying: {url[:80]}...")
            r = requests.get(url, timeout=60, allow_redirects=True,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            if len(r.content) < 1000:
                print(f"    Too small ({len(r.content)} bytes), skipping")
                continue
            out.write_bytes(r.content)
            print(f"  OK → {name} ({out.stat().st_size/1e3:.0f} KB)")
            return True
        except Exception as e:
            print(f"    Failed: {e}")
    return False


def main():
    for doc in DOCS:
        print(f"\n{doc['description']}")
        ok = download(doc["name"], doc["urls"], doc["description"])
        if not ok:
            print(f"  WARN: Could not download {doc['name']} — add manually if needed")

    files = list(OUT_DIR.glob("*.pdf"))
    print(f"\nDocs directory: {len(files)} PDF(s)")
    for f in files:
        print(f"  {f.name}: {f.stat().st_size/1e3:.0f} KB")


if __name__ == "__main__":
    main()
