"""Find Vantor tiles covering Kahramanmaraş / Antakya and pick demo area.

Fetches all acquisition collections, splits by pre/post earthquake date,
finds overlapping tiles.
"""
import json
import time

import requests

BASE = "https://maxar-opendata.s3.amazonaws.com/events/Kahramanmaras-turkey-earthquake-23"
QUAKE_DATE = "2023-02-06"

# Study area: rough bounding box covering both Antakya and Kahramanmaraş region
STUDY_BBOX = (35.8, 36.0, 37.2, 37.8)   # minlon, minlat, maxlon, maxlat


def fetch_json(url: str) -> dict | None:
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  WARN: {url} → {e}")
        return None


def bbox_overlaps(b1, b2) -> bool:
    return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])


def bbox_intersection(b1, b2):
    return [
        max(b1[0], b2[0]), max(b1[1], b2[1]),
        min(b1[2], b2[2]), min(b1[3], b2[3]),
    ]


def main():
    with open("reports/vantor_stac_summary.json") as f:
        summary = json.load(f)

    sub_links = summary["sub_links"]
    print(f"Fetching all {len(sub_links)} acquisition collections...\n")

    all_cols = []
    for lk in sub_links:
        href = lk["href"].lstrip("./")
        url = f"{BASE}/{href}"
        col = fetch_json(url)
        if col is None:
            continue
        extent = col.get("extent", {}).get("spatial", {}).get("bbox", [[]])[0]
        if not extent or len(extent) < 4:
            continue
        dt = col.get("extent", {}).get("temporal", {}).get("interval", [[None]])[0][0]
        if dt is None:
            continue
        if not bbox_overlaps(extent, STUDY_BBOX):
            continue
        all_cols.append({
            "id": col.get("id"),
            "datetime": dt,
            "bbox": extent,
            "url": url,
            "is_post": dt >= QUAKE_DATE,
        })
        phase = "POST" if dt >= QUAKE_DATE else "PRE "
        print(f"  [{phase}] {dt[:10]}  bbox={[round(x,4) for x in extent]}  id={col.get('id')}")
        time.sleep(0.05)

    pre  = [c for c in all_cols if not c["is_post"]]
    post = [c for c in all_cols if c["is_post"]]
    print(f"\nIn study area: {len(pre)} pre, {len(post)} post collections")

    # Find overlapping pre/post pairs
    pairs = []
    for pr in pre:
        for po in post:
            if bbox_overlaps(pr["bbox"], po["bbox"]):
                inter = bbox_intersection(pr["bbox"], po["bbox"])
                w = inter[2] - inter[0]
                h = inter[3] - inter[1]
                area_km2 = w * 111 * h * 111 * abs((inter[1]+inter[3])/2 * 3.14159/180 * 0 + 1)
                # rough km²
                area_km2 = w * 111 * h * 111
                pairs.append({
                    "pre_id": pr["id"], "pre_date": pr["datetime"][:10], "pre_bbox": pr["bbox"],
                    "post_id": po["id"], "post_date": po["datetime"][:10], "post_bbox": po["bbox"],
                    "intersection": inter,
                    "overlap_km2_approx": round(area_km2, 2),
                })

    print(f"Overlapping pre/post pairs: {len(pairs)}")
    for p in pairs:
        print(f"  PRE {p['pre_date']} × POST {p['post_date']}  "
              f"overlap≈{p['overlap_km2_approx']} km²  "
              f"inter={[round(x, 4) for x in p['intersection']]}")

    out = {"all_in_study_area": all_cols, "pre_post_pairs": pairs}
    with open("reports/vantor_matches.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved → reports/vantor_matches.json")


if __name__ == "__main__":
    main()
