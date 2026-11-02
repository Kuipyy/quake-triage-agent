"""Inspect the overlapping pre/post Vantor tile pair and fetch item details."""
from urllib.parse import urljoin

import requests

BASE = "https://maxar-opendata.s3.amazonaws.com/events/Kahramanmaras-turkey-earthquake-23"

PRE_ID  = "10300100E18A1000"
POST_ID = "1040010082B85E00"


def fetch_json(url: str) -> dict | None:
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  WARN: {url} → {e}")
        return None


def inspect_collection(col_id: str, label: str):
    url = f"{BASE}/ard/acquisition_collections/{col_id}_collection.json"
    col = fetch_json(url)
    if col is None:
        return
    print(f"\n=== {label}: {col_id} ===")
    print(f"  datetime: {col.get('extent',{}).get('temporal',{}).get('interval',[[None]])[0][0]}")
    print(f"  bbox: {col.get('extent',{}).get('spatial',{}).get('bbox',[[]])[0]}")

    # Get item links
    item_links = [lk for lk in col.get("links", []) if lk.get("rel") == "item"]
    print(f"  items: {len(item_links)}")

    for lk in item_links[:3]:
        # item links are relative to the collection file (e.g. ../37/<quadkey>/...)
        item_url = urljoin(url, lk["href"])
        item = fetch_json(item_url)
        if item is None:
            continue
        print(f"\n  Item: {item.get('id')}")
        print(f"    bbox: {item.get('bbox')}")
        assets = item.get("assets", {})
        for k, v in assets.items():
            href_a = v.get("href", "")
            print(f"    asset [{k}]: {urljoin(item_url, href_a)[:100]}")


def main():
    inspect_collection(PRE_ID,  "PRE  (2023-01-26)")
    inspect_collection(POST_ID, "POST (2023-02-07)")


if __name__ == "__main__":
    main()
