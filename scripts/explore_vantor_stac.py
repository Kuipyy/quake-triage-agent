"""Query Vantor (Maxar) Open Data STAC for Kahramanmaraş earthquake imagery.

Uses direct HTTP requests to navigate the static S3 catalog — much faster than
pystac_client's recursive crawl.
"""
import json
import requests

BASE = "https://maxar-opendata.s3.amazonaws.com/events"
EVENT_ID = "Kahramanmaras-turkey-earthquake-23"


def fetch_json(url: str) -> dict:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    # 1. Top-level catalog
    catalog = fetch_json(f"{BASE}/catalog.json")
    print(f"Catalog title: {catalog.get('title', 'N/A')}")

    # List child links (events)
    children = [lk for lk in catalog.get("links", []) if lk.get("rel") == "child"]
    print(f"Total events: {len(children)}")

    # Find Kahramanmaraş
    target = next((c for c in children if EVENT_ID in c.get("href", "")), None)
    if target is None:
        print(f"Event '{EVENT_ID}' not found. Available:")
        for c in children:
            print(f"  {c['href']}")
        return

    event_url = f"{BASE}/{EVENT_ID}/collection.json"
    print(f"\nFetching event collection: {event_url}")
    try:
        collection = fetch_json(event_url)
    except Exception:
        event_url = target["href"]
        if not event_url.startswith("http"):
            event_url = f"{BASE}/{event_url}"
        collection = fetch_json(event_url)

    print(f"Event title: {collection.get('title', 'N/A')}")
    print(f"Description: {collection.get('description', 'N/A')[:200]}")

    # List sub-catalogs (pre / post / ...)
    sub_links = [lk for lk in collection.get("links", []) if lk.get("rel") in ("child", "item")]
    print(f"\nSub-links ({len(sub_links)}):")
    for lk in sub_links[:20]:
        print(f"  [{lk['rel']}] {lk['href']}")

    # Save summary
    out = {
        "event_id": EVENT_ID,
        "event_url": event_url,
        "title": collection.get("title"),
        "sub_links": sub_links,
    }
    out_path = "reports/vantor_stac_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSummary saved → {out_path}")


if __name__ == "__main__":
    main()
