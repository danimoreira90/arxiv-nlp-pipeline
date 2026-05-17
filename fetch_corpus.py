#!/usr/bin/env python3
import heapq
import json
import os
from datetime import date

import pandas as pd

DATA_DIR   = "data"
SOURCE     = os.path.join(DATA_DIR, "arxiv-metadata-oai-snapshot.json")
OUTPUT     = os.path.join(DATA_DIR, "corpus_raw.parquet")

CATEGORIES = ["cs.LG", "stat.ML", "q-bio.NC", "econ.EM", "eess.SP"]
TARGET_SET = set(CATEGORIES)

TARGET: dict[str, int] = {
    "cs.LG":    2000,
    "stat.ML":  1200,
    "q-bio.NC":  500,
    "econ.EM":   500,
    "eess.SP":  1200,
}

MIN_MINORITY  = 300
MINORITY_CATS = {"econ.EM", "q-bio.NC"}


def _build_record(obj: dict, primary: str) -> dict:
    authors_str = obj.get("authors", "")
    return {
        "id":               obj["id"],
        "title":            obj.get("title", "").replace("\n", " ").strip(),
        "abstract":         obj.get("abstract", "").replace("\n", " ").strip(),
        "primary_category": primary,
        "authors":          [a.strip() for a in authors_str.split(",") if a.strip()],
        "published":        pd.Timestamp(obj.get("update_date", "1900-01-01")),
    }


def stream_top_n() -> list[dict]:
    """
    Single-pass stream over the JSON-lines file.
    Per category, keeps a min-heap of size TARGET keyed on update_date.
    After the scan the heap contains the TARGET most-recent papers.
    """
    heaps: dict[str, list] = {cat: [] for cat in CATEGORIES}
    seq   = 0   # tiebreaker so heap never compares raw dicts
    lines = 0

    print(f"Scanning {SOURCE} ...")
    with open(SOURCE, encoding="utf-8") as fh:
        for raw in fh:
            lines += 1
            if lines % 500_000 == 0:
                found = sum(len(h) for h in heaps.values())
                print(f"  {lines / 1e6:.1f}M lines scanned — {found:,} candidates")

            obj     = json.loads(raw)
            primary = obj.get("categories", "").split()[0]
            if primary not in TARGET_SET:
                continue

            date_key = obj.get("update_date", "0000-00-00")
            cap      = TARGET[primary]
            heap     = heaps[primary]

            if len(heap) < cap:
                heapq.heappush(heap, (date_key, seq, obj))
            elif date_key > heap[0][0]:
                heapq.heapreplace(heap, (date_key, seq, obj))
            seq += 1

    print(f"  {lines / 1e6:.2f}M lines total.")

    records: list[dict] = []
    for cat, heap in heaps.items():
        sorted_papers = sorted(heap, key=lambda x: x[0], reverse=True)
        for date_key, _, obj in sorted_papers:
            records.append(_build_record(obj, cat))

    return records


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    records = stream_top_n()
    df = pd.DataFrame(records)

    for cat in MINORITY_CATS:
        count = int((df["primary_category"] == cat).sum())
        if count < MIN_MINORITY:
            print(f"WARNING: {cat} — {count} abstracts (below MIN_MINORITY={MIN_MINORITY})")

    df.to_parquet(OUTPUT, index=False)

    snapshot = date.today().isoformat()
    counts   = df.groupby("primary_category").size().sort_values(ascending=False)
    print(f"\n{'=' * 42}")
    print(f"Snapshot date   : {snapshot}")
    print(f"{'=' * 42}")
    print(counts.to_string())
    print(f"\nTotal abstracts : {len(df)}")


if __name__ == "__main__":
    main()
