"""
Fetch Digital Humanities works from OpenAlex via pyalex.

Filters (mirroring the requested API URL):
  - primary_topic.field.id = 12 (Arts and Humanities)
  - primary_topic.id       = T12377 (Digital Humanities topic)
  - publication_year       = 1990-2025
  - has_abstract           = true

Provide your polite-pool email and (optional) OpenAlex Premium API key in a
.env file (default location: data1205/.env, or pass --env-file PATH):

    OPENALEX_EMAIL=you@example.com
    OPENALEX_API_KEY=...   # optional

Environment variables already set in the shell take precedence over .env.

Usage:
    python data1205/fetch_openalex_dh.py [--limit N] [--out PATH] [--env-file PATH]

Output is a JSON list of work records (with reconstructed abstracts) saved to
data1205/openalex_dh_works.json by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from pyalex import Works, config


def configure_pyalex() -> None:
    email = os.environ.get("OPENALEX_EMAIL")
    api_key = os.environ.get("OPENALEX_API_KEY")
    if email:
        config.email = email
    if api_key:
        config.api_key = api_key
    # Be polite with retries.
    config.max_retries = 5
    config.retry_backoff_factor = 0.3
    config.retry_http_codes = [429, 500, 503]


def reconstruct_abstract(inv_index: dict | None) -> str | None:
    """Rebuild plain-text abstract from OpenAlex's inverted index."""
    if not inv_index:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inv_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def build_query(field_id: int, topic_id: str, year_range: str):
    return (
        Works()
        .filter(primary_topic={"field": {"id": field_id}})
        .filter(primary_topic={"id": topic_id})
        .filter(publication_year=year_range)
        .filter(has_abstract=True)
    )


def fetch(
    query,
    limit: int | None,
    per_page: int = 200,
    skip_ids: set[str] | None = None,
) -> list[dict]:
    results: list[dict] = []
    skip_ids = skip_ids or set()
    skipped = 0
    # pyalex.paginate defaults to n_max=10000, which silently caps results.
    # Pass n_max=None (or a large number) to fetch everything via cursor.
    pager_kwargs: dict = {"per_page": per_page, "method": "cursor", "cursor": "*"}
    pager_kwargs["n_max"] = limit if limit else None
    pager = query.paginate(**pager_kwargs)
    t0 = time.time()
    for page in pager:
        for w in page:
            if w.get("id") in skip_ids:
                skipped += 1
                continue
            w["abstract"] = reconstruct_abstract(w.get("abstract_inverted_index"))
            results.append(w)
        print(
            f"  fetched {len(results)} new works "
            f"(skipped {skipped} existing, {time.time() - t0:.1f}s elapsed)",
            file=sys.stderr,
        )
        if limit and len(results) >= limit:
            results = results[:limit]
            break
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of works to fetch (default: all matching).",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent / "openalex_dh_works.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--per-page", type=int, default=200,
        help="Page size (max 200).",
    )
    parser.add_argument(
        "--count-only", action="store_true",
        help="Only print the total count and exit.",
    )
    parser.add_argument(
        "--env-file", type=Path,
        default=Path(__file__).parent / ".env",
        help="Path to .env file with OPENALEX_EMAIL / OPENALEX_API_KEY.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="If --out exists, load it and skip already-fetched IDs; "
             "append new results and rewrite the file.",
    )
    parser.add_argument(
        "--field-id", type=int, default=12,
        help="OpenAlex primary_topic.field.id (default: 12 = Arts & Humanities).",
    )
    parser.add_argument(
        "--topic-id", type=str, default="T12377",
        help="OpenAlex primary_topic.id (default: T12377 = Digital Humanities).",
    )
    parser.add_argument(
        "--year-range", type=str, default="1990-2025",
        help="Publication year range filter (default: 1990-2025).",
    )
    args = parser.parse_args()

    if args.env_file.exists():
        load_dotenv(args.env_file, override=False)
        print(f"Loaded env from {args.env_file}", file=sys.stderr)
    else:
        print(
            f"No .env file at {args.env_file}; using process environment only.",
            file=sys.stderr,
        )

    configure_pyalex()
    if not config.email:
        print(
            "WARNING: OPENALEX_EMAIL not set. Requests will use the anonymous "
            "(slower) pool. Set it for the polite pool.",
            file=sys.stderr,
        )

    query = build_query(args.field_id, args.topic_id, args.year_range)
    total = query.count()
    print(f"Total matching works on OpenAlex: {total}", file=sys.stderr)
    if args.count_only:
        return 0

    existing: list[dict] = []
    skip_ids: set[str] = set()
    if args.resume and args.out.exists():
        with args.out.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        skip_ids = {w["id"] for w in existing if w.get("id")}
        print(
            f"Resume: loaded {len(existing)} existing works from {args.out}",
            file=sys.stderr,
        )

    new_works = fetch(
        query, limit=args.limit, per_page=args.per_page, skip_ids=skip_ids
    )
    works = existing + new_works
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(works, f, ensure_ascii=False, indent=2)
    print(
        f"Wrote {len(works)} works ({len(new_works)} new) to {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
