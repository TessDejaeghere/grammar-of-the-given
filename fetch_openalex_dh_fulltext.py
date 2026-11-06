"""
Download full-paper PDFs for Digital Humanities works already fetched from
OpenAlex (data1205/openalex_dh_works.json) and extract plain text.

For each work, this script uses ``content_urls.pdf`` (OpenAlex's hosted PDF
mirror, available with an OpenAlex Premium key) by default, and falls back to
``best_oa_location.pdf_url`` / any locations[*].pdf_url when the OpenAlex
mirror is missing or fails.

Outputs (under --out-dir, default data1205/openalex_dh_fulltext/):
    pdfs/<WORK_ID>.pdf       # raw PDF
    text/<WORK_ID>.txt       # plain text extracted with pypdf
    manifest.json            # per-work status: source, http code, n_chars, errors

The script is resumable: works that already have a non-empty .txt are skipped,
and the manifest is merged on each run. Use --redo-failed to retry works whose
previous attempt failed.

Auth: set OPENALEX_EMAIL and OPENALEX_API_KEY in data1205/.env (same file
already used by fetch_openalex_dh.py). The api_key is sent as a query param
on content.openalex.org requests.

Usage:
    python data1205/fetch_openalex_dh_fulltext.py
    python data1205/fetch_openalex_dh_fulltext.py --limit 50 --workers 8
    python data1205/fetch_openalex_dh_fulltext.py --redo-failed
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import requests
from dotenv import load_dotenv


OPENALEX_CONTENT_HOST = "content.openalex.org"


def work_short_id(work: dict) -> str | None:
    wid = work.get("id") or ""
    # work id looks like "https://openalex.org/W2128136878"
    short = wid.rsplit("/", 1)[-1]
    return short or None


def candidate_pdf_urls(work: dict, prefer_openalex_mirror: bool) -> list[str]:
    urls: list[str] = []
    cu = work.get("content_urls") or {}
    mirror = cu.get("pdf")
    bol = (work.get("best_oa_location") or {}).get("pdf_url")
    loc_pdfs = [
        (loc or {}).get("pdf_url")
        for loc in (work.get("locations") or [])
    ]
    loc_pdfs = [u for u in loc_pdfs if u]

    if prefer_openalex_mirror and mirror:
        urls.append(mirror)
    if bol:
        urls.append(bol)
    for u in loc_pdfs:
        if u not in urls:
            urls.append(u)
    if not prefer_openalex_mirror and mirror and mirror not in urls:
        urls.append(mirror)
    # de-dup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u and u not in seen:
            out.append(u)
            seen.add(u)
    return out


@dataclass
class Result:
    work_id: str
    status: str  # ok | skip | http_error | no_pdf_url | download_error | extract_error
    source_url: str | None = None
    http_code: int | None = None
    pdf_bytes: int = 0
    text_chars: int = 0
    error: str | None = None
    elapsed_s: float = 0.0


def http_get(
    session: requests.Session,
    url: str,
    api_key: str | None,
    timeout: float,
) -> requests.Response:
    params = None
    if api_key and OPENALEX_CONTENT_HOST in url:
        params = {"api_key": api_key}
    return session.get(url, params=params, timeout=timeout, allow_redirects=True)


def extract_text(pdf_bytes: bytes) -> str:
    # pypdf is pure-python; good enough for OpenAlex's mostly text PDFs.
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
    pages_text: list[str] = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except (PdfReadError, Exception):  # noqa: BLE001 - pypdf raises many
            pages_text.append("")
    return "\n\n".join(pages_text).strip()


def process_one(
    work: dict,
    pdf_dir: Path,
    text_dir: Path,
    session: requests.Session,
    api_key: str | None,
    prefer_openalex_mirror: bool,
    keep_pdf: bool,
    timeout: float,
    overwrite: bool,
) -> Result:
    t0 = time.time()
    short = work_short_id(work)
    if not short:
        return Result("?", "no_pdf_url", error="missing work id")

    text_path = text_dir / f"{short}.txt"
    pdf_path = pdf_dir / f"{short}.pdf"
    if (not overwrite) and text_path.exists() and text_path.stat().st_size > 0:
        return Result(
            short, "skip", text_chars=text_path.stat().st_size,
            elapsed_s=time.time() - t0,
        )

    urls = candidate_pdf_urls(work, prefer_openalex_mirror)
    if not urls:
        return Result(short, "no_pdf_url", elapsed_s=time.time() - t0)

    last_err: str | None = None
    last_code: int | None = None
    pdf_bytes: bytes | None = None
    used_url: str | None = None
    for url in urls:
        try:
            r = http_get(session, url, api_key, timeout)
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            continue
        last_code = r.status_code
        ct = (r.headers.get("Content-Type") or "").lower()
        if r.status_code == 200 and (
            "pdf" in ct or r.content[:4] == b"%PDF"
        ):
            pdf_bytes = r.content
            used_url = url
            break
        last_err = f"http {r.status_code}, content-type={ct!r}"

    if pdf_bytes is None:
        return Result(
            short, "http_error" if last_code else "download_error",
            source_url=urls[0], http_code=last_code, error=last_err,
            elapsed_s=time.time() - t0,
        )

    try:
        text = extract_text(pdf_bytes)
    except Exception as e:  # noqa: BLE001
        # still save the pdf for later manual recovery
        if keep_pdf:
            pdf_path.write_bytes(pdf_bytes)
        return Result(
            short, "extract_error", source_url=used_url, http_code=200,
            pdf_bytes=len(pdf_bytes), error=f"{type(e).__name__}: {e}",
            elapsed_s=time.time() - t0,
        )

    if keep_pdf:
        pdf_path.write_bytes(pdf_bytes)
    # Some PDFs yield lone UTF-16 surrogates from broken CMap entries; encode
    # with errors="replace" so the write never blows up on \udxxx codepoints.
    try:
        text_path.write_bytes(text.encode("utf-8", errors="replace"))
    except OSError as e:
        return Result(
            short, "extract_error", source_url=used_url, http_code=200,
            pdf_bytes=len(pdf_bytes), error=f"write: {e}",
            elapsed_s=time.time() - t0,
        )
    return Result(
        short, "ok", source_url=used_url, http_code=200,
        pdf_bytes=len(pdf_bytes), text_chars=len(text),
        elapsed_s=time.time() - t0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--works", type=Path,
        default=Path(__file__).parent / "openalex_dh_works.json",
        help="Path to the previously fetched OpenAlex DH works JSON.",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).parent / "openalex_dh_fulltext",
        help="Output directory (pdfs/, text/, manifest.json).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--only-with-fulltext", action="store_true",
        help="Only attempt works where has_fulltext=True (OpenAlex flag).",
    )
    parser.add_argument(
        "--only-oa", action="store_true",
        help="Only attempt works where open_access.is_oa=True.",
    )
    parser.add_argument(
        "--no-keep-pdf", action="store_true",
        help="Delete PDF after extracting text (saves disk).",
    )
    parser.add_argument(
        "--prefer-publisher", action="store_true",
        help="Try best_oa_location.pdf_url before the OpenAlex content mirror.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-fetch even works that already have non-empty .txt.",
    )
    parser.add_argument(
        "--redo-failed", action="store_true",
        help="Retry works whose previous manifest entry was a failure.",
    )
    parser.add_argument(
        "--year-min", type=int, default=None,
        help="Only attempt works with publication_year >= this value.",
    )
    parser.add_argument(
        "--year-max", type=int, default=None,
        help="Only attempt works with publication_year <= this value.",
    )
    parser.add_argument(
        "--require-pdf-url", action="store_true",
        help="Skip (don't record) works that have no PDF URL anywhere.",
    )
    parser.add_argument(
        "--env-file", type=Path,
        default=Path(__file__).parent / ".env",
    )
    args = parser.parse_args()

    if args.env_file.exists():
        load_dotenv(args.env_file, override=False)

    email = os.environ.get("OPENALEX_EMAIL")
    api_key = os.environ.get("OPENALEX_API_KEY")

    pdf_dir = args.out_dir / "pdfs"
    text_dir = args.out_dir / "text"
    manifest_path = args.out_dir / "manifest.json"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading works from {args.works} ...", file=sys.stderr)
    with args.works.open("r", encoding="utf-8") as f:
        works = json.load(f)
    print(f"  loaded {len(works)} works", file=sys.stderr)

    # Load existing manifest (resume).
    manifest: dict[str, dict] = {}
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
        except json.JSONDecodeError:
            manifest = {}
        print(f"  resumed manifest with {len(manifest)} entries", file=sys.stderr)

    # Filter / select work set.
    targets: list[dict] = []
    failed_statuses = {"http_error", "download_error", "extract_error", "no_pdf_url"}
    for w in works:
        if args.only_with_fulltext and not w.get("has_fulltext"):
            continue
        if args.only_oa and not (w.get("open_access") or {}).get("is_oa"):
            continue
        y = w.get("publication_year")
        if args.year_min is not None and (not isinstance(y, int) or y < args.year_min):
            continue
        if args.year_max is not None and (not isinstance(y, int) or y > args.year_max):
            continue
        if args.require_pdf_url and not candidate_pdf_urls(w, True):
            continue
        short = work_short_id(w)
        if not short:
            continue
        existing = manifest.get(short)
        text_path = text_dir / f"{short}.txt"
        if (
            not args.overwrite
            and existing
            and existing.get("status") == "ok"
            and text_path.exists()
        ):
            continue
        if existing and existing.get("status") in failed_statuses and not args.redo_failed:
            continue
        targets.append(w)
        if args.limit and len(targets) >= args.limit:
            break

    print(
        f"Targeting {len(targets)} works "
        f"(workers={args.workers}, keep_pdf={not args.no_keep_pdf}, "
        f"api_key={'set' if api_key else 'unset'}, email={'set' if email else 'unset'})",
        file=sys.stderr,
    )

    session = requests.Session()
    ua = f"openalex-dh-fulltext/1.0"
    if email:
        ua += f" (mailto:{email})"
    session.headers.update({"User-Agent": ua, "Accept": "application/pdf,*/*"})

    counts = {"ok": 0, "skip": 0, "fail": 0}
    lock = Lock()
    t_start = time.time()
    flush_every = 100
    processed = 0

    def save_manifest() -> None:
        tmp = manifest_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        tmp.replace(manifest_path)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_one,
                w, pdf_dir, text_dir, session, api_key,
                not args.prefer_publisher,
                not args.no_keep_pdf,
                args.timeout,
                args.overwrite,
            ): w
            for w in targets
        }
        for fut in as_completed(futures):
            w = futures[fut]
            try:
                res: Result = fut.result()
            except Exception as e:  # noqa: BLE001 - never let one bad work kill the pool
                short = work_short_id(w) or "?"
                res = Result(
                    short, "extract_error",
                    error=f"worker {type(e).__name__}: {e}",
                )
            with lock:
                manifest[res.work_id] = {
                    "status": res.status,
                    "source_url": res.source_url,
                    "http_code": res.http_code,
                    "pdf_bytes": res.pdf_bytes,
                    "text_chars": res.text_chars,
                    "error": res.error,
                    "elapsed_s": round(res.elapsed_s, 2),
                }
                if res.status == "ok":
                    counts["ok"] += 1
                elif res.status == "skip":
                    counts["skip"] += 1
                else:
                    counts["fail"] += 1
                processed += 1
                if processed % 25 == 0 or processed == len(targets):
                    rate = processed / max(time.time() - t_start, 1e-3)
                    print(
                        f"  [{processed}/{len(targets)}] "
                        f"ok={counts['ok']} skip={counts['skip']} fail={counts['fail']} "
                        f"({rate:.1f} works/s)",
                        file=sys.stderr,
                    )
                if processed % flush_every == 0:
                    save_manifest()

    save_manifest()
    print(
        f"Done. ok={counts['ok']} skip={counts['skip']} fail={counts['fail']} "
        f"in {time.time() - t_start:.1f}s. Manifest: {manifest_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
