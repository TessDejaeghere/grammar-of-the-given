"""Cleaning helpers for OpenAlex-extracted fulltext.

Pipeline (in order):
  1. ftfy mojibake repair
  2. normalise line endings + remove form feeds
  3. dehyphenate words broken across line breaks ("inter-\nnationally")
  4. drop repeated header/footer lines (same line appearing on many pages)
  5. strip references section (heuristic)
  6. collapse runs of whitespace, normalise blank lines

Plus utilities:
  * detect_language(text) -> str | None
  * approx_tokens(text)   -> int   (whitespace tokens, cheap)
  * reconstruct_abstract_from_inverted_index(d) -> str | None
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import ftfy

# ---------------------------------------------------------------------------
# Low-level normalisation
# ---------------------------------------------------------------------------
_HYPHEN_BREAK = re.compile(r"(\w+)-\n(\w+)")
_MULTISPACE = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")
_FORM_FEED = re.compile(r"[\f\v]")


def _normalise(text: str) -> str:
    text = ftfy.fix_text(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _FORM_FEED.sub("\n", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return text


# ---------------------------------------------------------------------------
# Header / footer removal
# ---------------------------------------------------------------------------
def _strip_headers_footers(text: str, min_repeats: int = 5) -> str:
    """Drop lines that recur many times (page headers/footers, journal banners).

    Lines shorter than 4 chars or that look like pure page numbers are also
    dropped.
    """
    lines = text.split("\n")
    stripped = [ln.strip() for ln in lines]
    counts = Counter(s for s in stripped if len(s) >= 4)
    repeats = {s for s, c in counts.items() if c >= min_repeats}

    out = []
    for raw, s in zip(lines, stripped):
        if not s:
            out.append("")
            continue
        if s in repeats:
            continue
        # bare page numbers, e.g. "12", "p. 12", "Page 12"
        if re.fullmatch(r"(p\.?\s*)?(page\s*)?\d{1,4}", s, flags=re.I):
            continue
        out.append(raw)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# References-section stripping
# ---------------------------------------------------------------------------
_REF_HEADINGS = re.compile(
    r"^\s*(references|bibliography|works\s+cited|reference\s+list)\s*:?\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def _strip_references(text: str) -> str:
    """Cut text at the *last* References-like heading appearing in the second
    half of the document. Conservative — won't cut if no such heading found.
    """
    matches = list(_REF_HEADINGS.finditer(text))
    if not matches:
        return text
    # Only trust a heading that sits in the latter half of the document.
    half = len(text) // 2
    candidate = None
    for m in matches:
        if m.start() >= half:
            candidate = m
            break
    if candidate is None:
        return text
    return text[: candidate.start()].rstrip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = _normalise(text)
    text = _strip_headers_footers(text)
    text = _strip_references(text)
    text = _MULTISPACE.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def approx_tokens(text: str) -> int:
    return len(text.split())


def detect_language(text: str, sample_chars: int = 4000) -> str | None:
    """Return ISO-639-1 code, or None on failure."""
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0  # determinism
        snippet = text[:sample_chars]
        if len(snippet.split()) < 20:
            return None
        return detect(snippet)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# OpenAlex abstract reconstruction (inverted index → string)
# ---------------------------------------------------------------------------
def reconstruct_abstract_from_inverted_index(idx: dict | None) -> str | None:
    if not idx:
        return None
    positions: list[tuple[int, str]] = []
    for word, posns in idx.items():
        for p in posns:
            positions.append((p, word))
    if not positions:
        return None
    positions.sort()
    return " ".join(w for _, w in positions)


def iter_lines(text: str) -> Iterable[str]:
    yield from text.split("\n")
