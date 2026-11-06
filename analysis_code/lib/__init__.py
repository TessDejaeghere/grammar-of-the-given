"""Shared helpers for the CDA-of-AI-agency-in-DH project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPENALEX_JSON = ROOT / "openalex_data" / "openalex_dh_works.json"
FULLTEXT_DIR = ROOT / "post_2022_fulltext" / "text"
FULLTEXT_MANIFEST = ROOT / "post_2022_fulltext" / "manifest.json"
OUTPUTS = ROOT / "outputs"
OUT_MACRO = OUTPUTS / "macro"
OUT_CLEAN = OUTPUTS / "clean"
OUT_PARSED = OUTPUTS / "parsed"
OUT_ANALYSIS = OUTPUTS / "analysis"

for _p in (OUT_MACRO, OUT_CLEAN, OUT_PARSED, OUT_ANALYSIS):
    _p.mkdir(parents=True, exist_ok=True)
