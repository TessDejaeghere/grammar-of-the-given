"""Generate 05b_kwic.ipynb — KWIC concordance utility."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _nbbuild import write

HERE = Path(__file__).parent

cells = [
    ("md", """# 05b — KWIC concordance utility

Quick query interface over the cleaned AI corpus: given a regex pattern, return Key-Word-In-Context lines with ±N characters of context plus document metadata. Export to CSV for qualitative reading.
"""),
    ("code", """from __future__ import annotations
import sys, re
from pathlib import Path

ROOT = Path.cwd()
while ROOT.name and not (ROOT / "analysis_code").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "analysis_code"))

import pandas as pd
from tqdm.auto import tqdm
from lib import OUT_CLEAN, OUT_ANALYSIS

idx = pd.read_parquet(OUT_CLEAN / "index.parquet")
idx_ai = idx[idx["in_ai_corpus"]]
print("AI corpus docs:", len(idx_ai))


def kwic(pattern: str, window: int = 80, flags: int = re.I, sample: int | None = None,
         restrict_ids: list[str] | None = None) -> pd.DataFrame:
    rx = re.compile(pattern, flags)
    rows = []
    ids = restrict_ids if restrict_ids is not None else idx_ai["id"].tolist()
    for wid in tqdm(ids, desc="kwic"):
        text = (OUT_CLEAN / f"{wid}.txt").read_text(encoding="utf-8", errors="replace")
        for m in rx.finditer(text):
            s, e = m.start(), m.end()
            left = text[max(0, s - window):s].replace("\\n", " ")
            keyw = text[s:e]
            right = text[e:e + window].replace("\\n", " ")
            rows.append({
                "work_id": wid,
                "left": left,
                "match": keyw,
                "right": right,
                "offset": s,
            })
        if sample and len(rows) >= sample:
            break
    df = pd.DataFrame(rows)
    return df.merge(
        idx_ai[["id", "publication_year", "venue", "primary_field"]]
        .rename(columns={"id": "work_id"}),
        on="work_id", how="left",
    )
"""),
    ("md", "## Example queries"),
    ("code", """# 1. AI hallucinations
h = kwic(r"\\bhallucinat\\w+\\b", window=70)
print("matches:", len(h))
h.head(10)
"""),
    ("code", """# 2. AI agency verbs — 'understand', 'know', 'think'
verbs = kwic(r"\\b(?:GPT|ChatGPT|LLM|LLMs|AI|artificial intelligence)\\b[^.]{0,40}\\b(?:understand|think|know|believe|decide|hallucinate)s?\\b", window=70)
print("matches:", len(verbs))
verbs.head(10)
"""),
    ("code", """# 3. 'use of' nominalisations
use = kwic(r"\\b(use|deployment|adoption|integration|application)\\s+of\\s+(?:AI|ChatGPT|GPT|LLM|LLMs|generative\\s+AI|large\\s+language\\s+models?)\\b", window=70)
print("matches:", len(use))
use.head(10)
"""),
    ("md", "## Save the latest concordance to CSV"),
    ("code", """OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
out = OUT_ANALYSIS / "05b_kwic_last.csv"
use.to_csv(out, index=False, encoding="utf-8-sig")
print("saved", out)
"""),
]

write(HERE / "05b_kwic.ipynb", cells)
