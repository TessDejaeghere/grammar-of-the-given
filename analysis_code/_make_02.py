"""Generate 02_fulltext_clean_filter.ipynb."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _nbbuild import write

HERE = Path(__file__).parent

cells = [
    ("md", """# 02 — Clean & filter the post-2022 fulltext corpus

Goal: take the ~1,300 retrieved .txt files, clean them, drop short / non-English documents, count AI mentions, and select an `ai_corpus` for fine-grained parsing in Phase 3.

Inputs:
- `outputs/macro/works_metadata.parquet` (from Phase 1)
- `post_2022_fulltext/text/*.txt`
- `post_2022_fulltext/manifest.json`

Outputs:
- `outputs/clean/<id>.txt` — cleaned fulltexts kept after filtering
- `outputs/clean/index.parquet` — per-document index (lengths, language, mention counts, in_ai_corpus flag)
- Inline figures: mention-count histogram, year coverage, representativeness comparison.
"""),
    ("code", """from __future__ import annotations
import sys, json, re
from pathlib import Path

ROOT = Path.cwd()
while ROOT.name and not (ROOT / "analysis_code").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "analysis_code"))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from lib import FULLTEXT_DIR, FULLTEXT_MANIFEST, OUT_MACRO, OUT_CLEAN
from lib.text_clean import clean_text, approx_tokens, detect_language
from lib.ai_lexicon import compile_ai_regex

sns.set_theme(context="notebook", style="whitegrid")
AI_MIN_MENTIONS = 3   # threshold for `ai_corpus` membership

print("text dir:", FULLTEXT_DIR, "exists:", FULLTEXT_DIR.exists())
print("manifest:", FULLTEXT_MANIFEST, "exists:", FULLTEXT_MANIFEST.exists())
"""),
    ("md", "## 1. Join fulltext filenames with metadata + manifest"),
    ("code", """meta = pd.read_parquet(OUT_MACRO / "works_metadata.parquet")
print("meta rows:", len(meta))

manifest = json.loads(FULLTEXT_MANIFEST.read_text(encoding="utf-8"))
manifest_df = (
    pd.DataFrame.from_dict(manifest, orient="index")
    .reset_index().rename(columns={"index": "id"})
)
print("manifest rows:", len(manifest_df))

txt_ids = {p.stem for p in FULLTEXT_DIR.glob("*.txt")}
print("txt files on disk:", len(txt_ids))

base = (
    meta.merge(manifest_df, on="id", how="inner")
        .assign(has_txt=lambda d: d["id"].isin(txt_ids))
)
print("with txt:", base["has_txt"].sum(), "/", len(base))
base = base[base["has_txt"]].copy()
"""),
    ("code", """# Strict year filter
base = base[base["publication_year"] >= 2022].copy()
print("post-2022 with txt:", len(base))
base["publication_year"].value_counts().sort_index()
"""),
    ("md", "## 2. Clean + language-detect + token-count each document"),
    ("code", """ai_re = compile_ai_regex()

records = []
for wid in tqdm(base["id"].tolist(), desc="clean"):
    raw = (FULLTEXT_DIR / f"{wid}.txt").read_text(encoding="utf-8", errors="replace")
    cleaned = clean_text(raw)
    n_tok = approx_tokens(cleaned)
    lang = detect_language(cleaned)
    ai_hits = ai_re.findall(cleaned) if cleaned else []
    n_ai = len(ai_hits)
    records.append({
        "id": wid,
        "raw_chars": len(raw),
        "clean_chars": len(cleaned),
        "n_tokens_approx": n_tok,
        "language_detected": lang,
        "n_ai_mentions": n_ai,
        "cleaned_text": cleaned,
    })

doc_df = pd.DataFrame(records)
doc_df.describe(include="all").T
"""),
    ("md", "## 3. Apply filters and pick the AI corpus"),
    ("code", """MIN_TOKENS = 500

doc_df["pass_lang"] = doc_df["language_detected"] == "en"
doc_df["pass_len"]  = doc_df["n_tokens_approx"] >= MIN_TOKENS
doc_df["kept"]      = doc_df["pass_lang"] & doc_df["pass_len"]

print("language=en:", doc_df["pass_lang"].sum())
print("tokens>=", MIN_TOKENS, ":", doc_df["pass_len"].sum())
print("kept (both):", doc_df["kept"].sum())

doc_df["in_ai_corpus"] = doc_df["kept"] & (doc_df["n_ai_mentions"] >= AI_MIN_MENTIONS)
print(f"AI corpus (>= {AI_MIN_MENTIONS} mentions):", doc_df["in_ai_corpus"].sum())
"""),
    ("code", """# Mention-count histogram (kept docs only)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
kept = doc_df[doc_df["kept"]]
axes[0].hist(kept["n_ai_mentions"].clip(upper=50), bins=50)
axes[0].axvline(AI_MIN_MENTIONS, color="red", ls="--", label=f"threshold={AI_MIN_MENTIONS}")
axes[0].set_title("AI mentions per kept document (clipped at 50)")
axes[0].set_xlabel("# AI mentions"); axes[0].set_ylabel("docs"); axes[0].legend()

axes[1].hist(kept["n_tokens_approx"].clip(upper=30000), bins=50)
axes[1].axvline(MIN_TOKENS, color="red", ls="--")
axes[1].set_title("Token count per kept document (clipped at 30k)")
axes[1].set_xlabel("tokens"); axes[1].set_ylabel("docs")
plt.tight_layout(); plt.show()
"""),
    ("md", "## 4. Write cleaned files + index"),
    ("code", """# Write only kept (post-filter) docs to disk; mark ai_corpus in index
written = 0
for r in tqdm(doc_df[doc_df["kept"]].itertuples(index=False), total=int(doc_df["kept"].sum()), desc="write"):
    (OUT_CLEAN / f"{r.id}.txt").write_text(r.cleaned_text, encoding="utf-8")
    written += 1
print("written:", written)

index_cols = [
    "id", "raw_chars", "clean_chars", "n_tokens_approx",
    "language_detected", "n_ai_mentions",
    "pass_lang", "pass_len", "kept", "in_ai_corpus",
]
index = doc_df[index_cols].merge(
    base[["id", "publication_year", "type", "venue", "language",
          "primary_field", "first_country", "is_oa", "oa_status"]],
    on="id", how="left",
)
out_idx = OUT_CLEAN / "index.parquet"
index.to_parquet(out_idx, index=False)
print("saved", out_idx, "rows:", len(index))
index.head()
"""),
    ("md", "## 5. Representativeness check — `ai_corpus` vs. all kept post-2022"),
    ("code", """def _share(df: pd.DataFrame, col: str, top: int = 10) -> pd.Series:
    return df[col].value_counts(normalize=True, dropna=False).head(top)


kept_idx = index[index["kept"]]
ai_idx = index[index["in_ai_corpus"]]
print(f"kept docs: {len(kept_idx)}    ai_corpus: {len(ai_idx)}")

for col in ["publication_year", "type", "primary_field", "first_country", "venue"]:
    print(f"\\n=== {col} ===")
    comp = pd.concat(
        [_share(kept_idx, col).rename("kept"),
         _share(ai_idx, col).rename("ai_corpus")],
        axis=1,
    ).fillna(0).sort_values("kept", ascending=False).head(10)
    print(comp.round(3))
"""),
    ("code", """# Diachronic coverage
fig, ax = plt.subplots(figsize=(8, 3.5))
yr = (
    pd.crosstab(kept_idx["publication_year"], kept_idx["in_ai_corpus"])
    .rename(columns={False: "kept (non-AI)", True: "ai_corpus"})
)
yr.plot(kind="bar", stacked=True, ax=ax)
ax.set_title("Post-2022 fulltext docs by year (stacked: AI corpus vs rest)")
ax.set_ylabel("docs"); ax.set_xlabel("year")
plt.tight_layout(); plt.show()
"""),
    ("code", """summary = {
    "txt_files_total": int(len(base) + 0),
    "post_2022_with_txt": int(len(base)),
    "kept_after_filters": int(doc_df["kept"].sum()),
    "ai_corpus_size": int(doc_df["in_ai_corpus"].sum()),
    "ai_corpus_threshold_mentions": AI_MIN_MENTIONS,
    "median_tokens_kept": int(kept["n_tokens_approx"].median()),
    "median_ai_mentions_ai_corpus": int(ai_idx["n_ai_mentions"].median()) if len(ai_idx) else None,
}
summary
"""),
]

write(HERE / "02_fulltext_clean_filter.ipynb", cells)
