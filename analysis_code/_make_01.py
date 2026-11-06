"""Generate analysis_code/01_macro_openalex.ipynb."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from _nbbuild import write

HERE = Path(__file__).parent

cells = [
    ("md", """# 01 — Macro analysis of the OpenAlex DH corpus

Goal: characterise the full collection and its biases (publication years, work types, OA status, language, country, venues, concepts) and produce a per-year frequency series for AI-related terms in **abstracts**, alongside `machine learning` / `deep learning` comparators.

Inputs:
- `openalex_data/openalex_dh_works.json` (~470 MB, streamed with `ijson`).

Outputs:
- `outputs/macro/works_metadata.parquet` — flat metadata DataFrame.
- `outputs/macro/term_counts_by_year.parquet` — per-year AI / ML / DL counts.
- Inline figures (bias profile + diachronic curves).
"""),
    ("code", """from __future__ import annotations

import sys
from pathlib import Path

# allow importing analysis_code/lib from the notebook
ROOT = Path.cwd()
while ROOT.name and not (ROOT / "analysis_code").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "analysis_code"))

import ijson
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from lib import (
    OPENALEX_JSON, OUT_MACRO,
)
from lib.ai_lexicon import (
    AI_GROUPS,
    classify_text,
    compile_comparator_regex,
)
from lib.text_clean import reconstruct_abstract_from_inverted_index

sns.set_theme(context="notebook", style="whitegrid")
print("openalex json:", OPENALEX_JSON, OPENALEX_JSON.exists())
print("size MB:", round(OPENALEX_JSON.stat().st_size / 1e6, 1))
"""),
    ("md", "## 1. Stream-parse JSON → flat DataFrame"),
    ("code", """def _flatten_work(w: dict) -> dict:
    pl = w.get("primary_location") or {}
    src = (pl.get("source") or {}) if isinstance(pl, dict) else {}
    oa = w.get("open_access") or {}
    auths = w.get("authorships") or []
    countries: list[str] = []
    for a in auths:
        for c in a.get("countries", []) or []:
            countries.append(c)
    primary_topic = (w.get("primary_topic") or {})
    field = (primary_topic.get("field") or {}).get("display_name") if primary_topic else None
    subfield = (primary_topic.get("subfield") or {}).get("display_name") if primary_topic else None

    abstract = w.get("abstract")
    if not abstract:
        abstract = reconstruct_abstract_from_inverted_index(w.get("abstract_inverted_index"))

    wid = (w.get("id") or "").rsplit("/", 1)[-1]
    return {
        "id": wid,
        "doi": w.get("doi"),
        "title": w.get("title") or w.get("display_name"),
        "publication_year": w.get("publication_year"),
        "publication_date": w.get("publication_date"),
        "type": w.get("type"),
        "language": w.get("language"),
        "is_oa": oa.get("is_oa"),
        "oa_status": oa.get("oa_status"),
        "venue": src.get("display_name"),
        "venue_id": src.get("id"),
        "venue_type": src.get("type"),
        "host_organization": src.get("host_organization_name"),
        "n_authors": len(auths),
        "first_country": countries[0] if countries else None,
        "all_countries": "|".join(sorted(set(countries))) if countries else None,
        "primary_topic": primary_topic.get("display_name") if primary_topic else None,
        "primary_field": field,
        "primary_subfield": subfield,
        "concepts": "|".join(
            (c.get("display_name") or "")
            for c in (w.get("concepts") or [])[:5]
        ),
        "cited_by_count": w.get("cited_by_count"),
        "has_fulltext": w.get("has_fulltext"),
        "is_retracted": w.get("is_retracted"),
        "is_paratext": w.get("is_paratext"),
        "abstract": abstract,
    }


rows: list[dict] = []
with open(OPENALEX_JSON, "rb") as f:
    for w in tqdm(ijson.items(f, "item"), desc="works"):
        rows.append(_flatten_work(w))
df = pd.DataFrame(rows)
print(df.shape)
df.head(3)
"""),
    ("code", """# Quick sanity
assert df["id"].is_unique, "duplicate work IDs in JSON"
print("n works:", len(df))
print("n with abstract:", df["abstract"].notna().sum())
print("year range:", int(df["publication_year"].min()), "→", int(df["publication_year"].max()))

out_meta = OUT_MACRO / "works_metadata.parquet"
df.to_parquet(out_meta, index=False)
print("saved", out_meta)
"""),
    ("md", "## 2. Bias profile"),
    ("code", """def _bar(series: pd.Series, title: str, top: int | None = 15, ax=None):
    s = series.value_counts(dropna=False).head(top) if top else series.value_counts(dropna=False)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(s))))
    s.iloc[::-1].plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("works")
    return ax


# Year distribution
fig, ax = plt.subplots(figsize=(10, 3.5))
df["publication_year"].dropna().astype(int).value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_title("Works per publication year")
ax.set_xlabel("year"); ax.set_ylabel("works")
plt.tight_layout(); plt.show()
"""),
    ("code", """fig, axes = plt.subplots(2, 2, figsize=(14, 10))
_bar(df["type"], "Work type", ax=axes[0, 0])
_bar(df["oa_status"], "Open access status", ax=axes[0, 1])
_bar(df["language"], "Language", ax=axes[1, 0])
_bar(df["first_country"], "First-author country (top 15)", top=15, ax=axes[1, 1])
plt.tight_layout(); plt.show()
"""),
    ("code", """fig, axes = plt.subplots(1, 2, figsize=(15, 6))
_bar(df["venue"], "Top venues", top=20, ax=axes[0])
_bar(df["primary_field"], "Primary OpenAlex field", top=15, ax=axes[1])
plt.tight_layout(); plt.show()
"""),
    ("code", """# Top concepts (flatten the 'concepts' pipe-joined column)
concepts = (
    df["concepts"].dropna().str.split("|").explode().str.strip()
)
concepts = concepts[concepts != ""]
fig, ax = plt.subplots(figsize=(8, 6))
concepts.value_counts().head(20).iloc[::-1].plot(kind="barh", ax=ax)
ax.set_title("Top 20 OpenAlex concepts (across top-5 per work)")
ax.set_xlabel("works")
plt.tight_layout(); plt.show()
"""),
    ("md", """## 3. Term-frequency in abstracts, by year (grouped)

For each abstract we count non-overlapping hits per AI group from `AI_GROUPS`,
plus comparators (`machine learning`, `deep learning`).

**Overlap policy.** A sentence like *"using generative AI and an AI agent"*
yields `generative_ai=1, agent=1, general_ai=0`. The bare `general_ai` bucket
only catches free-standing "AI" / "artificial intelligence" / "AGI" tokens
that are not absorbed by a more specific group."""),
    ("code", """GROUP_COLS = [f"n_{g}" for g in AI_GROUPS]
cmp_re = compile_comparator_regex()

import re as _re
_ML_RE = _re.compile(r"\\bmachine\\s+learning\\b", flags=_re.IGNORECASE)
_DL_RE = _re.compile(r"\\bdeep\\s+learning\\b", flags=_re.IGNORECASE)


def _mention_row(text: str | None) -> dict:
    counts = classify_text(text)
    row = {f"n_{g}": counts[g] for g in AI_GROUPS}
    row["n_ai_total"] = sum(counts.values())
    row["has_ai"] = row["n_ai_total"] > 0
    if text:
        row["n_ml"] = len(_ML_RE.findall(text))
        row["n_dl"] = len(_DL_RE.findall(text))
    else:
        row["n_ml"] = row["n_dl"] = 0
    return row


flags = pd.DataFrame(df["abstract"].map(_mention_row).tolist())
df_terms = pd.concat([df[["id", "publication_year"]], flags], axis=1)
df_terms.head()
"""),
    ("code", """# Per-year aggregation: count of works whose abstract contains >=1 hit in each bucket.
has_cols = {f"has_{g}": (df_terms[f"n_{g}"] > 0) for g in AI_GROUPS}
has_df = pd.DataFrame(has_cols)
df_terms_b = pd.concat([df_terms, has_df], axis=1)

agg = {"n_works": ("id", "count"), "n_ai": ("has_ai", "sum"),
       "n_ml": ("n_ml", "sum"), "n_dl": ("n_dl", "sum")}
for g in AI_GROUPS:
    agg[f"n_{g}"] = (f"has_{g}", "sum")           # works whose abstract has >=1
    agg[f"hits_{g}"] = (f"n_{g}", "sum")          # total hit count

by_year = (
    df_terms_b.dropna(subset=["publication_year"])
    .assign(publication_year=lambda d: d["publication_year"].astype(int))
    .groupby("publication_year").agg(**agg).reset_index()
)
for g in AI_GROUPS:
    by_year[f"share_{g}"] = by_year[f"n_{g}"] / by_year["n_works"]
by_year["share_ai"] = by_year["n_ai"] / by_year["n_works"]

out_terms = OUT_MACRO / "term_counts_by_year.parquet"
by_year.to_parquet(out_terms, index=False)
by_year.tail(10)
"""),
    ("md", """### 3.1 Mentions per group over time (line plot)

The primary diachronic view: total **hit count** per AI group per publication
year. Each group has a distinct colour; the encoder/decoder/image-gen groups
plus `generative_ai`, `llm`, `chatbot`, `agent`, `general_ai` are all on the
same axes so their relative weight is directly comparable.

We show two panels:
* **all years** with a log y-axis (handles the sparse pre-2018 tail and the
  2024-2025 spike on the same chart);
* **2015 onward** with a linear y-axis to read off the recent absolute volume.
"""),
    ("code", """# Distinct color + marker per group (8 groups -> tab10 cycle).
GROUP_LIST = list(AI_GROUPS)
_PALETTE = plt.get_cmap("tab10").colors
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
GROUP_STYLE = {g: {"color": _PALETTE[i % 10], "marker": _MARKERS[i % len(_MARKERS)]}
               for i, g in enumerate(GROUP_LIST)}


def _plot_group_lines(ax, sub, ycol_fmt="hits_{}", **kwargs):
    for g in GROUP_LIST:
        ax.plot(sub["publication_year"], sub[ycol_fmt.format(g)],
                label=g, linewidth=1.6, markersize=4.5,
                **GROUP_STYLE[g], **kwargs)


fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

sub_all = by_year[by_year["publication_year"] >= 1990].sort_values("publication_year")
_plot_group_lines(axes[0], sub_all)
axes[0].set_yscale("log")
axes[0].set_title("AI-group mentions in abstracts per year (all years, log scale)")
axes[0].set_xlabel("publication year"); axes[0].set_ylabel("mentions (hits, log)")
axes[0].grid(True, which="both", linestyle=":", alpha=0.4)
axes[0].legend(fontsize=8, loc="upper left", ncol=2, frameon=True)

sub_recent = by_year[by_year["publication_year"] >= 2015].sort_values("publication_year")
_plot_group_lines(axes[1], sub_recent)
axes[1].set_title("AI-group mentions in abstracts per year (2015+, linear)")
axes[1].set_xlabel("publication year"); axes[1].set_ylabel("mentions (hits)")
axes[1].grid(True, linestyle=":", alpha=0.4)
axes[1].legend(fontsize=8, loc="upper left", ncol=2, frameon=True)

plt.tight_layout(); plt.show()
"""),
    ("md", """### 3.2 Share of works per group + ML/DL comparators (percentages)

Same groups, but normalised to the per-year work count (so the y-axis is a
**percentage of the works published that year** that contain at least one
hit from the group). The `machine learning` / `deep learning` comparator
curves are overlaid in black.

Two panels:
* **all years**, linear y-axis as %;
* **2015+ zoom**, linear y-axis as %, to read off the recent surge."""),
    ("code", """from matplotlib.ticker import PercentFormatter

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

sub_all = by_year[by_year["publication_year"] >= 1990].sort_values("publication_year")
_plot_group_lines(axes[0], sub_all, ycol_fmt="share_{}")
axes[0].plot(sub_all["publication_year"], sub_all["n_ml"] / sub_all["n_works"],
             marker="x", linestyle="--", color="black", linewidth=1.2,
             label="machine learning")
axes[0].plot(sub_all["publication_year"], sub_all["n_dl"] / sub_all["n_works"],
             marker="x", linestyle=":", color="dimgray", linewidth=1.2,
             label="deep learning")
axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
axes[0].set_title("Share of works per year with >=1 hit, by group (all years)")
axes[0].set_xlabel("publication year"); axes[0].set_ylabel("share of works")
axes[0].grid(True, linestyle=":", alpha=0.4)
axes[0].legend(fontsize=8, loc="upper left", ncol=2, frameon=True)

sub_recent = by_year[by_year["publication_year"] >= 2015].sort_values("publication_year")
_plot_group_lines(axes[1], sub_recent, ycol_fmt="share_{}")
axes[1].plot(sub_recent["publication_year"], sub_recent["n_ml"] / sub_recent["n_works"],
             marker="x", linestyle="--", color="black", linewidth=1.2,
             label="machine learning")
axes[1].plot(sub_recent["publication_year"], sub_recent["n_dl"] / sub_recent["n_works"],
             marker="x", linestyle=":", color="dimgray", linewidth=1.2,
             label="deep learning")
axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
axes[1].set_title("Share of works per year with >=1 hit, by group (2015+)")
axes[1].set_xlabel("publication year"); axes[1].set_ylabel("share of works")
axes[1].grid(True, linestyle=":", alpha=0.4)
axes[1].legend(fontsize=8, loc="upper left", ncol=2, frameon=True)

plt.tight_layout(); plt.show()
"""),
    ("md", """### 3.3 Composition of mentions (stacked, 2015+)

Stacked-bar view: of all AI-group mentions in a given year, what fraction is
contributed by each group? Useful for spotting the 2023+ shift from
encoder-style transformers (BERT/T5) toward decoder-only LLMs and
`generative_ai`/`llm` vocabulary."""),
    ("code", """recent = by_year[by_year["publication_year"] >= 2015].set_index("publication_year")
stack_cols = [f"hits_{g}" for g in GROUP_LIST]
colors = [GROUP_STYLE[g]["color"] for g in GROUP_LIST]
ax = recent[stack_cols].plot(kind="bar", stacked=True, figsize=(13, 4.5),
                              color=colors)
ax.set_title("Total AI-term hits in abstracts per year, by group (stacked)")
ax.set_ylabel("hits"); ax.set_xlabel("year")
ax.legend([c.replace("hits_", "") for c in stack_cols],
          fontsize=8, loc="upper left", ncol=2)
plt.tight_layout(); plt.show()
"""),
    ("md", """## 4. Headline numbers

A small summary printed for the write-up."""),
    ("code", """summary = {
    "total_works": int(len(df)),
    "with_abstract": int(df["abstract"].notna().sum()),
    "share_with_abstract": round(df["abstract"].notna().mean(), 3),
    "oa_share": round(df["is_oa"].mean(), 3) if df["is_oa"].notna().any() else None,
    "english_share": round((df["language"] == "en").mean(), 3),
    "year_min": int(df["publication_year"].min()),
    "year_max": int(df["publication_year"].max()),
    "ai_mentioning_abstracts": int(df_terms["has_ai"].sum()),
    "ml_mentioning_abstracts": int((df_terms["n_ml"] > 0).sum()),
    "dl_mentioning_abstracts": int((df_terms["n_dl"] > 0).sum()),
    "by_group_works": {g: int((df_terms[f"n_{g}"] > 0).sum()) for g in AI_GROUPS},
    "by_group_hits":  {g: int(df_terms[f"n_{g}"].sum()) for g in AI_GROUPS},
}
summary
"""),
]

write(HERE / "01_macro_openalex.ipynb", cells)
