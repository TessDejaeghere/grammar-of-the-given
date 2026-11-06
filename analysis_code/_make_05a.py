"""Generate 05a_diachronic.ipynb — synthesise diachronic trends from 4a-4d."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _nbbuild import write

HERE = Path(__file__).parent

cells = [
    ("md", """# 05a — Diachronic synthesis (2022 → present)

Pulls together the year-by-year signals produced by Phases 4a–4d so they can be read at a glance:

- AI-mention density per kept document.
- Grammatical-role mix (active subject / passive subject / object / oblique / nominal modifier).
- Modality mix (strong / weak / bare / mixed).
- Verb-class mix (material / cognitive / instrumental / verbal / stative).
- Actor-effaced rate of nominalisations.
"""),
    ("code", """from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path.cwd()
while ROOT.name and not (ROOT / "analysis_code").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "analysis_code"))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lib import OUT_PARSED, OUT_ANALYSIS, OUT_CLEAN
sns.set_theme(context="notebook", style="whitegrid")

idx = pd.read_parquet(OUT_CLEAN / "index.parquet")
idx_ai = idx[idx["in_ai_corpus"]]
print("AI corpus docs:", len(idx_ai))
"""),
    ("md", "## 1. AI-mention density per doc, by year"),
    ("code", """density = (
    idx_ai.assign(
        density_per_1k=lambda d: 1000 * d["n_ai_mentions"] / d["n_tokens_approx"].clip(lower=1)
    )
    .groupby("publication_year")["density_per_1k"]
    .agg(["mean", "median", "count"])
)
print(density)

fig, ax = plt.subplots(figsize=(8, 3.5))
density["median"].plot(kind="bar", ax=ax, color="#4c72b0")
ax.set_title("Median AI-mention density (mentions / 1k tokens) by year")
ax.set_ylabel("median density")
plt.tight_layout(); plt.show()
"""),
    ("md", "## 2. Role mix (from 4a)"),
    ("code", """ag = pd.read_parquet(OUT_ANALYSIS / "04a_agency_per_mention.parquet")
yr_role = pd.crosstab(ag["publication_year"], ag["role"], normalize="index")
ax = yr_role.plot(kind="bar", stacked=True, figsize=(9, 4), colormap="tab10")
ax.set_title("Role of AI mentions by year")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
ax.set_ylabel("share")
plt.tight_layout(); plt.show()
"""),
    ("md", "## 3. Modality mix (from 4b)"),
    ("code", """try:
    mod = pd.read_csv(OUT_ANALYSIS / "04b_modality_per_sentence.csv")
    yr_mod = pd.crosstab(mod["publication_year"], mod["auto_label"], normalize="index")
    ax = yr_mod.plot(kind="bar", stacked=True, figsize=(9, 4), colormap="Set2")
    ax.set_title("Modality mix of AI-clause sentences by year")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_ylabel("share")
    plt.tight_layout(); plt.show()
except FileNotFoundError:
    print("Run 04b first")
"""),
    ("md", "## 4. Verb-class mix (from 4c)"),
    ("code", """try:
    active = pd.read_parquet(OUT_ANALYSIS / "04a_agency_per_mention.parquet")
    active = active[active["role"] == "active_subject"]
    summary = pd.read_parquet(OUT_ANALYSIS / "04c_verb_class_summary.parquet")
    # Re-derive per-year using the seed lexicon from 04c
    from analysis_code._make_04c import VERB_CLASSES  # noqa: E402
except Exception:
    VERB_CLASSES = None

if VERB_CLASSES is None:
    print("Run 04c first to get verb classes")
else:
    LEMMA2CLASS = {}
    for cls, lemmas in VERB_CLASSES.items():
        for lem in lemmas:
            LEMMA2CLASS.setdefault(lem, cls)
    active = active.copy()
    active["auto_class"] = active["governor_lemma"].map(lambda l: LEMMA2CLASS.get(l, "OTHER"))
    yr_v = pd.crosstab(active["publication_year"], active["auto_class"], normalize="index")
    ax = yr_v.plot(kind="bar", stacked=True, figsize=(9, 4), colormap="tab10")
    ax.set_title("Verb-class mix (auto) for AI-as-subject by year")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_ylabel("share")
    plt.tight_layout(); plt.show()
"""),
    ("md", "## 5. Actor-effaced rate (from 4d)"),
    ("code", """try:
    nom = pd.read_parquet(OUT_ANALYSIS / "04d_nominalizations.parquet")
    yr_n = nom.groupby("publication_year")["agent_present"].agg(["mean", "count"])
    yr_n["effaced_rate"] = (1 - yr_n["mean"]).round(3)
    print(yr_n)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    yr_n["effaced_rate"].plot(kind="bar", ax=ax, color="#c44e52")
    ax.set_title("Actor-effaced rate of AI nominalisations by year")
    ax.set_ylabel("share without explicit agent")
    plt.tight_layout(); plt.show()
except FileNotFoundError:
    print("Run 04d first")
"""),
]

write(HERE / "05a_diachronic.ipynb", cells)
