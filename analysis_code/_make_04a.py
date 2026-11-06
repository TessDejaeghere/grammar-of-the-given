"""Generate 04a_agency.ipynb — grammatical agency of AI mentions."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _nbbuild import write

HERE = Path(__file__).parent

cells = [
    ("md", """# 04a — Grammatical agency of AI mentions

For every AI mention (head of the noun phrase that matched the lexicon), classify its **syntactic role** with respect to its governing verb:

| role | dependency cue |
|------|----------------|
| `active_subject`   | `nsubj` AND governor is not in passive voice |
| `passive_subject`  | `nsubjpass` (or `nsubj:pass`) |
| `direct_object`    | `dobj` / `obj` |
| `attribute`        | `attr` / `oprd` / `acomp` — Attribute/Value of a relational clause ("AI **is an umbrella term**") |
| `oblique_agent`    | head dep `pobj` whose governing preposition carries dep `agent` — the canonical "X is done **by AI**" pattern |
| `prepositional`    | other `pobj` / `pcomp` |
| `nominal_modifier` | `compound`, `amod`, `nmod`, `poss` |
| `other`            | anything else — largely `npadvmod` / `nummod` / `dep` parse noise from PDF furniture, plus `conj` mentions whose role is inherited from their conjunct |

Outputs: `outputs/analysis/04a_agency_per_mention.parquet`, role distribution plots, diachronic split.
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

ment = pd.read_parquet(OUT_PARSED / "mentions.parquet")
ai = ment[ment["category"] == "AI"].copy()
print("AI mentions:", len(ai))
ai.head()
"""),
    ("md", "## 1. Detect passive voice on the governing verb"),
    ("code", """# Heuristic: the governing verb is "passive" if the head dep was nsubjpass,
# or if the governor's morph contains VerbForm=Part + Voice=Pass, or there's
# a 'by'-agent dependency hanging off it. For per-mention classification we
# only have head_dep + governor info; passivity is recovered from:
#   (a) head_dep ∈ {nsubjpass, nsubj:pass, csubjpass} → passive subject
#   (b) head_dep == 'pobj' AND the PP preposition lemma is 'by' → look up
#       sentence DocBins to confirm governor is passive.
# To stay lightweight, classify roles purely from token-level info already
# stored in mentions.parquet (shared with 05d/06a via lib.roles).
from lib.roles import classify_role

ai["role"] = ai.apply(classify_role, axis=1)
ai["role"].value_counts()
"""),
    ("md", "## 2. Aggregate + per-document normalisation"),
    ("code", """role_counts = ai["role"].value_counts().reindex(
    ["active_subject", "passive_subject", "direct_object", "attribute",
     "oblique_agent", "prepositional", "nominal_modifier", "other"],
    fill_value=0,
)
role_share = (role_counts / role_counts.sum()).round(3)
pd.DataFrame({"n": role_counts, "share": role_share})
"""),
    ("code", """fig, ax = plt.subplots(figsize=(9, 4))
role_counts.plot(kind="bar", ax=ax)
ax.set_title("Grammatical role of AI mentions (whole AI corpus)")
ax.set_ylabel("mentions"); ax.set_xlabel("role")
plt.xticks(rotation=30, ha="right")
plt.tight_layout(); plt.show()
"""),
    ("md", "## 3. Diachronic split"),
    ("code", """idx = pd.read_parquet(OUT_CLEAN / "index.parquet")[["id", "publication_year"]]
ai_yr = ai.merge(idx.rename(columns={"id": "work_id"}), on="work_id", how="left")
year_role = (
    pd.crosstab(ai_yr["publication_year"], ai_yr["role"], normalize="index")
    .round(3)
)
year_role
"""),
    ("code", """ax = year_role.plot(kind="bar", stacked=True, figsize=(9, 4), colormap="tab10")
ax.set_title("Share of AI-mention grammatical roles by year")
ax.set_ylabel("share"); ax.set_xlabel("publication year")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(); plt.show()
"""),
    ("md", "## 4. Persist + sample sentences per role"),
    ("code", """OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
ai_yr.to_parquet(OUT_ANALYSIS / "04a_agency_per_mention.parquet", index=False)
print("saved", OUT_ANALYSIS / "04a_agency_per_mention.parquet")

# Sample 5 sentences per role for inspection
for r, g in ai_yr.groupby("role"):
    print(f"\\n=== role: {r}  (n={len(g)}) ===")
    for _, row in g.sample(min(5, len(g)), random_state=0).iterrows():
        print(f"- [{row['work_id']}] head='{row['head_text']}' gov='{row['governor_lemma']}' :: {row['sent_text'][:200]}")
"""),
]

write(HERE / "04a_agency.ipynb", cells)
