"""Generate 04d_nominalization.ipynb — nominalization & actor deletion."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _nbbuild import write

HERE = Path(__file__).parent

cells = [
    ("md", """# 04d — Nominalization & actor-deletion around AI

We hunt for *nominalised AI actions* — phrases like **"the deployment of LLMs"**, **"AI hallucination"**, **"the use of ChatGPT"** — and ask: *is a human/company agent realised in the same sentence?* If not, agency has been syntactically erased.

Detection patterns (run over parsed AI-relevant sentences):
1. **`Nom of AI`** — an AI mention with `head_dep == 'pobj'`, preposition lemma `of`, governor is a deverbal noun (suffixes `-tion`, `-ment`, `-sis`, `-ance`, `-ence`, `-ing`).
2. **`AI Nom`** — an AI mention with `head_dep ∈ {compound, amod, nmod}` whose governor is a deverbal noun.
3. **agentless `by` check** — also flag passive constructions with no `by`-agent.

For each detected nominalisation we then look in the same sentence for any HUMAN or COMPANY mention (from `mentions.parquet`); if none, mark `agent_present = False`.

Outputs:
- `outputs/analysis/04d_nominalizations.parquet`
- `outputs/analysis/04d_nominalizations_per_sentence.csv` — manual review sheet
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

NOMINAL_SUFFIXES = ("tion", "sion", "ment", "ance", "ence", "sis", "ing", "ure", "al")

ment = pd.read_parquet(OUT_PARSED / "mentions.parquet")
ai = ment[ment["category"] == "AI"].copy()
human = ment[ment["category"] == "HUMAN"]
company = ment[ment["category"] == "COMPANY"]
print("AI:", len(ai), " HUMAN:", len(human), " COMPANY:", len(company))
"""),
    ("md", "## 1. Tag candidate nominalisations"),
    ("code", """def _is_deverbal(word: str) -> bool:
    w = word.lower()
    return len(w) > 4 and any(w.endswith(s) for s in NOMINAL_SUFFIXES) and not w.endswith("ing")[:0]


def _classify(row) -> str:
    dep = row["head_dep"]
    gov_pos = row["governor_pos"]
    gov_lem = row["governor_lemma"]
    if gov_pos == "NOUN" and gov_lem.lower().endswith(NOMINAL_SUFFIXES):
        if dep == "pobj":
            return "of_AI"        # "deployment OF llms"  (gov is the noun)
        if dep in {"compound", "amod", "nmod", "poss"}:
            return "AI_Nom"       # "AI deployment"
    return ""


ai["nom_pattern"] = ai.apply(_classify, axis=1)
nom = ai[ai["nom_pattern"] != ""].copy()
print("nominalisations found:", len(nom))
nom["nom_pattern"].value_counts()
"""),
    ("md", "## 2. Is an agent (human or company) present in the same sentence?"),
    ("code", """actors = pd.concat([human, company], ignore_index=True)


def _agent_relation(row) -> str:
    candidates = actors[
        (actors["work_id"] == row["work_id"])
        & (actors["sent_id"] == row["sent_id"])
    ]
    nominal = str(row["governor_lemma"]).lower()
    for _, actor in candidates.iterrows():
        dep = actor["head_dep"]
        gov = str(actor["governor_lemma"]).lower()
        if dep == "poss" and gov == nominal:
            return "explicit_possessive"
        if dep == "pobj" and actor["governor_dep"] == "agent":
            return "explicit_by_agent"
        if dep in {"nsubj", "nsubjpass", "nsubj:pass"} and gov == nominal:
            return "explicit_subject"
    return "suppressed"


nom["agent_relation"] = nom.apply(_agent_relation, axis=1)
nom["agent_present"] = nom["agent_relation"].str.startswith("explicit")
print(nom["agent_present"].value_counts())
print("agent-effaced rate:", round(1 - nom["agent_present"].mean(), 3))
"""),
    ("md", "## 3. Aggregate by year"),
    ("code", """idx = pd.read_parquet(OUT_CLEAN / "index.parquet")[["id", "publication_year"]]
nom = nom.merge(idx.rename(columns={"id": "work_id"}), on="work_id", how="left")

yr = (
    nom.groupby("publication_year")
       .agg(n=("nom_pattern", "size"),
            agent_present_rate=("agent_present", "mean"))
)
yr["agent_effaced_rate"] = (1 - yr["agent_present_rate"]).round(3)
print(yr)

fig, ax = plt.subplots(figsize=(8, 3.5))
yr["agent_effaced_rate"].plot(kind="bar", ax=ax, color="#c44e52")
ax.set_title("Agent-effaced rate of AI-action nominalisations by year")
ax.set_ylabel("share without explicit agent")
plt.tight_layout(); plt.show()
"""),
    ("md", "## 4. Persist outputs"),
    ("code", """OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)

nom.to_parquet(OUT_ANALYSIS / "04d_nominalizations.parquet", index=False)
print("saved nominalizations.parquet rows:", len(nom))

# manual-review CSV
manual = nom[[
    "work_id", "sent_id", "publication_year",
    "nom_pattern", "mention_text", "governor_lemma",
    "agent_present", "sent_text",
]].copy()
manual["manual_pattern"] = ""    # confirm / reject / alternative
manual["manual_agent"] = ""      # name agent if implicit etc.
manual["notes"] = ""

out_csv = OUT_ANALYSIS / "04d_nominalizations_per_sentence.csv"
manual.to_csv(out_csv, index=False, encoding="utf-8-sig")
print("saved", out_csv)
manual.head(15)
"""),
]

write(HERE / "04d_nominalization.ipynb", cells)
