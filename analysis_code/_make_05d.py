"""Generate 05d_ai_vs_human.ipynb — AI vs human-actor agency comparison."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _nbbuild import write

HERE = Path(__file__).parent

cells = [
    ("md", """# 05d — AI vs. human agency comparison

Re-run the Phase 4a role-classification on **human** actor mentions (we / researchers / scholars / users) in the *same* corpus of AI-mentioning documents, then plot the two side-by-side. Reveals where authors grant AI more or less syntactic agency than humans.
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
print(ment["category"].value_counts())
"""),
    ("md", "## 1. Apply the role classifier to all three actor categories"),
    ("code", """from lib.roles import classify_role, ROLES_ORDER

ment["role"] = ment.apply(classify_role, axis=1)
roles_order = ROLES_ORDER

comp = (
    ment.groupby(["category", "role"]).size().unstack(fill_value=0)
    .reindex(columns=roles_order, fill_value=0)
)
comp_share = comp.div(comp.sum(axis=1), axis=0).round(3)
print("Role share by actor category:")
print(comp_share)
"""),
    ("code", """fig, ax = plt.subplots(figsize=(10, 4))
comp_share.T.plot(kind="bar", ax=ax)
ax.set_title("Role distribution: AI vs HUMAN vs COMPANY")
ax.set_ylabel("share of mentions")
ax.legend(title="actor", loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.xticks(rotation=30, ha="right")
plt.tight_layout(); plt.show()
"""),
    ("md", "## 2. Top governing verbs per actor category (active subjects only)"),
    ("code", """active = ment[(ment["role"] == "active_subject") & ment["governor_pos"].isin(["VERB", "AUX"])]

top = (
    active.groupby("category")["governor_lemma"]
    .apply(lambda s: s.value_counts().head(15))
)
print(top)
"""),
    ("md", "## 3. Diachronic comparison"),
    ("code", """idx = pd.read_parquet(OUT_CLEAN / "index.parquet")[["id", "publication_year"]]
m2 = ment.merge(idx.rename(columns={"id": "work_id"}), on="work_id", how="left")

# Share of active-subject role per category, per year
act_share = (
    m2.assign(is_active=lambda d: d["role"] == "active_subject")
       .groupby(["publication_year", "category"])["is_active"].mean()
       .unstack()
       .round(3)
)
print("Active-subject rate by year and actor:")
print(act_share)

ax = act_share.plot(kind="bar", figsize=(9, 4))
ax.set_title("Active-subject share by actor category, by year")
ax.set_ylabel("share")
ax.legend(title="actor", loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(); plt.show()
"""),
    ("md", "## 4. Persist"),
    ("code", """OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
comp_share.to_parquet(OUT_ANALYSIS / "05d_role_share_by_actor.parquet")
act_share.to_parquet(OUT_ANALYSIS / "05d_active_share_by_year.parquet")
print("saved 05d outputs")
"""),
]

write(HERE / "05d_ai_vs_human.ipynb", cells)
