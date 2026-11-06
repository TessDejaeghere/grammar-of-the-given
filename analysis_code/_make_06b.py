"""Generate 06b_term_groups.ipynb — general 'AI' vs named models, grammatically."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _nbbuild import write

HERE = Path(__file__).parent

cells = [
    ("md", """# 06b — Umbrella term vs named model: the same split, grammatically

The macro-analysis shows that abstracts increasingly use umbrella vocabulary *without* naming a
model. If that de-specification is doing ideological work, the two kinds of mention should behave
differently in the clause, not merely differ in frequency.

We tag every AI mention with its lexicon group and contrast the **abstraction** cluster
(`general_ai`, `generative_ai`, `llm`) with the **named-system** cluster (`named_decoder_only`,
`named_encoder_decoder`, `named_image_generator`) on role, process type, modality and polarity.

Outputs:
- `outputs/analysis/06b_mentions_by_group.parquet`
- `outputs/analysis/06b_group_contrasts.csv`
"""),
    ("code", """from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path.cwd()
while ROOT.name and not (ROOT / "analysis_code").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "analysis_code"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import statsmodels.formula.api as smf

from lib import OUT_PARSED, OUT_ANALYSIS, OUT_CLEAN
from lib.roles import classify_role
from lib.ai_lexicon import build_phrase_matcher

sns.set_theme(context="notebook", style="whitegrid")

ABSTRACTION = {"general_ai", "generative_ai", "llm"}
NAMED = {"named_decoder_only", "named_encoder_decoder", "named_image_generator"}

# The grouped matcher assigns each span to exactly one group (most specific
# first), so we can classify a mention by re-matching its own text.
nlp = spacy.blank("en")
pm = build_phrase_matcher(nlp, label="AI")


def group_of(text: str) -> str:
    doc = nlp.make_doc(text)
    hits = pm.find_grouped(doc)
    return hits[0][0] if hits else "unmatched"


ment = pd.read_parquet(OUT_PARSED / "mentions.parquet")
ai = ment[ment["category"] == "AI"].copy()
uniq = pd.Series(ai["mention_text"].unique())
lookup = {t: group_of(t) for t in uniq}
ai["ai_group"] = ai["mention_text"].map(lookup)
ai["cluster"] = np.where(ai["ai_group"].isin(ABSTRACTION), "abstraction",
                  np.where(ai["ai_group"].isin(NAMED), "named", "other"))
print(ai["ai_group"].value_counts())
print()
print(ai["cluster"].value_counts())
"""),
    ("code", """ai["role"] = ai.apply(classify_role, axis=1)
idx = pd.read_parquet(OUT_CLEAN / "index.parquet")[["id", "publication_year"]]
ai = ai.merge(idx.rename(columns={"id": "work_id"}), on="work_id", how="left")
ai = ai[ai["cluster"].isin(["abstraction", "named"])].copy()
print("mentions in the two clusters:", len(ai))
"""),
    ("md", "## 1. Grammatical role by cluster"),
    ("code", """role_tab = pd.crosstab(ai["cluster"], ai["role"], normalize="index").round(3)
cols = [c for c in ["active_subject", "passive_subject", "direct_object", "attribute",
                    "oblique_agent", "prepositional", "nominal_modifier", "other"]
        if c in role_tab.columns]
role_tab = role_tab[cols]
print(role_tab.to_string())

ax = role_tab.T.plot(kind="bar", figsize=(10, 4), color=["#4c72b0", "#dd8452"])
ax.set_title("Grammatical role: umbrella terms vs named models")
ax.set_ylabel("share of mentions"); ax.set_xlabel("")
ax.legend(title="cluster")
plt.xticks(rotation=30, ha="right")
plt.tight_layout(); plt.show()
"""),
    ("code", """ai["is_active"] = ai["role"].eq("active_subject").astype(int)
ai["is_classifier"] = ai["role"].eq("nominal_modifier").astype(int)
ai["is_named"] = ai["cluster"].eq("named").astype(int)
ai["year_c"] = ai["publication_year"] - 2024


def or_table(res, model_name):
    p, s, pv = res.params, res.bse, res.pvalues
    return pd.DataFrame({
        "model": model_name, "term": list(p.index),
        "odds_ratio": np.exp(p).round(3).values,
        "ci_low": np.exp(p - 1.96 * s).round(3).values,
        "ci_high": np.exp(p + 1.96 * s).round(3).values,
        "p": pv.round(5).values,
    })


r_act = smf.logit("is_active ~ is_named + year_c", data=ai).fit(
    disp=False, cov_type="cluster", cov_kwds={"groups": ai["work_id"]})
r_cls = smf.logit("is_classifier ~ is_named + year_c", data=ai).fit(
    disp=False, cov_type="cluster", cov_kwds={"groups": ai["work_id"]})
tabs = [or_table(r_act, "active ~ named"), or_table(r_cls, "classifier ~ named")]
print(pd.concat(tabs).to_string(index=False))
"""),
    ("md", "## 2. Process type by cluster"),
    ("code", """vr = pd.read_csv(OUT_ANALYSIS / "04c_verbs_ranked.csv")
lemma2class = dict(zip(vr["verb_lemma"], vr["auto_class"]))

act = ai[(ai["role"] == "active_subject") & ai["governor_pos"].isin(["VERB", "AUX"])].copy()
act["verb_class"] = act["governor_lemma"].str.lower().map(lemma2class).fillna("OTHER")
vc = pd.crosstab(act["cluster"], act["verb_class"], normalize="index").round(3)
print(vc.to_string())
print()
print("n active-subject clauses per cluster:", act["cluster"].value_counts().to_dict())

ax = vc.T.plot(kind="bar", figsize=(10, 4), color=["#4c72b0", "#dd8452"])
ax.set_title("Process type of AI-as-subject clauses: umbrella vs named")
ax.set_ylabel("share"); ax.set_xlabel("")
plt.xticks(rotation=30, ha="right")
plt.tight_layout(); plt.show()
"""),
    ("md", "## 3. Modality and polarity by cluster"),
    ("code", """mod = pd.read_csv(OUT_ANALYSIS / "04b_modality_per_sentence.csv")[
    ["work_id", "sent_id", "auto_label"]]
m = ai.merge(mod, on=["work_id", "sent_id"], how="inner")
mod_tab = pd.crosstab(m["cluster"], m["auto_label"], normalize="index").round(3)
print("modality:"); print(mod_tab.to_string())

pol = pd.read_parquet(OUT_ANALYSIS / "04e_polarity_per_mention.parquet")[
    ["work_id", "sent_id", "negated", "projected"]].drop_duplicates(["work_id", "sent_id"])
p = ai.merge(pol, on=["work_id", "sent_id"], how="inner")
pol_tab = p.groupby("cluster").agg(
    n=("negated", "size"), negated=("negated", "mean"), projected=("projected", "mean")
).round(3)
print(); print("polarity:"); print(pol_tab.to_string())
"""),
    ("code", """m["is_bare"] = m["auto_label"].eq("bare").astype(int)
m["is_named"] = m["cluster"].eq("named").astype(int)
m["year_c"] = m["publication_year"] - 2024
r_bare = smf.logit("is_bare ~ is_named + year_c", data=m).fit(
    disp=False, cov_type="cluster", cov_kwds={"groups": m["work_id"]})
tab_bare = or_table(r_bare, "bare ~ named")
print(tab_bare.to_string(index=False))
"""),
    ("md", "## 4. Diachrony of the split"),
    ("code", """yr = (
    ai.assign(named=lambda d: d["is_named"])
    .groupby("publication_year")["named"].agg(["mean", "size"]).round(3)
)
print("share of AI mentions that name a system, by year:")
print(yr)

ax = yr["mean"].plot(kind="bar", figsize=(7, 3.5), color="#4c72b0")
ax.set_title("Share of AI mentions that name a concrete system")
ax.set_ylabel("share"); ax.set_xlabel("year")
plt.tight_layout(); plt.show()
"""),
    ("md", "## 5. Persist"),
    ("code", """OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
ai.to_parquet(OUT_ANALYSIS / "06b_mentions_by_group.parquet", index=False)
contrasts = pd.concat(tabs + [tab_bare], ignore_index=True)
contrasts.to_csv(OUT_ANALYSIS / "06b_group_contrasts.csv", index=False)
role_tab.to_csv(OUT_ANALYSIS / "06b_role_by_cluster.csv")
vc.to_csv(OUT_ANALYSIS / "06b_verbclass_by_cluster.csv")
print("saved")
contrasts
"""),
]

write(HERE / "06b_term_groups.ipynb", cells)
