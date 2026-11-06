"""Generate 06e_provenance.ipynb — who wrote these papers, and does it predict the grammar?"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _nbbuild import write

HERE = Path(__file__).parent

cells = [
    ("md", """# 06e — Corpus provenance and its effect on the findings

Every result so far treats "Digital Humanities" as one discourse community. It is not. The corpus
mixes arXiv preprints from NLP groups with diamond-OA humanities journals and edited book chapters,
and \u00a75.8 already showed that two registers coexist. This notebook asks whether the register split
is **a property of the papers' origins** rather than of the field as a whole.

We characterise every work by venue class, an OpenAlex-concept discipline proxy, country and OA
route, then re-compute the headline measures per document and compare across those strata.

Outputs:
- `outputs/analysis/06e_work_profiles.parquet` \u2014 one row per work, provenance + all measures
- `outputs/analysis/06e_by_stratum.csv`
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
import statsmodels.formula.api as smf

from lib import OUT_PARSED, OUT_ANALYSIS, OUT_CLEAN, OUT_MACRO
from lib.roles import classify_role

sns.set_theme(context="notebook", style="whitegrid")
RNG = np.random.default_rng(20260907)

idx = pd.read_parquet(OUT_CLEAN / "index.parquet")
works = idx[idx["in_ai_corpus"]].copy()
meta = pd.read_parquet(OUT_MACRO / "works_metadata.parquet")
prof = works[["id", "publication_year", "n_tokens_approx", "n_ai_mentions"]].merge(
    meta[["id", "title", "type", "venue", "venue_type", "first_country",
          "oa_status", "concepts", "cited_by_count", "n_authors"]],
    on="id", how="left")
prof = prof.rename(columns={"id": "work_id"})
print("works in the AI corpus:", len(prof))
"""),
    ("md", "## 1. What is actually in the corpus?"),
    ("code", """print("document type:")
print(prof["type"].value_counts().head(10))
print()
print("venue type:")
print(prof["venue_type"].value_counts(dropna=False))
print()
print("top venues:")
print(prof["venue"].value_counts().head(12))
print()
print("first-author country (39% missing):")
print(prof["first_country"].value_counts().head(10))
print()
print("OA route:")
print(prof["oa_status"].value_counts())
"""),
    ("code", """# Venue class: the practical proxy for 'which community published this'.
def venue_class(row) -> str:
    t = str(row["type"]).lower()
    vt = str(row["venue_type"]).lower()
    v = str(row["venue"]).lower()
    if t in ("book", "book-chapter") or vt in ("ebook platform", "book series"):
        return "book / chapter"
    if t == "conference-paper" or vt == "conference":
        return "conference"
    if t == "preprint" or vt == "repository" or "arxiv" in v:
        return "preprint / repository"
    if t in ("dissertation", "report"):
        return "other"
    return "journal article"


prof["venue_class"] = prof.apply(venue_class, axis=1)
print(prof["venue_class"].value_counts())
"""),
    ("code", """# Discipline proxy from OpenAlex concepts.
COMPUTATIONAL = {
    "Computer science", "Artificial intelligence", "Natural language processing",
    "Information retrieval", "Data science", "Machine learning", "Deep learning",
    "Pipeline (software)", "Speech recognition", "Pattern recognition (psychology)",
    "Computer vision", "Data mining", "Programming language", "World Wide Web",
}
HUMANISTIC = {
    "Art", "History", "Humanities", "Literature", "Epistemology", "Philosophy",
    "Media studies", "Poetry", "Narrative", "Library science", "Sociology",
    "Aesthetics", "Archaeology", "Classics", "Art history", "Visual arts",
    "Cultural heritage", "Scholarship", "Reading (process)",
}


def discipline(concepts: str) -> str:
    cs = set(str(concepts).split("|"))
    n_comp = len(cs & COMPUTATIONAL)
    n_hum = len(cs & HUMANISTIC)
    if n_comp > n_hum:
        return "computational"
    if n_hum > n_comp:
        return "humanistic"
    return "mixed / unclear"


prof["discipline"] = prof["concepts"].map(discipline)
print(prof["discipline"].value_counts())
print()
print(pd.crosstab(prof["venue_class"], prof["discipline"]))
"""),
    ("code", """print("most AI-dense works (mentions per 1k tokens):")
prof["ai_density"] = 1000 * prof["n_ai_mentions"] / prof["n_tokens_approx"].clip(lower=1)
top = prof.nlargest(12, "ai_density")[
    ["title", "venue_class", "discipline", "publication_year", "ai_density"]]
for r in top.itertuples():
    print(f"  {r.ai_density:6.1f}  [{r.venue_class[:12]:<12} {r.discipline[:13]:<13} "
          f"{r.publication_year:.0f}]  {str(r.title)[:70]}")
"""),
    ("md", "## 2. Per-work measures"),
    ("code", """ment = pd.read_parquet(OUT_PARSED / "mentions.parquet")
ment["role"] = ment.apply(classify_role, axis=1)
ai = ment[ment["category"] == "AI"]

per_work = ai.groupby("work_id").agg(
    n_ai=("role", "size"),
    active_share=("role", lambda s: (s == "active_subject").mean()),
    classifier_share=("role", lambda s: (s == "nominal_modifier").mean()),
).reset_index()

# Human active-subject share in the same work, as a within-document baseline.
hum = ment[ment["category"] == "HUMAN"].groupby("work_id")["role"].agg(
    human_active_share=lambda s: (s == "active_subject").mean(),
    n_human="size").reset_index()
per_work = per_work.merge(hum, on="work_id", how="left")

# Named vs umbrella (from 06b).
grp = pd.read_parquet(OUT_ANALYSIS / "06b_mentions_by_group.parquet")
named = grp.groupby("work_id")["cluster"].agg(
    named_share=lambda s: (s == "named").mean()).reset_index()
per_work = per_work.merge(named, on="work_id", how="left")

# Modality (from 04b).
mod = pd.read_csv(OUT_ANALYSIS / "04b_modality_per_sentence.csv")
mods = mod.groupby("work_id")["auto_label"].agg(
    bare_share=lambda s: (s == "bare").mean(),
    hedged_share=lambda s: (s == "weak/hedged").mean(),
    n_modality="size").reset_index()
per_work = per_work.merge(mods, on="work_id", how="left")

# Agent effacement (04d) and agentless passives (06d).
nom = pd.read_parquet(OUT_ANALYSIS / "04d_nominalizations.parquet")
noms = nom.groupby("work_id")["agent_present"].agg(
    effaced_share=lambda s: 1 - s.astype(bool).mean(), n_nom="size").reset_index()
per_work = per_work.merge(noms, on="work_id", how="left")

pas = pd.read_parquet(OUT_ANALYSIS / "06d_passives.parquet")
pss = pas.groupby("work_id")["agentless"].agg(
    agentless_share="mean", n_passive="size").reset_index()
per_work = per_work.merge(pss, on="work_id", how="left")

prof = prof.merge(per_work, on="work_id", how="left")
print(prof[["active_share", "classifier_share", "named_share", "bare_share",
            "effaced_share", "agentless_share"]].describe().round(3).to_string())
"""),
    ("md", "## 3. Does provenance predict the grammar?"),
    ("code", """def boot_ci(vals, n_boot=2000):
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 3:
        return (np.nan, np.nan)
    draws = RNG.choice(vals, (n_boot, len(vals)), replace=True).mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


MEASURES = ["named_share", "active_share", "classifier_share", "bare_share",
            "hedged_share", "effaced_share", "agentless_share", "ai_density"]


def stratum_table(col, min_n=15):
    rows = []
    for val, g in prof.groupby(col):
        if len(g) < min_n:
            continue
        row = {col: val, "n_works": len(g)}
        for m in MEASURES:
            row[m] = round(g[m].mean(), 3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_works", ascending=False)


by_venue = stratum_table("venue_class")
print("BY VENUE CLASS (document-level means)")
print(by_venue.to_string(index=False))
print()
by_disc = stratum_table("discipline")
print("BY DISCIPLINE PROXY")
print(by_disc.to_string(index=False))
"""),
    ("code", """# Confidence intervals on the two measures that carry the register argument.
for col in ["venue_class", "discipline"]:
    print(f"\\n=== {col}")
    for m in ["named_share", "bare_share", "active_share"]:
        print(f"  {m}")
        for val, g in prof.groupby(col):
            if len(g) < 15:
                continue
            lo, hi = boot_ci(g[m])
            print(f"    {str(val):<24} n={len(g):>3}  {g[m].mean():.3f}  [{lo:.3f}, {hi:.3f}]")
"""),
    ("code", """fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, m, title in zip(
        axes,
        ["named_share", "bare_share", "hedged_share"],
        ["Share of AI mentions naming a system", "Bare assertion share", "Hedged share"]):
    sub = prof[prof["venue_class"].isin(by_venue["venue_class"])]
    sns.boxplot(data=sub, x="venue_class", y=m, ax=ax, showfliers=False,
                order=by_venue["venue_class"])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=25)
plt.suptitle("Grammar by venue class (one point per work)")
plt.tight_layout(); plt.show()
"""),
    ("code", """# Mention-level models: does venue class predict the outcome, controlling for year?
aim = ai.merge(prof[["work_id", "venue_class", "discipline", "publication_year"]],
               on="work_id", how="inner")
aim["year_c"] = aim["publication_year"] - 2024
aim["is_active"] = aim["role"].eq("active_subject").astype(int)
aim["vc"] = pd.Categorical(
    aim["venue_class"],
    categories=["journal article", "preprint / repository", "conference", "book / chapter", "other"])

res = smf.logit("is_active ~ C(vc) + year_c", data=aim).fit(
    disp=False, cov_type="cluster", cov_kwds={"groups": aim["work_id"]})
tab = pd.DataFrame({
    "term": list(res.params.index),
    "odds_ratio": np.exp(res.params).round(3).values,
    "ci_low": np.exp(res.params - 1.96 * res.bse).round(3).values,
    "ci_high": np.exp(res.params + 1.96 * res.bse).round(3).values,
    "p": res.pvalues.round(5).values,
})
print("AI as active subject, by venue class (ref = journal article):")
print(tab.to_string(index=False))
"""),
    ("code", """# 06b already carries publication_year, so only take provenance columns here.
grp2 = grp.merge(prof[["work_id", "venue_class", "discipline"]],
                 on="work_id", how="inner")
grp2["is_named"] = grp2["cluster"].eq("named").astype(int)
grp2["year_c"] = grp2["publication_year"] - 2024
grp2["vc"] = pd.Categorical(
    grp2["venue_class"],
    categories=["journal article", "preprint / repository", "conference", "book / chapter", "other"])
res2 = smf.logit("is_named ~ C(vc) + year_c", data=grp2).fit(
    disp=False, cov_type="cluster", cov_kwds={"groups": grp2["work_id"]})
tab2 = pd.DataFrame({
    "term": list(res2.params.index),
    "odds_ratio": np.exp(res2.params).round(3).values,
    "ci_low": np.exp(res2.params - 1.96 * res2.bse).round(3).values,
    "ci_high": np.exp(res2.params + 1.96 * res2.bse).round(3).values,
    "p": res2.pvalues.round(5).values,
})
print("Mention names a concrete system, by venue class (ref = journal article):")
print(tab2.to_string(index=False))
"""),
    ("md", "## 4. Process types by provenance"),
    ("code", """vr = pd.read_csv(OUT_ANALYSIS / "04c_verbs_ranked.csv")
lemma2class = dict(zip(vr["verb_lemma"], vr["auto_class"]))
act = aim[(aim["role"] == "active_subject") & aim["governor_pos"].isin(["VERB", "AUX"])].copy()
act["verb_class"] = act["governor_lemma"].str.lower().map(lemma2class).fillna("OTHER")

vc_venue = pd.crosstab(act["venue_class"], act["verb_class"], normalize="index").round(3)
print(vc_venue.to_string())
print()
print("n per venue class:", act["venue_class"].value_counts().to_dict())

keep = [c for c in ["instrumental_tool", "evaluative_performance", "cognitive_mental",
                    "verbal_communicative", "stative_relational", "material_action"]
        if c in vc_venue.columns]
ax = vc_venue[keep].plot(kind="bar", figsize=(10, 4), colormap="tab10")
ax.set_title("Process type of AI-as-subject clauses, by venue class")
ax.set_ylabel("share"); ax.set_xlabel("")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.xticks(rotation=20, ha="right")
plt.tight_layout(); plt.show()
"""),
    ("md", "## 5. Country and open-access route"),
    ("code", """ctry = prof[prof["first_country"].notna()].copy()
top_c = ctry["first_country"].value_counts()
top_c = top_c[top_c >= 15].index
sub = ctry[ctry["first_country"].isin(top_c)]
print("by first-author country (>=15 works; 39% of the corpus has no country):")
print(sub.groupby("first_country")[["named_share", "active_share", "bare_share",
                                    "effaced_share"]].mean().round(3).to_string())
print()
print("by OA route:")
print(prof.groupby("oa_status")[["named_share", "active_share", "bare_share",
                                 "effaced_share", "ai_density"]]
      .agg(["mean", "size"]).round(3).to_string())
"""),
    ("md", "## 6. Persist"),
    ("code", """OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
prof.to_parquet(OUT_ANALYSIS / "06e_work_profiles.parquet", index=False)
out = pd.concat([by_venue.rename(columns={"venue_class": "stratum"}).assign(dim="venue_class"),
                 by_disc.rename(columns={"discipline": "stratum"}).assign(dim="discipline")],
                ignore_index=True)
out.to_csv(OUT_ANALYSIS / "06e_by_stratum.csv", index=False)
pd.concat([tab.assign(model="active ~ venue"), tab2.assign(model="named ~ venue")]).to_csv(
    OUT_ANALYSIS / "06e_models.csv", index=False)
print("saved profiles for", len(prof), "works")
out
"""),
]

write(HERE / "06e_provenance.ipynb", cells)
