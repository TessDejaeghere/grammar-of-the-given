"""Generate 06a_statistics.ipynb — document-level rates, CIs and regression models."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _nbbuild import write

HERE = Path(__file__).parent

cells = [
    ("md", """# 06a — Statistical robustness: document clustering, CIs and models

Every result so far is a pooled proportion over *mentions*. Two problems follow:

1. **Clustering.** Mentions are nested in documents, and a handful of book-length works contribute
   hundreds of mentions each. A pooled proportion silently weights those works most heavily and its
   nominal standard error is far too small.
2. **No uncertainty.** "AI 0.137 vs HUMAN 0.501" is reported without an interval or a test.

This notebook re-expresses the headline claims as (a) document-level rates with cluster bootstrap
CIs and (b) logistic regressions with cluster-robust standard errors by `work_id`.

Outputs:
- `outputs/analysis/06a_doc_level_rates.parquet`
- `outputs/analysis/06a_models.csv`
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

from lib import OUT_PARSED, OUT_ANALYSIS, OUT_CLEAN
from lib.roles import classify_role

sns.set_theme(context="notebook", style="whitegrid")
RNG = np.random.default_rng(20260907)

ment = pd.read_parquet(OUT_PARSED / "mentions.parquet")
ment["role"] = ment.apply(classify_role, axis=1)
idx = pd.read_parquet(OUT_CLEAN / "index.parquet")[["id", "publication_year"]]
ment = ment.merge(idx.rename(columns={"id": "work_id"}), on="work_id", how="left")
ment["is_active"] = ment["role"].eq("active_subject").astype(int)
print(ment.groupby("category").size())
"""),
    ("md", "## 1. Pooled vs document-level active-subject rate"),
    ("code", """def cluster_bootstrap(df, value_col, cluster_col="work_id", n_boot=2000, stat="mean"):
    \"\"\"Resample whole documents, not mentions.\"\"\"
    groups = [g[value_col].values for _, g in df.groupby(cluster_col, sort=False)]
    k = len(groups)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = RNG.integers(0, k, k)
        vals = np.concatenate([groups[i] for i in pick])
        out[b] = vals.mean()
    return np.percentile(out, [2.5, 97.5])


rows = []
for cat, g in ment.groupby("category"):
    pooled = g["is_active"].mean()
    per_doc = g.groupby("work_id")["is_active"].mean()
    lo, hi = cluster_bootstrap(g, "is_active")
    rows.append({
        "category": cat,
        "n_mentions": len(g),
        "n_docs": g["work_id"].nunique(),
        "pooled_rate": round(pooled, 3),
        "boot_ci_low": round(lo, 3),
        "boot_ci_high": round(hi, 3),
        "doc_mean": round(per_doc.mean(), 3),
        "doc_sd": round(per_doc.std(), 3),
    })
rates = pd.DataFrame(rows).sort_values("pooled_rate", ascending=False)
print(rates.to_string(index=False))
"""),
    ("code", """# How concentrated is the corpus? Share of mentions from the top 5% of works.
conc = (
    ment.groupby(["category", "work_id"]).size().rename("n").reset_index()
    .sort_values("n", ascending=False)
)
for cat, g in conc.groupby("category"):
    top = int(np.ceil(0.05 * len(g)))
    print(f"{cat}: top 5% of works ({top}/{len(g)}) hold "
          f"{g.head(top)['n'].sum() / g['n'].sum():.1%} of mentions")
"""),
    ("code", """fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(rates))
ax.bar(x, rates["pooled_rate"], color="#4c72b0", width=0.55)
ax.errorbar(x, rates["pooled_rate"],
            yerr=[rates["pooled_rate"] - rates["boot_ci_low"],
                  rates["boot_ci_high"] - rates["pooled_rate"]],
            fmt="none", ecolor="black", capsize=5, lw=1.4)
ax.scatter(x, rates["doc_mean"], color="#c44e52", zorder=5,
           label="document-level mean")
ax.set_xticks(x); ax.set_xticklabels(rates["category"])
ax.set_ylabel("active-subject share")
ax.set_title("Active-subject rate by actor category\\n(bars = pooled, whiskers = cluster bootstrap 95% CI)")
ax.legend()
plt.tight_layout(); plt.show()
"""),
    ("md", "## 2. Logistic regression with cluster-robust SEs"),
    ("code", """d = ment[ment["category"].isin(["AI", "HUMAN", "COMPANY"])].copy()
d["year_c"] = d["publication_year"] - 2024          # centred, so the intercept is meaningful
d["cat"] = pd.Categorical(d["category"], categories=["HUMAN", "AI", "COMPANY"])

m1 = smf.logit("is_active ~ C(cat) + year_c", data=d)
r1 = m1.fit(disp=False, cov_type="cluster", cov_kwds={"groups": d["work_id"]})
print(r1.summary())
"""),
    ("code", """def or_table(res, model_name):
    params = res.params
    ses = res.bse
    pv = res.pvalues
    out = pd.DataFrame({
        "model": model_name,
        "term": list(params.index),
        "odds_ratio": np.exp(params).round(3).values,
        "ci_low": np.exp(params - 1.96 * ses).round(3).values,
        "ci_high": np.exp(params + 1.96 * ses).round(3).values,
        "p": pv.round(5).values,
    })
    return out.reset_index(drop=True)


tab1 = or_table(r1, "active ~ actor + year")
print(tab1.to_string(index=False))
"""),
    ("md", """### 2b. Does AI's agency actually grow? (AI mentions only)"""),
    ("code", """ai = d[d["category"] == "AI"].copy()
r2 = smf.logit("is_active ~ year_c", data=ai).fit(
    disp=False, cov_type="cluster", cov_kwds={"groups": ai["work_id"]})
tab2 = or_table(r2, "AI: active ~ year")
print(tab2.to_string(index=False))

# Robustness: drop the post-dated 2026 slice.
ai25 = ai[ai["publication_year"] <= 2025]
r2b = smf.logit("is_active ~ year_c", data=ai25).fit(
    disp=False, cov_type="cluster", cov_kwds={"groups": ai25["work_id"]})
tab2b = or_table(r2b, "AI: active ~ year (<=2025)")
print(tab2b.to_string(index=False))
"""),
    ("md", """### 2c. Is the AI/HUMAN gap stable, or is it closing?"""),
    ("code", """ah = d[d["category"].isin(["AI", "HUMAN"])].copy()
ah["is_ai"] = ah["category"].eq("AI").astype(int)
r3 = smf.logit("is_active ~ is_ai * year_c", data=ah).fit(
    disp=False, cov_type="cluster", cov_kwds={"groups": ah["work_id"]})
tab3 = or_table(r3, "active ~ is_ai * year")
print(tab3.to_string(index=False))
print("\\nInteraction OR > 1 would mean the AI/HUMAN gap narrows over time.")
"""),
    ("md", "## 3. Naturalisation: does classifier position predict bare assertion?"),
    ("code", """mod = pd.read_csv(OUT_ANALYSIS / "04b_modality_per_sentence.csv")[
    ["work_id", "sent_id", "auto_label"]]
aim = ment[ment["category"] == "AI"].merge(mod, on=["work_id", "sent_id"], how="inner")
aim["is_bare"] = aim["auto_label"].eq("bare").astype(int)
aim["is_classifier"] = aim["role"].eq("nominal_modifier").astype(int)
aim["year_c"] = aim["publication_year"] - 2024

m4 = smf.logit("is_bare ~ is_classifier + year_c", data=aim)
r4 = m4.fit(disp=False, cov_type="cluster", cov_kwds={"groups": aim["work_id"]})
tab4 = or_table(r4, "bare ~ classifier + year")
print(tab4.to_string(index=False))
"""),
    ("md", "## 4. Agent effacement in nominalisations"),
    ("code", """nom = pd.read_parquet(OUT_ANALYSIS / "04d_nominalizations.parquet")
nom["effaced"] = (~nom["agent_present"].astype(bool)).astype(int)
lo, hi = cluster_bootstrap(nom, "effaced")
print(f"agent-effaced rate {nom['effaced'].mean():.3f}  "
      f"cluster bootstrap 95% CI [{lo:.3f}, {hi:.3f}]  "
      f"({len(nom)} nominalisations in {nom['work_id'].nunique()} works)")

nom["year_c"] = nom["publication_year"] - 2024
r5 = smf.logit("effaced ~ year_c", data=nom).fit(
    disp=False, cov_type="cluster", cov_kwds={"groups": nom["work_id"]})
tab5 = or_table(r5, "effaced ~ year")
print(tab5.to_string(index=False))
"""),
    ("md", "## 5. Persist"),
    ("code", """OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)

per_doc = (
    ment.groupby(["work_id", "category", "publication_year"])
    .agg(n_mentions=("is_active", "size"), active_rate=("is_active", "mean"))
    .reset_index()
)
per_doc.to_parquet(OUT_ANALYSIS / "06a_doc_level_rates.parquet", index=False)

models = pd.concat([tab1, tab2, tab2b, tab3, tab4, tab5], ignore_index=True)
models.to_csv(OUT_ANALYSIS / "06a_models.csv", index=False)
rates.to_csv(OUT_ANALYSIS / "06a_rates_with_ci.csv", index=False)
print("saved", len(per_doc), "document rows and", len(models), "model terms")
models
"""),
]

write(HERE / "06a_statistics.ipynb", cells)
