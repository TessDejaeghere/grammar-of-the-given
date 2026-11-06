"""Generate 06d_passives.ipynb — agentless passives as a second suppression device."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _nbbuild import write

HERE = Path(__file__).parent

cells = [
    ("md", """# 06d — Agentless passives

Nominalisation is only one of Van Leeuwen's suppression devices; the other is the agentless
passive. "The text was generated", "the model was trained", "the corpus was annotated" all describe
processes with an obligatory human or corporate Agent, and all omit it.

For every passive clause in an AI-bearing sentence we record whether a `by`-agent is realised and,
if so, which actor category it names. Because the `oblique_agent` role covers only 1.5% of AI
mentions (04a), the expectation is a very high agentless rate.

Outputs: `outputs/analysis/06d_passives.parquet`, `outputs/analysis/06d_passives.csv`
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
from spacy.tokens import DocBin

from lib import OUT_PARSED, OUT_ANALYSIS, OUT_CLEAN
from lib.ai_lexicon import build_phrase_matcher
from lib.actors_lexicon import HUMAN_TERMS, COMPANY_TERMS

sns.set_theme(context="notebook", style="whitegrid")
RNG = np.random.default_rng(20260907)

HUMAN_LOWER = {t.lower() for t in HUMAN_TERMS}
COMPANY_LOWER = {t.lower() for t in COMPANY_TERMS}

PASSIVE_SUBJ_DEPS = {"nsubjpass", "nsubj:pass"}

nlp = spacy.blank("en")
ai_pm = build_phrase_matcher(nlp, label="AI")
PARSED = OUT_PARSED / "docs"
work_ids = sorted(p.stem for p in PARSED.glob("*.spacy"))
print("works:", len(work_ids))
"""),
    ("md", "## 1. Extract every passive clause"),
    ("code", """def actor_category(tok, ai_ids):
    if tok is None:
        return ""
    words = {t.text.lower() for t in tok.subtree}
    text = " ".join(t.text.lower() for t in tok.subtree)
    if any(t.i in ai_ids for t in tok.subtree):
        return "AI"
    if any((w in words) if " " not in w else (w in text) for w in HUMAN_LOWER):
        return "HUMAN"
    if any((c in words) if " " not in c else (c in text) for c in COMPANY_LOWER):
        return "COMPANY"
    return "OTHER"


rows = []
for wid in work_ids:
    db = DocBin().from_disk(PARSED / f"{wid}.spacy")
    for sent_idx, doc in enumerate(db.get_docs(nlp.vocab)):
        hits = ai_pm(doc)
        if not hits:
            continue
        ai_ids = {t for _, s, e in hits for t in range(s, e)}
        for tok in doc:
            if tok.pos_ not in ("VERB", "AUX"):
                continue
            subj = next((c for c in tok.children if c.dep_ in PASSIVE_SUBJ_DEPS), None)
            if subj is None:
                continue
            agent_prep = next((c for c in tok.children if c.dep_ == "agent"), None)
            agent_obj = None
            if agent_prep is not None:
                agent_obj = next((c for c in agent_prep.children if c.dep_ == "pobj"), None)
            rows.append({
                "work_id": wid,
                "sent_idx": sent_idx,
                "verb": tok.lemma_.lower(),
                "subj_text": " ".join(t.text for t in subj.subtree)[:80],
                "subj_is_ai": any(t.i in ai_ids for t in subj.subtree),
                "agent_present": agent_obj is not None,
                "agent_text": (" ".join(t.text for t in agent_obj.subtree)[:60]
                               if agent_obj is not None else ""),
                "agent_cat": actor_category(agent_obj, ai_ids),
                "sent_text": doc.text,
            })
    del db

pas = pd.DataFrame(rows)
print("passive clauses in AI-bearing sentences:", len(pas))
print("agentless:", f"{(~pas['agent_present']).mean():.3f}")
"""),
    ("code", """# sent_idx is the within-work order; map to the global sent_id used elsewhere.
all_sents = pd.read_parquet(OUT_PARSED / "sentences.parquet")
all_sents = all_sents.sort_values(["work_id", "sent_id"]).reset_index(drop=True)
all_sents["sent_idx"] = all_sents.groupby("work_id").cumcount()
pas = pas.merge(all_sents[["work_id", "sent_idx", "sent_id"]],
                on=["work_id", "sent_idx"], how="left")

idx = pd.read_parquet(OUT_CLEAN / "index.parquet")[["id", "publication_year"]]
pas = pas.merge(idx.rename(columns={"id": "work_id"}), on="work_id", how="left")
pas["agentless"] = (~pas["agent_present"]).astype(int)
print(pas["publication_year"].value_counts().sort_index())
"""),
    ("md", "## 2. How often is the Agent recoverable?"),
    ("code", """def cluster_bootstrap(df, col, cluster="work_id", n_boot=2000):
    groups = [g[col].values for _, g in df.groupby(cluster, sort=False)]
    k = len(groups)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = RNG.integers(0, k, k)
        out[b] = np.concatenate([groups[i] for i in pick]).mean()
    return np.percentile(out, [2.5, 97.5])


lo, hi = cluster_bootstrap(pas, "agentless")
print(f"agentless passive rate: {pas['agentless'].mean():.3f} "
      f"[{lo:.3f}, {hi:.3f}]  (n={len(pas)} in {pas['work_id'].nunique()} works)")

sub = pas[pas["subj_is_ai"]]
lo2, hi2 = cluster_bootstrap(sub, "agentless")
print(f"  when the Goal is an AI mention: {sub['agentless'].mean():.3f} "
      f"[{lo2:.3f}, {hi2:.3f}]  (n={len(sub)})")

print()
print("when an agent IS realised, which actor is it?")
print(pas[pas["agent_present"]]["agent_cat"].value_counts(normalize=True).round(3))
print(pas[pas["agent_present"]]["agent_cat"].value_counts())
"""),
    ("code", """yr = pas.groupby("publication_year")["agentless"].agg(["mean", "size"]).round(3)
print(yr)

ax = yr["mean"].plot(kind="bar", figsize=(7, 3.5), color="#c44e52")
ax.set_ylim(0.5, 1.0)
ax.axhline(pas["agentless"].mean(), color="black", ls="--", lw=1,
           label=f"overall {pas['agentless'].mean():.3f}")
ax.set_title("Agentless rate of passive clauses in AI-bearing sentences")
ax.set_ylabel("share without a by-agent"); ax.set_xlabel("year")
ax.legend()
plt.tight_layout(); plt.show()
"""),
    ("md", "## 3. Which processes lose their Agent?"),
    ("code", """by_verb = (
    pas.groupby("verb")
    .agg(n=("agentless", "size"), agentless=("agentless", "mean"))
    .query("n >= 25")
    .sort_values("n", ascending=False)
)
print("most frequent passive processes in AI-bearing sentences:")
print(by_verb.head(25).round(3).to_string())
"""),
    ("code", """# The accountability-critical set: processes with an obligatory human agent.
HUMAN_AGENT_VERBS = ["train", "develop", "design", "build", "deploy", "create",
                     "annotate", "collect", "curate", "fine-tune", "release",
                     "publish", "select", "implement"]
crit = pas[pas["verb"].isin(HUMAN_AGENT_VERBS)]
print(f"processes with an obligatory human agent: n={len(crit)}, "
      f"agentless {crit['agentless'].mean():.3f}")
print(crit.groupby("verb")["agentless"].agg(["size", "mean"]).round(3)
      .sort_values("size", ascending=False).to_string())
"""),
    ("code", """import re
def clean(s):
    return re.sub(r"\\s+", " ", str(s)).strip()

print("Examples — agentless passives of obligatory-agent processes:\\n")
ex = crit[crit["agentless"].eq(1) & crit["sent_text"].str.len().between(50, 170)]
for r in ex.drop_duplicates("sent_text").head(8).itertuples():
    print(f"  [{r.verb}] {clean(r.sent_text)[:165]}")

print("\\nExamples — agent realised:\\n")
ex2 = pas[pas["agent_present"] & pas["sent_text"].str.len().between(50, 170)]
for r in ex2.drop_duplicates("sent_text").head(6).itertuples():
    print(f"  [{r.verb} <- {r.agent_cat}: {clean(r.agent_text)}] {clean(r.sent_text)[:150]}")
"""),
    ("md", "## 4. Model + persist"),
    ("code", """# A few DocBins exist for works that did not survive the clean-index filter,
# so drop rows with no year before modelling (groups must match the design matrix).
fit_df = pas.dropna(subset=["publication_year"]).copy()
fit_df["year_c"] = fit_df["publication_year"] - 2024
res = smf.logit("agentless ~ year_c", data=fit_df).fit(
    disp=False, cov_type="cluster", cov_kwds={"groups": fit_df["work_id"]})
tab = pd.DataFrame({
    "term": list(res.params.index),
    "odds_ratio": np.exp(res.params).round(3).values,
    "ci_low": np.exp(res.params - 1.96 * res.bse).round(3).values,
    "ci_high": np.exp(res.params + 1.96 * res.bse).round(3).values,
    "p": res.pvalues.round(5).values,
})
print(tab.to_string(index=False))

OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
keep = ["work_id", "sent_id", "publication_year", "verb", "subj_text", "subj_is_ai",
        "agent_present", "agent_text", "agent_cat", "agentless", "sent_text"]
pas[keep].to_parquet(OUT_ANALYSIS / "06d_passives.parquet", index=False)
pas[keep].to_csv(OUT_ANALYSIS / "06d_passives.csv", index=False, encoding="utf-8-sig")
by_verb.to_csv(OUT_ANALYSIS / "06d_passives_by_verb.csv")
print("saved", len(pas), "passive clauses")
"""),
    ("md", """## 5. Backgrounding or radical exclusion?

Van Leeuwen distinguishes **backgrounding** (the actor is deleted here but named elsewhere in the
text) from **radical exclusion** (the actor never appears at all). The clause-level rates above
cannot tell these apart, so for every agentless passive and every suppressed nominalisation we ask
whether a company is mentioned (a) in the same sentence, (b) within ±3 sentences, (c) anywhere in
the same work.

Only companies named as such count. Generic terms (*developers*, *vendors*, *tech companies*) and
matches that do not refer to a company as an actor (*Google Scholar*, *metadata*, *meta-analysis*,
*NVIDIA A100*, *XAI*, ...) are filtered out with `filter_named_companies` from `lib/actors_lexicon.py`.
"""),
    ("code", """from lib.actors_lexicon import filter_named_companies

ment = pd.read_parquet(OUT_PARSED / "mentions.parquet")
comp_all = ment[ment["category"] == "COMPANY"]
# Named companies only: drop generic terms (developers, vendors, tech companies) and
# non-actor matches (Google Scholar, metadata/meta-analysis, NVIDIA A100, XAI, ...).
comp = filter_named_companies(comp_all)
print(f"company matches: {len(comp_all)} -> named companies: {len(comp)}\n")
works_company = set(comp["work_id"])
comp_sent_keys = set(map(tuple, comp[["work_id", "sent_id"]].values))
comp_sents_by_work = comp[["work_id", "sent_id"]].drop_duplicates() \\
    .groupby("work_id")["sent_id"].apply(set).to_dict()


def recoverability(df, label):
    same_work = df["work_id"].isin(works_company)
    same_sent = df.apply(lambda r: (r["work_id"], r["sent_id"]) in comp_sent_keys, axis=1)

    def near(r, w=3):
        s = comp_sents_by_work.get(r["work_id"])
        return bool(s) and any(abs(r["sent_id"] - x) <= w for x in s)

    window = df.apply(near, axis=1)
    print(f"{label} (n={len(df)}):")
    print(f"  company in same sentence:      {same_sent.mean():.3f}")
    print(f"  company within +/-3 sentences: {window.mean():.3f}")
    print(f"  company anywhere in the work:  {same_work.mean():.3f}")
    print(f"  -> radical exclusion (never named in the work): {1 - same_work.mean():.3f}")
    return same_work


ag = pas[pas["agentless"].eq(1)].dropna(subset=["sent_id"])
sw_pas = recoverability(ag, "agentless passives")
print()
recoverability(ag[ag["verb"].isin(HUMAN_AGENT_VERBS)],
               "  ... obligatory-agent subset")
print()
nom = pd.read_parquet(OUT_ANALYSIS / "04d_nominalizations.parquet")
sup = nom[nom["agent_relation"] == "suppressed"]
recoverability(sup, "suppressed nominalisations")

print(f"\\ncontext: {len(works_company)} of {ment['work_id'].nunique()} works "
      f"({len(works_company) / ment['work_id'].nunique():.1%}) mention a company at all; "
      f"median company mentions per such work: "
      f"{comp.groupby('work_id').size().median():.0f}")
"""),
]

write(HERE / "06d_passives.ipynb", cells)
