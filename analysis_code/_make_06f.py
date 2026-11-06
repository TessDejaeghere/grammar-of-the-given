"""Generate 06f_anthropomorphization.ipynb — Inie/Zukerman/Bender taxonomy applied at scale."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _nbbuild import write

HERE = Path(__file__).parent

cells = [
    ("md", """# 06f — Anthropomorphization by description (Inie, Zukerman & Bender 2026)

Inie, Zukerman and Bender (*First Monday* 31(2)) build an eight-category taxonomy of
anthropomorphizing language from 29 texts / 1,368 sentences of news, company blogs and CHI papers,
with unitized Krippendorff's alpha of 0.84–0.91. We apply it to 21,687 AI-bearing sentences of DH
research prose, which lets us answer the question they leave open — *"different domains are likely
to show different distributions of categories — we welcome further evaluation"*.

Three things make this more than a re-application.

1. **Dependency anchoring.** Their annotation is word-level and manual. We require the
   anthropomorphizing item to stand in a specific grammatical relation to an AI mention (verb
   governing an AI subject, noun possessed by or compounded with AI, modifier attached to AI), so
   the count is a count of *predications about AI* rather than of words in the vicinity.
2. **Both polarity rules.** Their guideline 4.1 counts a property even when it is negated, on the
   grounds that *"failed to grasp"* presupposes grasping was expected. Our 04e coding says the
   opposite: negation marks refusal of the frame. Both are defensible and they answer different
   questions, so every category is reported twice — **presupposed** (their criterion) and
   **asserted** (ours). They cannot make this split; we can.
3. **The entrenched-term question, quantified.** Their §4.2.1 and §5.2 insist that *artificial
   intelligence*, *machine learning*, *neural network* and *training* count. That single decision
   is the difference between a marginal and a pervasive result, so we report the rate with and
   without the entrenched set rather than choosing silently.

Outputs:
- `outputs/analysis/06f_anthro_per_sentence.parquet`
- `outputs/analysis/06f_category_summary.csv`
- `outputs/analysis/06f_anthro_sample.csv` — review sheet using their category labels
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
from lib.actors_lexicon import HUMAN_TERMS
from lib.anthro_lexicon import CATEGORIES, ENTRENCHED, ANIMATE_PRONOUNS

sns.set_theme(context="notebook", style="whitegrid")
RNG = np.random.default_rng(20260908)

CATS = list(CATEGORIES)
HUMAN_LOWER = {t.lower() for t in HUMAN_TERMS}
POSS_DEPS = {"poss", "nmod", "compound"}

nlp = spacy.blank("en")
ai_pm = build_phrase_matcher(nlp, label="AI")
PARSED = OUT_PARSED / "docs"
work_ids = sorted(p.stem for p in PARSED.glob("*.spacy"))
print("works:", len(work_ids))
print("categories:", CATS)
"""),
    ("md", "## 1. Tag each AI-bearing sentence, dependency-anchored"),
    ("code", """def governing_verb(tok):
    cur, hops = tok, 0
    while cur.pos_ not in ("VERB", "AUX") and cur.head is not cur and hops < 12:
        cur = cur.head
        hops += 1
    return cur if cur.pos_ in ("VERB", "AUX") else None


def tag_sentence(doc, ai_spans):
    \"\"\"Return {category: [(token_text, anchor_kind), ...]} for AI-anchored hits.\"\"\"
    ai_ids = {i for sp in ai_spans for i in range(sp.start, sp.end)}
    roots = [sp.root for sp in ai_spans]
    hits = {c: [] for c in CATS}

    # (a) verbs that govern an AI mention as subject
    for r in roots:
        if r.dep_ in ("nsubj", "nsubjpass", "nsubj:pass"):
            v = r.head if r.head.pos_ in ("VERB", "AUX") else governing_verb(r)
            if v is not None:
                lem = v.lemma_.lower()
                for c in CATS:
                    if lem in CATEGORIES[c].get("verbs", set()):
                        hits[c].append((lem, "subject_of_verb"))

    # (b) nouns possessed by / compounded with an AI mention, and AI as their
    #     Attribute ('AI is a co-author')
    for r in roots:
        head = r.head
        if r.dep_ in POSS_DEPS and head.pos_ in ("NOUN", "PROPN"):
            lem = head.lemma_.lower()
            for c in CATS:
                if lem in CATEGORIES[c].get("nouns", set()):
                    hits[c].append((lem, "modifies_noun"))
        if r.dep_ in ("attr", "appos", "oprd"):
            lem = r.lemma_.lower()
            for c in CATS:
                if lem in CATEGORIES[c].get("nouns", set()):
                    hits[c].append((lem, "attribute"))
        # 'AI has capabilities' / 'AI shows bias'
        if r.dep_ == "nsubj" and r.head.lemma_.lower() in ("have", "possess", "show", "exhibit", "lack"):
            for child in r.head.children:
                if child.dep_ in ("dobj", "obj"):
                    lem = child.lemma_.lower()
                    for c in CATS:
                        if lem in CATEGORIES[c].get("nouns", set()):
                            hits[c].append((lem, "possessed_noun"))

    # (c) modifiers attached to an AI mention
    for r in roots:
        for child in list(r.children) + ([r.head] if r.dep_ in ("attr", "acomp") else []):
            if child.pos_ in ("ADJ", "ADV"):
                lem = child.lemma_.lower()
                for c in CATS:
                    if lem in CATEGORIES[c].get("modifiers", set()):
                        hits[c].append((lem, "modifier"))
        for t in doc:
            if t.pos_ in ("ADJ", "ADV") and t.head.i in ai_ids:
                lem = t.lemma_.lower()
                for c in CATS:
                    if lem in CATEGORIES[c].get("modifiers", set()):
                        hits[c].append((lem, "modifier"))

    # (d) animate pronouns. Possessive determiners ('his article') are excluded:
    #     in this corpus they overwhelmingly possess a human. Relative pronouns
    #     must attach to an AI mention; personal pronouns must be a core argument
    #     of a clause with no human-actor term and no PERSON entity.
    words = {t.text.lower() for t in doc}
    has_person = any(e.label_ == "PERSON" for e in doc.ents)
    for t in doc:
        low = t.text.lower()
        if low not in ANIMATE_PRONOUNS or t.pos_ != "PRON":
            continue
        if low in ("who", "whom", "whose"):
            anchor = t.head
            while anchor.dep_ in ("relcl", "acl") and anchor.head is not anchor:
                anchor = anchor.head
            if anchor.i in ai_ids:
                hits["Names_and_pronouns"].append((low, "relative_pronoun"))
        elif t.dep_ in ("nsubj", "nsubjpass", "dobj", "obj") and not has_person \
                and not (words & HUMAN_LOWER):
            hits["Names_and_pronouns"].append((low, "personal_pronoun"))

    return hits
"""),
    ("code", """all_sents = pd.read_parquet(OUT_PARSED / "sentences.parquet")
all_sents = all_sents.sort_values(["work_id", "sent_id"]).reset_index(drop=True)
all_sents["sent_idx"] = all_sents.groupby("work_id").cumcount()
lookup = all_sents.set_index(["work_id", "sent_idx"])["sent_id"].to_dict()

rows = []
for wid in work_ids:
    db = DocBin().from_disk(PARSED / f"{wid}.spacy")
    for i, doc in enumerate(db.get_docs(nlp.vocab)):
        raw = ai_pm(doc)
        if not raw:
            continue
        spans = [doc[s:e] for _, s, e in raw]
        hits = tag_sentence(doc, spans)
        mention_terms = " | ".join(sp.text.lower() for sp in spans)
        row = {
            "work_id": wid,
            "sent_id": lookup.get((wid, i)),
            "mention_terms": mention_terms[:120],
            "sent_text": doc.text,
        }
        for c in CATS:
            row[c] = len(hits[c]) > 0
            row[f"{c}_cue"] = "|".join(sorted({h[0] for h in hits[c]}))
            row[f"{c}_anchor"] = "|".join(sorted({h[1] for h in hits[c]}))
        # Does the AI term itself belong to the entrenched set (their 4.2.1)?
        row["entrenched_term"] = any(
            sp.text.lower().strip() in ENTRENCHED
            or any(w in ENTRENCHED for w in sp.text.lower().split())
            for sp in spans
        )
        rows.append(row)
    del db

anthro = pd.DataFrame(rows)
anthro["any_category"] = anthro[CATS].any(axis=1)
print("AI-bearing sentences tagged:", len(anthro))
print(anthro[CATS].sum().sort_values(ascending=False))
"""),
    ("md", "## 2. Headline rates, with and without the entrenched terms"),
    ("code", """idx = pd.read_parquet(OUT_CLEAN / "index.parquet")[["id", "publication_year"]]
anthro = anthro.merge(idx.rename(columns={"id": "work_id"}), on="work_id", how="left")

n = len(anthro)
print(f"sentences with >=1 dependency-anchored category: "
      f"{anthro['any_category'].sum()} ({anthro['any_category'].mean():.1%})")
print(f"sentences whose AI term is itself an entrenched Cognizer/Biological term: "
      f"{anthro['entrenched_term'].sum()} ({anthro['entrenched_term'].mean():.1%})")
either = anthro["any_category"] | anthro["entrenched_term"]
print(f"anthropomorphizing under Inie et al.'s full rule (either): "
      f"{either.sum()} ({either.mean():.1%})")
"""),
    ("code", """summary = pd.DataFrame({
    "n_sentences": anthro[CATS].sum(),
    "share": anthro[CATS].mean().round(4),
}).sort_values("n_sentences", ascending=False)
print(summary.to_string())

ax = summary["n_sentences"].plot(kind="barh", figsize=(8, 4.5), color="#4c72b0")
ax.invert_yaxis()
ax.set_title("Anthropomorphizing predications about AI, by category\\n(dependency-anchored)")
ax.set_xlabel("sentences")
plt.tight_layout(); plt.show()
"""),
    ("md", """## 3. Presupposed vs asserted

Inie et al. 4.1 counts a category even under negation. Our 04e coding treats negation as refusal.
Reporting both turns an incompatibility into a measurement."""),
    ("code", """pol = pd.read_parquet(OUT_ANALYSIS / "04e_polarity_per_mention.parquet")
sent_pol = (pol.groupby(["work_id", "sent_id"])
            .agg(negated=("negated", "max"), projected=("projected", "max"))
            .reset_index())
a = anthro.merge(sent_pol, on=["work_id", "sent_id"], how="left")
for col in ("negated", "projected"):
    a[col] = a[col].fillna(False).astype(bool)

rows = []
for c in CATS:
    sub = a[a[c]]
    if len(sub) == 0:
        continue
    rows.append({
        "category": c,
        "presupposed_n": len(sub),                      # Inie et al. criterion
        "asserted_n": int((~sub["negated"]).sum()),     # our criterion
        "negated_share": round(sub["negated"].mean(), 3),
        "attributed_share": round(sub["projected"].mean(), 3),
    })
pol_tab = pd.DataFrame(rows).sort_values("negated_share", ascending=False)
print(pol_tab.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4.5))
y = np.arange(len(pol_tab))
ax.barh(y, pol_tab["presupposed_n"], color="#c9d3e6", label="presupposed (Inie et al. 4.1)")
ax.barh(y, pol_tab["asserted_n"], color="#4c72b0", label="asserted (this study)")
ax.set_yticks(y); ax.set_yticklabels(pol_tab["category"])
ax.invert_yaxis(); ax.set_xlabel("sentences")
ax.set_title("Anthropomorphizing predications: presupposed vs asserted")
ax.legend()
plt.tight_layout(); plt.show()
"""),
    ("md", "## 4. Which cues carry each category?"),
    ("code", """for c in CATS:
    sub = anthro[anthro[c]]
    if len(sub) == 0:
        continue
    cues = (sub[f"{c}_cue"].str.split("|").explode().value_counts().head(8).to_dict())
    anchors = sub[f"{c}_anchor"].str.split("|").explode().value_counts().to_dict()
    print(f"{c}  (n={len(sub)})")
    print(f"   cues: {cues}")
    print(f"   anchors: {anchors}")
"""),
    ("md", "## 5. Diachrony and register"),
    ("code", """yr = anthro.groupby("publication_year")[CATS + ["entrenched_term"]].mean().round(3)
print(yr.to_string())

top = summary.head(5).index.tolist()
ax = yr[top].plot(kind="bar", figsize=(10, 4), colormap="tab10")
ax.set_title("Share of AI sentences carrying each anthropomorphization category, by year")
ax.set_ylabel("share"); ax.set_xlabel("year")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(); plt.show()
"""),
    ("code", """prof = pd.read_parquet(OUT_ANALYSIS / "06e_work_profiles.parquet")[
    ["work_id", "venue_class", "discipline"]]
av = anthro.merge(prof, on="work_id", how="inner")
by_venue = av.groupby("venue_class")[CATS + ["entrenched_term", "any_category"]].mean().round(3)
by_venue["n_sentences"] = av.groupby("venue_class").size()
print(by_venue.to_string())

show = [c for c in ["Cognizer", "Communication", "Agent", "Human_role_analogy",
                    "Products_of_cognition", "Emotion"] if c in by_venue.columns]
ax = by_venue[show].plot(kind="bar", figsize=(11, 4), colormap="tab10")
ax.set_title("Anthropomorphization categories by venue class")
ax.set_ylabel("share of AI sentences"); ax.set_xlabel("")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.xticks(rotation=20, ha="right")
plt.tight_layout(); plt.show()
"""),
    ("code", """# Is the humanistic register more anthropomorphizing than the technical one?
av["is_book"] = av["venue_class"].eq("book / chapter").astype(int)
av["is_preprint"] = av["venue_class"].eq("preprint / repository").astype(int)
av["year_c"] = av["publication_year"] - 2024
av["anthro"] = av["any_category"].astype(int)
fit = av.dropna(subset=["publication_year"])
res = smf.logit("anthro ~ is_book + is_preprint + year_c", data=fit).fit(
    disp=False, cov_type="cluster", cov_kwds={"groups": fit["work_id"]})
tab = pd.DataFrame({
    "term": list(res.params.index),
    "odds_ratio": np.exp(res.params).round(3).values,
    "ci_low": np.exp(res.params - 1.96 * res.bse).round(3).values,
    "ci_high": np.exp(res.params + 1.96 * res.bse).round(3).values,
    "p": res.pvalues.round(5).values,
})
print("Any anthropomorphizing predication, vs journal articles:")
print(tab.to_string(index=False))
"""),
    ("md", "## 6. Examples per category"),
    ("code", """import re
def clean(s):
    return re.sub(r"\\s+", " ", str(s)).strip()

for c in CATS:
    sub = anthro[anthro[c] & anthro["sent_text"].str.len().between(50, 175)]
    if len(sub) == 0:
        continue
    print(f"\\n### {c}")
    for r in sub.drop_duplicates("sent_text").head(4).itertuples():
        cue = getattr(r, f"{c}_cue")
        print(f"  [{cue}] {clean(r.sent_text)[:170]}")
"""),
    ("md", "## 7. Persist + blind review sheet"),
    ("code", """OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
anthro.to_parquet(OUT_ANALYSIS / "06f_anthro_per_sentence.parquet", index=False)
summary.join(pol_tab.set_index("category")).to_csv(
    OUT_ANALYSIS / "06f_category_summary.csv")
by_venue.to_csv(OUT_ANALYSIS / "06f_by_venue.csv")

# Quota sample for manual coding against Inie et al.'s guidebook definitions.
parts = []
for c in CATS:
    sub = anthro[anthro[c]]
    if len(sub) == 0:
        continue
    sub = sub.sample(frac=1.0, random_state=int(RNG.integers(1 << 31)))
    sub = sub.groupby("work_id", sort=False).head(3)
    parts.append(sub.head(40).assign(auto_category=c))
neg = anthro[~anthro["any_category"]].sample(
    n=60, random_state=int(RNG.integers(1 << 31))).assign(auto_category="None")
parts.append(neg)
sample = pd.concat(parts, ignore_index=True)
sample = sample.sample(frac=1.0, random_state=20260908).reset_index(drop=True)
sample["row_uid"] = [f"06f-{i:04d}" for i in range(len(sample))]

sample[["row_uid", "work_id", "sent_id", "auto_category"]].to_csv(
    OUT_ANALYSIS / "06f_anthro_sample_key.csv", index=False)
sheet = sample[["row_uid", "mention_terms", "sent_text"]].copy()
sheet["manual_categories"] = ""   # semicolon-separated; see legend
sheet["notes"] = ""
sheet.to_csv(OUT_ANALYSIS / "06f_anthro_sample.csv", index=False, encoding="utf-8-sig")
print("saved", len(sheet), "rows for manual coding")
summary
"""),
]

write(HERE / "06f_anthropomorphization.ipynb", cells)
