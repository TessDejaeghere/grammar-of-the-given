"""Generate 04e_negation.ipynb — negation scope & projection around AI mentions."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _nbbuild import write

HERE = Path(__file__).parent

cells = [
    ("md", """# 04e — Negation scope & projection

Frame *presence* is not frame *endorsement*. "AI replaces philology" and "the article does not
claim that AI replaces philology" both fire the AI-as-threat cue, but they make opposite claims.
This notebook codes, for every AI mention, whether the process construing it is

1. **negated** — and by what: a `neg` dependency on its own verb, a negative-polarity item inside
   its clause, or a negation on a verb that *projects* it;
2. **projected** — embedded under a saying/thinking verb, in which case the proposition is
   attributed to someone rather than asserted by the author, and we record who.

Crossing the two gives four stances: `asserted_positive`, `asserted_negative`,
`projected_positive`, `projected_negative`.

Outputs:
- `outputs/analysis/04e_polarity_per_mention.parquet`
- `outputs/analysis/04e_polarity_per_mention.csv` — manual-review sheet
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
import spacy
from spacy.tokens import DocBin

from lib import OUT_PARSED, OUT_ANALYSIS, OUT_CLEAN
from lib.ai_lexicon import build_phrase_matcher
from lib.actors_lexicon import HUMAN_TERMS, COMPANY_TERMS

sns.set_theme(context="notebook", style="whitegrid")

# Negative-polarity items that negate a clause without a `neg` dependency.
NPI_LEMMAS = {
    "never", "no", "none", "nothing", "nobody", "neither", "nor",
    "without", "hardly", "barely", "rarely", "seldom", "scarcely",
    "fail", "failure", "unable", "incapable", "lack", "absent",
}
NPI_PHRASES = ("rather than", "instead of", "far from", "little more than")

# Verbs that project a proposition (Halliday) \u2014 the clause below them is
# attributed, not asserted.
PROJECTING_LEMMAS = {
    "say", "claim", "argue", "suggest", "note", "observe", "remark",
    "believe", "think", "assume", "contend", "conclude", "report",
    "show", "demonstrate", "indicate", "reveal", "find", "prove",
    "fear", "worry", "hope", "expect", "predict", "warn",
    "deny", "doubt", "question", "acknowledge", "admit", "recognize",
    "state", "write", "hypothesize", "posit", "propose", "assert",
}
# Clausal dependencies that make the AI clause subordinate to a projector.
PROJECTED_DEPS = {"ccomp", "xcomp", "acl", "advcl", "relcl", "pcomp", "csubj"}

HUMAN_LOWER = {t.lower() for t in HUMAN_TERMS}
COMPANY_LOWER = {t.lower() for t in COMPANY_TERMS}

ment = pd.read_parquet(OUT_PARSED / "mentions.parquet")
ai = ment[ment["category"] == "AI"].copy()
# Matcher hits are appended per sentence in regex order, so a running count
# within (work_id, sent_id) recovers the same key on both sides of the join.
ai["ai_idx"] = ai.groupby(["work_id", "sent_id"]).cumcount()
print("AI mentions:", len(ai))
"""),
    ("md", "## 1. Clause scope, negation and projection helpers"),
    ("code", """def _governing_verb(tok):
    \"\"\"Nearest VERB/AUX at or above the mention head.\"\"\"
    cur = tok
    seen = 0
    while cur.pos_ not in ("VERB", "AUX") and cur.head is not cur and seen < 12:
        cur = cur.head
        seen += 1
    return cur if cur.pos_ in ("VERB", "AUX") else None


def _local_clause(verb):
    \"\"\"Subtree of `verb` minus any embedded clauses hanging off it.\"\"\"
    blocked = {c.i for ch in verb.children if ch.dep_ in PROJECTED_DEPS
               for c in ch.subtree}
    return [t for t in verb.subtree if t.i not in blocked]


def _negation(verb, clause):
    \"\"\"Return (is_negated, source, cue).\"\"\"
    for t in verb.children:
        if t.dep_ == "neg":
            return True, "direct", t.text.lower()
        if t.pos_ == "AUX":
            for c in t.children:
                if c.dep_ == "neg":
                    return True, "direct", c.text.lower()
    for t in clause:
        if t.lemma_.lower() in NPI_LEMMAS:
            return True, "npi", t.text.lower()
    text_l = " ".join(t.text for t in clause).lower()
    for p in NPI_PHRASES:
        if p in text_l:
            return True, "npi", p
    return False, "none", ""


def _subject_of(verb):
    for c in verb.children:
        if c.dep_ in ("nsubj", "nsubjpass", "csubj"):
            return c
    return None


def _actor_category(tok, ai_token_ids):
    if tok is None:
        return ""
    words = {t.text.lower() for t in tok.subtree}
    span_text = " ".join(t.text.lower() for t in tok.subtree)
    if any(t.i in ai_token_ids for t in tok.subtree):
        return "AI"
    if any((w in words) if " " not in w else (w in span_text) for w in HUMAN_LOWER):
        return "HUMAN"
    if any((c in words) if " " not in c else (c in span_text) for c in COMPANY_LOWER):
        return "COMPANY"
    return "OTHER"


def _projection(verb):
    \"\"\"Walk up from `verb` to the nearest projecting verb, if any.\"\"\"
    cur = verb
    hops = 0
    while cur.head is not cur and hops < 6:
        dep = cur.dep_
        parent = cur.head
        if dep in PROJECTED_DEPS and parent.pos_ in ("VERB", "AUX"):
            return parent, dep
        cur = parent
        hops += 1
    return None, ""
"""),
    ("md", "## 2. Walk the DocBins and code every AI mention"),
    ("code", """nlp = spacy.blank("en")
ai_pm = build_phrase_matcher(nlp, label="AI")
PARSED = OUT_PARSED / "docs"

all_sents = pd.read_parquet(OUT_PARSED / "sentences.parquet")
all_sents = all_sents.sort_values(["work_id", "sent_id"]).reset_index(drop=True)
all_sents["local_idx"] = all_sents.groupby("work_id").cumcount()
local_lookup = all_sents.set_index(["work_id", "sent_id"])["local_idx"].to_dict()

wanted_sents = ai[["work_id", "sent_id"]].drop_duplicates()
print("sentences to code:", len(wanted_sents))

rows = []
for work_id, grp in wanted_sents.groupby("work_id", sort=False):
    wanted = {local_lookup[(work_id, s)]: s for s in grp["sent_id"]}
    last = max(wanted)
    db = DocBin().from_disk(PARSED / f"{work_id}.spacy")
    for i, doc in enumerate(db.get_docs(nlp.vocab)):
        if i in wanted:
            hits = ai_pm(doc)
            ai_token_ids = {t for _, s, e in hits for t in range(s, e)}
            for k, (_, start, end) in enumerate(hits):
                span = doc[start:end]
                verb = _governing_verb(span.root)
                if verb is None:
                    continue
                clause = _local_clause(verb)
                neg, neg_source, neg_cue = _negation(verb, clause)
                projector, proj_dep = _projection(verb)
                proj_neg = False
                proj_lemma = proj_subj = proj_cat = ""
                projecting = False
                if projector is not None:
                    proj_lemma = projector.lemma_.lower()
                    projecting = proj_lemma in PROJECTING_LEMMAS
                    proj_neg = _negation(projector, _local_clause(projector))[0]
                    subj = _subject_of(projector)
                    proj_subj = subj.text if subj is not None else ""
                    proj_cat = _actor_category(subj, ai_token_ids)
                if not neg and projecting and proj_neg:
                    neg, neg_source, neg_cue = True, "projector", proj_lemma
                rows.append({
                    "work_id": work_id,
                    "sent_id": wanted[i],
                    "ai_idx": k,
                    "mention_text": span.text,
                    "gov_verb": verb.lemma_.lower(),
                    "negated": neg,
                    "neg_source": neg_source,
                    "neg_cue": neg_cue,
                    "projected": projecting,
                    "proj_verb": proj_lemma if projecting else "",
                    "proj_dep": proj_dep if projecting else "",
                    "proj_subject": proj_subj if projecting else "",
                    "proj_subject_cat": proj_cat if projecting else "",
                    "sent_text": doc.text,
                })
        if i >= last:
            break
    del db

pol = pd.DataFrame(rows)
pol["polarity"] = pol["negated"].map({True: "negative", False: "positive"})
pol["stance"] = (
    pol["projected"].map({True: "projected", False: "asserted"})
    + "_" + pol["polarity"]
)
print("coded mentions:", len(pol))
print(pol["stance"].value_counts())
print(pol["neg_source"].value_counts())
"""),
    ("md", "## 3. How much of the corpus is negated or attributed?"),
    ("code", """idx = pd.read_parquet(OUT_CLEAN / "index.parquet")[["id", "publication_year"]]
pol = pol.merge(idx.rename(columns={"id": "work_id"}), on="work_id", how="left")

overall = pol["stance"].value_counts(normalize=True).round(3)
print(overall)

yr = pd.crosstab(pol["publication_year"], pol["stance"], normalize="index").round(3)
print(yr)

ax = yr.plot(kind="bar", stacked=True, figsize=(9, 4), colormap="Set2")
ax.set_title("Stance of AI-construing clauses by year")
ax.set_ylabel("share"); ax.set_xlabel("year")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(); plt.show()
"""),
    ("code", """# Who is credited when the proposition is attributed rather than asserted?
proj = pol[pol["projected"]]
print("projected mentions:", len(proj))
print(proj["proj_subject_cat"].value_counts(normalize=True).round(3))
print()
print("top projecting verbs:")
print(proj["proj_verb"].value_counts().head(15))
"""),
    ("md", "## 4. Which processes get denied? (negation by verb)"),
    ("code", """by_verb = (
    pol.groupby("gov_verb")
    .agg(n=("negated", "size"), neg_rate=("negated", "mean"))
    .query("n >= 20")
    .sort_values("neg_rate", ascending=False)
)
print("most-negated verbs governing AI (n >= 20):")
print(by_verb.head(20).round(3))
print()
print("least-negated:")
print(by_verb.tail(10).round(3))
"""),
    ("md", "## 5. Cross with 04c verb classes and 05c frames"),
    ("code", """vr = pd.read_csv(OUT_ANALYSIS / "04c_verbs_ranked.csv")
lemma2class = dict(zip(vr["verb_lemma"], vr["auto_class"]))
pol["verb_class"] = pol["gov_verb"].map(lemma2class).fillna("OTHER")

cls = (
    pol.groupby("verb_class")
    .agg(n=("negated", "size"), neg_rate=("negated", "mean"),
         proj_rate=("projected", "mean"))
    .sort_values("n", ascending=False)
    .round(3)
)
print(cls)
"""),
    ("code", """frames = pd.read_parquet(OUT_ANALYSIS / "05c_frames_per_sentence.parquet")
frame_cols = [c for c in frames.columns if c.startswith("AI_as_") and c.endswith("_strict")]

# Sentence-level polarity: a sentence counts as negated if any AI-construing
# clause in it is negated.
sent_pol = (
    pol.groupby(["work_id", "sent_id"])
    .agg(any_neg=("negated", "max"), any_proj=("projected", "max"))
    .reset_index()
)
fr = frames.merge(sent_pol, on=["work_id", "sent_id"], how="left")

rowsf = []
for c in frame_cols:
    sub = fr[fr[c].fillna(False)]
    if len(sub) == 0:
        continue
    rowsf.append({
        "frame": c.replace("_strict", ""),
        "n": len(sub),
        "negated_share": round(sub["any_neg"].mean(), 3),
        "projected_share": round(sub["any_proj"].mean(), 3),
    })
frame_pol = pd.DataFrame(rowsf).sort_values("negated_share", ascending=False)
print("strict frame hits, share that are negated / attributed:")
print(frame_pol.to_string(index=False))
"""),
    ("md", "## 6. Persist + examples"),
    ("code", """OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
pol.to_parquet(OUT_ANALYSIS / "04e_polarity_per_mention.parquet", index=False)

sheet = pol[[
    "work_id", "sent_id", "ai_idx", "publication_year", "mention_text",
    "gov_verb", "verb_class", "negated", "neg_source", "neg_cue",
    "projected", "proj_verb", "proj_subject", "proj_subject_cat",
    "stance", "sent_text",
]].copy()
sheet["manual_stance"] = ""
sheet["notes"] = ""
sheet.to_csv(OUT_ANALYSIS / "04e_polarity_per_mention.csv",
             index=False, encoding="utf-8-sig")
print("saved", len(sheet), "rows")

import re
def _clean(s):
    return re.sub(r"\\s+", " ", str(s)).strip()

for st in ["asserted_negative", "projected_positive", "projected_negative"]:
    print("\\n##", st)
    sub = pol[(pol["stance"] == st) & pol["sent_text"].str.len().between(60, 180)]
    for r in sub.head(5).itertuples():
        print("  -", _clean(r.sent_text)[:180])
"""),
]

write(HERE / "04e_negation.ipynb", cells)
