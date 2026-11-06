"""Generate 04b_modality.ipynb — clause-level modality & hedging."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _nbbuild import write

HERE = Path(__file__).parent

cells = [
    ("md", """# 04b — Modality & hedging, one row per AI predicate clause

A sentence is too large a unit for modality: "GenAI **can** assist philologists, but it **must not**
substitute human experience" contains two AI propositions with different commitments, and a
sentence-level tagger either mislabels it `mixed` or discards it as uncertain.

This notebook therefore anchors on **AI mention spans** (same matcher as the pipeline), finds each
mention's governing predicate, and labels the predicate's **local clause** — embedded clauses
excluded. The unit of analysis is the AI predicate clause; the sentence-level file is derived from
it by aggregation, for backwards compatibility.

Outputs:
- `outputs/analysis/04b_modality_per_clause.parquet` / `.csv` — the primary output
- `outputs/analysis/04b_modality_per_sentence.csv` — derived aggregate + manual-coding sheet
- `outputs/analysis/04b_modality_summary.parquet` — clause-level counts
"""),
    ("code", """from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path.cwd()
while ROOT.name and not (ROOT / "analysis_code").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "analysis_code"))

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from spacy.tokens import DocBin

from lib import OUT_PARSED, OUT_ANALYSIS, OUT_CLEAN
from lib.ai_lexicon import build_phrase_matcher
from lib.predicate_scope import clause_tokens, verbal_group

sns.set_theme(context="notebook", style="whitegrid")

STRONG_MODALS = {"will", "shall", "must", "cannot"}
WEAK_MODALS = {"may", "might", "could", "would", "can", "should", "ought"}
HEDGE_LEMMAS = {
    "seem", "appear", "tend", "suggest", "indicate", "imply",
    "likely", "possible", "potential", "probable",
}
HEDGE_ADVERBS = {
    "perhaps", "possibly", "arguably", "potentially", "often",
    "generally", "presumably", "apparently", "probably",
    "supposedly", "seemingly", "allegedly",
}
BOOSTER_ADVERBS = {
    "clearly", "obviously", "certainly", "undoubtedly",
    "definitely", "evidently", "indeed",
}

HEDGE_ADV_RE = re.compile(rf"\\b(?:{'|'.join(sorted(HEDGE_ADVERBS))})\\b")
BOOSTER_RE = re.compile(rf"\\b(?:{'|'.join(sorted(BOOSTER_ADVERBS))})\\b")

SUBJ_OBJ_DEPS = {"nsubj", "nsubjpass", "nsubj:pass", "dobj", "obj"}

ai = pd.read_parquet(OUT_PARSED / "mentions.parquet")
ai = ai[ai["category"] == "AI"]
clause_ai = ai[ai["head_dep"].isin(SUBJ_OBJ_DEPS)]
wanted_sents = clause_ai.drop_duplicates(["work_id", "sent_id"])
print("AI mentions:", len(ai), " - sentences with a clause-participant AI mention:",
      len(wanted_sents))
"""),
    ("md", "## 1. Clause segmentation: one record per AI-governing predicate"),
    ("code", """# POS/dep/lemma are baked into the DocBins; a blank vocab deserialises them.
nlp = spacy.blank("en")
ai_pm = build_phrase_matcher(nlp, label="AI")
PARSED = OUT_PARSED / "docs"


def _governing_predicate(tok):
    \"\"\"Nearest VERB/AUX at or above the mention head.\"\"\"
    cur, hops = tok, 0
    while cur.pos_ not in ("VERB", "AUX") and cur.head is not cur and hops < 12:
        cur, hops = cur.head, hops + 1
    return cur if cur.pos_ in ("VERB", "AUX") else None


def _label(strong, weak, hedge_lem, hedge_adv, boosters):
    if weak or hedge_lem or hedge_adv:
        return "mixed" if (strong or boosters) else "weak/hedged"
    if strong or boosters:
        return "strong"
    return "bare"


def probe_clauses(doc) -> list[dict]:
    \"\"\"One record per predicate that governs an AI clause participant.

    Anchoring on AI mention spans (not on any subject/object token) keeps
    human-subject predicates out. Coordinated predicates sharing one auxiliary
    both inherit the modal, because conjuncts stay inside clause_tokens while
    embedded clauses are cut.
    \"\"\"
    hits = ai_pm(doc)
    if not hits:
        return []
    by_pred: dict[int, list] = {}
    for _, start, end in hits:
        span = doc[start:end]
        if span.root.dep_ not in SUBJ_OBJ_DEPS:
            continue
        pred = _governing_predicate(span.root)
        if pred is None:
            continue
        by_pred.setdefault(pred.i, []).append(span)

    records = []
    for pred_i, spans in sorted(by_pred.items()):
        pred = doc[pred_i]
        local = clause_tokens(pred)
        lemmas = {t.lemma_.lower() for t in local}
        text_l = " ".join(t.text for t in local).lower()

        strong = sorted(lemmas & STRONG_MODALS)
        weak = sorted(lemmas & WEAK_MODALS)
        hedge_lem = sorted(lemmas & HEDGE_LEMMAS)
        hedge_adv = sorted(set(HEDGE_ADV_RE.findall(text_l)))
        boosters = sorted(set(BOOSTER_RE.findall(text_l)))
        aux = sorted({t.lemma_.lower() for t in verbal_group(pred) if t.pos_ == "AUX"})

        records.append({
            "predicate_i": pred_i,
            "predicate_lemma": pred.lemma_.lower(),
            "clause_text": " ".join(t.text for t in local),
            "ai_mentions": " | ".join(s.text for s in spans),
            "n_ai_mentions": len(spans),
            "auto_label": _label(strong, weak, hedge_lem, hedge_adv, boosters),
            "modal_strong": "|".join(strong),
            "modal_weak": "|".join(weak),
            "hedge_lemmas": "|".join(hedge_lem),
            "hedge_adverbs": "|".join(hedge_adv),
            "boosters": "|".join(boosters),
            "aux_lemmas": "|".join(aux),
        })
    return records
"""),
    ("code", """all_sents = pd.read_parquet(OUT_PARSED / "sentences.parquet")
all_sents = all_sents.sort_values(["work_id", "sent_id"]).reset_index(drop=True)
all_sents["local_idx"] = all_sents.groupby("work_id").cumcount()
local_lookup = all_sents.set_index(["work_id", "sent_id"])["local_idx"].to_dict()

rows = []
for work_id, grp in wanted_sents.groupby("work_id", sort=False):
    wanted = {local_lookup[(work_id, sid)]: sid for sid in grp["sent_id"]}
    last = max(wanted)
    db = DocBin().from_disk(PARSED / f"{work_id}.spacy")
    for i, doc in enumerate(db.get_docs(nlp.vocab)):
        if i in wanted:
            for k, rec in enumerate(probe_clauses(doc)):
                rec["work_id"] = work_id
                rec["sent_id"] = wanted[i]
                rec["clause_id"] = f"{work_id}-{wanted[i]}-{k}"
                rec["sent_text"] = doc.text
                rows.append(rec)
        if i >= last:
            break
    del db

clauses = pd.DataFrame(rows)
print("AI predicate clauses:", len(clauses),
      "in", clauses[["work_id", "sent_id"]].drop_duplicates().shape[0], "sentences")
clauses["auto_label"].value_counts()
"""),
    ("md", "## 2. Clause-level results"),
    ("code", """idx = pd.read_parquet(OUT_CLEAN / "index.parquet")[["id", "publication_year"]]
clauses = clauses.merge(idx.rename(columns={"id": "work_id"}), on="work_id", how="left")

summary = clauses["auto_label"].value_counts().rename("n").to_frame()
summary["share"] = (summary["n"] / summary["n"].sum()).round(3)
print(summary)

yr_mod = pd.crosstab(clauses["publication_year"], clauses["auto_label"],
                     normalize="index").round(3)
print(yr_mod)

ax = yr_mod.plot(kind="bar", stacked=True, figsize=(9, 4), colormap="Set2")
ax.set_title("Modality of AI predicate clauses (auto-labels) by year")
ax.set_ylabel("share"); ax.set_xlabel("year")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(); plt.show()
"""),
    ("code", """# How much did clause segmentation matter?
per_sent = clauses.groupby(["work_id", "sent_id"])["auto_label"].agg(
    n_clauses="size", labels=lambda s: "|".join(sorted(set(s))))
multi = per_sent[per_sent["n_clauses"] > 1]
conflicting = multi["labels"].str.contains("|", regex=False).sum()
print(f"sentences with >1 AI clause: {len(multi)} "
      f"({len(multi) / len(per_sent):.1%} of sentences)")
print(f"  of which with conflicting labels: {conflicting}")
print("genuinely mixed clauses (strong+weak in ONE clause):",
      int((clauses["auto_label"] == "mixed").sum()))
"""),
    ("md", "## 3. Persist: clause-level primary + sentence-level derived"),
    ("code", """OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)

clause_cols = ["work_id", "sent_id", "clause_id", "publication_year",
               "predicate_lemma", "ai_mentions", "n_ai_mentions",
               "clause_text", "sent_text", "auto_label",
               "modal_strong", "modal_weak", "hedge_lemmas", "hedge_adverbs",
               "boosters", "aux_lemmas"]
clauses[clause_cols].to_parquet(OUT_ANALYSIS / "04b_modality_per_clause.parquet",
                                index=False)
csv = clauses[clause_cols].copy()
csv["manual_label"] = ""
csv["notes"] = ""
csv.to_csv(OUT_ANALYSIS / "04b_modality_per_clause.csv",
           index=False, encoding="utf-8-sig")
print("clause-level:", len(csv), "rows")

summary.to_parquet(OUT_ANALYSIS / "04b_modality_summary.parquet")
"""),
    ("code", """# Sentence-level file, derived by aggregation (legacy schema).
def _sent_label(labels):
    uniq = set(labels)
    return labels.iloc[0] if len(uniq) == 1 else "scope_uncertain"


def _join(series):
    return "|".join(sorted({v for x in series for v in str(x).split("|") if v}))


sent = (clauses.groupby(["work_id", "sent_id"], as_index=False)
        .agg(publication_year=("publication_year", "first"),
             sent_text=("sent_text", "first"),
             auto_label=("auto_label", _sent_label),
             n_clauses=("clause_id", "size"),
             scope_labels=("auto_label", lambda s: "|".join(sorted(set(s)))),
             modal_strong=("modal_strong", _join),
             modal_weak=("modal_weak", _join),
             hedge_lemmas=("hedge_lemmas", _join),
             hedge_adverbs=("hedge_adverbs", _join),
             boosters=("boosters", _join),
             aux_lemmas=("aux_lemmas", _join)))
sent["scope_confidence"] = (sent["n_clauses"] == 1).map({True: "high", False: "medium"})
sent["scope_note"] = (sent["n_clauses"] == 1).map(
    {True: "single AI clause", False: "aggregated over multiple AI clauses"})
sent["manual_label"] = ""
sent["notes"] = ""
sent.to_csv(OUT_ANALYSIS / "04b_modality_per_sentence.csv",
            index=False, encoding="utf-8-sig")
print("sentence-level (derived):", len(sent), "rows")
print(sent["auto_label"].value_counts())
"""),
]

write(HERE / "04b_modality.ipynb", cells)
