"""Generate 04c_valency.ipynb — semantic valency / transitivity of verbs
governing AI-as-subject mentions."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _nbbuild import write

HERE = Path(__file__).parent

cells = [
    ("md", """# 04c — Semantic valency & transitivity

For every AI mention that is the **active subject** of a verb (from 04a), we collect the verb lemma and auto-bucket it into seed classes:

- **material/action** — *produce, generate, create, write, transform, replace, destroy, automate, process*
- **cognitive/mental** — *think, know, understand, learn, decide, infer, perceive, imagine, recognise*
- **instrumental/tool** — *assist, support, enable, help, facilitate, allow, serve*
- **verbal/communicative** — *say, claim, suggest, explain, describe, respond, answer, report*
- **stative/relational** — *be, become, remain, seem, appear*, plus possessive-relational *have, possess, lack* and existential *exist, emerge, evolve*
- **behavioural** — *operate, function, work, act, interact, collaborate* (Halliday's sixth process type)
- **evaluative/performance** — *achieve, outperform, excel, struggle, fail, score, lag*
- **semiotic/evidential** — *demonstrate, show, exhibit, reveal, indicate*

The last three buckets were added on 2026-09-07 after auditing the `OTHER` bin, which held 56% of tokens under the original five-way scheme.

The ranked-verbs CSV is the **manual-coding sheet**: every verb gets a row with `auto_class` as a suggestion and `manual_class` for the analyst to fill.

Outputs:
- `outputs/analysis/04c_verbs_ranked.csv` — manual sheet
- `outputs/analysis/04c_verb_class_summary.parquet`
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

VERB_CLASSES = {
    "material_action": {
        "produce", "generate", "create", "write", "make", "build", "design",
        "transform", "replace", "destroy", "automate", "process", "perform",
        "compute", "calculate", "render", "draw", "paint", "compose",
        "extract", "detect", "classify", "predict", "analyze", "analyse",
        "translate", "summarize", "summarise", "rewrite", "edit", "annotate",
        # added after auditing the OTHER bin (2026-09-07)
        "identify", "simulate", "encode", "replicate", "reproduce", "transcribe",
        "decompose", "construct", "train", "correct", "output", "use", "apply",
        "reshape", "change", "revolutionize", "revolutionise", "disrupt",
        "impact", "influence", "affect", "shape", "accelerate", "enhance",
        "improve", "introduce", "handle", "adapt",
    },
    "cognitive_mental": {
        "think", "know", "understand", "learn", "decide", "infer", "perceive",
        "imagine", "recognize", "recognise", "remember", "comprehend",
        "interpret", "consider", "believe", "assume", "reason",
        "hallucinate", "dream",
        # added after auditing the OTHER bin
        "distinguish", "compare", "evaluate", "assess", "approximate",
    },
    "instrumental_tool": {
        "assist", "support", "enable", "help", "facilitate", "allow",
        "serve", "provide", "offer", "deliver", "augment", "complement",
        "aid",
        # added after auditing the OTHER bin
        "empower", "benefit", "contribute", "foster", "promote",
    },
    "verbal_communicative": {
        "say", "claim", "suggest", "explain", "describe", "respond",
        "answer", "report", "tell", "argue", "state", "mention",
        "express", "discuss", "communicate", "write",  # 'write' is dual-listed
    },
    "stative_relational": {
        "be", "become", "remain", "seem", "appear", "represent",
        "constitute", "comprise", "form",
        # possessive-relational + existential, added after auditing OTHER
        "have", "possess", "lack", "hold", "contain", "include", "feature",
        "exist", "emerge", "evolve", "grow", "expand", "continue", "tend",
    },
    # Behavioural is Halliday's sixth process type: 'AI operates/functions/works'
    # construes conduct without a Goal.
    "behavioural": {
        "operate", "function", "work", "act", "behave", "run",
        "interact", "collaborate", "engage", "participate", "play",
    },
    # Benchmark/competition talk — the register-specific cluster the original
    # five buckets had no home for.
    "evaluative_performance": {
        "achieve", "outperform", "excel", "struggle", "fail", "succeed",
        "surpass", "exceed", "score", "match", "lag", "underperform", "rank",
        "judge", "evaluate", "assess",
    },
    # 'the model demonstrates/exhibits X' — AI as source of evidence, adjacent
    # to Sayer but not speech.
    "semiotic_evidential": {
        "demonstrate", "show", "exhibit", "reveal", "present", "indicate",
        "illustrate", "highlight", "signal", "prove",
    },
}

PHASE_OR_PURPOSIVE = {
    "aim", "begin", "cease", "continue", "finish", "keep", "plan", "seek",
    "start", "stop", "tend", "try", "attempt", "intend", "hope",
}
# Build reverse lookup; resolve ties by class order above
LEMMA2CLASS: dict[str, str] = {}
for cls, lemmas in VERB_CLASSES.items():
    for lem in lemmas:
        LEMMA2CLASS.setdefault(lem, cls)


def _classify_predicate(lemma: str) -> tuple[str, str]:
    lemma = str(lemma).lower()
    if lemma in PHASE_OR_PURPOSIVE:
        return "OTHER", "phase_or_catenative"
    return LEMMA2CLASS.get(lemma, "OTHER"), "lexical"


print("seed lemmas:", len(LEMMA2CLASS))
"""),
    ("md", "## 1. AI-as-active-subject rows from Phase 4a"),
    ("code", """ag = pd.read_parquet(OUT_ANALYSIS / "04a_agency_per_mention.parquet")
active = ag[ag["role"] == "active_subject"].copy()
print("AI-as-active-subject mentions:", len(active))

# governor_lemma is the verb (we filtered to gov POS VERB/AUX in spec, but
# keep only verbs here)
active = active[active["governor_pos"].isin(["VERB", "AUX"])].copy()
active["predicate_lemma"] = active.get("governor_predicate", active["governor_lemma"])
active["construction"] = active.get("governor_construction", "simple")
active["class_lemma"] = active["predicate_lemma"]
if "governor_complement_lemma" in active:
    catenative = active["construction"].eq("catenative")
    active.loc[catenative, "class_lemma"] = active.loc[catenative, "governor_complement_lemma"]
print("with verbal governor:", len(active))
active.head()
"""),
    ("code", """vcounts = (
    active["predicate_lemma"].value_counts()
    .rename_axis("verb_lemma").reset_index(name="n")
)
class_lookup = active.groupby("predicate_lemma")["class_lemma"].first().to_dict()
construction_lookup = active.groupby("predicate_lemma")["construction"].first().to_dict()
vcounts["auto_class"] = vcounts["verb_lemma"].map(
    lambda lemma: _classify_predicate(class_lookup.get(lemma, lemma))[0])
vcounts["construction"] = vcounts["verb_lemma"].map(construction_lookup).fillna("simple")
print("distinct verbs:", len(vcounts))
print("top 30:")
print(vcounts.head(30).to_string(index=False))
"""),
    ("code", """fig, ax = plt.subplots(figsize=(10, 6))
top = vcounts.head(30).iloc[::-1]
colors = {
    "material_action": "#4c72b0",
    "cognitive_mental": "#dd8452",
    "instrumental_tool": "#55a467",
    "verbal_communicative": "#c44e52",
    "stative_relational": "#8172b3",
    "behavioural": "#937860",
    "evaluative_performance": "#da8bc3",
    "semiotic_evidential": "#8c8c8c",
    "OTHER": "#cccccc",
}
ax.barh(top["verb_lemma"], top["n"], color=[colors.get(c, "#cccccc") for c in top["auto_class"]])
ax.set_title("Top 30 verbs governing AI-as-active-subject (auto-coloured by class)")
ax.set_xlabel("mentions")
plt.tight_layout(); plt.show()
"""),
    ("md", "## 3. Class summary + diachronic"),
    ("code", """active[["auto_class", "construction"]] = active["governor_lemma"].apply(
    lambda lemma: pd.Series(_classify_predicate(lemma)))
class_summary = active["auto_class"].value_counts()
class_summary
"""),
    ("code", """yr_class = pd.crosstab(active["publication_year"], active["auto_class"], normalize="index").round(3)
print(yr_class)
ax = yr_class.plot(kind="bar", stacked=True, figsize=(9, 4), colormap="tab10")
ax.set_title("Verb-class distribution of AI-as-subject by year (auto)")
ax.set_ylabel("share"); ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(); plt.show()
"""),
    ("md", "## 4. Manual-coding CSV (per-verb)"),
    ("code", """OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
ranked = vcounts.copy()
ranked["manual_class"] = ""
ranked["notes"] = ""
# Include up to 3 example sentences per verb
examples = (
    active.groupby("governor_lemma")["sent_text"]
    .apply(lambda s: " | ".join(s.head(3).str.slice(0, 200)))
    .rename("examples")
)
ranked = ranked.merge(examples, left_on="verb_lemma", right_index=True, how="left")

out_csv = OUT_ANALYSIS / "04c_verbs_ranked.csv"
ranked.to_csv(out_csv, index=False, encoding="utf-8-sig")
print("saved", out_csv, "rows:", len(ranked))

class_summary.to_frame("n").assign(
    share=lambda d: (d["n"] / d["n"].sum()).round(3)
).to_parquet(OUT_ANALYSIS / "04c_verb_class_summary.parquet")
ranked.head(20)
"""),
]

write(HERE / "04c_valency.ipynb", cells)
