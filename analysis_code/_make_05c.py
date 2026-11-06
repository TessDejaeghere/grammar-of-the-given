"""Generate 05c_metaphor.ipynb — metaphor / frame detection around AI."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _nbbuild import write

HERE = Path(__file__).parent

cells = [
    ("md", """# 05c — Metaphor / frame detection around AI

Rule-based frame tagging of AI-mention sentences using seed lexicons:

| frame | cues |
|-------|------|
| **AI-as-human** | cognitive/mental verbs ("think", "understand", "imagine", "feel"); perception verbs ("see", "hear", "perceive"); social verbs ("collaborate", "converse", "lie") |
| **AI-as-tool**  | instrumental verbs ("use", "deploy", "leverage", "apply", "employ"); possessive frames ("our AI", "the AI system") |
| **AI-as-threat / agent of change** | "replace", "threaten", "disrupt", "transform", "automate", "displace", "obsolete"; military / disaster vocabulary |
| **AI-as-oracle / authority** | "predict", "decide", "judge", "evaluate", "recommend"; "according to" |

Sentences can carry multiple frames; we record all that fire.
"""),
    ("code", """from __future__ import annotations
import sys, re
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

FRAMES = {
    "AI_as_human": {
        "think", "understand", "know", "believe", "imagine", "feel",
        "perceive", "remember", "dream", "intend",
        "collaborate", "converse", "speak", "lie", "deceive",
        "want", "desire", "consider", "judge",
    },
    "AI_as_tool": {
        "use", "deploy", "leverage", "apply", "employ", "utilize", "utilise",
        "operate", "run", "configure",
    },
    "AI_as_threat": {
        "replace", "threaten", "disrupt", "displace", "obsolete",
        "destabilize", "destabilise", "endanger", "undermine",
        "automate", "eliminate",
    },
    "AI_as_oracle": {
        "predict", "forecast", "decide", "judge", "evaluate",
        "recommend", "diagnose", "rank", "score",
    },
    "AI_as_transformer": {
        "transform", "reshape", "revolutionize", "revolutionise",
        "change", "reinvent", "remake",
    },
}

ment = pd.read_parquet(OUT_PARSED / "mentions.parquet")
ai = ment[ment["category"] == "AI"].copy()
print("AI mentions:", len(ai))
"""),
    ("md", "## 1. Tag each AI-bearing sentence with frames"),
    ("code", """# Work per sentence (dedup AI rows by sent)
sents = ai.drop_duplicates(["work_id", "sent_id"]).copy()
sent_texts = sents.set_index(["work_id", "sent_id"])["sent_text"]

# Verbs that actually govern an AI subject in each sentence \u2014 the anchor for
# the strict variant, which ignores cues that merely occur somewhere nearby
# ("see Bergmann", "Judging llm-as-a-judge").
ai_subj_verbs = (
    ai[ai["head_dep"] == "nsubj"]
    .groupby(["work_id", "sent_id"])["governor_lemma"]
    .apply(lambda s: set(s.str.lower()))
)


def _tag(text: str) -> dict:
    tl = text.lower()
    out = {}
    for frame, lemmas in FRAMES.items():
        hit = ""
        for lem in sorted(lemmas):
            if (re.search(rf"\\b{lem}s?\\b", tl) or re.search(rf"\\b{lem}ing\\b", tl)
                    or re.search(rf"\\b{lem}ed\\b", tl)):
                hit = lem
                break
        out[frame] = bool(hit)
        out[f"{frame}_cue"] = hit
    return out


tagged = sent_texts.map(_tag).apply(pd.Series)
tagged.index.names = ["work_id", "sent_id"]
for frame, lemmas in FRAMES.items():
    verbs = tagged.index.map(lambda k: ai_subj_verbs.get(k, frozenset()))
    tagged[f"{frame}_strict"] = [bool(v & lemmas) for v in verbs]
tagged["text"] = sent_texts.values
print(tagged[[f for f in FRAMES]].sum())
print(tagged[[f"{f}_strict" for f in FRAMES]].sum())
"""),
    ("md", "## 2. Frame counts + per-year"),
    ("code", """tagged = tagged.reset_index()
idx = pd.read_parquet(OUT_CLEAN / "index.parquet")[["id", "publication_year"]]
tagged = tagged.merge(idx.rename(columns={"id": "work_id"}), on="work_id", how="left")

frame_cols = list(FRAMES.keys())
totals = tagged[frame_cols].sum().sort_values(ascending=False)
print("total sentences per frame:")
print(totals)

fig, ax = plt.subplots(figsize=(8, 3.5))
totals.plot(kind="bar", ax=ax, color="#4c72b0")
ax.set_title("Frame counts (AI-mention sentences)")
ax.set_ylabel("sentences")
plt.tight_layout(); plt.show()
"""),
    ("code", """yr = tagged.groupby("publication_year")[frame_cols].mean()
print("share of AI sentences carrying each frame, by year:")
print(yr.round(3))

ax = yr.plot(kind="bar", figsize=(10, 4))
ax.set_title("Share of AI sentences carrying each frame, by year")
ax.set_ylabel("share")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(); plt.show()
"""),
    ("md", """### 2b. Strict (dependency-anchored) frames

The loose tagger fires on any occurrence of a cue lemma in the sentence, which
lets citation furniture through. The strict columns require the cue to be the
verb that actually governs an AI subject.
"""),
    ("code", """strict_cols = [f"{f}_strict" for f in frame_cols]
comp = pd.DataFrame({
    "loose_n": tagged[frame_cols].sum().values,
    "strict_n": tagged[strict_cols].sum().values,
}, index=frame_cols)
comp["precision_proxy"] = (comp["strict_n"] / comp["loose_n"]).round(3)
print(comp)

yr_strict = tagged.groupby("publication_year")[strict_cols].mean()
print(yr_strict.round(4))

ax = yr_strict.plot(kind="bar", figsize=(10, 4))
ax.set_title("Share of AI sentences carrying each frame (strict), by year")
ax.set_ylabel("share")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(); plt.show()

# Which cue lemma is carrying each loose frame?
for f in frame_cols:
    print(f, "->", tagged.loc[tagged[f], f"{f}_cue"].value_counts().head(6).to_dict())
"""),
    ("md", "## 3. Persist + sample sentences per frame"),
    ("code", """OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
tagged.to_parquet(OUT_ANALYSIS / "05c_frames_per_sentence.parquet", index=False)
print("saved frames_per_sentence.parquet")

for f in frame_cols:
    sub = tagged[tagged[f]]
    if not len(sub):
        continue
    print(f"\\n=== {f}  (n={len(sub)}) ===")
    for _, r in sub.sample(min(3, len(sub)), random_state=1).iterrows():
        print(f"  [{r['work_id']}] {r['text'][:220]}")
"""),
]

write(HERE / "05c_metaphor.ipynb", cells)
