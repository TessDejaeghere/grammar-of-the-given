"""Generate 06c_collocations.ipynb — log-Dice / MI collocates of AI node terms."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _nbbuild import write

HERE = Path(__file__).parent

cells = [
    ("md", """# 06c — Collocation analysis

The SFL layers describe how AI is positioned in the clause; collocation describes the lexical
company it keeps. We compute association measures over the parsed sentences for three node
categories — the umbrella term, the model-class term and named systems — using a +/-5 token window
over content words only.

Two measures are reported, because they answer different questions:

- **log-Dice** is frequency-robust and comparable across nodes of different sizes; it is the
  standard measure in Sketch Engine-style CADS work.
- **MI** rewards exclusivity and surfaces the terminologically distinctive collocates, but is
  unstable at low frequency, so a minimum joint frequency is enforced.

Outputs: `outputs/analysis/06c_collocations.csv`
"""),
    ("code", """from __future__ import annotations
import sys
from collections import Counter, defaultdict
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
from spacy.tokens import DocBin

from lib import OUT_PARSED, OUT_ANALYSIS, OUT_CLEAN
from lib.ai_lexicon import build_phrase_matcher

sns.set_theme(context="notebook", style="whitegrid")

WINDOW = 5
MIN_JOINT = 8
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}

NODE_OF_GROUP = {
    "general_ai": "AI (umbrella)",
    "generative_ai": "AI (umbrella)",
    "llm": "LLM / language model",
    "named_decoder_only": "named system",
    "named_encoder_decoder": "named system",
    "named_image_generator": "named system",
}

nlp = spacy.blank("en")
pm = build_phrase_matcher(nlp, label="AI")
PARSED = OUT_PARSED / "docs"
work_ids = sorted(p.stem for p in PARSED.glob("*.spacy"))
print("works:", len(work_ids))
"""),
    ("md", "## 1. Single pass: corpus frequencies and windowed co-occurrence"),
    ("code", """corpus_freq = Counter()          # lemma -> corpus frequency (content words)
node_total = Counter()           # node -> occurrences
joint = defaultdict(Counter)     # node -> lemma -> co-occurrence in window
n_tokens = 0

for wid in work_ids:
    db = DocBin().from_disk(PARSED / f"{wid}.spacy")
    for doc in db.get_docs(nlp.vocab):
        lemmas = [t.lemma_.lower() for t in doc]
        keep = [t.pos_ in CONTENT_POS and not t.is_stop and t.is_alpha and len(t) > 2
                for t in doc]
        for lem, k in zip(lemmas, keep):
            if k:
                corpus_freq[lem] += 1
                n_tokens += 1
        hits = pm.find_grouped(doc)
        for grp, s, e in hits:
            node = NODE_OF_GROUP.get(grp)
            if node is None:
                continue
            node_total[node] += 1
            lo, hi = max(0, s - WINDOW), min(len(doc), e + WINDOW)
            span_ids = set(range(s, e))
            for i in range(lo, hi):
                if i in span_ids or not keep[i]:
                    continue
                joint[node][lemmas[i]] += 1
    del db

print("content tokens:", n_tokens)
print(pd.Series(node_total).sort_values(ascending=False))
"""),
    ("md", "## 2. Association measures"),
    ("code", """rows = []
for node, counter in joint.items():
    n_node = node_total[node]
    win_size = 2 * WINDOW
    for lem, f_joint in counter.items():
        if f_joint < MIN_JOINT:
            continue
        f_coll = corpus_freq[lem]
        # log-Dice: 14 + log2(2*f_joint / (f_node + f_coll))
        dice = 14 + np.log2(2 * f_joint / (n_node + f_coll))
        expected = n_node * win_size * f_coll / n_tokens
        mi = np.log2(f_joint / expected) if expected > 0 else np.nan
        rows.append({"node": node, "collocate": lem, "f_joint": f_joint,
                     "f_collocate": f_coll, "log_dice": round(dice, 3),
                     "mi": round(mi, 3)})

coll = pd.DataFrame(rows)
print("candidate collocations:", len(coll))
for node in sorted(coll["node"].unique()):
    sub = coll[coll["node"] == node].nlargest(20, "log_dice")
    print(f"\\n=== {node} — top 20 by log-Dice")
    print(sub[["collocate", "f_joint", "log_dice", "mi"]].to_string(index=False))
"""),
    ("code", """for node in sorted(coll["node"].unique()):
    sub = coll[(coll["node"] == node) & (coll["f_joint"] >= 15)].nlargest(15, "mi")
    print(f"\\n=== {node} — top 15 by MI (f_joint >= 15)")
    print(sub[["collocate", "f_joint", "mi", "log_dice"]].to_string(index=False))
"""),
    ("md", "## 3. What is distinctive to each node?"),
    ("code", """# Contrast log-Dice between the umbrella node and the named-system node:
# positive = more strongly associated with the umbrella term.
piv = coll.pivot_table(index="collocate", columns="node", values="log_dice")
piv = piv.dropna(subset=["AI (umbrella)", "named system"], how="any")
piv["delta_umbrella_vs_named"] = (piv["AI (umbrella)"] - piv["named system"]).round(3)
top_um = piv.nlargest(20, "delta_umbrella_vs_named")
top_nm = piv.nsmallest(20, "delta_umbrella_vs_named")
print("Collocates skewed to the UMBRELLA term:")
print(top_um[["AI (umbrella)", "named system", "delta_umbrella_vs_named"]].to_string())
print("\\nCollocates skewed to NAMED systems:")
print(top_nm[["AI (umbrella)", "named system", "delta_umbrella_vs_named"]].to_string())
"""),
    ("code", """fig, ax = plt.subplots(figsize=(9, 6))
plot_df = pd.concat([top_um.head(12), top_nm.head(12)]).sort_values("delta_umbrella_vs_named")
colors = ["#dd8452" if v < 0 else "#4c72b0" for v in plot_df["delta_umbrella_vs_named"]]
ax.barh(plot_df.index, plot_df["delta_umbrella_vs_named"], color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("log-Dice difference  (positive = umbrella term, negative = named system)")
ax.set_title("Collocates distinctive to the umbrella term vs to named systems")
plt.tight_layout(); plt.show()
"""),
    ("md", "## 4. Diachronic check on a few key collocates"),
    ("code", """OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
coll.sort_values(["node", "log_dice"], ascending=[True, False]).to_csv(
    OUT_ANALYSIS / "06c_collocations.csv", index=False)
piv.to_csv(OUT_ANALYSIS / "06c_collocation_contrast.csv")
print("saved", len(coll), "collocation rows")
coll.nlargest(10, "log_dice")
"""),
]

write(HERE / "06c_collocations.ipynb", cells)
