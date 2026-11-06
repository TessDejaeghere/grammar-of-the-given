"""Forest plot of the two modality odds ratios (bare assertion).

Values from 06a/06b logits (document-clustered SEs); see draft_paper.md §5.5/§5.8.
Output: outputs/paper/figures/fig31_modality_or_forest.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "outputs" / "paper" / "figures" / "fig31_modality_or_forest.png"

rows = [  # label, OR, lo, hi, p
    ("Classifier position\n(vs clause participant)", 1.318, 1.083, 1.605, "p = .006"),
    ("Named system\n(vs umbrella term)", 1.862, 1.468, 2.361, "p < .0001"),
]

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(8.5, 3.2))
ax.set_facecolor("#fbfbf9")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.grid(True, axis="x", color="#e8e8e4", lw=0.8)
ax.set_axisbelow(True)

ys = np.arange(len(rows))[::-1]
for (label, orr, lo, hi, p), yy in zip(rows, ys):
    ax.plot([lo, hi], [yy, yy], color="#3b6fb6", lw=2.2, solid_capstyle="round", zorder=3)
    ax.scatter([orr], [yy], s=110, color="#3b6fb6", edgecolors="white", lw=1.2, zorder=4)
    ax.annotate(f"OR {orr:.2f}  [{lo:.2f}, {hi:.2f}]   {p}",
                (hi, yy), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=10, color="#2f2f2f")

ax.axvline(1.0, color="#b5b5b0", lw=1.2, ls=(0, (6, 4)), zorder=2)
ax.text(1.0, len(rows) - 0.42, "no effect", fontsize=9, color="#9a9a94",
        ha="center", va="bottom")

ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows], fontsize=11)
ax.set_xscale("log")
ax.set_xticks([1.0, 1.25, 1.5, 2.0, 2.5])
ax.set_xticklabels(["1.0", "1.25", "1.5", "2.0", "2.5"], fontsize=10)
ax.minorticks_off()
ax.set_xlim(0.92, 4.6)
ax.set_ylim(-0.5, len(rows) - 0.3)
ax.set_xlabel("odds ratio for bare assertion (log scale, 95% CI, document-clustered)",
              fontsize=10.5)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200)
print(f"wrote {OUT}")
