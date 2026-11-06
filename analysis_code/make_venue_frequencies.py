"""Venue-class frequencies of the full-text AI corpus.

Output: outputs/paper/figures/fig34_venue_frequencies.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "analysis" / "06e_work_profiles.parquet"
OUT = ROOT / "outputs" / "paper" / "figures" / "fig34_venue_frequencies.png"

p = pd.read_parquet(SRC)
counts = p["venue_class"].value_counts().sort_values()

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.set_facecolor("#fbfbf9")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.grid(True, axis="x", color="#e8e8e4", lw=0.8)
ax.set_axisbelow(True)

bars = ax.barh(counts.index, counts.values, color="#3b6fb6", height=0.62)
total = counts.sum()
for b, v in zip(bars, counts.values):
    ax.text(v + 5, b.get_y() + b.get_height() / 2,
            f"{v}  ({v / total:.0%})", va="center", fontsize=10.5, color="#2f2f2f")

ax.tick_params(labelsize=11.5)
ax.set_xlim(0, counts.max() * 1.18)
ax.set_xlabel("works in the full-text AI corpus", fontsize=11)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200)
print(f"wrote {OUT}  (total n = {total})")
