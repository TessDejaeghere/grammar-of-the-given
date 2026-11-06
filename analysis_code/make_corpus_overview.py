"""Two-panel corpus overview: works per year (macro) + venue classes (micro AI corpus).

Output: outputs/paper/figures/fig35_corpus_overview.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "paper" / "figures" / "fig35_corpus_overview.png"

TEAL = "#2a9d8f"
TEAL_LIGHT = "#b8e0da"
VENUE_COLORS = {
    "journal article": "#2a9d8f",
    "preprint / repository": "#e76f51",
    "book / chapter": "#e9c46a",
    "conference": "#5f7fbf",
    "other": "#b0b0aa",
}

w = pd.read_parquet(ROOT / "outputs" / "macro" / "works_metadata_filtered.parquet")
counts = w["publication_year"].value_counts().sort_index()
counts = counts.reindex(range(int(counts.index.min()), int(counts.index.max()) + 1), fill_value=0)

p = pd.read_parquet(ROOT / "outputs" / "analysis" / "06e_work_profiles.parquet")
venues = p["venue_class"].value_counts().sort_values()

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.4), width_ratios=[1.55, 1])
for ax in (ax1, ax2):
    ax.set_facecolor("#fbfbf9")
    ax.set_axisbelow(True)

# (a) works per year
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)
ax1.grid(True, axis="y", color="#e8e8e4", lw=0.8)
ax1.bar(counts.index, counts.values, width=0.8,
        color=[TEAL_LIGHT if y == 2026 else TEAL for y in counts.index],
        edgecolor=[TEAL if y == 2026 else "none" for y in counts.index], linewidth=0.8)
ax1.set_xticks(range(1990, 2027, 4))
ax1.tick_params(labelsize=10.5)
ax1.set_xlim(1989, 2027)
ax1.set_ylabel("works", fontsize=11.5)
ax1.set_xlabel("publication year", fontsize=11.5)
ax1.set_title("(a)", fontsize=12, loc="left", pad=10)

# (b) venue classes of the AI corpus
for s in ("top", "right", "left"):
    ax2.spines[s].set_visible(False)
ax2.grid(True, axis="x", color="#e8e8e4", lw=0.8)
bars = ax2.barh(venues.index, venues.values, height=0.62,
                color=[VENUE_COLORS.get(v, "#b0b0aa") for v in venues.index])
total = venues.sum()
for b, v in zip(bars, venues.values):
    ax2.text(v + 6, b.get_y() + b.get_height() / 2,
             f"{v}  ({v / total:.0%})", va="center", fontsize=10.5, color="#2f2f2f")
ax2.tick_params(labelsize=11)
ax2.set_xlim(0, venues.max() * 1.24)
ax2.set_xlabel("works in the full-text AI corpus", fontsize=11)
ax2.set_title("(b)", fontsize=12, loc="left", pad=10)

fig.tight_layout(w_pad=2.5)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200)
print(f"wrote {OUT}")
