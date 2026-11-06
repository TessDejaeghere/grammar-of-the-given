"""Small multiples: anthropomorphizing predications per category (Inie, Zukerman & Bender 2026), by year.

Output: outputs/paper/figures/fig37_anthro_categories_by_year.png
Shares are computed from outputs/analysis/06f_anthro_per_sentence.parquet (notebook 06f).
No figure title: the paper caption describes the figure.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "analysis" / "06f_anthro_per_sentence.parquet"
OUT = ROOT / "outputs" / "paper" / "figures" / "fig37_anthro_categories_by_year.png"

# (column, panel label, colour, example cues); order and colours as in the earlier figure
PANELS = [
    ("Cognizer", "Cognizer", "#CC79A7", "understands, knows, interprets"),
    ("Agent", "Agent", "#0072B2", "helps, operates, enables"),
    ("Communication", "Communication", "#009E73", "responds, claims, answers"),
    ("Products_of_cognition", "Products of cognition", "#E69F00", "capability, ability, judgment"),
    ("Human_role_analogy", "Human role analogy", "#56B4E9", "author, assistant, judge"),
    ("Names_and_pronouns", "Names and pronouns", "#8C8C8C", "FAILED AUDIT — excluded"),
    ("Biological_metaphors", "Biological metaphors", "#D55E00", "neural, hallucinate, evolve"),
    ("Emotion", "Emotion", "#A0522D", "struggle, feel, suffer"),
]

d = pd.read_parquet(SRC)
d = d[d["publication_year"].between(2022, 2026)]
d["publication_year"] = d["publication_year"].astype(int)
share = d.groupby("publication_year")[[c for c, *_ in PANELS]].mean() * 100
years = share.index.tolist()
ymax = max(6.0, share.values.max() * 1.35)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 4, figsize=(14, 5.6), sharex=True)
for ax, (col, label, colour, cues) in zip(axes.flat, PANELS):
    excluded = col == "Names_and_pronouns"
    line = "#BFBFBF" if excluded else colour
    vals = share[col].values
    ax.plot(years, vals, marker="o", lw=2.2, ms=5, color=line)
    ax.fill_between(years, vals, color=line, alpha=0.07)
    for x, v in zip(years, vals):
        ax.text(x, v + ymax * 0.04, f"{v:.1f}", ha="center", va="bottom",
                fontsize=8.5, color=colour)
    ax.set_title(label, color=colour, fontweight="bold", fontsize=12)
    ax.text(0.02, 0.96, cues, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, style="italic", color="#888888")
    ax.set_ylim(0, ymax)
    ax.set_xticks(years)
    ax.tick_params(labelsize=9)
    ax.yaxis.grid(True, color="#e5e5e5")
    ax.set_axisbelow(True)
for ax in axes[:, 0]:
    ax.set_ylabel("% of AI sentences")

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("n per year:", d.groupby("publication_year").size().to_dict())
print(share.round(1).to_string())
print("saved", OUT)
