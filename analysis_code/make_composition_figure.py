"""Slide-ready corpus-composition figure -> outputs/paper/figures/fig26_corpus_composition.png."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import OUT_CLEAN, OUT_PARSED

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "outputs" / "paper" / "figures"

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
      "grey": "#8C8C8C", "dark": "#333333"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

idx = pd.read_parquet(OUT_CLEAN / "index.parquet")
sub = idx[idx["in_ai_corpus"]]
yearly = sub["publication_year"].value_counts().sort_index()

prof = pd.read_parquet(ROOT / "outputs" / "analysis" / "06e_work_profiles.parquet")
venue = prof["venue_class"].value_counts()
venue = venue[venue.index != "other"]
disc = prof["discipline"].value_counts()

ment = pd.read_parquet(OUT_PARSED / "mentions.parquet")
mcounts = ment["category"].value_counts()

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
fig.subplots_adjust(wspace=0.45)

# funnel
ax = axes[0]
steps = [("abstracts", 22_827), ("OA full texts", 795),
         ("parsed sentences", 79_360), ("actor mentions", 112_826)]
labels = [s[0] for s in steps]
y = np.arange(len(steps))[::-1]
for yi, (lab, n) in zip(y, steps):
    ax.barh(yi, 1.0, color=OI["blue"], alpha=0.25 + 0.18 * (3 - yi), height=0.62)
    ax.text(0.03, yi, f"{lab}:  {n:,}", va="center", fontsize=12.5,
            fontweight="bold", color=OI["dark"])
ax.set_xlim(0, 1.05)
ax.axis("off")
ax.set_title("Corpus")

# works per year
ax = axes[1]
ax.bar(yearly.index.astype(int).astype(str), yearly.values, color=OI["blue"],
       width=0.65)
for xi, v in enumerate(yearly.values):
    ax.text(xi, v + 6, str(v), ha="center", fontsize=11.5)
ax.set_ylim(0, yearly.max() * 1.18)
ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
ax.set_title("Full texts per year")

# mentions by actor
ax = axes[2]
order_m = ["HUMAN", "AI", "COMPANY"]
vals_m = mcounts.reindex(order_m)
colors_m = [OI["green"], OI["blue"], OI["orange"]]
ym = np.arange(len(vals_m))[::-1]
ax.barh(ym, vals_m.values, color=colors_m, height=0.62)
ax.set_yticks(ym, order_m)
for yi, v in zip(ym, vals_m.values):
    ax.text(v + 1200, yi, f"{v:,}", va="center", fontsize=11.5)
ax.set_xlim(0, vals_m.max() * 1.28)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_title("Actor mentions (same docs, same rules)")

fig.suptitle("22,827 DH abstracts (1990–2026)  ·  795 open-access full texts (2022–2026, ≈8.9M words)",
             x=0.005, ha="left", fontweight="bold", fontsize=16, y=1.06)
fig.savefig(FIGS / "fig26_corpus_composition.png")
print("wrote", FIGS / "fig26_corpus_composition.png")
plt.close(fig)

# standalone venue-class figure
fig, ax = plt.subplots(figsize=(7, 3.4))
order = ["journal article", "preprint / repository", "book / chapter", "conference"]
vals = venue.reindex(order)
short = ["journal article", "preprint / repository", "book / chapter", "conference"]
yv = np.arange(len(vals))[::-1]
colors = [OI["green"], OI["orange"], OI["purple"], OI["sky"]]
ax.barh(yv, vals.values, color=colors, height=0.62)
ax.set_yticks(yv, short)
for yi, v in zip(yv, vals.values):
    share = v / vals.sum()
    ax.text(v + 6, yi, f"{int(v)}  ({share:.0%})", va="center", fontsize=12)
ax.set_xlim(0, vals.max() * 1.28)
ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_title("Venue class of the 795 full texts")
fig.savefig(FIGS / "fig27_venue_class.png")
print("wrote", FIGS / "fig27_venue_class.png")
