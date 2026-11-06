"""Scatter of collocate attraction: umbrella "AI" vs named systems (log-Dice).

Reads outputs/analysis/06c_collocations.csv; collocates missing under one node
(below the top-list threshold) are plotted at an axis floor with open markers.
Output: outputs/paper/figures/fig30_collocation_scatter.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "analysis" / "06c_collocations.csv"
OUT = ROOT / "outputs" / "paper" / "figures" / "fig30_collocation_scatter.png"

UMBRELLA = "AI (umbrella)"
NAMED = "named system"
GOVERNANCE = {
    "ethical", "ethic", "ethics", "responsible", "responsibly", "responsibility",
    "explainable", "explainability", "trustworthy", "disclosure", "ethically",
}
NOISE = {"enviado", "parte"}  # known OCR/non-English MI instability

C_UMB = "#3b6fb6"
C_NAM = "#d1682a"
C_GOV = "#1f8a3b"
C_GREY = "#9a9a9a"

df = pd.read_csv(SRC)
df = df[~df["collocate"].isin(NOISE)]
piv_ld = df.pivot_table(index="collocate", columns="node", values="log_dice")
piv_f = df.pivot_table(index="collocate", columns="node", values="f_joint", aggfunc="sum")
piv = piv_ld[[UMBRELLA, NAMED]].copy()
piv["freq"] = piv_f.sum(axis=1)
piv = piv.dropna(how="all", subset=[UMBRELLA, NAMED])

floor_x = np.nanmin(piv[NAMED]) - 0.8
floor_y = np.nanmin(piv[UMBRELLA]) - 0.8
x = piv[NAMED].fillna(floor_x)
y = piv[UMBRELLA].fillna(floor_y)
delta = y - x
shared = piv[UMBRELLA].notna() & piv[NAMED].notna()
gov = piv.index.isin(GOVERNANCE)

lim_lo = min(floor_x, floor_y) - 0.4
lim_hi = max(x.max(), y.max()) + 0.5

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(13, 11))
ax.set_facecolor("#fbfbf9")
for s in ax.spines.values():
    s.set_visible(False)
ax.grid(True, color="#e8e8e4", lw=0.8, zorder=0)
ax.set_axisbelow(True)

# tinted register zones either side of the diagonal
tri = np.array([lim_lo, lim_hi])
ax.fill_between(tri, tri, lim_hi, color=C_UMB, alpha=0.045, zorder=0)
ax.fill_between(tri, lim_lo, tri, color=C_NAM, alpha=0.045, zorder=0)

sizes = 10 + 3.6 * np.sqrt(piv["freq"])

sc = ax.scatter(
    x[shared & ~gov], y[shared & ~gov], s=sizes[shared & ~gov],
    c=delta[shared & ~gov], cmap="coolwarm_r", vmin=-5.5, vmax=5.5,
    alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3,
)
only = shared.eq(False) & ~gov
ax.scatter(x[only], y[only], s=sizes[only], facecolors="none",
           edgecolors=C_GREY, linewidths=0.9, alpha=0.65, zorder=2)
ax.scatter(x[gov], y[gov], s=sizes[gov] + 40, marker="D",
           facecolors=C_GOV, edgecolors="white", linewidths=1.1, zorder=4,
           label="governance vocabulary")

ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], ls=(0, (6, 4)), c="#b5b5b0", lw=1.2, zorder=1)
ax.text(lim_hi - 0.55, lim_hi - 0.9, "equal attraction", rotation=45,
        color="#9a9a94", fontsize=9.5, ha="center")

ax.text(lim_lo + 0.15, lim_hi - 0.25, "not a named-system collocate →",
        rotation=90, fontsize=9, color=C_GREY, va="top")
ax.text(lim_lo + 2.0, lim_lo + 0.12, "← not an umbrella collocate",
        fontsize=9, color=C_GREY, ha="left")


def stacked_column(names, colour, x_off, min_gap=0.36, fontsize=9.5):
    """Label wall-pinned points as an evenly spaced column with leader lines."""
    prev_ty = None
    for py, name in sorted(((y[n], n) for n in names), reverse=True):
        ty = py if prev_ty is None else min(py, prev_ty - min_gap)
        prev_ty = ty
        ax.annotate(name, (x[name] + 0.07, py), xytext=(x[name] + x_off, ty),
                    fontsize=fontsize, ha="left", va="center", style="italic",
                    color=colour, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec=colour, lw=0.5, alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color=colour, lw=0.7,
                                    shrinkA=0, shrinkB=1))


# label selection
lab = pd.Series(delta[shared], index=piv.index[shared])
freq_top = piv.loc[shared, "freq"].nlargest(8).index
floor_nam = piv.loc[~shared & piv[UMBRELLA].isna(), NAMED].nlargest(5).index
interior = (set(lab.nlargest(12).index) | set(lab.nsmallest(12).index) | set(freq_top))
interior -= {"artificial", "literature", "development", "drive"}

# wall column: governance (green)
stacked_column(piv.index[gov], C_GOV, x_off=0.85)

# bottom wall: named-only collocates, staggered labels below their points
for i, name in enumerate(sorted(floor_nam, key=lambda n: x[n])):
    dy = -0.28 - 0.30 * (i % 3)
    ax.annotate(name, (x[name], y[name]), xytext=(x[name], y[name] + dy),
                fontsize=9, ha="center", va="top", style="italic", color="#555555",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8),
                arrowprops=dict(arrowstyle="-", color=C_GREY, lw=0.6,
                                shrinkA=0, shrinkB=1),
                zorder=5)

# interior labels: let adjustText resolve overlaps
texts = [
    ax.text(x[n], y[n], n, fontsize=9.5, style="italic", color="#2f2f2f", zorder=5)
    for n in interior if x[n] > floor_x + 0.3 and y[n] > floor_y + 0.3
]
adjust_text(
    texts, x=x[shared].to_numpy(), y=y[shared].to_numpy(), ax=ax,
    expand=(1.25, 1.45), force_text=(0.3, 0.6),
    arrowprops=dict(arrowstyle="-", color="#c0c0ba", lw=0.6),
)

cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.015)
cbar.outline.set_visible(False)
cbar.set_label("skew (log-Dice difference)   ← named systems side   ·   umbrella \"AI\" side →",
               fontsize=10)
ax.set_xlabel("attraction to named systems (log-Dice)", fontsize=12)
ax.set_ylabel("attraction to umbrella \"AI\" (log-Dice)", fontsize=12)
ax.legend(loc="upper left", frameon=False, fontsize=10.5, borderaxespad=1.2)
ax.set_xlim(lim_lo, lim_hi)
ax.set_ylim(lim_lo - 0.45, lim_hi)
ax.set_aspect("equal")
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200)
print(f"wrote {OUT}")
