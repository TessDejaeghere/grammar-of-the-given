"""Works per publication year in the filtered macro-corpus (1990-2026).

Output: outputs/paper/figures/fig33_works_per_year.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "macro" / "works_metadata_filtered.parquet"
OUT = ROOT / "outputs" / "paper" / "figures" / "fig33_works_per_year.png"

w = pd.read_parquet(SRC)
counts = w["publication_year"].value_counts().sort_index()
counts = counts.reindex(range(int(counts.index.min()), int(counts.index.max()) + 1), fill_value=0)

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_facecolor("#fbfbf9")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(True, axis="y", color="#e8e8e4", lw=0.8)
ax.set_axisbelow(True)

colors = ["#c9dced" if y == 2026 else "#3b6fb6" for y in counts.index]
ax.bar(counts.index, counts.values, color=colors, width=0.8,
       edgecolor=["#3b6fb6" if y == 2026 else "none" for y in counts.index],
       linewidth=0.8)

ax.set_xticks(range(1990, 2027, 4))
ax.tick_params(labelsize=10.5)
ax.set_xlim(1989, 2027)
ax.set_ylabel("works", fontsize=11.5)
ax.set_xlabel("publication year", fontsize=11.5)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200)
print(f"wrote {OUT}  (total n = {counts.sum():,})")
