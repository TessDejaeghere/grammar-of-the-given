"""Role distribution figure (fig04 style) restricted to umbrella-term mentions.

Output: outputs/paper/figures/fig32_role_distribution_umbrella.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "analysis" / "06b_mentions_by_group.parquet"
OUT = ROOT / "outputs" / "paper" / "figures" / "fig32_role_distribution_umbrella.png"

LABELS = {
    "nominal_modifier": "nominal modifier (Classifier)",
    "prepositional": "prepositional",
    "other": "other",
    "active_subject": "active subject",
    "direct_object": "direct object",
    "passive_subject": "passive subject",
    "oblique_agent": "oblique agent",
    "attribute": "attribute",
}

df = pd.read_parquet(SRC)
umb = df[df["cluster"] == "abstraction"]
share = umb["role"].value_counts(normalize=True).sort_values()

fig, ax = plt.subplots(figsize=(9.5, 4.8))
bars = ax.barh([LABELS.get(r, r) for r in share.index], share.values * 100,
               color="#1170aa")
for b, v in zip(bars, share.values * 100):
    ax.text(v + 0.4, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
            va="center", fontsize=12)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(True, axis="x", color="#e5e5e5", lw=0.8)
ax.set_axisbelow(True)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.tick_params(labelsize=12)
ax.set_xlim(0, share.max() * 100 + 5)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200)
print(f"wrote {OUT}")
