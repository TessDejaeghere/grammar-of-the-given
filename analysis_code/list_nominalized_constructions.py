"""Rank AI + nominalised-head constructions from 04d output."""
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
src = ROOT / "outputs" / "analysis" / "04d_nominalizations.parquet"
out = ROOT / "outputs" / "analysis" / "04d_nominalized_constructions_ranked.csv"

df = pd.read_parquet(src).copy()
df["mention_clean"] = df["mention_text"].astype(str).map(
    lambda value: re.sub(r"\s+", " ", value).strip())
df["construction"] = (
    df["mention_clean"] + " " + df["governor_lemma"].astype(str).str.lower()
)

rows = []
for construction, group in df.groupby("construction", sort=False):
    rows.append({
        "construction": construction,
        "n": len(group),
        "n_docs": group["work_id"].nunique(),
        "agent_present_n": int(group["agent_present"].sum()),
        "agent_present_share": round(group["agent_present"].mean(), 3),
        "example": re.sub(r"\s+", " ", str(group.iloc[0]["sent_text"])).strip()[:300],
    })

ranked = pd.DataFrame(rows).sort_values(["n", "construction"], ascending=[False, True])
ranked.to_csv(out, index=False, encoding="utf-8-sig")
print(ranked.head(50).to_string(index=False))
print("\nsaved", out, "rows:", len(ranked))
