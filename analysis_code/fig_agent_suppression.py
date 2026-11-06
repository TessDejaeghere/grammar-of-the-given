"""Figure: agent suppression at group, clause and discourse rank.

Reads the outputs of 04d (nominalisations), 06d (passives) and 03 (mentions)
and recomputes every percentage shown, using the same definitions as 06d.
Run from anywhere inside the Discourse_analysis folder.
"""
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd()
while ROOT.name and not (ROOT / "analysis_code").exists():
    ROOT = ROOT.parent
ANALYSIS = ROOT / "outputs" / "analysis"
PARSED = ROOT / "outputs" / "parsed"
OUT = ROOT / "outputs" / "paper" / "figures" / "fig36_agent_suppression.png"

# Group rank: AI-action nominals with no grammatically linked doer
nom = pd.read_parquet(ANALYSIS / "04d_nominalizations.parquet")
nom_no_agent = 100 * (~nom["agent_present"].astype(bool)).mean()

# Clause rank: passives in AI-bearing sentences without a by-agent
pas = pd.read_parquet(ANALYSIS / "06d_passives.parquet")
pas_no_agent = 100 * (~pas["agent_present"].astype(bool)).mean()

# Discourse rank: is a named company mentioned near an agentless passive? (as in 06d, section 5,
# but with false company matches filtered out)
ment = pd.read_parquet(PARSED / "mentions.parquet")
comp = ment[ment["category"] == "COMPANY"].copy()

# Keep named companies only: drop generic terms and matches that are not a company as actor
GENERIC = {"developers", "developer", "vendors", "vendor", "tech companies", "tech company",
           "ai companies", "ai company"}
G_TOOLS = {"books", "scholar", "colab", "arts", "search", "docs", "drive", "sheets", "maps",
           "ngram", "forms", "earth", "trends", "images", "news", "sites", "classroom", "street"}
MS_TOOLS = {"excel", "office", "teams", "word", "powerpoint", "books", "outlook", "onedrive", "coco"}
NV_HW = re.compile(r"^(a\d+|v\d+|h\d+|l\d+|t\d+|geforce|gforce|rtx|dgx|tesla|quadro|titan|cuvs|gtx)", re.I)


def is_named_company(r):
    text, sent = r["mention_text"], str(r["sent_text"])
    low = text.lower()
    if low in GENERIC:
        return False
    m = re.search(re.escape(text) + r"[\s\-]*([^\s,.;:()]*)", sent)
    nxt = m.group(1) if m else ""
    hyphen = bool(re.search(re.escape(text) + r"\s*-", sent))
    if low == "meta" and (text != "Meta" or hyphen or re.match(r"^[a-z]", nxt)):
        return False
    if text == "XAI":
        return False
    if low == "google" and nxt.lower() in G_TOOLS:
        return False
    if low == "microsoft" and nxt.lower() in MS_TOOLS:
        return False
    if low == "amazon" and nxt.lower() in {"kindle", "river", "rainforest"}:
        return False
    if low == "nvidia" and NV_HW.match(nxt):
        return False
    return True


comp = comp[comp.apply(is_named_company, axis=1)]
works_company = set(comp["work_id"])
comp_sent_keys = set(map(tuple, comp[["work_id", "sent_id"]].values))
comp_sents_by_work = (comp[["work_id", "sent_id"]].drop_duplicates()
                      .groupby("work_id")["sent_id"].apply(set).to_dict())

ag = pas[~pas["agent_present"].astype(bool)].dropna(subset=["sent_id"])
same_sent = 100 * ag.apply(lambda r: (r["work_id"], r["sent_id"]) in comp_sent_keys, axis=1).mean()


def near(r, w=3):
    s = comp_sents_by_work.get(r["work_id"])
    return bool(s) and any(abs(r["sent_id"] - x) <= w for x in s)


window = 100 * ag.apply(near, axis=1).mean()
same_work = 100 * ag["work_id"].isin(works_company).mean()

print(f"nominals without agent:        {nom_no_agent:.1f}% (n={len(nom)})")
print(f"passives without by-agent:     {pas_no_agent:.1f}% (n={len(pas)})")
print(f"agentless passives (n={len(ag)}): company same sentence {same_sent:.1f}%, "
      f"+/-3 sentences {window:.1f}%, same work {same_work:.1f}%")

# Plot (same style as the other paper figures: seaborn "colorblind" palette)
from matplotlib.ticker import PercentFormatter

BLUE, VERMILLION = "#0173B2", "#D55E00"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.1]})


def bars(ax, labels, values, color, title, xlabel):
    ax.barh(labels, values, color=color, height=0.6)
    for i, v in enumerate(values):
        ax.text(v + 1.5, i, f"{v:.1f}%", va="center", ha="left")
    ax.set_xlim(0, 112)
    ax.set_xticks(range(0, 101, 20))
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.xaxis.grid(True, color="#e5e5e5")
    ax.set_axisbelow(True)


bars(a, [f"AI-action nominals\n(group rank, n = {len(nom):,})",
         f"Passives in AI-bearing\nsentences (clause rank, n = {len(pas):,})"],
     [nom_no_agent, pas_no_agent], BLUE,
     "A", "share with no agent")
bars(b, ["same sentence", "within ±3 sentences", "same work"],
     [same_sent, window, same_work], VERMILLION,
     "B",
     "share of agentless passives")

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print("saved", OUT)
