"""Regenerate the paper figures with a consistent, publication-grade style.

Usage: ../.venv/Scripts/python make_paper_figures.py
Writes to outputs/paper/figures/ (overwrites). Figures whose source data is not
persisted (fig03, fig09, fig21) are left untouched.
"""
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
from lib import OUT_ANALYSIS, OUT_CLEAN
from lib.roles import classify_role

ROOT = Path(__file__).resolve().parents[1]
MACRO = ROOT / "outputs" / "macro"
FIGS = ROOT / "outputs" / "paper" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# Okabe-Ito palette, colourblind safe
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
      "yellow": "#F0E442", "grey": "#8C8C8C", "dark": "#333333"}

ACTOR_C = {"AI": OI["blue"], "HUMAN": OI["green"], "COMPANY": OI["orange"]}
MOD_C = {"bare": "#9FB4C7", "weak/hedged": OI["orange"],
         "strong": OI["red"], "mixed": OI["purple"]}
CLASS_C = {
    "material_action": OI["blue"], "stative_relational": "#9FB4C7",
    "instrumental_tool": OI["green"], "evaluative_performance": OI["orange"],
    "cognitive_mental": OI["purple"], "semiotic_evidential": OI["sky"],
    "behavioural": OI["yellow"], "verbal_communicative": OI["red"],
    "OTHER": "#CFCFCF",
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#DDDDDD", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 11,
    "legend.frameon": False, "legend.fontsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

PCT = mticker.FuncFormatter(lambda v, _: f"{v:.0%}")


def _new_ax(ax):
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    return ax


_orig_subplots = plt.subplots


def subplots(*args, **kwargs):
    fig, axes = _orig_subplots(*args, **kwargs)
    for a in (axes.flat if hasattr(axes, "flat") else [axes]):
        _new_ax(a)
    return fig, axes


plt.subplots = subplots


def save(fig, name):
    fig.savefig(FIGS / name)
    plt.close(fig)
    print("wrote", name)


def hbar(ax, labels, values, color, fmt="{:.1%}", pad=0.002):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=color, height=0.62)
    ax.set_yticks(y, labels)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, values):
        ax.text(v + pad, yi, fmt.format(v), va="center", fontsize=9.5,
                color=OI["dark"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


# ---------------------------------------------------------------- fig01 / 02
tc = pd.read_parquet(MACRO / "term_counts_by_year.parquet")
tc = tc[tc["publication_year"].between(2010, 2026)]

fig, ax = plt.subplots(figsize=(8, 3.4))
ax.bar(tc["publication_year"], tc["n_works"], color=OI["blue"], width=0.75)
ax.set_title("DH abstracts per year in the corpus")
ax.set_ylabel("works")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
save(fig, "fig01_works_per_year.png")

groups = {"general_ai": ("umbrella “AI”", OI["blue"]),
          "llm": ("LLM", OI["orange"]),
          "generative_ai": ("generative AI", OI["green"]),
          "named_decoder_only": ("named decoder-only", OI["red"]),
          "agent": ("agent", OI["purple"]),
          "chatbot": ("chatbot", OI["sky"])}
fig, ax = plt.subplots(figsize=(8, 4.2))
recent = tc[tc["publication_year"] >= 2018]
for g, (label, colr) in groups.items():
    col = f"hits_{g}"
    if col not in recent.columns:
        continue
    rate = recent[col] / recent["n_works"] * 1000
    ax.plot(recent["publication_year"], rate, marker="o", ms=4, lw=2,
            color=colr, label=label)
ax.set_ylabel("hits per 1,000 abstracts")
ax.legend(ncols=2)
save(fig, "fig02_ai_terms_per_1000.png")

# ---------------------------------------------------------------- fig03
from lib.ai_lexicon import AI_GROUPS
import re as _re

wm = pd.read_parquet(MACRO / "works_metadata_filtered.parquet")
wm = wm[wm["publication_year"].between(2018, 2026)].dropna(subset=["abstract"])
umbrella_re = _re.compile("|".join(AI_GROUPS["general_ai"] + AI_GROUPS["generative_ai"]),
                          _re.IGNORECASE)
named_re = _re.compile("|".join(AI_GROUPS["named_decoder_only"]
                                + AI_GROUPS["named_encoder_decoder"]
                                + AI_GROUPS["named_image_generator"]),
                       _re.IGNORECASE)
# Arrow-backed str.contains uses re2, which rejects these patterns; match in Python.
wm["has_umbrella"] = wm["abstract"].map(lambda s: bool(umbrella_re.search(s)))
wm["has_named"] = wm["abstract"].map(lambda s: bool(named_re.search(s)))
co = (wm[wm["has_umbrella"]]
      .groupby("publication_year")["has_named"].agg(["mean", "size"]))
co = co.reindex(range(2018, 2027))

fig, ax = plt.subplots(figsize=(8.5, 3.8))
years = co.index.astype(int).astype(str)
thin = co["size"].fillna(0) < 10  # too few umbrella abstracts for a stable share
ax.plot(years, co["mean"], marker="o", ms=7, lw=2.4, color=OI["blue"],
        markerfacecolor="white", markeredgewidth=0, zorder=2)
ax.plot(years[~thin], co["mean"][~thin], marker="o", ms=7, lw=0,
        color=OI["blue"], zorder=3)
ax.plot(years[thin], co["mean"][thin], marker="o", ms=7, lw=0,
        color=OI["blue"], alpha=0.35, zorder=3)
ax.fill_between(range(len(co)), 0, co["mean"].fillna(0), color=OI["blue"],
                alpha=0.08)
for xi, (m, n) in enumerate(zip(co["mean"], co["size"])):
    if pd.isna(m):
        continue
    faded = n < 10
    near_zero = m < 0.02  # stack both labels above the point, off the axis
    ax.annotate(f"{m:.1%}", (xi, m),
                xytext=(0, 22 if near_zero else 9), textcoords="offset points",
                ha="center", fontweight="bold",
                color=OI["grey"] if faded else OI["blue"])
    ax.annotate(f"n={int(n)}", (xi, m),
                xytext=(0, 9 if near_zero else -14),
                textcoords="offset points", ha="center", fontsize=8.5,
                color=OI["grey"])
ax.set_ylim(0, np.nanmax(co["mean"]) * 1.35)
ax.yaxis.set_major_formatter(PCT)
ax.set_ylabel("share also naming a model family")
ax.set_title("Abstracts using umbrella AI vocabulary that also name a concrete model"
             " (grey = n < 10)")
save(fig, "fig03_cooccurrence_named_models.png")

# ---------------------------------------------------------------- fig04 / 05
ment = pd.read_parquet(OUT_ANALYSIS / "04a_agency_per_mention.parquet")
ai = ment[ment["category"] == "AI"].copy()
if "role" not in ai.columns:
    ai["role"] = ai.apply(classify_role, axis=1)
if "publication_year" not in ai.columns:
    idx = pd.read_parquet(OUT_CLEAN / "index.parquet")[["id", "publication_year"]]
    ai = ai.merge(idx.rename(columns={"id": "work_id"}), on="work_id", how="left")

ROLE_LABEL = {"nominal_modifier": "nominal modifier (Classifier)",
              "prepositional": "prepositional", "other": "other",
              "active_subject": "active subject", "direct_object": "direct object",
              "passive_subject": "passive subject", "oblique_agent": "oblique agent",
              "attribute": "attribute"}
dist = ai["role"].value_counts(normalize=True)
fig, ax = plt.subplots(figsize=(7.5, 3.8))
hbar(ax, [ROLE_LABEL[r] for r in dist.index], dist.values, OI["blue"])
ax.xaxis.set_major_formatter(PCT)
ax.set_title("Grammatical role of AI mentions (n = 29,163)")
save(fig, "fig04_role_distribution.png")

ROLE_ORDER = ["active_subject", "passive_subject", "direct_object", "attribute",
              "oblique_agent", "prepositional", "nominal_modifier", "other"]
ROLE_C = {"active_subject": OI["blue"], "passive_subject": OI["sky"],
          "direct_object": OI["green"], "attribute": OI["yellow"],
          "oblique_agent": OI["purple"], "prepositional": OI["orange"],
          "nominal_modifier": "#9FB4C7", "other": "#CFCFCF"}
yr_role = pd.crosstab(ai["publication_year"], ai["role"],
                      normalize="index")[ROLE_ORDER]
fig, ax = plt.subplots(figsize=(8, 4.2))
bottom = np.zeros(len(yr_role))
for r in ROLE_ORDER:
    ax.bar(yr_role.index.astype(int).astype(str), yr_role[r], bottom=bottom,
           color=ROLE_C[r], width=0.7, label=ROLE_LABEL[r])
    bottom += yr_role[r].values
ax.set_title("Grammatical roles of AI mentions by year")
ax.set_ylabel("share")
ax.yaxis.set_major_formatter(PCT)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
save(fig, "fig05_role_by_year.png")

# ---------------------------------------------------------------- fig06 / 07
rba = pd.read_parquet(OUT_ANALYSIS / "05d_role_share_by_actor.parquet")
ACTOR_N = {"AI": 29163, "HUMAN": 80247, "COMPANY": 3416}
show = ["active_subject", "direct_object", "prepositional", "nominal_modifier"]
fig, ax = plt.subplots(figsize=(8, 3.8))
x = np.arange(len(show))
for i, actor in enumerate(["AI", "HUMAN", "COMPANY"]):
    vals = rba.loc[actor, show].values
    ax.bar(x + (i - 1) * 0.26, vals, width=0.24, color=ACTOR_C[actor],
           label=f"{actor} (n = {ACTOR_N[actor]:,})")
    for xi, v in zip(x + (i - 1) * 0.26, vals):
        ax.text(xi, v + 0.008, f"{v:.0%}", ha="center", fontsize=8.5,
                color=OI["dark"])
ax.set_xticks(x, [ROLE_LABEL[r] for r in show])
ax.set_title("Who gets which grammatical role?")
ax.set_ylabel("share of that actor's mentions")
ax.yaxis.set_major_formatter(PCT)
ax.legend()
fig.text(0.01, -0.04,
         "Bars are normalised within each actor: each group of bars sums over "
         "all roles to 100% of that actor's own mentions,\nso groups are "
         "comparable in profile, not in volume.",
         fontsize=8, color="#666666", ha="left")
save(fig, "fig06_role_by_actor.png")

asy = pd.read_parquet(OUT_ANALYSIS / "05d_active_share_by_year.parquet")
fig, ax = plt.subplots(figsize=(8, 3.8))
for actor in ["HUMAN", "AI", "COMPANY"]:
    ax.plot(asy.index.astype(int).astype(str), asy[actor], marker="o", ms=5,
            lw=2.2, color=ACTOR_C[actor], label=actor)
    ax.annotate(actor, (len(asy) - 1, asy[actor].iloc[-1]),
                xytext=(6, 0), textcoords="offset points",
                color=ACTOR_C[actor], fontweight="bold", va="center")
ax.set_title("Active-subject share by actor category")
ax.set_ylabel("share of mentions")
ax.set_ylim(0, 0.62)
ax.yaxis.set_major_formatter(PCT)
save(fig, "fig07_active_share_by_year.png")

# ---------------------------------------------------------------- fig08
verbs = pd.read_csv(OUT_ANALYSIS / "04c_verbs_ranked.csv").nlargest(30, "n")
fig, ax = plt.subplots(figsize=(8, 7))
y = np.arange(len(verbs))[::-1]
colors = [CLASS_C.get(c, "#CFCFCF") for c in verbs["auto_class"]]
ax.barh(y, verbs["n"], color=colors, height=0.66)
ax.set_yticks(y, verbs["verb_lemma"])
ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_title("30 most frequent verbs governing AI as active subject")
ax.set_xlabel("occurrences")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_C.values()]
ax.legend(handles, [k.replace("_", "/") for k in CLASS_C],
          loc="lower right", fontsize=8.5)
save(fig, "fig08_top_verbs.png")

# ---------------------------------------------------------------- fig10
cl = pd.read_parquet(OUT_ANALYSIS / "04b_modality_per_clause.parquet")
yr_mod = pd.crosstab(cl["publication_year"], cl["auto_label"], normalize="index")
yr_mod = yr_mod[["bare", "weak/hedged", "strong", "mixed"]]
fig, ax = plt.subplots(figsize=(8, 4))
bottom = np.zeros(len(yr_mod))
for lab in yr_mod.columns:
    ax.bar(yr_mod.index.astype(int).astype(str), yr_mod[lab], bottom=bottom,
           color=MOD_C[lab], width=0.7, label=lab)
    bottom += yr_mod[lab].values
for i, (_, row) in enumerate(yr_mod.iterrows()):
    ax.text(i, row["bare"] / 2, f"{row['bare']:.0%}", ha="center",
            va="center", color="white", fontweight="bold")
ax.set_title("Modality of AI predicate clauses by year (bare share labelled)")
ax.set_ylabel("share of clauses")
ax.yaxis.set_major_formatter(PCT)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
save(fig, "fig10_modality_by_year.png")

# ---------------------------------------------------------------- fig11
noms = pd.read_parquet(OUT_ANALYSIS / "04d_nominalizations.parquet")
sent_flags = pd.read_csv(OUT_ANALYSIS / "04d_nominalizations_per_sentence.csv")
fig, ax = plt.subplots(figsize=(8, 3.8))
if "agent_in_sentence" in sent_flags.columns:
    sm = 1 - sent_flags.groupby("publication_year")["agent_in_sentence"].mean()
else:  # fall back to what is persisted: local relation
    sm = None
loc = 1 - noms.groupby("publication_year")["agent_present"].mean()
years = loc.index.astype(int).astype(str)
if sm is not None:
    ax.plot(years, sm.reindex(loc.index), marker="o", ms=5, lw=2,
            color=OI["orange"], label="sentence-level (co-occurrence)")
ax.plot(years, loc, marker="s", ms=5, lw=2, color=OI["blue"],
        label="local (grammatically linked)")
ax.set_ylim(0.5, 1.02)
ax.yaxis.set_major_formatter(PCT)
ax.set_title("Agent-effaced rate of AI-action nominalisations")
ax.set_ylabel("share with no agent")
ax.legend(loc="lower right")
save(fig, "fig11_agent_effacement.png")

# ---------------------------------------------------------------- fig12
pol = pd.read_parquet(OUT_ANALYSIS / "04e_polarity_per_mention.parquet")
if "publication_year" not in pol.columns:
    idx = pd.read_parquet(OUT_CLEAN / "index.parquet")[["id", "publication_year"]]
    pol = pol.merge(idx.rename(columns={"id": "work_id"}), on="work_id", how="left")
pol["stance"] = np.select(
    [pol["projected"] & pol["negated"], pol["projected"],
     pol["negated"]],
    ["projected negative", "projected positive", "asserted negative"],
    default="asserted positive")
ORDER = ["asserted positive", "asserted negative",
         "projected positive", "projected negative"]
ST_C = {"asserted positive": "#9FB4C7", "asserted negative": OI["red"],
        "projected positive": OI["sky"], "projected negative": OI["purple"]}
yr_st = pd.crosstab(pol["publication_year"], pol["stance"], normalize="index")[ORDER]
fig, ax = plt.subplots(figsize=(8, 4))
bottom = np.zeros(len(yr_st))
for lab in ORDER:
    ax.bar(yr_st.index.astype(int).astype(str), yr_st[lab], bottom=bottom,
           color=ST_C[lab], width=0.7, label=lab)
    bottom += yr_st[lab].values
ax.set_title("Stance of AI-construing clauses by year")
ax.set_ylabel("share")
ax.yaxis.set_major_formatter(PCT)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
save(fig, "fig12_stance_by_year.png")

# ---------------------------------------------------------------- fig13
fr = pd.read_parquet(OUT_ANALYSIS / "05c_frames_per_sentence.parquet")
strict_cols = [c for c in fr.columns if c.endswith("_strict")]
shares = fr[strict_cols].mean().sort_values()
labels = [c.replace("_strict", "").replace("AI_as_", "AI as ") for c in shares.index]
fig, ax = plt.subplots(figsize=(7, 3.2))
hbar(ax, labels[::-1], shares.values[::-1], OI["blue"], fmt="{:.2%}", pad=0.00006)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2%}"))
ax.set_title("Share of AI sentences carrying each frame (strict, dependency-anchored)")
save(fig, "fig13_frames_strict.png")

# ---------------------------------------------------------------- fig14
ci = pd.read_csv(OUT_ANALYSIS / "06a_rates_with_ci.csv")
fig, ax = plt.subplots(figsize=(6.5, 3.6))
x = np.arange(len(ci))
ax.bar(x, ci["pooled_rate"], width=0.5,
       color=[ACTOR_C[c] for c in ci["category"]], alpha=0.85)
ax.errorbar(x, ci["pooled_rate"],
            yerr=[ci["pooled_rate"] - ci["boot_ci_low"],
                  ci["boot_ci_high"] - ci["pooled_rate"]],
            fmt="none", ecolor=OI["dark"], capsize=4, lw=1.4)
ax.scatter(x, ci["doc_mean"], color=OI["red"], zorder=5, s=45, marker="D",
           label="document-level mean")
ax.set_xticks(x, ci["category"])
ax.set_title("Active-subject rate: pooled, 95% cluster-bootstrap CI, doc-level mean")
ax.set_ylabel("rate")
ax.yaxis.set_major_formatter(PCT)
ax.legend()
save(fig, "fig14_active_rate_ci.png")

# ---------------------------------------------------------------- fig15 / 16
rbc = pd.read_csv(OUT_ANALYSIS / "06b_role_by_cluster.csv").set_index("cluster")
show = ["active_subject", "passive_subject", "direct_object",
        "prepositional", "nominal_modifier"]
fig, ax = plt.subplots(figsize=(8, 3.8))
x = np.arange(len(show))
for i, (cluster, colr) in enumerate([("abstraction", OI["blue"]),
                                     ("named", OI["orange"])]):
    vals = rbc.loc[cluster, show].values
    ax.bar(x + (i - 0.5) * 0.34, vals, width=0.32, color=colr,
           label="umbrella terms" if cluster == "abstraction" else "named systems")
    for xi, v in zip(x + (i - 0.5) * 0.34, vals):
        ax.text(xi, v + 0.006, f"{v:.0%}", ha="center", fontsize=8.5)
ax.set_xticks(x, [ROLE_LABEL[r] for r in show])
ax.set_title("Grammatical role: umbrella terms vs named systems")
ax.yaxis.set_major_formatter(PCT)
ax.legend()
save(fig, "fig15_role_by_cluster.png")

vbc = pd.read_csv(OUT_ANALYSIS / "06b_verbclass_by_cluster.csv").set_index("cluster")
classes = [c for c in vbc.columns if c != "OTHER"]
diffs = (vbc.loc["abstraction", classes] - vbc.loc["named", classes]).sort_values()
fig, ax = plt.subplots(figsize=(7.5, 3.8))
y = np.arange(len(diffs))
colors = [OI["orange"] if v < 0 else OI["blue"] for v in diffs.values]
ax.barh(y, diffs.values, color=colors, height=0.62)
ax.axvline(0, color=OI["dark"], lw=0.8)
ax.set_yticks(y, [c.replace("_", "/") for c in diffs.index])
ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlabel("share difference (umbrella − named)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0%}"))
save(fig, "fig16_verbclass_by_cluster.png")

# ---------------------------------------------------------------- fig17
mg = pd.read_parquet(OUT_ANALYSIS / "06b_mentions_by_group.parquet")
named = mg[mg["cluster"].isin(["abstraction", "named"])]
share = (named.assign(is_named=named["cluster"].eq("named"))
         .groupby("publication_year")["is_named"].mean())
fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.plot(share.index.astype(int).astype(str), share.values, marker="o", ms=6,
        lw=2.2, color=OI["orange"])
for xi, v in enumerate(share.values):
    ax.text(xi, v + 0.012, f"{v:.1%}", ha="center", fontsize=9)
ax.set_title("Share of AI mentions naming a concrete system")
ax.set_ylabel("named-system share")
ax.set_ylim(0, 0.45)
ax.yaxis.set_major_formatter(PCT)
save(fig, "fig17_named_share_by_year.png")

# ---------------------------------------------------------------- fig18
coll = pd.read_csv(OUT_ANALYSIS / "06c_collocation_contrast.csv")
coll = coll.dropna(subset=["delta_umbrella_vs_named"])
top = pd.concat([coll.nlargest(12, "delta_umbrella_vs_named"),
                 coll.nsmallest(12, "delta_umbrella_vs_named")])
top = top.sort_values("delta_umbrella_vs_named")
fig, ax = plt.subplots(figsize=(7.5, 5.6))
y = np.arange(len(top))
colors = [OI["orange"] if v < 0 else OI["blue"]
          for v in top["delta_umbrella_vs_named"]]
ax.barh(y, top["delta_umbrella_vs_named"], color=colors, height=0.66)
ax.axvline(0, color=OI["dark"], lw=0.8)
ax.set_yticks(y, top["collocate"])
ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_title("Collocates skewed to the umbrella term (→) vs named systems (←)")
ax.set_xlabel("log-Dice difference")
save(fig, "fig18_collocation_contrast.png")

# ---------------------------------------------------------------- fig19
pas = pd.read_parquet(OUT_ANALYSIS / "06d_passives.parquet")
ag = pas.dropna(subset=["publication_year"]).groupby("publication_year")[
    "agentless"].agg(["mean", "size"])
fig, ax = plt.subplots(figsize=(7.5, 3.4))
years = ag.index.astype(int).astype(str)
ax.bar(years, ag["mean"], color=OI["red"], width=0.62, alpha=0.85)
overall = pas["agentless"].mean()
ax.axhline(overall, color=OI["dark"], ls="--", lw=1.2)
ax.text(len(ag) - 0.4, overall + 0.012, f"overall {overall:.1%}", ha="right",
        fontsize=9.5)
for xi, (m, n) in enumerate(zip(ag["mean"], ag["size"])):
    ax.text(xi, m - 0.05, f"n={n}", ha="center", color="white", fontsize=8.5)
ax.set_ylim(0.5, 1.0)
ax.yaxis.set_major_formatter(PCT)
ax.set_title("Agentless rate of passive clauses in AI-bearing sentences")
save(fig, "fig19_agentless_by_year.png")

# ---------------------------------------------------------------- fig20
strat = pd.read_csv(OUT_ANALYSIS / "06e_by_stratum.csv")
ven = strat[strat["dim"] == "venue_class"].set_index("stratum")
ven = ven.reindex(["journal article", "preprint / repository",
                   "conference", "book / chapter"]).dropna(how="all")
metrics = [("named_share", "names a system", OI["orange"]),
           ("bare_share", "bare assertion", "#9FB4C7"),
           ("hedged_share", "hedged", OI["green"]),
           ("active_share", "AI active subject", OI["blue"])]
fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=False)
for ax, (col, label, colr) in zip(axes, metrics):
    vals = ven[col]
    ax.bar(range(len(vals)), vals, color=colr, width=0.6)
    ax.set_xticks(range(len(vals)),
                  ["journal", "preprint", "conf.", "book"], fontsize=9)
    ax.set_title(label, fontsize=11, pad=12)
    ax.set_ylim(0, vals.max() * 1.28)
    ax.yaxis.set_major_formatter(PCT)
    for xi, v in enumerate(vals):
        ax.text(xi, v + vals.max() * 0.04, f"{v:.0%}", ha="center", fontsize=8.5)
fig.suptitle("Register profile by venue class (document-level means)",
             x=0.01, ha="left", fontweight="bold", fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "fig20_grammar_by_venue.png")

# ---------------------------------------------------------------- fig22 / 23
cats = pd.read_csv(OUT_ANALYSIS / "06f_category_summary.csv")
cats = cats.rename(columns={"Unnamed: 0": "category"})
cats["label"] = cats["category"].str.replace("_", " ")
cats = cats.sort_values("n_sentences")
fig, ax = plt.subplots(figsize=(7.5, 3.8))
y = np.arange(len(cats))
colors = [OI["grey"] if c == "Names_and_pronouns" else OI["blue"]
          for c in cats["category"]]
ax.barh(y, cats["n_sentences"], color=colors, height=0.62)
ax.set_yticks(y, cats["label"])
ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
for yi, (n, c) in enumerate(zip(cats["n_sentences"], cats["category"])):
    note = "  (failed audit — excluded)" if c == "Names_and_pronouns" else ""
    ax.text(n + 4, yi, f"{n}{note}", va="center", fontsize=9)
ax.set_title("Anthropomorphizing predications about AI, by category (sentences)")
save(fig, "fig22_anthro_categories.png")

fig, ax = plt.subplots(figsize=(7.5, 3.8))
c2 = cats.sort_values("n_sentences")
y = np.arange(len(c2))
ax.barh(y - 0.18, c2["presupposed_n"], height=0.34, color="#9FB4C7",
        label="presupposed (entrenched term)")
ax.barh(y + 0.18, c2["asserted_n"], height=0.34, color=OI["blue"],
        label="asserted (predication)")
ax.set_yticks(y, c2["label"])
ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_title("Presupposed vs asserted anthropomorphization")
ax.set_xscale("log")
ax.set_xlabel("sentences (log scale)")
ax.legend(loc="lower right")
save(fig, "fig23_presupposed_vs_asserted.png")

# ---------------------------------------------------------------- fig24 / 25
an = pd.read_parquet(OUT_ANALYSIS / "06f_anthro_per_sentence.parquet")
an = an.dropna(subset=["publication_year"])
fig, ax = plt.subplots(figsize=(7.5, 3.8))
for cat, colr in [("Cognizer", OI["purple"]), ("Agent", OI["blue"]),
                  ("Communication", OI["green"])]:
    s = an.groupby("publication_year")[cat].mean()
    ax.plot(s.index.astype(int).astype(str), s.values, marker="o", ms=5, lw=2,
            color=colr, label=cat)
ax.set_title("Anthropomorphizing predication rates by year")
ax.set_ylabel("share of AI sentences")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1%}"))
ax.legend()
save(fig, "fig24_anthro_by_year.png")

bv = pd.read_csv(OUT_ANALYSIS / "06f_by_venue.csv")
bv = bv[bv["venue_class"] != "other"]
fig, ax = plt.subplots(figsize=(7.5, 3.6))
x = np.arange(len(bv))
for i, (col, colr, lab) in enumerate([
        ("any_category", OI["blue"], "any predication"),
        ("entrenched_term", "#9FB4C7", "entrenched term")]):
    ax.bar(x + (i - 0.5) * 0.36, bv[col], width=0.34, color=colr, label=lab)
    for xi, v in zip(x + (i - 0.5) * 0.36, bv[col]):
        ax.text(xi, v + 0.012, f"{v:.0%}", ha="center", fontsize=8.5)
ax.set_xticks(x, [v.replace(" / ", "/\n") for v in bv["venue_class"]], fontsize=9)
ax.set_title("Anthropomorphization by venue class")
ax.yaxis.set_major_formatter(PCT)
ax.legend()
save(fig, "fig25_anthro_by_venue.png")

print("\ndone — untouched (source data not persisted): fig09, fig21")
