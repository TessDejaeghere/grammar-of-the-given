"""Build blind, stratified annotation samples and score them.

Usage
-----
    python annotation_sampler.py sample     # writes outputs/annotation/*_sample.csv
    python annotation_sampler.py score      # merges your codings back, reports precision + kappa

The samples are *blind*: auto-labels are stripped and rows are shuffled, so the
coder is not anchored. The mapping from `row_uid` back to the auto-label lives in
outputs/annotation/keys/ and is only read by `score`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "outputs" / "analysis"
ANNO = ROOT / "outputs" / "annotation"
KEYS = ANNO / "keys"

SEED = 20260907
WORK_CAP = 5          # max rows per work per stratum, to stop long books dominating
QUOTA = 100           # rows per stratum
PASS2_N = 100         # rows re-coded for the reliability estimate

SHEETS = {
    "04b_modality": {
        "source": "04b_modality_per_sentence.csv",
        "stratum": "auto_label",
        "keep": ["sent_text"],
        "manual": {"manual_label": "strong | weak/hedged | mixed | bare | NA"},
        "quota": 100,
    },
    "04d_nominalization": {
        "source": "04d_nominalizations_per_sentence.csv",
        "stratum": "agent_present",
        "keep": ["governor_lemma", "sent_text"],
        "manual": {"manual_agent_present": "TRUE | FALSE | NA"},
        "quota": 125,
    },
    "04e_polarity": {
        "source": "04e_polarity_per_mention.csv",
        "stratum": "stance",
        "keep": ["mention_text", "gov_verb", "sent_text"],
        "manual": {
            "manual_negated": "TRUE | FALSE | NA",
            "manual_projected": "TRUE | FALSE | NA",
        },
        # The coder judges two booleans; the auto label is their combination.
        "derive": lambda d: (
            d["manual_projected"].astype(str).str.strip().str.lower()
            .map({"true": "projected", "false": "asserted"})
            + "_"
            + d["manual_negated"].astype(str).str.strip().str.lower()
            .map({"true": "negative", "false": "positive"})
        ),
        "quota": 100,
    },
}


def _stratified(df: pd.DataFrame, stratum: str, quota: int, rng) -> pd.DataFrame:
    out = []
    for _, grp in df.groupby(stratum, dropna=False):
        if "work_id" in grp.columns:
            grp = (
                grp.sample(frac=1.0, random_state=rng.integers(1 << 31))
                .groupby("work_id", sort=False)
                .head(WORK_CAP)
            )
        take = min(quota, len(grp))
        out.append(grp.sample(n=take, random_state=rng.integers(1 << 31)))
    return pd.concat(out, ignore_index=True)


def _ai_index() -> pd.DataFrame:
    """AI mentions with their occurrence index within the sentence.

    Matcher hits are appended in regex order, so the running count matches the
    order the same regex finds them in when the workbook highlights a cell.
    """
    ment = pd.read_parquet(ROOT / "outputs" / "parsed" / "mentions.parquet")
    ai = ment[ment["category"] == "AI"].copy()
    ai["ai_idx"] = ai.groupby(["work_id", "sent_id"]).cumcount()
    return ai


def _attach_focus(name: str, picked: pd.DataFrame) -> pd.DataFrame:
    """Add focus_mention / focus_occurrence so only one span gets highlighted."""
    picked = picked.copy()
    picked["focus_mention"] = ""
    picked["focus_occurrence"] = -1
    if not {"work_id", "sent_id"} <= set(picked.columns):
        return picked

    if name == "04b_modality":
        ai = _ai_index()
        subj_obj = {"nsubj", "nsubjpass", "nsubj:pass", "dobj", "obj"}
        first = (ai[ai["head_dep"].isin(subj_obj)]
                 .sort_values(["work_id", "sent_id", "ai_idx"])
                 .groupby(["work_id", "sent_id"]).head(1)
                 [["work_id", "sent_id", "mention_text", "ai_idx"]])
        picked = picked.merge(first, on=["work_id", "sent_id"], how="left",
                              suffixes=("", "_focus"))
    elif name == "04e_polarity":
        # The sheet already names its mention; ai_idx came along with the sample.
        picked["ai_idx"] = picked.get("ai_idx", -1)
    elif name == "04d_nominalization":
        ai = _ai_index()[["work_id", "sent_id", "mention_text", "ai_idx"]]
        ai = ai.drop_duplicates(["work_id", "sent_id", "mention_text"])
        picked = picked.merge(ai, on=["work_id", "sent_id", "mention_text"], how="left")

    if "mention_text" in picked.columns:
        picked["focus_mention"] = picked["mention_text"].fillna("").astype(str)
    if "ai_idx" in picked.columns:
        picked["focus_occurrence"] = (
            pd.to_numeric(picked["ai_idx"], errors="coerce").fillna(-1).astype(int))
    return picked


def sample() -> None:
    ANNO.mkdir(parents=True, exist_ok=True)
    KEYS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    legend = []

    for name, spec in SHEETS.items():
        src = ANALYSIS / spec["source"]
        if not src.exists():
            print(f"skip {name}: {src.name} not found")
            continue
        df = pd.read_csv(src)
        picked = _stratified(df, spec["stratum"], spec["quota"], rng)
        picked = picked.sample(frac=1.0, random_state=rng.integers(1 << 31)).reset_index(drop=True)
        picked["row_uid"] = [f"{name}-{i:04d}" for i in range(len(picked))]
        picked = _attach_focus(name, picked)

        key_cols = ["row_uid", "work_id", "sent_id", spec["stratum"]]
        key_cols = [c for c in key_cols if c in picked.columns]
        picked[key_cols].to_csv(KEYS / f"{name}_key.csv", index=False)

        blind = picked[["row_uid"] + [c for c in spec["keep"] if c in picked.columns]].copy()
        blind["focus_mention"] = picked["focus_mention"]
        blind["focus_occurrence"] = picked["focus_occurrence"]
        for col in spec["manual"]:
            blind[col] = ""
        blind["notes"] = ""
        blind.to_csv(ANNO / f"{name}_sample.csv", index=False, encoding="utf-8-sig")

        pass2 = blind.sample(n=min(PASS2_N, len(blind)),
                             random_state=rng.integers(1 << 31))
        pass2 = pass2.sample(frac=1.0, random_state=rng.integers(1 << 31))
        pass2.to_csv(ANNO / f"{name}_sample_pass2.csv", index=False, encoding="utf-8-sig")

        counts = picked[spec["stratum"]].value_counts().to_dict()
        print(f"{name}: {len(blind)} rows  (pass2 {len(pass2)})  strata {counts}")
        for col, allowed in spec["manual"].items():
            legend.append({"sheet": name, "column": col, "allowed_values": allowed})

    # 04c is a *type* sheet: coding a verb fixes every token of it, so it is
    # coded down to n >= 2 rather than sampled.
    vsrc = ANALYSIS / "04c_verbs_ranked.csv"
    if vsrc.exists():
        vr = pd.read_csv(vsrc).sort_values("n", ascending=False)
        keep = vr[vr["n"] >= 2].copy()
        keep[["verb_lemma", "n", "examples"]].assign(manual_class="", notes="").to_csv(
            ANNO / "04c_verb_types.csv", index=False, encoding="utf-8-sig")
        vr[["verb_lemma", "n", "auto_class"]].to_csv(KEYS / "04c_verb_types_key.csv", index=False)
        cover = keep["n"].sum() / vr["n"].sum()
        print(f"04c_verb_types: {len(keep)} types covering {cover:.1%} of tokens")
        legend.append({
            "sheet": "04c_verb_types", "column": "manual_class",
            "allowed_values": ("material_action | cognitive_mental | instrumental_tool | "
                               "verbal_communicative | stative_relational | behavioural | "
                               "evaluative_performance | semiotic_evidential | OTHER"),
        })

    # 06f is sampled inside its own notebook (quota per category, capped per
    # work), so here we only copy the sheet in and register its legend.
    src6f = ANALYSIS / "06f_anthro_sample.csv"
    if src6f.exists():
        sheet = pd.read_csv(src6f)
        # mention_terms lists every AI match in the sentence; highlight the first.
        sheet["focus_mention"] = (sheet["mention_terms"].fillna("").astype(str)
                                  .str.split("|").str[0].str.strip())
        sheet["focus_occurrence"] = 0
        sheet.to_csv(ANNO / "06f_anthropomorphization_sample.csv",
                     index=False, encoding="utf-8-sig")
        key = pd.read_csv(ANALYSIS / "06f_anthro_sample_key.csv")
        key.to_csv(KEYS / "06f_anthropomorphization_key.csv", index=False)
        pass2 = sheet.sample(n=min(PASS2_N, len(sheet)), random_state=SEED)
        pass2.to_csv(ANNO / "06f_anthropomorphization_sample_pass2.csv",
                     index=False, encoding="utf-8-sig")
        print(f"06f_anthropomorphization: {len(sheet)} rows  (pass2 {len(pass2)})")
        legend.append({
            "sheet": "06f_anthropomorphization", "column": "manual_categories",
            "allowed_values": ("semicolon-separated, per Inie, Zukerman & Bender (2026) Table 2: "
                               "Cognizer | Products_of_cognition | Emotion | Communication | "
                               "Agent | Human_role_analogy | Names_and_pronouns | "
                               "Biological_metaphors | None"),
        })

    pd.DataFrame(legend).to_csv(ANNO / "legend.csv", index=False)
    print(f"\nwrote {ANNO}")
    print("Code the *_sample.csv files, then re-code *_sample_pass2.csv blind for kappa.")


def _clean_str(s: pd.Series) -> pd.Series:
    """Blank cells survive as pd.NA under string dtype, so normalise first."""
    return s.astype("object").fillna("").astype(str).str.strip()


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _kappa(a: pd.Series, b: pd.Series) -> float:
    cats = sorted(set(a.dropna()) | set(b.dropna()))
    n = len(a)
    if n == 0 or len(cats) < 2:
        return float("nan")
    po = (a.values == b.values).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def _score_sheet(name: str, spec: dict) -> None:
    path = ANNO / f"{name}_sample.csv"
    keypath = KEYS / f"{name}_key.csv"
    if not path.exists() or not keypath.exists():
        return
    coded = pd.read_csv(path)
    manual_col = list(spec["manual"])[0]
    for col in spec["manual"]:
        coded[col] = _clean_str(coded[col])
    coded = coded[coded[manual_col].ne("") & coded[manual_col].str.lower().ne("nan")]
    if coded.empty:
        print(f"{name}: not coded yet")
        return
    judged = "_judged"
    coded[judged] = spec["derive"](coded) if "derive" in spec else coded[manual_col]

    key = pd.read_csv(keypath)
    m = coded.merge(key, on="row_uid", how="left")
    auto = spec["stratum"]
    m[auto] = m[auto].astype(str).str.strip()
    m[judged] = m[judged].astype(str).str.strip()

    # 'NA' marks a row the coder judged unusable (PDF furniture, truncated
    # sentence, wrong language). Those rows say nothing about precision.
    n_na = int(m[judged].str.upper().eq("NA").sum())
    m = m[~m[judged].str.upper().eq("NA")]
    if m.empty:
        print(f"{name}: only NA rows coded")
        return

    print(f"\n=== {name}  ({len(m)} coded" + (f", {n_na} NA excluded)" if n_na else ")"))
    rows = []
    for lab, grp in m.groupby(auto):
        hits = int((grp[judged].str.lower() == lab.lower()).sum())
        lo, hi = _wilson(hits, len(grp))
        rows.append({"auto_label": lab, "n_coded": len(grp),
                     "precision": round(hits / len(grp), 3),
                     "ci95": f"[{lo:.2f}, {hi:.2f}]"})
    print(pd.DataFrame(rows).to_string(index=False))

    # Corpus-level accuracy, reweighted by the true class proportions (the
    # sample is stratified, so the raw mean would be biased).
    full = pd.read_csv(ANALYSIS / spec["source"])
    w = full[auto].astype(str).value_counts(normalize=True)
    acc = sum(w.get(r["auto_label"], 0.0) * r["precision"] for r in rows)
    print(f"reweighted overall accuracy: {acc:.3f}")

    p2path = ANNO / f"{name}_sample_pass2.csv"
    if p2path.exists():
        p2 = pd.read_csv(p2path)
        for col in spec["manual"]:
            p2[col] = _clean_str(p2[col])
        p2 = p2[p2[manual_col].ne("") & p2[manual_col].str.lower().ne("nan")]
        if p2.empty:
            return
        p2[judged] = spec["derive"](p2) if "derive" in spec else p2[manual_col]
        both = coded[["row_uid", judged]].merge(
            p2[["row_uid", judged]], on="row_uid", suffixes=("_1", "_2"))
        if len(both) >= 20:
            k = _kappa(both[f"{judged}_1"], both[f"{judged}_2"])
            agree = (both[f"{judged}_1"] == both[f"{judged}_2"]).mean()
            print(f"reliability on {len(both)} double-coded rows: "
                  f"agreement {agree:.3f}, Cohen's kappa {k:.3f}")


def score() -> None:
    for name, spec in SHEETS.items():
        _score_sheet(name, spec)

    vpath = ANNO / "04c_verb_types.csv"
    if vpath.exists():
        coded = pd.read_csv(vpath)
        coded["manual_class"] = _clean_str(coded["manual_class"])
        done = coded[coded["manual_class"].ne("") & coded["manual_class"].str.lower().ne("nan")]
        if len(done):
            key = pd.read_csv(KEYS / "04c_verb_types_key.csv")
            m = done.merge(key, on="verb_lemma", how="left")
            agree = (m["manual_class"] == m["auto_class"]).mean()
            n_col = "n_x" if "n_x" in m.columns else "n"
            tok_cov = m[n_col].sum() / pd.read_csv(
                ANALYSIS / "04c_verbs_ranked.csv")["n"].sum()
            print(f"\n=== 04c_verb_types ({len(m)} types coded, {tok_cov:.1%} of tokens)")
            print(f"auto/manual agreement on types: {agree:.3f}")
            tok = m.assign(ok=m["manual_class"] == m["auto_class"])
            print("token-weighted agreement: "
                  f"{(tok['ok'] * tok[n_col]).sum() / tok[n_col].sum():.3f}")
        else:
            print("\n04c_verb_types: not coded yet")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sample"
    if cmd == "sample":
        sample()
    elif cmd == "score":
        score()
    else:
        print(__doc__)
