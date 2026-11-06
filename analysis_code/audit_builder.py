"""Fast verification audit: confirm/reject auto labels instead of blind coding.

Usage:
    python audit_builder.py build   -> outputs/annotation/audit_workbook.xlsx
    python audit_builder.py score   -> precision + Wilson CIs from the workbook
    python audit_builder.py rescore -> re-validate the NEW clause-level modality
                                       labels against the EXISTING 400 manual codes

Design: verification (y/n per row, auto label SHOWN) is 3-5x faster than blind
labelling. Strata are skewed towards rare classes and never-validated headline
claims; abundant classes already piloted get small top-up quotas.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotation_workbook import _rich, _verb_regex  # rich-text bolding, shared look

ROOT = Path(__file__).resolve().parents[1]
OUT_ANALYSIS = ROOT / "outputs" / "analysis"
OUT_ANNOT = ROOT / "outputs" / "annotation"
XLSX = OUT_ANNOT / "audit_workbook.xlsx"
SEED = 20260909

# sheet -> (source description, quotas)
CHECKS = {
    "modality clauses": dict(
        question="Is the modality label correct for THIS clause (see clause_text)?",
        quotas={"strong": 30, "mixed": 20, "weak/hedged": 30, "bare": 30},
    ),
    "agentless passives": dict(
        question="Is this really a passive clause with NO by-agent for this verb?",
        quotas={"agentless": 30, "agent_present": 15},
    ),
    "nominalization agents": dict(
        question="Is the agent relation correct (suppressed = no human/company grammatically linked to the nominal)?",
        quotas={"suppressed": 25, "explicit": 9},  # explicit: take all
    ),
    "anthro predications": dict(
        question="Does the cue really predicate the category of AI here (not a citation/homonym/negated-only mention)?",
        quotas={"any": 40},
    ),
}


def _sample(df, n, rng):
    return df.sample(min(n, len(df)), random_state=rng)


def build():
    rng = SEED
    sheets = {}

    cl = pd.read_parquet(OUT_ANALYSIS / "04b_modality_per_clause.parquet")
    parts = []
    for lab, n in CHECKS["modality clauses"]["quotas"].items():
        parts.append(_sample(cl[cl["auto_label"] == lab], n, rng))
    mod = pd.concat(parts).sample(frac=1, random_state=rng)

    def _markers(r):
        bits = []
        for label, col in (("strong", "modal_strong"), ("weak", "modal_weak"),
                           ("hedge", "hedge_lemmas"), ("hedge-adv", "hedge_adverbs"),
                           ("boost", "boosters")):
            v = str(r[col] or "").replace("|", ", ")
            if v:
                bits.append(f"{label}: {v}")
        return "   ".join(bits)

    mod["markers"] = mod.apply(_markers, axis=1)
    mod["focus"] = mod["ai_mentions"].str.split(" | ", regex=False).str[0]
    mod["bold_extra"] = mod["predicate_lemma"]
    sheets["modality clauses"] = mod[[
        "sent_text", "clause_text", "markers", "predicate_lemma",
        "auto_label", "focus", "bold_extra", "clause_id"]].copy()

    pas = pd.read_parquet(OUT_ANALYSIS / "06d_passives.parquet")
    ag = _sample(pas[pas["agentless"].astype(bool)],
                 CHECKS["agentless passives"]["quotas"]["agentless"], rng)
    ap = _sample(pas[~pas["agentless"].astype(bool)],
                 CHECKS["agentless passives"]["quotas"]["agent_present"], rng)
    pa = pd.concat([ag, ap]).sample(frac=1, random_state=rng)
    pa["auto_label"] = np.where(pa["agentless"].astype(bool),
                                "agentless", "agent_present")
    pa["focus"] = ""
    pa["bold_extra"] = pa["verb"]
    sheets["agentless passives"] = pa[[
        "sent_text", "verb", "subj_text", "agent_text", "agent_cat",
        "auto_label", "focus", "bold_extra", "work_id", "sent_id"]].copy()

    nom = pd.read_parquet(OUT_ANALYSIS / "04d_nominalizations.parquet")
    sup = _sample(nom[nom["agent_relation"] == "suppressed"],
                  CHECKS["nominalization agents"]["quotas"]["suppressed"], rng)
    exp = nom[nom["agent_relation"] != "suppressed"]  # only 9 exist: take all
    nm = pd.concat([sup, exp]).sample(frac=1, random_state=rng)
    nm = nm.rename(columns={"agent_relation": "auto_label"})
    nm["focus"] = nm["mention_text"]
    nm["bold_extra"] = nm["head_lemma"]
    sheets["nominalization agents"] = nm[[
        "sent_text", "mention_text", "head_lemma", "nom_pattern",
        "auto_label", "focus", "bold_extra", "work_id", "sent_id"]].copy()

    an = pd.read_parquet(OUT_ANALYSIS / "06f_anthro_per_sentence.parquet")
    cats = ["Cognizer", "Agent", "Emotion", "Communication",
            "Human_role_analogy", "Names_and_pronouns",
            "Biological_metaphors", "Products_of_cognition"]
    hits = []
    per_cat = max(3, CHECKS["anthro predications"]["quotas"]["any"] // len(cats))
    for c in cats:
        sub = an[an[c].astype(bool)]
        take = _sample(sub, per_cat, rng)
        take = take.assign(auto_label=c, cue=take[f"{c}_cue"],
                           anchor=take[f"{c}_anchor"])
        hits.append(take)
    at = pd.concat(hits).sample(frac=1, random_state=rng)
    at["focus"] = ""
    at["bold_extra"] = at["cue"].astype(str).str.split("[;,]", regex=True).str[0]
    sheets["anthro predications"] = at[[
        "sent_text", "cue", "anchor", "auto_label",
        "focus", "bold_extra", "work_id", "sent_id"]].copy()

    OUT_ANNOT.mkdir(parents=True, exist_ok=True)
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    HEADER_FILL = PatternFill("solid", fgColor="1F3864")
    HEADER_FONT = Font(color="FFFFFF", bold=True, size=11)
    INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")
    THIN = Side(style="thin", color="BFBFBF")
    BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
    HIDDEN = {"focus", "bold_extra", "clause_id", "work_id", "sent_id"}
    WIDTHS = {"sent_text": 95, "clause_text": 45, "markers": 30, "notes": 30,
              "subj_text": 24, "agent_text": 20, "auto_label": 15, "correct": 10}

    wb = Workbook()
    wb.remove(wb.active)

    guide = wb.create_sheet("Read me")
    guide.sheet_properties.tabColor = "C00000"
    guide_rows = [
        ("Verification audit", ""),
        ("", ""),
        ("The idea", "The auto label IS shown. You only confirm or reject it - "
                     "no relabelling. Bold marks the span the label is about."),
        ("How", "Fill the yellow `correct` column: y = label right, n = label "
                "wrong, na = PDF noise / unjudgeable."),
        ("", "If n, a one-word reason in notes helps but is optional."),
        ("", "Sheets stand alone: finish them in order, stop when time runs out."),
        ("Score", "python analysis_code/audit_builder.py score"),
        ("", ""),
        ("Sheet", "Question to answer per row"),
    ]
    for a, b in guide_rows:
        guide.append([a, b])
    for name, cfg in CHECKS.items():
        guide.append([name, cfg["question"]])
    guide.column_dimensions["A"].width = 24
    guide.column_dimensions["B"].width = 110
    for row in guide.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    guide["A1"].font = Font(bold=True, size=14)
    for c in ("A3", "A4", "A7", "A9"):
        guide[c].font = Font(bold=True)

    progress = wb.create_sheet("Progress")
    progress.append(["Sheet", "Rows", "Judged", "Remaining", "% done"])

    for name, df in sheets.items():
        df = df.copy()
        df.insert(len(df.columns) - len(HIDDEN & set(df.columns)), "correct", "")
        df.insert(df.columns.get_loc("correct") + 1, "notes", "")
        cols = list(df.columns)

        ws = wb.create_sheet(name[:31])
        ws.append(cols)
        j_text = cols.index("sent_text")
        j_focus, j_extra = cols.index("focus"), cols.index("bold_extra")
        for rec in df.itertuples(index=False):
            values = ["" if v is None else v for v in rec]
            extra = _verb_regex(str(values[j_extra]))
            values[j_text] = _rich(str(values[j_text]), extra,
                                   focus_mention=str(values[j_focus]))
            ws.append(values)

        n_rows = len(df) + 1
        for j, col in enumerate(cols, start=1):
            letter = get_column_letter(j)
            ws.column_dimensions[letter].width = WIDTHS.get(col, 16)
            head = ws.cell(row=1, column=j)
            head.fill = HEADER_FILL
            head.font = HEADER_FONT
            head.alignment = Alignment(vertical="center", wrap_text=True)
            if col in HIDDEN:
                ws.column_dimensions[letter].hidden = True
            if col in ("correct", "notes"):
                for i in range(2, n_rows + 1):
                    ws.cell(row=i, column=j).fill = INPUT_FILL
            if col == "correct":
                dv = DataValidation(type="list", formula1='"y,n,na"',
                                    allow_blank=True, showDropDown=False)
                dv.prompt = CHECKS[name]["question"]
                ws.add_data_validation(dv)
                dv.add(f"{letter}2:{letter}{n_rows}")

        for row in ws.iter_rows(min_row=2, max_row=n_rows):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                cell.border = BORDER

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{n_rows}"
        ws.sheet_properties.tabColor = "1F3864"

        c_letter = get_column_letter(cols.index("correct") + 1)
        r = progress.max_row + 1
        progress.append([
            name, len(df),
            f"=COUNTA('{name[:31]}'!{c_letter}2:{c_letter}{n_rows})",
            f"=B{r}-C{r}", f"=IF(B{r}=0,0,C{r}/B{r})"])

    for j, w in enumerate((24, 8, 8, 10, 8), start=1):
        progress.column_dimensions[get_column_letter(j)].width = w
    for cell in progress[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT

    try:
        wb.save(XLSX)
    except PermissionError:
        sys.exit(f"Close {XLSX.name} in Excel first.")

    total = sum(len(d) for d in sheets.values())
    print(f"wrote {XLSX}")
    for name, d in sheets.items():
        print(f"  {name}: {len(d)} rows")
    print(f"total to verify: {total} rows (y/n each)")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def score():
    from openpyxl import load_workbook
    wb = load_workbook(XLSX, read_only=True)
    print("Verification audit - precision of auto labels (verified y / judged):\n")
    for name in CHECKS:
        if name not in wb.sheetnames:
            continue
        rows = list(wb[name].values)
        header = list(rows[0])
        df = pd.DataFrame(rows[1:], columns=header)
        df["correct"] = df["correct"].astype(object).fillna("").astype(str).str.strip().str.lower()
        judged = df[df["correct"].isin(["y", "n"])]
        if not len(judged):
            print(f"{name}: no rows judged yet")
            continue
        print(f"{name} ({len(judged)} judged, {len(df) - len(judged)} pending/na):")
        for lab, g in judged.groupby("auto_label"):
            k, n = (g["correct"] == "y").sum(), len(g)
            lo, hi = wilson(k, n)
            print(f"  {str(lab):22s} n={n:3d} precision={k / n:.3f} [{lo:.3f},{hi:.3f}]")
        k, n = (judged["correct"] == "y").sum(), len(judged)
        lo, hi = wilson(k, n)
        print(f"  {'OVERALL':22s} n={n:3d} precision={k / n:.3f} [{lo:.3f},{hi:.3f}]\n")


def rescore():
    """Existing sentence-level manual codes vs new clause-level auto labels.

    The manual instruction was to code the focus mention's clause, so the fair
    target is the clause containing that mention, not the sentence aggregate.
    """
    s = pd.read_csv(OUT_ANNOT / "04b_modality_sample.csv")
    k = pd.read_csv(OUT_ANNOT / "keys" / "04b_modality_key.csv")
    cl = pd.read_parquet(OUT_ANALYSIS / "04b_modality_per_clause.parquet")

    df = s.merge(k[["row_uid", "work_id", "sent_id"]], on="row_uid")
    df["manual_label"] = (df["manual_label"].astype(object).fillna("")
                          .astype(str).str.strip())
    df = df[df["manual_label"].ne("") & df["manual_label"].ne("NA")]

    by_sent = {key: g for key, g in cl.groupby(["work_id", "sent_id"])}
    rows = []
    for r in df.itertuples(index=False):
        g = by_sent.get((r.work_id, r.sent_id))
        if g is None:
            continue
        focus = str(r.focus_mention).lower()
        hit = g[g["ai_mentions"].str.lower().str.contains(focus, regex=False)]
        pick = (hit if len(hit) else g).iloc[0]
        rows.append({"manual": r.manual_label, "auto": pick["auto_label"]})
    m = pd.DataFrame(rows)

    print(f"matched: {len(m)} of {len(df)} coded rows\n")
    print("confusion (rows=clause auto, cols=manual):")
    print(pd.crosstab(m["auto"], m["manual"]).to_string())
    print("\nper-class precision of clause-level auto labels:")
    for lab, g in m.groupby("auto"):
        k_, n = (g["manual"] == lab).sum(), len(g)
        lo, hi = wilson(k_, n)
        print(f"  {lab:12s} n={n:3d} precision={k_ / n:.3f} [{lo:.3f},{hi:.3f}]")
    print(f"\nraw agreement: {(m['auto'] == m['manual']).mean():.3f}")
    corpus = cl["auto_label"].value_counts(normalize=True)
    w = sum(corpus.get(lab, 0) * (g["manual"] == lab).mean()
            for lab, g in m.groupby("auto"))
    print(f"corpus-reweighted accuracy: {w:.3f}")
    binary = ((m["auto"] == "bare") == (m["manual"] == "bare")).mean()
    print(f"binary bare vs modalised agreement: {binary:.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    {"build": build, "score": score, "rescore": rescore}[cmd]()
