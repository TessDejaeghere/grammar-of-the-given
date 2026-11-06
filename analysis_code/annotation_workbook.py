"""Build a working Excel workbook from the annotation CSVs, and read it back.

    python annotation_workbook.py build     # CSVs  -> outputs/annotation/annotation_workbook.xlsx
    python annotation_workbook.py import    # workbook -> CSVs (so `annotation_sampler.py score` works)

The CSVs stay the source of truth for scoring; the workbook is just a comfortable
surface to code in. Dropdowns are generated from the same legend the sampler writes,
so the two cannot drift apart.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import regex as re
from openpyxl import Workbook, load_workbook
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.ai_lexicon import compile_ai_regex

ROOT = Path(__file__).resolve().parent.parent
ANNO = ROOT / "outputs" / "annotation"
XLSX = ANNO / "annotation_workbook.xlsx"

AI_RE = compile_ai_regex()
BOLD = InlineFont(b=True)


def _verb_regex(lemma: str) -> re.Pattern | None:
    """Inflected forms of a verb lemma, good enough to highlight in examples."""
    lemma = str(lemma).strip()
    if not lemma or not lemma.isalpha():
        return None
    stem = re.escape(lemma)
    alts = [stem, stem + "s", stem + "es", stem + "ed", stem + "d", stem + "ing"]
    if len(lemma) > 3 and lemma.endswith("e"):
        base = re.escape(lemma[:-1])
        alts += [base + "ing", base + "ed"]
    return re.compile(r"\b(?:" + "|".join(alts) + r")\b", re.IGNORECASE)


def _rich(text: str, extra: re.Pattern | None = None,
          focus_mention: str = "", focus_occurrence: int = -1,
          extra_all: bool = False) -> CellRichText | str:
    """Bold the one span the coder is meant to judge, plus its anchor token.

    A sentence often contains several AI mentions; only the one this row is
    about is highlighted. Falls back to the first exact-string match, then to
    every match, so a row is never left with nothing marked.
    """
    text = "" if text is None else str(text)
    if not text:
        return text

    matches = [(m.start(), m.end()) for m in AI_RE.finditer(text)]
    spans: list[tuple[int, int]] = []
    if matches:
        if 0 <= focus_occurrence < len(matches):
            spans = [matches[focus_occurrence]]
        elif focus_mention:
            hit = re.search(re.escape(focus_mention.strip()), text, re.IGNORECASE)
            spans = [(hit.start(), hit.end())] if hit else [matches[0]]
        else:
            spans = matches
    if extra is not None:
        found = [(m.start(), m.end()) for m in extra.finditer(text)]
        spans += found if extra_all else found[:1]
    if not spans:
        return text

    merged: list[list[int]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    parts: list = []
    cursor = 0
    for s, e in merged:
        if s > cursor:
            parts.append(text[cursor:s])
        parts.append(TextBlock(BOLD, text[s:e]))
        cursor = e
    if cursor < len(text):
        parts.append(text[cursor:])
    return CellRichText(*parts)

ANTHRO_CATEGORIES = [
    "Cognizer", "Products_of_cognition", "Emotion", "Communication",
    "Agent", "Human_role_analogy", "Names_and_pronouns", "Biological_metaphors",
]

# sheet name -> (csv file, {column: dropdown options}, guidance line)
SHEETS: dict[str, tuple[str, dict[str, list[str]], str]] = {
    "04b modality": (
        "04b_modality_sample.csv",
        {"manual_label": ["strong", "weak/hedged", "mixed", "scope_uncertain", "bare", "NA"]},
        "Judge the clause around the AI mention. bare = no modal or hedge at all.",
    ),
    "04b modality p2": (
        "04b_modality_sample_pass2.csv",
        {"manual_label": ["strong", "weak/hedged", "mixed", "scope_uncertain", "bare", "NA"]},
        "Reliability pass. Code without looking at the first pass.",
    ),
    "04c verb types": (
        "04c_verb_types.csv",
        {"manual_class": [
            "material_action", "cognitive_mental", "instrumental_tool",
            "verbal_communicative", "stative_relational", "behavioural",
            "evaluative_performance", "semiotic_evidential", "OTHER"]},
        "One row per verb type; coding a verb fixes every token of it. 92% of tokens.",
    ),
    "04d nominalisation": (
        "04d_nominalization_sample.csv",
        {"manual_agent_present": ["TRUE", "FALSE", "NA"]},
        "Is a human or corporate agent realised anywhere in the sentence?",
    ),
    "04d nominalisation p2": (
        "04d_nominalization_sample_pass2.csv",
        {"manual_agent_present": ["TRUE", "FALSE", "NA"]},
        "Reliability pass.",
    ),
    "04e polarity": (
        "04e_polarity_sample.csv",
        {"manual_negated": ["TRUE", "FALSE", "NA"],
         "manual_projected": ["TRUE", "FALSE", "NA"]},
        "negated = the process is denied. projected = the claim is attributed to someone.",
    ),
    "04e polarity p2": (
        "04e_polarity_sample_pass2.csv",
        {"manual_negated": ["TRUE", "FALSE", "NA"],
         "manual_projected": ["TRUE", "FALSE", "NA"]},
        "Reliability pass.",
    ),
    "06f anthropomorphization": (
        "06f_anthropomorphization_sample.csv",
        {c: ["y"] for c in ANTHRO_CATEGORIES},
        "Mark every category that applies (Inie, Zukerman & Bender 2026, Table 2). "
        "Leave all blank for None.",
    ),
    "06f anthro p2": (
        "06f_anthropomorphization_sample_pass2.csv",
        {c: ["y"] for c in ANTHRO_CATEGORIES},
        "Reliability pass.",
    ),
}

HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(color="FFFFFF", bold=True, size=11)
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

WIDE_COLS = {"sent_text", "examples"}
NARROW_COLS = {"row_uid", "n", "verb_lemma", "gov_verb", "governor_lemma",
               "mention_text", "mention_terms"}


def _prepare(df: pd.DataFrame, sheet: str, options: dict[str, list[str]]) -> pd.DataFrame:
    """06f ships one free-text column; the workbook uses one column per category."""
    if sheet.startswith("06f"):
        df = df.drop(columns=[c for c in ("manual_categories",) if c in df.columns])
        for c in ANTHRO_CATEGORIES:
            df[c] = ""
        if "notes" in df.columns:
            df = df[[c for c in df.columns if c != "notes"] + ["notes"]]
    for col in options:
        if col not in df.columns:
            df[col] = ""
    return df


def build() -> None:
    wb = Workbook()
    wb.remove(wb.active)

    guide = wb.create_sheet("Read me")
    guide.sheet_properties.tabColor = "C00000"
    guide_rows = [
        ("Annotation workbook", ""),
        ("", ""),
        ("Codebook", "outputs/annotation/CODEBOOK.md \u2014 read section 0 before you start, "
                     "then the section for the sheet you are on."),
        ("How to use", "Code the yellow columns only. Everything else is context."),
        ("", "Yellow cells have dropdowns; type the first letter to jump."),
        ("", "Leave a cell blank if you have not coded that row yet."),
        ("", "Put NA (and 'noise' in notes) for PDF furniture: reference lists, figure "
             "captions, truncated or non-English rows. NA rows are excluded from scoring."),
        ("", "'p2' sheets are the reliability pass \u2014 code them later, without "
             "looking at your first pass."),
        ("", "Save the file, then run:  python analysis_code/annotation_workbook.py import"),
        ("", "then:  python analysis_code/annotation_sampler.py score"),
        ("", ""),
        ("Order of work", "1. 04c verb types (best value: 315 rows fixes 92% of verb tokens)"),
        ("", "2. 04b modality, 3. 04e polarity, 4. 06f anthropomorphization, 5. 04d"),
        ("", ""),
        ("Sheet", "What to judge"),
    ]
    for a, b in guide_rows:
        guide.append([a, b])
    for sheet, (_, _, note) in SHEETS.items():
        guide.append([sheet, note])
    guide.column_dimensions["A"].width = 26
    guide.column_dimensions["B"].width = 110
    for row in guide.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    guide["A1"].font = Font(bold=True, size=14)
    for cell in ("A3", "A4", "A12", "A15"):
        guide[cell].font = Font(bold=True)

    progress = wb.create_sheet("Progress")
    progress.append(["Sheet", "Rows", "Coded", "Remaining", "% done"])

    for sheet, (fname, options, _note) in SHEETS.items():
        src = ANNO / fname
        if not src.exists():
            print(f"skip {sheet}: {fname} not found")
            continue
        df = pd.read_csv(src).fillna("")
        df = _prepare(df, sheet, options)

        ws = wb.create_sheet(sheet[:31])
        ws.append(list(df.columns))
        cols = list(df.columns)
        text_col = "sent_text" if "sent_text" in cols else (
            "examples" if "examples" in cols else None)
        # 04c is judged on the verb, 04d on the nominalised noun, 04e on the
        # governing verb; bold those too, alongside the AI mention.
        focus_col = next((c for c in ("verb_lemma", "governor_lemma", "gov_verb")
                          if c in cols), None)
        has_focus = "focus_mention" in cols and "focus_occurrence" in cols
        for rec in df.itertuples(index=False):
            values = list(rec)
            if text_col is not None:
                j = cols.index(text_col)
                extra = _verb_regex(values[cols.index(focus_col)]) if focus_col else None
                mention, occurrence = "", -1
                if has_focus:
                    mention = str(values[cols.index("focus_mention")] or "")
                    try:
                        occurrence = int(values[cols.index("focus_occurrence")])
                    except (TypeError, ValueError):
                        occurrence = -1
                # 04c examples concatenate three unrelated sentences, so the
                # per-row occurrence index does not apply there.
                if text_col == "examples":
                    mention, occurrence = "", -1
                values[j] = _rich(values[j], extra, mention, occurrence,
                                  extra_all=(text_col == "examples"))
            ws.append(["" if v is None else v for v in values])

        n_rows = len(df) + 1
        for j, col in enumerate(df.columns, start=1):
            letter = get_column_letter(j)
            if col in WIDE_COLS:
                width = 95
            elif col in options:
                width = 22
            elif col == "notes":
                width = 34
            elif col in NARROW_COLS:
                width = 16
            else:
                width = 20
            ws.column_dimensions[letter].width = width

            cell = ws.cell(row=1, column=j)
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
            cell.alignment = Alignment(vertical="center", wrap_text=True)

            if col in ("focus_mention", "focus_occurrence"):
                ws.column_dimensions[letter].hidden = True

            if col in options or col == "notes":
                for i in range(2, n_rows + 1):
                    ws.cell(row=i, column=j).fill = INPUT_FILL

            if col in options:
                opts = ",".join(options[col])
                dv = DataValidation(type="list", formula1=f'"{opts}"',
                                    allow_blank=True, showDropDown=False)
                dv.error = "Pick a value from the list."
                dv.prompt = f"Allowed: {opts}"
                ws.add_data_validation(dv)
                dv.add(f"{letter}2:{letter}{n_rows}")

        for row in ws.iter_rows(min_row=2, max_row=n_rows):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                cell.border = BORDER

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(df.columns))}{n_rows}"
        ws.sheet_properties.tabColor = "1F3864" if " p2" not in sheet else "808080"

        first_input = get_column_letter(list(df.columns).index(list(options)[0]) + 1)
        r = progress.max_row + 1
        progress.append([
            sheet, len(df),
            f"=COUNTA('{sheet[:31]}'!{first_input}2:{first_input}{n_rows})",
            f"=B{r}-C{r}", f"=IF(B{r}=0,0,C{r}/B{r})",
        ])
        print(f"{sheet}: {len(df)} rows, {len(options)} coded column(s)")

    for j, w in enumerate([28, 10, 10, 12, 10], start=1):
        progress.column_dimensions[get_column_letter(j)].width = w
    for cell in progress[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
    for i in range(2, progress.max_row + 1):
        progress.cell(row=i, column=5).number_format = "0%"
    progress.freeze_panes = "A2"
    progress.sheet_properties.tabColor = "375623"

    ANNO.mkdir(parents=True, exist_ok=True)
    try:
        wb.save(XLSX)
    except PermissionError:
        print(f"\nCannot write {XLSX.name} - it is open in Excel.")
        print("Close the workbook and run this again. If you have already coded "
              "rows in it, run 'import' first so your work is not lost.")
        return
    print(f"\nwrote {XLSX}")


def import_back() -> None:
    if not XLSX.exists():
        print(f"{XLSX} not found - run 'build' first")
        return
    wb = load_workbook(XLSX, data_only=True)
    for sheet, (fname, options, _note) in SHEETS.items():
        name = sheet[:31]
        if name not in wb.sheetnames:
            continue
        ws = wb[name]
        rows = list(ws.values)
        if not rows:
            continue
        cols = list(rows[0])
        edited = pd.DataFrame(rows[1:], columns=cols)

        target = ANNO / fname
        df = pd.read_csv(target)
        key = "row_uid" if "row_uid" in df.columns else "verb_lemma"
        edited[key] = edited[key].astype(str)
        df[key] = df[key].astype(str)

        if sheet.startswith("06f"):
            def join(r):
                return ";".join(c for c in ANTHRO_CATEGORIES
                                if str(r.get(c) or "").strip().lower() == "y")
            edited["manual_categories"] = edited.apply(join, axis=1)
            cols_to_take = ["manual_categories", "notes"]
        else:
            cols_to_take = [c for c in list(options) + ["notes"] if c in edited.columns]

        merged = df.drop(columns=[c for c in cols_to_take if c in df.columns]).merge(
            edited[[key] + cols_to_take], on=key, how="left")
        merged.to_csv(target, index=False, encoding="utf-8-sig")
        filled = (merged[cols_to_take[0]].astype("object").fillna("")
                  .astype(str).str.strip())
        n = int((~filled.isin(["", "None", "nan"])).sum())
        print(f"{sheet}: wrote {fname}  ({n} coded)")


def prefill() -> None:
    """Fill only blank first-pass labels from automatic key columns."""
    targets = {
        "04b_modality": ("04b_modality_sample.csv", "04b_modality_key.csv", ["manual_label"]),
        "04c_verb_types": ("04c_verb_types.csv", "04c_verb_types_key.csv", ["manual_class"]),
        "04d_nominalization": ("04d_nominalization_sample.csv", "04d_nominalization_key.csv", ["manual_agent_present"]),
        "04e_polarity": ("04e_polarity_sample.csv", "04e_polarity_key.csv", ["manual_negated", "manual_projected"]),
        "06f_anthropomorphization": ("06f_anthropomorphization_sample.csv", "06f_anthropomorphization_key.csv", ["manual_categories"]),
    }

    for name, (sample_name, key_name, manual_cols) in targets.items():
        sample_path = ANNO / sample_name
        key_path = ANNO / "keys" / key_name
        if not sample_path.exists() or not key_path.exists():
            continue
        sample = pd.read_csv(sample_path).fillna("")
        key = pd.read_csv(key_path).fillna("")
        join = "verb_lemma" if name == "04c_verb_types" else "row_uid"
        sample = sample.drop(
            columns=[c for c in key.columns if c != join and c in sample.columns],
            errors="ignore",
        )
        sample = sample.merge(key, on=join, how="left", suffixes=("", "_auto"))

        for col in manual_cols:
            sample[col] = sample[col].astype(object)
            blank = sample[col].astype(str).str.strip().isin(["", "nan", "None"])
            if name == "04e_polarity":
                suggestion = sample["stance"].map({
                    "asserted_positive": (False, False),
                    "asserted_negative": (True, False),
                    "projected_positive": (False, True),
                    "projected_negative": (True, True),
                })
                suggestion = suggestion.map(lambda pair: {
                    "manual_negated": str(pair[0]).upper(),
                    "manual_projected": str(pair[1]).upper(),
                }[col])
            elif name == "06f_anthropomorphization":
                suggestion = sample["auto_category"].replace("None", "")
            else:
                auto_col = {
                    "manual_label": "auto_label",
                    "manual_class": "auto_class",
                    "manual_agent_present": "agent_present",
                }[col]
                suggestion = sample[auto_col]
            sample.loc[blank, col] = suggestion[blank]

        if "notes" in sample.columns:
            blank_note = sample["notes"].astype(str).str.strip().isin(["", "nan", "None"])
            sample.loc[blank_note, "notes"] = "AUTO SUGGESTION - post-correct"
        sample = sample.drop(
            columns=[c for c in key.columns if c != join and c in sample.columns],
            errors="ignore",
        )
        sample.to_csv(sample_path, index=False, encoding="utf-8-sig")
        print(f"{name}: prefilled blank labels")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build()
    elif cmd in ("import", "import-excel"):
        import_back()
    elif cmd == "prefill":
        prefill()
    else:
        print(__doc__)
