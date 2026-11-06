# The Grammar of the Given: code and data

Code and derived data for

> Dejaeghere, T. *The Grammar of the Given: How Does Digital Humanities Discuss A.I., and Why Does It Matter?* Submitted to *Computational Linguistics in the Netherlands Journal*.

The study combines Systemic Functional Linguistics and Critical Discourse Analysis to examine how digital humanities (DH) scholarship talks about AI. It has two layers:

- a **macro-corpus** of 22,827 DH abstracts from OpenAlex (1990–2026), used for term frequencies over time;
- a **micro-corpus** of 795 full-text DH articles (2022–2026) that mention AI, parsed with spaCy and analysed for grammatical role, process type, modality, nominalisation, negation, passives and anthropomorphism.

## What is (and is not) in this repository

| Included | Not included |
|---|---|
| All analysis code (fetch scripts, notebooks, helper library, figure scripts) | Full texts and PDFs of the articles |
| OpenAlex work IDs and metadata (titles, venues, years, etc.) | Sentences, clauses and keyword-in-context snippets |
| Every derived table, with one row per AI mention, sentence, clause or work, keyed by OpenAlex work ID and sentence ID | Abstracts |
| Automatic labels, manual annotation keys and the audit judgements | spaCy `DocBin` files |
| The figures as printed in the paper | |

The articles have mixed licences, so we do not redistribute their text. Every row in `outputs/` carries an OpenAlex `work_id` (e.g. `W4415144040`) and, where relevant, a `sent_id`. You can rebuild the text layer yourself with the fetch scripts below. Re-parsing the same documents with the same spaCy version yields the same sentence IDs.

## Repository layout

```
fetch_openalex_dh.py            # step 0a: fetch DH works + abstracts from OpenAlex
fetch_openalex_dh_fulltext.py   # step 0b: download PDFs and extract text
analysis_code/
  01_macro_openalex.ipynb       # macro-corpus: filtering, term counts per year
  02_fulltext_clean_filter.ipynb# full-text cleaning, language/length filter, AI corpus
  03_parse_and_extract.ipynb    # spaCy (en_core_web_trf) parsing, AI-mention extraction
  04a_agency.ipynb              # grammatical role of AI mentions
  04b_modality.ipynb            # clause-level modality / hedging
  04c_valency.ipynb             # verb (process) types
  04d_nominalization.ipynb      # nominalisations and agent presence
  04e_negation.ipynb            # negation, projection and stance
  05a_diachronic.ipynb … 05d_ai_vs_human.ipynb   # diachronic, KWIC, metaphor frames, AI vs human actors
  06a_statistics.ipynb … 06f_anthropomorphization.ipynb  # models, term groups, collocations, passives, provenance, anthropomorphism
  _make_*.py, _nbbuild.py       # generators that write the notebooks above
  lib/                          # lexicons (AI terms, actors, anthropomorphism cues), role rules, text cleaning
  annotation_sampler.py, annotation_workbook.py, audit_builder.py   # validation samples and scoring
  make_*.py, fig_agent_suppression.py   # figure scripts
outputs/                        # derived data (see data dictionary below)
paper_figures/                  # figures exactly as included in the paper
```

## Reproducing the analysis

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_trf
cp .env.example .env            # add your OpenAlex email (and optional API key)

# 0. Data collection (OpenAlex snapshot changes over time, so counts will drift)
python fetch_openalex_dh.py --year-range 1990-2026 --out openalex_data/openalex_dh_works.json
python fetch_openalex_dh_fulltext.py --works openalex_data/openalex_dh_works.json --out-dir openalex_dh_fulltext

# 1. Run the notebooks in order (01 → 06f) from analysis_code/
# 2. Regenerate the figures
cd analysis_code && python make_corpus_overview.py   # etc., see table below
```

Our data were collected in November 2026, so the 2026 slice is partial. `outputs/fulltext/fulltext_manifest.csv` lists every work we tried to download, with its source URL and status. The full-text PDF mirror (`content.openalex.org`) needs an OpenAlex Premium key; without one the script falls back to open-access PDF links, which may give a smaller corpus.

If you only want to check the statistics or figures, skip step 0. Steps 04–06 and the figure scripts read the tables in `outputs/` directly. Cells that need the sentence text (e.g. KWIC printing) will not run without it.

### Figures in the paper

| Paper figure | Script (in `analysis_code/`) |
|---|---|
| `corpus_overview.png` | `make_corpus_overview.py` |
| `despecification.png` | `make_paper_figures.py` (fig02) |
| `grammatical_roles.png` | `make_role_distribution_umbrella.py` |
| `process_types.png` | `make_paper_figures.py` (fig16) |
| `anthropomorphism_categories.png` | `make_anthro_categories_by_year.py` |
| `collocations.png` | `make_collocation_scatter.py` |
| `agent_suppression.png` | `fig_agent_suppression.py` |

## Data dictionary

All tables are Parquet or CSV. Common keys: `work_id` (OpenAlex ID), `sent_id` (sentence index within a work), `publication_year`.

**`outputs/macro/`**
- `works_metadata.parquet`: all fetched OpenAlex works, with metadata but without abstracts.
- `works_metadata_filtered.parquet`: the 22,827 works of the macro-corpus.
- `term_counts_by_year.parquet`: number and share of works per year that mention each term group (umbrella AI, ML, DL, LLM, named models, …).

**`outputs/clean/index.parquet`**: the 1,894 retrieved full texts, with cleaning statistics, filter flags (`kept`, `in_ai_corpus`) and metadata.

**`outputs/parsed/`**
- `mentions.parquet`: every AI/human/company mention, with head and governor token features (lemma, POS, dependency, morphology). `mention_text` is the mention span only.
- `sentences.parquet`: per sentence, the counts of AI, human and company mentions.

**`outputs/analysis/`**: one table per analysis step. The prefix matches the notebook. For example:
- `04a_agency_per_mention`: grammatical role (active subject, passive subject, object, classifier, …).
- `04b_modality_per_clause` / `_per_sentence`: modality label, hedges, boosters.
- `04c_verbs_ranked`: verb lemmas with process class.
- `04d_nominalizations`: nominalisation pattern and agent presence.
- `04e_polarity_per_mention`: negation, projection and stance.
- `05c_frames_per_sentence`: metaphor-frame flags and cue words.
- `06a_*`: regression models and rates with CIs.
- `06b_*`: umbrella vs named-system contrasts.
- `06c_*`: collocations.
- `06d_passives`: passives and agentlessness.
- `06e_*`: provenance (venue, discipline) profiles.
- `06f_*`: anthropomorphism categories, cues and anchors.

**`outputs/annotation/`**
- `CODEBOOK.md`: annotation guidelines.
- `legend.csv`: allowed values.
- `keys/`: the automatic labels behind each blind validation sample, by `row_uid`.
- `audit/`: the verification audit, with auto label, `correct` (y/n/na) and short notes per row. Run `python analysis_code/audit_builder.py score` to compute precision with Wilson CIs. In `modality_clauses.csv`, `clause_id` = `<work_id>-<sent_id>-<clause index>`.

## Licence

- Code: MIT (see `LICENSE`).
- Derived data and annotations: CC BY 4.0 (see `DATA_LICENSE.md`).
- OpenAlex metadata: CC0, from [OpenAlex](https://openalex.org) (Priem, Piwowar & Orr 2022).

## Citation

Please cite the paper above. A `CITATION.cff` will be added once the paper is published.
