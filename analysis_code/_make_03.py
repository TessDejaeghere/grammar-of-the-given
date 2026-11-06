"""Generate 03_parse_and_extract.ipynb (fast version)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _nbbuild import write

HERE = Path(__file__).parent

cells = [
    ("md", """# 03 — Parse the AI corpus & extract actor mentions

For ~316 documents in `ai_corpus`:
1. Sentence-split with a lightweight blank-English pipeline.
2. Run the AI/HUMAN/COMPANY `PhraseMatcher`s on the **full doc** (one pass per matcher per doc).
3. For each match, resolve its containing sentence; keep only those sentences.
4. Re-parse the kept sentences with `en_core_web_trf` to get full dependency parses.
5. Persist `DocBin` per work + flat `mentions.parquet` / `sentences.parquet`.
"""),
    ("code", """from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path.cwd()
while ROOT.name and not (ROOT / "analysis_code").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "analysis_code"))

import pandas as pd
import spacy
from spacy.tokens import DocBin
from tqdm.auto import tqdm

from lib import OUT_CLEAN, OUT_PARSED
from lib.ai_lexicon import build_phrase_matcher
from lib.actors_lexicon import build_actor_matchers
from lib.spacy_io import load_nlp

PARSED_DOCS_DIR = OUT_PARSED / "docs"
PARSED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
print("clean:", OUT_CLEAN, "  parsed:", OUT_PARSED)
"""),
    ("md", "## 1. Load index"),
    ("code", """index = pd.read_parquet(OUT_CLEAN / "index.parquet")
ai_ids = index.loc[index["in_ai_corpus"], "id"].tolist()
print("ai corpus size:", len(ai_ids))
"""),
    ("md", "## 2. Collect sentences-of-interest with the blank pipeline"),
    ("code", """splitter = spacy.blank("en")
splitter.add_pipe("sentencizer")
ai_m = build_phrase_matcher(splitter, label="AI")
hu_m, co_m = build_actor_matchers(splitter)


def _sentences_of_interest(text: str) -> list[str]:
    doc = splitter(text)
    keep_starts: set[int] = set()
    for matcher in (ai_m, hu_m, co_m):
        for _, start, _end in matcher(doc):
            keep_starts.add(doc[start].sent.start)
    by_start = {s.start: s for s in doc.sents}
    out, seen = [], set()
    for st in sorted(keep_starts):
        if st in by_start:
            txt = by_start[st].text.strip()
            if txt and txt not in seen:
                seen.add(txt)
                out.append(txt)
    return out


records = []
for wid in tqdm(ai_ids, desc="sent-split+match"):
    text = (OUT_CLEAN / f"{wid}.txt").read_text(encoding="utf-8")
    for s in _sentences_of_interest(text):
        records.append({"work_id": wid, "text": s})
sent_df = pd.DataFrame(records)
print("sentences of interest:", len(sent_df))
sent_df.groupby("work_id").size().describe()
"""),
    ("md", "## 3. Parse with `en_core_web_trf`"),
    ("code", """nlp = load_nlp("en_core_web_trf")
nlp.max_length = max(nlp.max_length, 2_000_000)
ai_pm = build_phrase_matcher(nlp, label="AI")
hu_pm, co_pm = build_actor_matchers(nlp)
print("pipeline:", nlp.pipe_names)
"""),
    ("code", """from collections import defaultdict

mention_rows = []
parsed_sent_rows = []
# store_user_data=False: with trf, user_data holds the wordpiece tensors
# (~100x the size of the annotations) and nothing downstream needs them.
docbins_by_work: dict[str, DocBin] = defaultdict(DocBin)

texts = sent_df["text"].tolist()
work_ids = sent_df["work_id"].tolist()

BATCH = 32
for sent_idx, (doc, work_id) in enumerate(tqdm(
    zip(nlp.pipe(texts, batch_size=BATCH), work_ids),
    total=len(texts), desc="parse"
)):
    docbins_by_work[work_id].add(doc)

    ai_hits = ai_pm(doc)
    hu_hits = hu_pm(doc)
    co_hits = co_pm(doc)

    parsed_sent_rows.append({
        "work_id": work_id, "sent_id": sent_idx, "text": doc.text,
        "n_ai": len(ai_hits), "n_human": len(hu_hits), "n_company": len(co_hits),
    })

    for label, hits in (("AI", ai_hits), ("HUMAN", hu_hits), ("COMPANY", co_hits)):
        for _, start, end in hits:
            span = doc[start:end]
            head = span.root
            governor = head.head
            particles = [t.lemma_.lower() for t in governor.children if t.dep_ == "prt"]
            xcomp = next((t for t in governor.children
                          if t.dep_ in {"xcomp", "ccomp"}
                          and t.pos_ in {"VERB", "AUX"}), None)
            mention_rows.append({
                "work_id": work_id, "sent_id": sent_idx, "category": label,
                "mention_text": span.text,
                "mention_lemma": " ".join(t.lemma_.lower() for t in span),
                "head_text": head.text, "head_lemma": head.lemma_.lower(),
                "head_pos": head.pos_, "head_tag": head.tag_,
                "head_dep": head.dep_, "head_morph": str(head.morph),
                "governor_text": governor.text, "governor_lemma": governor.lemma_.lower(),
                "governor_particle": "|".join(particles),
                "governor_predicate": "_".join([governor.lemma_.lower(), *particles]),
                "governor_construction": (
                    "phrasal" if particles else
                    "catenative" if xcomp is not None else "simple"
                ),
                "governor_complement_lemma": xcomp.lemma_.lower() if xcomp is not None else "",
                "governor_pos": governor.pos_, "governor_tag": governor.tag_,
                "governor_dep": governor.dep_,
                "governor_is_root": governor.dep_ == "ROOT",
                "sent_text": doc.text,
            })

print("mentions extracted:", len(mention_rows))
"""),
    ("md", "## 4. Persist DocBins + tables"),
    ("code", """for wid, db in tqdm(docbins_by_work.items(), desc="save docbins"):
    db.to_disk(PARSED_DOCS_DIR / f"{wid}.spacy")

ment_df = pd.DataFrame(mention_rows)
sent_parsed_df = pd.DataFrame(parsed_sent_rows)
ment_df.to_parquet(OUT_PARSED / "mentions.parquet", index=False)
sent_parsed_df.to_parquet(OUT_PARSED / "sentences.parquet", index=False)
print("mentions:", ment_df.shape, " sentences:", sent_parsed_df.shape)
ment_df.head()
"""),
    ("code", """ai_only = ment_df[ment_df["category"] == "AI"]
print("AI mentions:", len(ai_only))
print("\\ntop verbal governors:")
print(
    ai_only[ai_only["governor_pos"].isin(["VERB", "AUX"])]["governor_lemma"]
    .value_counts().head(20)
)
print("\\ntop head dep labels:")
print(ai_only["head_dep"].value_counts().head(15))
"""),
]

write(HERE / "03_parse_and_extract.ipynb", cells)
