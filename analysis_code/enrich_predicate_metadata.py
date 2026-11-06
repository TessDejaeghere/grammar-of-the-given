"""Add governing particle/predicate fields to an existing mentions.parquet."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import pandas as pd
import spacy
from spacy.tokens import DocBin

ROOT = Path(__file__).resolve().parent.parent
PARSED = ROOT / "outputs" / "parsed"
DOCS = PARSED / "docs"
MENTIONS = PARSED / "mentions.parquet"


def main() -> None:
    nlp = spacy.blank("en")
    mentions = pd.read_parquet(MENTIONS)
    mentions["governor_particle"] = ""
    mentions["governor_predicate"] = mentions["governor_lemma"].str.lower()
    mentions["governor_construction"] = "simple"
    mentions["governor_complement_lemma"] = ""

    by_work = defaultdict(list)
    for index, row in mentions.iterrows():
        by_work[row["work_id"]].append((index, row))

    for work_id, rows in by_work.items():
        path = DOCS / f"{work_id}.spacy"
        if not path.exists():
            continue
        docs = list(DocBin().from_disk(path).get_docs(nlp.vocab))
        by_sent = defaultdict(list)
        for index, row in rows:
            by_sent[row["sent_id"]].append((index, row))
        for sent_id, sent_rows in by_sent.items():
            if sent_id >= len(docs):
                continue
            doc = docs[sent_id]
            occurrences = defaultdict(int)
            for index, row in sent_rows:
                text = str(row["mention_text"])
                key = text.lower()
                matches = list(re.finditer(re.escape(text), doc.text, flags=re.IGNORECASE))
                occurrence = occurrences[key]
                occurrences[key] += 1
                if occurrence >= len(matches):
                    continue
                match = matches[occurrence]
                span = doc.char_span(match.start(), match.end(), alignment_mode="expand")
                if span is None:
                    continue
                governor = span.root.head
                particles = [child.lemma_.lower() for child in governor.children
                             if child.dep_ == "prt"]
                complement = next((child for child in governor.children
                                   if child.dep_ in {"xcomp", "ccomp"}
                                   and child.pos_ in {"VERB", "AUX"}), None)
                mentions.at[index, "governor_particle"] = "|".join(particles)
                mentions.at[index, "governor_predicate"] = "_".join(
                    [governor.lemma_.lower(), *particles])
                mentions.at[index, "governor_construction"] = (
                    "phrasal" if particles else
                    "catenative" if complement is not None else "simple"
                )
                mentions.at[index, "governor_complement_lemma"] = (
                    complement.lemma_.lower() if complement is not None else "")

    mentions.to_parquet(MENTIONS, index=False)
    print("saved", len(mentions), "mentions with predicate metadata")
    print("phrasal:", (mentions["governor_construction"] == "phrasal").sum())
    print("catenative:", (mentions["governor_construction"] == "catenative").sum())
    print(mentions.loc[mentions["governor_construction"] == "phrasal",
                       ["governor_lemma", "governor_particle", "governor_predicate"]]
          .drop_duplicates().head(20).to_string(index=False))


if __name__ == "__main__":
    main()
