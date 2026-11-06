"""spaCy loader + DocBin save/load helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

_NLP = None


def load_nlp(model: str = "en_core_web_trf", disable: tuple[str, ...] = ()):
    """Lazy-load and cache a spaCy pipeline."""
    global _NLP
    if _NLP is None or _NLP.meta.get("name") != model.split("_", 1)[-1]:
        import spacy

        _NLP = spacy.load(model, disable=list(disable))
    return _NLP


def save_docbin(docs: Iterable, path: Path) -> None:
    from spacy.tokens import DocBin

    # store_user_data=False: with trf docs, user_data holds huge tensors.
    db = DocBin()
    for d in docs:
        db.add(d)
    path.parent.mkdir(parents=True, exist_ok=True)
    db.to_disk(path)


def load_docbin(path: Path, nlp):
    from spacy.tokens import DocBin

    db = DocBin().from_disk(path)
    return list(db.get_docs(nlp.vocab))
