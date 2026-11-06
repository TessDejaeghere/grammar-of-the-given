"""Dependency-aware helpers shared by modality, nominalisation and valency."""
from __future__ import annotations

from collections.abc import Iterable

EMBEDDED_DEPS = {"ccomp", "xcomp", "advcl", "acl", "relcl", "csubj", "pcomp"}
PHASE_LEMMAS = {"begin", "start", "continue", "keep", "stop", "cease", "finish"}
PURPOSIVE_LEMMAS = {"aim", "seek", "try", "attempt", "intend", "plan", "hope"}


def clause_tokens(head, include_head: bool = True) -> list:
    """Return the local predicate subtree, excluding embedded clauses."""
    blocked = {
        child.i
        for child in head.children
        if child.dep_ in EMBEDDED_DEPS
        for token in child.subtree
    }
    tokens = [token for token in head.subtree if token.i not in blocked]
    if include_head and head not in tokens:
        tokens.append(head)
    return sorted(tokens, key=lambda token: token.i)


def verbal_group(head) -> list:
    """Return a verb and its directly attached auxiliaries and particles."""
    return [head] + [child for child in head.children if child.pos_ == "AUX" or child.dep_ == "prt"]


def predicate_lemma(head) -> str:
    """Combine phrasal particles into a single predicate key."""
    particles = sorted(child.lemma_.lower() for child in head.children if child.dep_ == "prt")
    return "_".join([head.lemma_.lower(), *particles]) if particles else head.lemma_.lower()


def xcomp_verb(head):
    """Return a verbal complement for catenative/phase constructions."""
    for child in head.children:
        if child.dep_ in {"xcomp", "ccomp"} and child.pos_ in {"VERB", "AUX"}:
            return child
    return None


def construction(head) -> str:
    key = head.lemma_.lower()
    if any(child.dep_ == "prt" for child in head.children):
        return "phrasal"
    if xcomp_verb(head) is not None:
        if key in PHASE_LEMMAS:
            return "phase"
        if key in PURPOSIVE_LEMMAS:
            return "purposive"
        return "catenative"
    return "simple"


def marker_scope(head, token) -> bool:
    """Whether a token belongs to this predicate's local clause."""
    return token.i in {item.i for item in clause_tokens(head)}


def first_child(head, deps: Iterable[str]):
    deps = set(deps)
    return next((child for child in head.children if child.dep_ in deps), None)
