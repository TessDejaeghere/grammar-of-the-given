"""Single source of truth for the syntactic -> SFL participant role mapping.

Used by 04a (AI mentions), 05d (AI vs HUMAN vs COMPANY) and 06a (statistics) so
the three cannot drift apart.
"""
from __future__ import annotations

PASSIVE_SUBJ_DEPS = {"nsubjpass", "nsubj:pass", "csubjpass"}
ACTIVE_SUBJ_DEPS = {"nsubj"}
OBJECT_DEPS = {"dobj", "obj"}
MOD_DEPS = {"compound", "amod", "nmod", "poss", "appos"}
# Relational complements: 'AI is an umbrella term' -- Attribute/Value.
ATTR_DEPS = {"attr", "oprd", "acomp"}

ROLES_ORDER = [
    "active_subject", "passive_subject", "direct_object", "attribute",
    "oblique_agent", "prepositional", "nominal_modifier", "other",
]


def classify_role(row) -> str:
    dep = row["head_dep"]
    if dep in PASSIVE_SUBJ_DEPS:
        return "passive_subject"
    if dep in ACTIVE_SUBJ_DEPS:
        return "active_subject"
    if dep in OBJECT_DEPS:
        return "direct_object"
    if dep in ATTR_DEPS:
        return "attribute"
    if dep == "pobj":
        # The governor of a pobj is the preposition itself; spaCy tags the 'by'
        # of a passive with dep 'agent'. (Testing gov_tag == 'VBN' never fires.)
        return "oblique_agent" if row["governor_dep"] == "agent" else "prepositional"
    if dep in MOD_DEPS:
        return "nominal_modifier"
    return "other"
