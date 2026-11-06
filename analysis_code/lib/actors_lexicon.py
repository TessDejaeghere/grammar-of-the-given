"""Human-actor and AI-company lexicons for nominalization & AI-vs-human comparison."""
from __future__ import annotations

HUMAN_TERMS: list[str] = [
    "we", "us", "our",
    "the author", "the authors",
    "researcher", "researchers",
    "scholar", "scholars",
    "humanist", "humanists",
    "scientist", "scientists",
    "user", "users",
    "human", "humans",
    "people",
]

COMPANY_TERMS: list[str] = [
    "OpenAI",
    "Anthropic",
    "Google", "Google DeepMind", "DeepMind",
    "Meta", "Meta AI", "Facebook",
    "Microsoft",
    "Amazon",
    "Mistral AI",
    "xAI",
    "HuggingFace",
    "Nvidia",
    "AI company", "AI companies",
    "tech company", "tech companies",
    "developer", "developers",
    "vendor", "vendors",
]


def build_actor_matchers(nlp):
    """Return two PhraseMatchers: (human, company), both with LOWER attribute."""
    from spacy.matcher import PhraseMatcher

    human = PhraseMatcher(nlp.vocab, attr="LOWER")
    human.add("HUMAN", [nlp.make_doc(t) for t in HUMAN_TERMS])

    company = PhraseMatcher(nlp.vocab, attr="LOWER")
    company.add("COMPANY", [nlp.make_doc(t) for t in COMPANY_TERMS])

    return human, company

# ---------------------------------------------------------------------------
# Named-company filter (used for the recoverability measure in 06d, section 5)
#
# The COMPANY PhraseMatcher also catches generic terms ("developers", "vendors",
# "tech companies") and strings that do not refer to a company as an actor:
# "Google Books/Scholar/Colab", "meta" in "metadata"/"meta-analysis",
# "Microsoft Word/Excel", "NVIDIA A100" (hardware), "XAI" (explainable AI),
# "Amazon Kindle". filter_named_companies() keeps only companies named as such.
# ---------------------------------------------------------------------------
import re as _re

GENERIC_COMPANY_TERMS = {"developer", "developers", "vendor", "vendors",
                         "tech company", "tech companies", "ai company", "ai companies"}
_GOOGLE_TOOLS = {"books", "scholar", "colab", "arts", "search", "docs", "drive", "sheets",
                 "maps", "ngram", "forms", "earth", "trends", "images", "news", "sites",
                 "classroom", "street"}
_MICROSOFT_TOOLS = {"excel", "office", "teams", "word", "powerpoint", "books", "outlook",
                    "onedrive", "coco"}
_NVIDIA_HARDWARE = _re.compile(
    r"^(a\d+|v\d+|h\d+|l\d+|t\d+|geforce|gforce|rtx|dgx|tesla|quadro|titan|cuvs|gtx)", _re.I)


def is_named_company(mention_text: str, sent_text: str) -> bool:
    """True if a COMPANY match names a company as an actor (not a generic term or a tool)."""
    low = mention_text.lower()
    if low in GENERIC_COMPANY_TERMS:
        return False
    sent = str(sent_text)
    m = _re.search(_re.escape(mention_text) + r"[\s\-]*([^\s,.;:()]*)", sent)
    nxt = m.group(1) if m else ""
    hyphen = bool(_re.search(_re.escape(mention_text) + r"\s*-", sent))
    if low == "meta" and (mention_text != "Meta" or hyphen or _re.match(r"^[a-z]", nxt)):
        return False
    if mention_text == "XAI":
        return False
    if low == "google" and nxt.lower() in _GOOGLE_TOOLS:
        return False
    if low == "microsoft" and nxt.lower() in _MICROSOFT_TOOLS:
        return False
    if low == "amazon" and nxt.lower() in {"kindle", "river", "rainforest"}:
        return False
    if low == "nvidia" and _NVIDIA_HARDWARE.match(nxt):
        return False
    return True


def filter_named_companies(mentions):
    """Keep only COMPANY rows of a mentions DataFrame that name a company as an actor."""
    keep = mentions.apply(lambda r: is_named_company(r["mention_text"], r["sent_text"]), axis=1)
    return mentions[keep]
