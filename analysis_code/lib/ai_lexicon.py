"""AI / GenAI / LLM / agent lexicon, expressed as REGEX patterns per group.

Every entry in ``AI_GROUPS`` is a regular expression (Python ``re`` syntax)
rather than a literal phrase. This lets one pattern cover a whole model
family with its numeric variants:

    r"\\b(?:Chat)?GPT(?:[-\\s]?[0-9]+(?:\\.[0-9]+)?[a-zA-Z]*)?\\b"
        matches: GPT, GPT-2, GPT-3, GPT-3.5, GPT-4, GPT-4o, ChatGPT
    r"\\b[A-Za-z]*BERTa?\\b"
        matches: BERT, RoBERTa, DistilBERT, ALBERT, DeBERTa
    r"\\bQwen[-\\s]?[0-9]*(?:[-\\s]?VL)?\\b"
        matches: Qwen, Qwen2, Qwen-2.5, Qwen2-VL

Group precedence (mutually-exclusive matching)
----------------------------------------------
Python's ``re`` alternation picks the FIRST listed alternative that matches
at a given start position. We therefore list groups in order from most
specific to most general. At the start of ``"AI agent"`` both the ``agent``
and the ``general_ai`` pattern can match; because ``agent`` comes first
in the combined regex, it wins and consumes the span, so ``general_ai``
never double-counts. Likewise ``"LLM agent"`` -> ``agent`` (beats ``llm``),
``"generative AI"`` -> ``generative_ai`` (beats ``general_ai``),
``"conversational AI"`` -> ``chatbot`` (beats ``general_ai``).

Group order (precedence, high -> low):
    agent, chatbot, named_image_generator,
    named_decoder_only, named_encoder_decoder,
    generative_ai, llm, general_ai

Public API
----------
* ``AI_GROUPS``                   -- dict[str, list[str]] of group -> regex patterns
* ``AI_TERMS``                    -- flat union of patterns (back-compat alias)
* ``compile_ai_regex()``          -- combined compiled regex (flat)
* ``compile_grouped_ai_regex()``  -- regex with named groups + name->category map
* ``classify_text(text)``         -- dict[group, count] of non-overlapping hits
* ``compile_comparator_regex()``  -- ``machine learning`` / ``deep learning`` / ...
* ``build_phrase_matcher(nlp, label="AI")``
        -- PhraseMatcher-compatible regex-backed matcher (one label)
* ``build_grouped_phrase_matchers(nlp)``
        -- dict[group, matcher], one per group, all regex-backed
"""
from __future__ import annotations

import regex as re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from spacy.language import Language
    from spacy.tokens import Doc


# ---------------------------------------------------------------------------
# Grouped regex patterns. ORDER MATTERS: most-specific groups FIRST.
# ---------------------------------------------------------------------------
AI_GROUPS: dict[str, list[str]] = {
    # ---- agent (must precede llm + general_ai so "LLM agent" / "AI agent" win) ----
    "agent": [
        r"\b(?:AI|LLM|autonomous)[\s-]agents?\b",
        r"\bagentic\s+(?:AI|systems?|behaviou?rs?)\b",
        r"\bmulti[\s-]?agent\s+(?:system|framework|LLM)s?\b",
    ],

    # ---- chatbot (must precede general_ai so "conversational AI" wins) ----
    "chatbot": [
        r"\bchat[\s-]?bots?\b",
        r"\bconversational\s+(?:AI|agents?)\b",
        r"\bdialogue\s+systems?\b",
    ],

    # ---- named image / multimodal generators ----
    "named_image_generator": [
        r"\bDALL[-\u00b7.\s]?E\b",          # DALL-E, DALL.E, DALL E, DALL\u00b7E
        r"\bDALLE\b",
        r"\bMidjourney\b",
        r"\bStable\s+Diffusion(?:\s+(?:XL|[0-9](?:\.[0-9])?))?\b",
        r"\bImagen\b",
        r"\bFlux(?:\.[0-9])?\b",
        r"\bSora\b",
    ],

    # ---- named decoder-only LLMs (model families incl. numeric variants) ----
    "named_decoder_only": [
        # OpenAI: GPT, GPT-2..5, GPT-3.5, GPT-4o, ChatGPT, ChatGPT-4
        r"\b(?:Chat)?GPT(?:[-\s]?[0-9]+(?:\.[0-9]+)?[a-zA-Z]*)?\b",
        # Anthropic
        r"\bClaude(?:[-\s]?[0-9](?:\.[0-9])?)?(?:\s+(?:Opus|Sonnet|Haiku))?\b",
        # Google
        r"\bGemini(?:[-\s]?[0-9](?:\.[0-9])?)?(?:\s+(?:Pro|Flash|Ultra|Nano))?\b",
        r"\bBard\b",
        r"\bPaLM[-\s]?[0-9]*\b",
        r"\bGemma[-\s]?[0-9]*\b",
        # Meta
        r"\b(?:LLa?MA|Llama)[-\s]?[0-9]*\b",
        # Mistral / Mixtral (incl. Mixtral-8x7B)
        r"\bMi[xs]tral(?:[-\s]?[0-9]+x?[0-9]*[Bb]?)?\b",
        # Microsoft / GitHub
        r"\bGitHub\s+Copilot\b",
        r"\bCopilot\b",
        r"\bPhi[-\s]?[0-9]*\b",
        # xAI
        r"\bGrok(?:[-\s]?[0-9]+)?\b",
        # Cohere
        r"\bCommand\s+(?:R(?:\+)?|A)\b",
        r"\bAya(?:[-\s]?(?:Expanse|[0-9]+[Bb]?))?\b",
        r"\bTiny[-\s]?Aya\b",
        # DeepSeek
        r"\bDeep[\s-]?Seek(?:[-\s]?(?:V[0-9]+|R[0-9]+|Coder|Chat))?\b",
        # TII
        r"\bFalcon(?:[-\s]?[0-9]+[Bb]?)?\b",
        # Z.AI
        r"\bGLM(?:[-\s]?[0-9]+(?:\.[0-9]+)?[Bb]?)?\b",
        # IBM
        r"\bGranite(?:[-\s]?[0-9]+[Bb]?)?\b",
        # Moonshot AI
        r"\bKimi(?:[-\s]?(?:K[0-9]+|VL|Chat))?\b",
        # MiniMax
        r"\bMiniMax(?:[-\s]?(?:Text|M[0-9]+))?\b",
        # NVIDIA
        r"\bNemotron(?:[-\s]?[0-9]+[Bb]?)?\b",
        # Alibaba
        r"\bQwen[-\s]?[0-9]*(?:\.[0-9]+)?(?:[-\s]?(?:VL|Coder|Chat))?\b",
        # 01.AI
        r"\bYi[-\s]?[0-9]+[Bb]?\b",
        # AllenAI
        r"\bOLMo[-\s]?[0-9]*\b",
        r"\bAllenAI\b",
    ],

    # ---- pre-LLM / encoder & encoder-decoder transformers ----
    "named_encoder_decoder": [
        # BERT family: BERT, RoBERTa, DistilBERT, ALBERT, DeBERTa,
        # XLM-RoBERTa, CamemBERT, FlauBERT, SciBERT, BioBERT, ...
        # NOTE: "BERT" is case-sensitive (via (?-i:BERT)) to avoid matching
        # the proper names "Robert", "Hubert", "Lambert", "Norbert", etc.
        r"\b[A-Za-z]*(?-i:BERT)a?\b",
        r"\bELECTRA\b",
        r"\bXLNet\b",
        r"\bXLM[-\s]?R\b",
        # T5 family: T5, mT5, FLAN-T5, ByT5
        r"\b(?:m|FLAN[-\s]?|By)?T5\b",
        # BART family: BART, mBART
        r"\bm?BART\b",
        r"\bPegasus\b",
    ],

    # ---- generative AI (must precede general_ai so "generative AI" wins) ----
    "generative_ai": [
        r"\bgenerative\s+(?:AI|artificial\s+intelligence|models?)\b",
        r"\bGen[-\s]?AI\b",
    ],

    # ---- LLM family terms ----
    "llm": [
        r"\bLLMs?\b",
        r"\bSLMs?\b",
        r"\bMLLMs?\b",
        r"\bVLMs?\b",
        r"\b(?:large|small)\s+language\s+models?\b",
        r"\blanguage\s+models?\b",
        r"\bmultimodal\s+(?:large\s+)?language\s+models?\b",
        r"\bvision[-\s]language\s+models?\b",
    ],

    # ---- general AI (catches only what nothing more specific consumed) ----
    "general_ai": [
        r"\bAI\b",
        r"\bA\.\s?I\.",
        r"\bartificial\s+(?:general\s+)?intelligence\b",
        r"\bAGI\b",
        r"\bfoundation\s+models?\b",
    ],
}

# Flat union (back-compat).
AI_TERMS: list[str] = [p for pats in AI_GROUPS.values() for p in pats]

COMPARATOR_TERMS: list[str] = [
    r"\bmachine\s+learning\b",
    r"\bdeep\s+learning\b",
    r"\bneural\s+networks?\b",
    r"\bML\b",
    r"\btransformers?\b",
]


# ---------------------------------------------------------------------------
# Regex compilation
# ---------------------------------------------------------------------------
def _ordered_group_pattern_pairs() -> list[tuple[str, str]]:
    """All (group, pattern) pairs in declared precedence order."""
    return [(g, p) for g, pats in AI_GROUPS.items() for p in pats]


def compile_ai_regex() -> re.Pattern:
    """Combined flat regex over every (group, pattern) entry."""
    pairs = _ordered_group_pattern_pairs()
    pat = "|".join(p for _, p in pairs)
    return re.compile(pat, flags=re.IGNORECASE | re.UNICODE)


def compile_grouped_ai_regex() -> tuple[re.Pattern, dict[str, str]]:
    """Named-group combined regex + ``name -> category`` map.

    Each entry becomes ``(?P<g{idx}>...)``; ``m.lastgroup`` on a hit
    resolves to that unique name, which the returned dict maps back
    to the group.
    """
    pairs = _ordered_group_pattern_pairs()
    chunks: list[str] = []
    name_to_category: dict[str, str] = {}
    for idx, (group, pattern) in enumerate(pairs):
        name = f"g{idx}"
        name_to_category[name] = group
        chunks.append(f"(?P<{name}>{pattern})")
    pat = "|".join(chunks)
    return re.compile(pat, flags=re.IGNORECASE | re.UNICODE), name_to_category


_GROUPED_RE_CACHE: tuple[re.Pattern, dict[str, str]] | None = None


def _grouped_re_cache() -> tuple[re.Pattern, dict[str, str]]:
    global _GROUPED_RE_CACHE
    if _GROUPED_RE_CACHE is None:
        _GROUPED_RE_CACHE = compile_grouped_ai_regex()
    return _GROUPED_RE_CACHE


def classify_text(text: str | None) -> dict[str, int]:
    """Non-overlapping per-group hit counts (every group always present)."""
    out = {g: 0 for g in AI_GROUPS}
    if not text:
        return out
    pat, name_to_cat = _grouped_re_cache()
    for m in pat.finditer(text):
        out[name_to_cat[m.lastgroup]] += 1
    return out


def compile_comparator_regex() -> re.Pattern:
    pat = "|".join(COMPARATOR_TERMS)
    return re.compile(pat, flags=re.IGNORECASE | re.UNICODE)


# ---------------------------------------------------------------------------
# spaCy-compatible regex matchers
# ---------------------------------------------------------------------------
class RegexSpanMatcher:
    """Drop-in replacement for ``spacy.matcher.PhraseMatcher``.

    Calling an instance with a ``Doc`` returns a list of
    ``(match_id, start, end)`` token-index tuples, exactly like spaCy's
    PhraseMatcher, so existing callers do not have to change.

    Internally it runs a compiled regex on ``doc.text`` and projects each
    character span back to token indices via ``doc.char_span``.
    """

    def __init__(
        self,
        vocab,
        label: str,
        pattern: re.Pattern,
        name_to_category: dict[str, str] | None = None,
    ):
        self.vocab = vocab
        self.label = label
        self.label_id = vocab.strings.add(label)
        self.pattern = pattern
        self.name_to_category = name_to_category or {}
        # Pre-register category labels so callers can use vocab.strings later.
        for g in AI_GROUPS:
            vocab.strings.add(g)

    def __call__(self, doc: "Doc"):
        out: list[tuple[int, int, int]] = []
        for m in self.pattern.finditer(doc.text):
            span = doc.char_span(m.start(), m.end(), alignment_mode="expand")
            if span is None:
                continue
            out.append((self.label_id, span.start, span.end))
        return out

    def find_grouped(self, doc: "Doc") -> list[tuple[str, int, int]]:
        """Like ``__call__`` but yields ``(category, start, end)``."""
        out: list[tuple[str, int, int]] = []
        if not self.name_to_category:
            return out
        for m in self.pattern.finditer(doc.text):
            span = doc.char_span(m.start(), m.end(), alignment_mode="expand")
            if span is None:
                continue
            cat = self.name_to_category.get(m.lastgroup)
            if cat is None:
                continue
            out.append((cat, span.start, span.end))
        return out


def build_phrase_matcher(nlp: "Language", label: str = "AI") -> RegexSpanMatcher:
    """Regex-backed, PhraseMatcher-compatible matcher over the full lexicon."""
    pat, name_to_cat = _grouped_re_cache()
    return RegexSpanMatcher(nlp.vocab, label=label, pattern=pat,
                            name_to_category=name_to_cat)


def build_grouped_phrase_matchers(nlp: "Language") -> dict[str, RegexSpanMatcher]:
    """One regex-backed matcher per group, each labelled with its group name."""
    matchers: dict[str, RegexSpanMatcher] = {}
    for group, patterns in AI_GROUPS.items():
        combined = "|".join(patterns)
        compiled = re.compile(combined, flags=re.IGNORECASE | re.UNICODE)
        matchers[group] = RegexSpanMatcher(nlp.vocab, label=group,
                                           pattern=compiled)
    return matchers
