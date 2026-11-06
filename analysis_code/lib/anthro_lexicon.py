"""Anthropomorphization-by-description lexicon.

Operationalises the eight-category taxonomy of Inie, Zukerman and Bender (2026),
"De-anthropomorphizing 'AI': From wishful mnemonics to accurate nomenclature",
First Monday 31(2). Categories and seed items follow their Table 2 and the
worked examples in their section 4.2; the split into VERB / NOUN / MODIFIER
members is ours, added so that each category can be dependency-anchored rather
than matched anywhere in the sentence.

Two of their annotation decisions are encoded here explicitly because they drive
the headline numbers:

* ENTRENCHED holds the terms they insist on counting (their 4.2.1 and 5.2) even
  though the terms are established technical vocabulary -- 'artificial
  intelligence' itself, 'machine learning', 'neural network', 'training data'.
  Reporting with and without this set is the only honest way to present the
  result, since it alone decides whether ~5% or ~50% of the corpus counts as
  anthropomorphizing.
* Their guideline that a negated property still counts (their 4.1: 'failed to
  grasp' presupposes that grasping was expected) is NOT applied here. It is
  applied downstream in 06f, which reports every category twice -- presupposed
  (their rule) and asserted (ours) -- using the polarity coding from 04e.
"""
from __future__ import annotations

# --- verbs: anchored when they govern an AI mention as subject ---------------

COGNIZER_VERBS = {
    "think", "believe", "reason", "learn", "understand", "know", "comprehend",
    "infer", "perceive", "recognize", "recognise", "interpret", "realise",
    "realize", "remember", "memorise", "memorize", "imagine", "conclude",
    "assume", "grasp", "confuse", "misunderstand", "consider",
    "train",  # 'the model is trained' -- Cognizer per their 4.2.1 discussion
}
# 'reflect' is deliberately absent: their 4.2.2 holds that a model that
# *reflects* bias is not anthropomorphized, only one that *has* it.
COMMUNICATION_VERBS = {
    "answer", "ask", "tell", "say", "explain", "suggest", "respond", "reply",
    "discuss", "converse", "chat", "talk", "speak", "describe", "report",
    "communicate", "instruct", "prompt", "state", "mention", "argue", "claim",
    "advise", "recommend", "inform", "narrate", "declare",
}
AGENT_VERBS = {
    "help", "assist", "aid", "support", "facilitate", "enable", "leverage",
    "curate", "impersonate", "allow", "decide", "act", "operate", "choose",
    "select", "manage", "handle", "join", "collaborate", "cooperate",
    "intervene", "attempt", "try", "seek", "aim", "refuse", "avoid",
}
EMOTION_VERBS = {
    "feel", "empathise", "empathize", "care", "struggle", "suffer", "enjoy",
    "prefer", "desire", "want", "wish", "hope", "fear", "worry", "love",
    "hate", "like", "coax", "chide", "bond",
}
BIOLOGICAL_VERBS = {
    "see", "hear", "watch", "listen", "ingest", "digest", "consume", "feed",
    "sleep", "wake", "breathe", "grow", "mature", "evolve", "embody",
    "stumble", "hallucinate", "dream", "absorb",
}

# --- nouns -----------------------------------------------------------------

COGNIZER_NOUNS = {
    "intelligence", "cognition", "reasoning", "understanding", "knowledge",
    "thought", "belief", "comprehension", "inference", "perception",
    "mind", "intellect", "whiz",
}
PRODUCTS_OF_COGNITION_NOUNS = {
    "skill", "skills", "capability", "capabilities", "capacity", "capacities",
    "ability", "abilities", "competence", "competency", "expertise",
    "mistake", "mistakes", "judgment", "judgement", "intuition", "creativity",
    "insight", "insights", "bias", "biases", "opinion", "opinions",
    "preference", "preferences", "aesthetics", "taste", "foresight",
}
COMMUNICATION_NOUNS = {
    "answer", "response", "reply", "explanation", "suggestion", "advice",
    "conversation", "dialogue", "chat", "chatbot", "prompt", "instruction",
    "instructions", "recommendation", "narrative", "statement", "claim",
}
EMOTION_NOUNS = {
    "emotion", "emotions", "feeling", "feelings", "empathy", "bond",
    "friend", "friendship", "companion", "companionship", "fondness",
    "desire", "anxiety", "confidence",
}
HUMAN_ROLE_NOUNS = {
    "tutor", "assistant", "teammate", "editor", "therapist", "co-creator",
    "cocreator", "co-author", "coauthor", "collaborator", "partner", "mentor",
    "advisor", "adviser", "critic", "judge", "author", "writer", "researcher",
    "scholar", "expert", "apprentice", "companion", "colleague", "copilot",
    "co-pilot", "curator", "translator", "interpreter", "reviewer", "teacher",
    "student", "coach", "guide", "assistant", "helper", "secretary",
    "librarian", "historian", "philologist", "artist", "poet", "novelist",
}
BIOLOGICAL_NOUNS = {
    "neuron", "neurons", "brain", "brains", "memory", "attention", "eye",
    "eyes", "hand", "hands", "body", "hallucination", "hallucinations",
    "ancestor", "descendant",
}
# Excluded as technical homonyms in this register, after inspecting the hits:
# 'generation' (text generation), 'family' (model family), 'vision' (computer
# vision). Inie et al.'s corpus of news and CHI papers could not surface these;
# an NLP-heavy corpus can, and counting them would inflate the category.
TECHNICAL_HOMONYMS = {"generation", "family", "vision"}

# --- modifiers: adjectives / adverbs that carry the attribution -------------

COGNIZER_MODIFIERS = {"intelligent", "smart", "knowledgeable", "thoughtful",
                      "aware", "conscious", "sentient", "rational"}
AGENT_MODIFIERS = {"autonomous", "autonomously", "independent", "independently",
                   "intentional", "intentionally", "deliberate", "deliberately",
                   "proactive", "proactively", "agentic"}
EMOTION_MODIFIERS = {"empathetic", "empathic", "caring", "friendly", "warm",
                     "anxious", "confident", "enthusiastic"}
BIOLOGICAL_MODIFIERS = {"neural", "cognitive", "tired", "tireless", "alive",
                        "born", "blind", "sensory", "organic"}

# Animate pronouns; Inie et al. 4.2.7. 'it' is excluded -- it is the
# non-anthropomorphizing choice.
ANIMATE_PRONOUNS = {"he", "him", "his", "she", "her", "hers", "who", "whom",
                    "whose"}

CATEGORIES = {
    "Cognizer": {
        "verbs": COGNIZER_VERBS, "nouns": COGNIZER_NOUNS,
        "modifiers": COGNIZER_MODIFIERS,
    },
    "Products_of_cognition": {
        "verbs": set(), "nouns": PRODUCTS_OF_COGNITION_NOUNS, "modifiers": set(),
    },
    "Emotion": {
        "verbs": EMOTION_VERBS, "nouns": EMOTION_NOUNS,
        "modifiers": EMOTION_MODIFIERS,
    },
    "Communication": {
        "verbs": COMMUNICATION_VERBS, "nouns": COMMUNICATION_NOUNS,
        "modifiers": set(),
    },
    "Agent": {
        "verbs": AGENT_VERBS, "nouns": set(), "modifiers": AGENT_MODIFIERS,
    },
    "Human_role_analogy": {
        "verbs": set(), "nouns": HUMAN_ROLE_NOUNS, "modifiers": set(),
    },
    "Names_and_pronouns": {
        "verbs": set(), "nouns": set(), "modifiers": set(),
        "pronouns": ANIMATE_PRONOUNS,
    },
    "Biological_metaphors": {
        "verbs": BIOLOGICAL_VERBS, "nouns": BIOLOGICAL_NOUNS,
        "modifiers": BIOLOGICAL_MODIFIERS,
    },
}

# Established terms of art that Inie et al. count anyway (their 5.2). Reported
# separately so the headline rate can be given with and without them.
ENTRENCHED = {
    "artificial intelligence", "ai", "agi", "machine learning", "deep learning",
    "neural network", "neural networks", "neural", "neuron", "neurons",
    "intelligence", "intelligent", "learn", "learning", "train", "training",
    "attention", "memory", "hallucination", "hallucinations", "hallucinate",
    "agent", "agents", "prompt", "recognition", "generation",
}

# Inie et al. Table 3: functionality-first replacements.
DE_ANTHRO_SUGGESTIONS = {
    "artificial intelligence": "probabilistic automation",
    "hybrid intelligence": "augmented human intelligence",
    "image recognition": "image labeling",
    "speech recognition": "automatic transcription",
    "the model shows bias": "the model reflects bias",
    "model mistakes": "model errors",
    "good at": "good for",
    "prompt": "text input",
    "answer": "output",
    "chatbot": "conversation simulator",
    "conversational agent": "conversation simulator",
    "AI assisted students": "the students used AI",
    "revealing the solution": "displaying the solution",
    "neural network": "weighted network",
    "the model consumes data": "data is used in setting model weights",
}


def all_members(category: str) -> set[str]:
    spec = CATEGORIES[category]
    out: set[str] = set()
    for key in ("verbs", "nouns", "modifiers", "pronouns"):
        out |= spec.get(key, set())
    return out
