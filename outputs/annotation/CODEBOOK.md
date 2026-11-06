# Annotation codebook

Companion to `annotation_workbook.xlsx`. Every rule here matches the operational definitions in
the paper (§2.2–2.8) and the lexicons in `analysis_code/lib/`, so your codings are directly
comparable with the automatic labels.

You are coding **five separate tasks**. They are independent: a sentence can be `bare` in the
modality task and `Cognizer` in the anthropomorphization task with no contradiction.

| Sheet | Rows | You decide | Paper |
|---|---|---|---|
| 04c verb types | 315 | which process type a verb realises | §2.4 |
| 04b modality | 400 | how strongly the claim is committed to | §2.3 |
| 04e polarity | 400 | is it negated? is it attributed? | §2.6 |
| 06f anthropomorphization | 380 | which human attributions are made | §2.8 |
| 04d nominalisation | 250 | is a human/corporate agent recoverable? | §2.5 |

Suggested order: **04c first** (315 rows fixes 92% of verb tokens — the best value per hour), then
04b, 04e, 06f, 04d.

---

## 0. General principles

These apply to every task.

**0.1 Judge the AI mention, not the sentence.** Every row contains at least one AI-lexicon match
(*AI*, *LLM*, *GPT-4o*, *BERT*, *Stable Diffusion*, …). Code what the sentence does with **that**
mention. If a sentence contains both an AI mention and a human actor, the human is not your object.

**0.2 Take the reader's perspective.** If a sentence has more than one plausible reading and one of
them is the marked one, code the marked one — even if you suspect the author did not intend it.
This follows Inie et al. (2026, §4.1) and keeps coding reproducible.

**0.3 Be conservative when genuinely unsure.** If you cannot decide after re-reading, choose the
unmarked option (`bare`, `FALSE`, `OTHER`, no category) rather than guessing the interesting one.
Systematic over-coding of the interesting category is the main way this kind of study goes wrong.

**0.4 Unusable rows → `NA`.** The corpus is PDF-derived and some rows are furniture: reference
lists, figure captions, table fragments, running heads, Greek or Chinese text, sentences truncated
mid-clause. Put **`NA`** in the label column and write `noise` in notes. These rows are excluded
from the precision calculation, so `NA` costs nothing — but do not use it to avoid a hard judgement
on a real sentence.

Examples of genuine `NA` rows in these samples:

> "2023) uses a separate language model to project labels by generating candidate spans, while CLaP
> (Parekh et al.,"  ← truncated citation

> "Direct preference optimization: Your language model is secretly a reward model, 2024."
> ← bibliography entry

> "Greek Translation DBBE 18534 ἡ βίβλος ἡδε τῶν θεοπνεύστων λόγων: …"  ← wrong language

**0.5 Don't look at the automatic label.** It is not in the workbook, deliberately. Don't go
looking for it in `outputs/analysis/`.

**0.6 Notes are cheap.** If a row made you hesitate, say why in one phrase. Those notes are what
you'll use to defend or revise a category later.

---

## 1. Verb types (sheet `04c verb types`) — §2.4

**Unit:** one verb *type*, not a sentence. The `examples` column shows up to three corpus sentences
where that verb governs AI as its active subject. `n` is how many times it occurs.

**Question:** *when AI is the subject of this verb, what kind of process is being construed?*

| Code | Construal | Test | Seed items |
|---|---|---|---|
| `material_action` | AI brings about a change in the world | Does something exist or differ afterwards? | generate, produce, create, transform, process, translate, detect |
| `cognitive_mental` | AI as a Senser — the anthropomorphic case | Does it require a mind, not just a computation? | think, know, understand, learn, infer, comprehend, believe |
| `verbal_communicative` | AI as a Sayer in symbolic exchange | Could the object be a quotation or a proposition? | say, claim, suggest, explain, respond, tell, argue |
| `instrumental_tool` | AI as a facilitator of someone else's action | Is a human beneficiary implied? | assist, support, enable, help, facilitate, allow, serve |
| `stative_relational` | AI as a settled state or possessor | Does it attribute being or having, not doing? | be, become, remain, seem, have, possess, lack, exist, emerge |
| `behavioural` | conduct without a Goal (Halliday's sixth type) | Is there no thing acted upon? | operate, function, work, act, interact, behave |
| `evaluative_performance` | AI as a competitor in a benchmark | Could you put a number or a rival next to it? | achieve, outperform, excel, struggle, fail, score, lag |
| `semiotic_evidential` | AI as a source of evidence, not of speech | Is it "the results show" rather than "the model says"? | demonstrate, show, exhibit, reveal, indicate, prove |
| `OTHER` | delexical or genuinely unclassifiable | Is the verb carrying no process meaning of its own? | do, take, give, bring, need, get |

**Boundary rules**

- **Computation vs cognition.** `analyze`, `classify`, `detect`, `predict` are `material_action`:
  they describe operations. Reserve `cognitive_mental` for verbs that would be odd about a
  calculator — *understand*, *believe*, *know*.
- **`show`/`demonstrate` are `semiotic_evidential`, not verbal.** "DeepSeek-V3 shows the best
  results" reports evidence; it is not speech.
- **`struggle`, `fail`, `achieve` are `evaluative_performance`, not emotional or material.** In this
  corpus they are benchmark vocabulary: "Gemini struggles with compound edits". (This is a
  deliberate departure from Inie et al., who classify *struggle* as Emotion — see §4.6 below.)
- **`play` is almost always `OTHER`**: "plays a role" is delexical, not behavioural.
- **Dual-use verbs:** decide from the examples shown, not from the verb in the abstract. `write` is
  `material_action` in "AI writes code" and `verbal_communicative` in "AI writes that…".
- **Passive-looking examples:** if all the examples show AI as Goal rather than Actor, the row was
  mis-extracted — code `OTHER` and note `bad extraction`.
- **Phase and purpose verbs:** `aim to`, `continue to`, `try to`, `begin to`, `seek to` and
  `keep -ing` borrow their process from the following verb. Code the row `OTHER` and note
  `phase/catenative`; classify the complement separately if it appears in a usable AI-subject
  example. A phrasal verb such as `take over` is one predicate, not bare `take`.

---

## 2. Modality (sheet `04b modality`) — §2.3

**Unit:** the **AI predicate clause** — the clause built around the verb that governs the AI
mention. Since the clause-level rebuild of 04b, the pipeline emits one row per AI predicate
clause (`04b_modality_per_clause.csv`); a sentence with two AI clauses ("AI *can* assist, but it
*must not* replace human judgement") yields two rows with independent labels. The sentence-level
file is a derived aggregate kept for compatibility.

**Question:** *how strongly is the claim about AI committed to?*

| Code | Definition | Markers |
|---|---|---|
| `bare` | Categorical assertion; no modal, no hedge | "AI supports monitoring health data." |
| `strong` | High affinity: authority or inevitability | *will*, *shall*, *must*, *cannot*, *clearly*, *obviously*, *certainly*, *undoubtedly* |
| `weak/hedged` | Low affinity: potentiality or uncertainty | *may*, *might*, *could*, *can*, *should*, *perhaps*, *possibly*, *potentially*, *seems*, *appears*, *likely* |
| `mixed` | Both a high- and a low-affinity marker in the same AI clause | "AI will clearly help, though it may struggle with nuance." |
| `scope_uncertain` | *(sentence-level file only)* the sentence's AI clauses carry different labels; code each clause separately at clause level | "AI can assist, but it must not replace human judgement." |

**Boundary rules**

- **`can` is weak, `cannot` is strong.** Positive *can* marks mere potentiality; negative *cannot*
  is a strong epistemic commitment to a limit (Hao 2020). This asymmetry is deliberate — don't
  normalise it.
- **`bare` is the default, and it is a finding, not a failure.** About three-quarters of AI
  predicate clauses are bare. If there is no modal and no hedge, code `bare` even when the claim
  is bold.
- **Deontic *must* counts as `strong`.** "Policies must state that AI cannot be listed as an author"
  — code `strong`, and note `deontic` if you want to separate them later.
- **Modality must be in the AI clause.** "We *may* have shown that AI supports X" hedges the
  authors' claim, not AI's capability. If the hedge attaches to a different clause, code `bare` and
  note it.
- **Questions:** "Can AI be a historian?" — code `weak/hedged` (the modal is present and the claim
  is not asserted).
- **Reported speech:** code the modality of the AI clause itself, not the reporting verb. The
  attribution is captured separately in task 3.
- **Multiple AI clauses:** each AI clause is its own row in the clause-level file — code each on
  its own. `mixed` means strong and weak markers in the *same* AI clause. `scope_uncertain` only
  survives in the derived sentence-level file, where it flags sentences whose clauses disagree; at
  clause level it should never be needed.

---

## 3. Polarity and projection (sheet `04e polarity`) — §2.6

Two independent yes/no judgements per row. The `gov_verb` column shows the verb we anchored to.

### 3.1 `manual_negated`

**Question:** *is the process involving AI denied?*

`TRUE` when the AI clause is negated by any means:

- a negative particle — "GPT-3 does **not** understand its answer"
- a negative-polarity item — "AI limits, **instead of** expands, our imagination"; "**without**
  mastery of language"; "**rather than** replace"
- an inherently negative verb — "the model **lacks** context", "LLMs **fail** to recognise"
- a negation on a verb that projects the AI clause — "the article does **not** claim that AI
  replaces philology"

**Boundary rules**

- **Scope matters.** Negation must reach the AI process. "AI supports work that is **not**
  well-funded" → `FALSE`; the negation is in a different clause.
- **Inherently negative verbs are `TRUE`** even without a particle. Note that the automatic coder
  scores *lack* and *fail* as negated by construction, so your judgement here is what tells us
  whether that was right.
- **Rhetorical questions:** "Can AI be a historian?" is not a denial → `FALSE`.
- **Double negation** resolving to a positive → `FALSE`, and note it.

### 3.2 `manual_projected`

**Question:** *is the claim attributed to someone rather than asserted by the author?*

`TRUE` when the AI clause is embedded under a saying or thinking verb:

- "Weizenbaum **believed that** AI should be understood in terms of illusory effect."
- "The study **shows that** increasing the parameter count does not yield a boost."
- "Elvia Wilk **notes**, 'AI may actually change the way we think.'"

`FALSE` for a direct authorial assertion, even a hedged one: "AI might not have been trained to
recognise poetic meters" is the author's own hedged claim.

**Boundary rules**

- **Inanimate projectors count.** "Results suggest…", "experiments demonstrate…" are projection
  (that is the point — §5.7 shows most projectors are inanimate).
- **Citations alone are not projection.** "AI supports X (Smith 2023)" is asserted by the author,
  who is merely citing support → `FALSE`.
- If projected, it is worth noting *who* projects — human, study, company — in the notes.

---

## 4. Anthropomorphization (sheet `06f anthropomorphization`) — §2.8

**Unit:** the sentence, judged with respect to its AI mention(s), shown in `mention_terms`.

**Question:** *which human attributions does this sentence make about AI?* Mark `y` in **every**
column that applies. Leave all eight blank for none.

Categories follow Inie, Zukerman & Bender (2026, Table 2).

| Category | Attributes to AI | Corpus examples |
|---|---|---|
| `Cognizer` | engaging in cognition | *understand*, *know*, *learn*, *reason*, *interpret*; also the noun **intelligence** and *training* (see 4.1) |
| `Products_of_cognition` | something obtainable only through cognition | *capability*, *ability*, *capacity*, *skill*, *competence*, *judgment*, *creativity*, *bias* (when **has**, not **reflects**) |
| `Emotion` | feeling, or an emotional bond | *feel*, *empathise*, *care*, *companion*, *friend*, *fondness* |
| `Communication` | participating in symbolic exchange | *answer*, *respond*, *suggest*, *explain*, *dialogue*, *chatbot*, *prompt*, *instruction* |
| `Agent` | acting with intent or independence | *help*, *assist*, *enable*, *facilitate*, *decide*, *operate*; modifiers *autonomous*, *agentic*, *on its own* |
| `Human_role_analogy` | occupying a human role | *author*, *assistant*, *judge*, *editor*, *critic*, *co-author*, *copilot*, *tutor*, *researcher* |
| `Names_and_pronouns` | referred to as a person | animate pronouns *he*, *she*, *who*; human first names given to systems |
| `Biological_metaphors` | having a body or biology | *neural*, *neuron*, *brain*, *hallucinate*, *ingest*, *digest*, *feed*, *see*, *tireless*, *evolve* |

### 4.1 The entrenched-term rule — read this before you start

Inie et al. hold that established terms of art still count. **"Artificial intelligence" and "AI" are
Cognizer**, because the term attributes intelligence in the act of naming; so are *machine learning*,
*neural network* and *training data*.

If you applied this literally, ~52% of rows would be `Cognizer` and the task would be pointless.
So, for this sheet:

> **Mark a category only when the sentence makes a *predication* beyond the name itself.**
> The bare occurrence of "AI", "machine learning" or "neural network" is **not** enough.

The entrenched-term rate is measured separately and automatically (§5.12), so nothing is lost.

- "We explore **AI** within the humanities." → nothing marked.
- "**AI** *understands* the text." → `Cognizer`.
- "the **neural network** was trained on Wikipedia" → nothing marked (both terms entrenched).
- "the model *ingested* the ugliest material on the internet" → `Biological_metaphors`.

### 4.2 Negation still counts here

This task is the one place where the rule inverts. Per Inie et al. §4.1, **a denied attribution is
still an attribution**: "GPT-3 does not *understand* its answer" presupposes that understanding was
on the table → mark `Cognizer`. Whether it was denied is recorded in task 3, not here.

### 4.3 Flagged comparisons don't count

"AI should not be imagined as an autopilot, but as a co-pilot **who** assists" — the simile is
explicitly flagged as a comparison, so per their §4.2.6 do **not** mark `Human_role_analogy`. The
relative pronoun *who*, however, **is** `Names_and_pronouns`.

### 4.4 Product names are not the author's choice

*ChatGPT* contains "Chat" and *OpenAI* contains "AI", but an author reusing an established product
name is not making an anthropomorphizing choice (their §4.1). Don't mark those.

### 4.5 Multiple categories are normal

"The **AI assistant** *explained* its *reasoning*" → `Human_role_analogy` + `Communication` +
`Cognizer`. Mark all three.

### 4.6 Where we depart from Inie et al.

**`struggle`.** They classify it as Emotion. In our corpus it is overwhelmingly benchmark prose:
"LLMs struggle to discern important information". **Do not mark Emotion for benchmark uses of
*struggle*, *suffer* or *prefer*.** Mark Emotion only where a feeling is genuinely attributed
("a bot can form an empathetic bond with its users"). Note `struggle-benchmark` when you skip one —
we are reporting this disagreement in the paper and your codings are the evidence for it.

---

## 5. Nominalisation and agent (sheet `04d nominalisation`) — §2.5

**Unit:** the nominalised AI action shown in `governor_lemma` (e.g. *development*, *training*,
*adoption*), within its sentence.

**Question:** *is a human or corporate agent for that process realised anywhere in the sentence?*

`TRUE` — an agent is recoverable:

> "Copilot is based on **OpenAI's** Codex, a GPT version especially trained on code."

> "**We** therefore combine rule-based detection with GPT-5's assistance."

`FALSE` — no agent is recoverable:

> "But the recent wave of AI **development** has inverted this curve."  ← whose development?

> "AI **hallucinations** and cultural errors necessitate human-in-the-loop workflows."

**Boundary rules**

- **Sentence scope only.** An agent named in the previous sentence does not count. This is why the
  automatic 67.2% is described as an upper bound.
- **Who counts as an agent:** people (*we*, *researchers*, *scholars*, *developers*, *users*) and
  organisations (*OpenAI*, *Google*, *the lab*, *the team*). A named *model* does not — "BERT's
  training" is still agentless.
- **The agent must plausibly be the agent of *that* process.** "Scholars have noted the rapid
  development of AI" — the scholars did the noting, not the developing → `FALSE`.
- **Passive with a by-agent** → `TRUE`: "trained **by Google AI**".
- The stricter automatic measure `agent_relation` only counts a possessive linked to the
  nominalisation, a local `by`-agent, or a subject directly governing the nominalisation. A person
  mentioned elsewhere in the sentence is not enough. This is the preferred interpretation for the
  paper; the older sentence-level measure is retained only as a comparison.

---

## 6. Quick reference

| Task | Column(s) | Values |
|---|---|---|
| Verb types | `manual_class` | material_action · cognitive_mental · instrumental_tool · verbal_communicative · stative_relational · behavioural · evaluative_performance · semiotic_evidential · OTHER |
| Modality | `manual_label` | bare · strong · weak/hedged · mixed · NA |
| Polarity | `manual_negated`, `manual_projected` | TRUE · FALSE · NA |
| Anthropomorphization | 8 category columns | `y` or blank |
| Nominalisation | `manual_agent_present` | TRUE · FALSE · NA |

**When you finish a sheet**

```
python analysis_code/annotation_workbook.py import
python analysis_code/annotation_sampler.py score
```

Close Excel first, or the import will read a stale file.

**The `p2` sheets** are the reliability pass: the same 100 rows, reshuffled. Code them at least a
week later without consulting your first pass — or, better, have a second coder do them. That
comparison produces Cohen's κ, which is what lets the paper claim reliability rather than merely
assert it.

---

## 7. Revising the scheme

If you find yourself repeatedly wanting a category that does not exist, or repeatedly unable to
separate two that do, that is a finding about the scheme rather than a coding failure. Write it in
the notes, keep coding consistently with the current definitions, and we will revise afterwards —
that is exactly how `behavioural`, `evaluative_performance` and `semiotic_evidential` were added
(§10 of the paper). Do not change definitions mid-pass; it makes the agreement statistic
meaningless.
