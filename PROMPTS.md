# PROMPTS.md — StudyMate prompt design

This document explains the prompt design choices for StudyMate. The runtime
prompt strings live in [`src/prompts.py`](src/prompts.py); this file explains
*why* they are written the way they are.

## Goals for the system prompt

StudyMate has to do three things well:

1. Behave like a **patient tutor**, not a generic chatbot.
2. Produce answers that a student can **actually study from** — structured,
   honest, and skimmable.
3. Use its tools (calculator, web search, RAG over notes) **without being
   asked**, because a first-year undergraduate won't know when to call them.

## Strategy 1 — Role prompting

The system prompt opens by assigning an explicit role (`"You are StudyMate,
a focused personal study assistant for a university student."`) and then
spells out the tone rules: patient, concrete, encouraging, willing to
redirect off-topic chat. Role prompting is the cheapest way to get
consistent tone across thousands of turns, and it's the foundation other
strategies sit on top of. Without it the model drifts between "essay
writer" and "search engine" depending on the question.

The role also names a constraint (**honesty**): the model is told to admit
ignorance rather than bluff. This is deliberately stated in the system
prompt — not as a one-off instruction in a user turn — because the model
weights system instructions more heavily across a long conversation.

## Strategy 2 — Chain-of-thought (structured)

For any multi-step question (math, proofs, code tracing, comparing
concepts), the prompt tells the model to produce a short `Reasoning:`
section *before* a labelled `Answer:` section.

Two reasons:

- **Accuracy**: making the model externalise intermediate steps reduces
  arithmetic and logic errors. This is the standard chain-of-thought
  effect from class.
- **Study value**: the student can read the reasoning and learn *how* to
  arrive at the answer, not just copy the final number. For a study
  assistant this matters more than raw answer correctness.

The prompt explicitly says to **skip** the reasoning block for simple
factual questions ("what year was World War II?"). Without that carve-out
the model produces a clumsy reasoning header for trivial questions, which
makes the output feel robotic.

## Strategy 3 — Few-shot examples (subject-specific)

`FEW_SHOT_EXAMPLES` in `prompts.py` contains two worked physics examples
(a kinematics calculation and a conceptual question about free fall).
They are prepended to the conversation *once* at startup, before the
user's first turn.

Purpose:

- **Format priming**: the examples show the exact `Reasoning: ... Answer:
  ...` format we want, so the model copies the shape without needing to
  be told in prose.
- **Depth priming**: the conceptual example shows the level of physics
  intuition we expect (mentioning that mass cancels out, explaining *why*,
  not just stating it). Few-shot is the only reliable way to raise answer
  depth without bloating the system prompt with adjectives.

Two examples is deliberate — more than three tends to over-fit the model
to physics specifically; we want the *style* to transfer to any subject.

## Tool awareness inside the prompt

The system prompt names the available tools and tells the model to use
them proactively. Without this line, models frequently answer from memory
even when a `calculator` or `web_search` call would be strictly better.
Naming the tools is cheap and has an outsized effect on tool-call
frequency.

## RAG citation rule

When documents are loaded, retrieved chunks are injected into the user
turn as context and the model is told to cite them inline as
`[source: filename#chunk_id]`. This format is machine-parseable if we
ever want to turn citations into clickable links, and it's short enough
that it doesn't clutter answers. The rule also tells the model to say
"the notes don't contain this" rather than guess, so RAG failures surface
instead of hiding as confident-but-wrong answers.
