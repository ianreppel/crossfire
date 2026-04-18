# Crossfire launch blog post prompt

Write a spiffy, fun blog post announcing **Crossfire**, a multi-agent adversarial refinement orchestrator a.k.a. Ralph Wiggum on steroids that offers an agentic build→break→rebuild loop for research, coding, writing, and editing.

**Audience:** Engineers who use LLMs for code or tech-savvy people who write a lot and are tired of brittle one-shot prompts.

**Tone:** Witty, conversational, human. **The Simpsons** references welcome (Ralph Wiggum). Jokes must not bury the facts. No corporate-LLM filler. Short paragraphs, punchy headings, skimmable.

**Must cover:**

1. **What it is** — plain English and the mental model.
2. **Why it exists** — pain (fragile one-shots, unexamined reasoning, code that looks fine but isn’t, endless copy-pasting when doing it manually).
3. **What it solves** — and **what it doesn’t** (limits).
4. **The loop** — **build → shoot down from many angles → rebuild** (or build / break / rebuild). Stress multiple reviewers and iterative rounds.
5. **How to use** — real commands and flags from the README; one minimal example and one richer one. 
6. **Dogfooding** — you built this to be rigorous; optional nod to using Crossfire on Crossfire-style work. Just do not say dogfooding as I hate corporate lingo and that would be flagged by edit mode.
7. **Example “live” output** — a **short** realistic-looking stub: command, a line or two of progress, tiny reviewer excerpt (STRENGTHS / WEAKNESSES / SEVERITY), a line of synthesis, a tiny cost line. **Label it clearly** as representative example output, not a paste of a real run.
8. **Quick checklist** — keys, `uv sync`, sample `crossfire run`.
9. **Close** — strong ending + practical CTA (try it, read the repo) without grovelling.

**Constraints:**

- Ground commands and flags in the provided context; don’t invent CLI or config.
- **~900–1400 words.** Markdown only.

---

## Where this file lives

`docs/` is a reasonable home for **prompts and examples** that are not the main README. If you prefer them more visible, a top-level `prompts/` folder works too; keep paths in commands in sync.

---

## Step 1 — `write`: generate the draft

Use the README so commands and behaviour stay accurate.

**Naming:** “Step” here means **this two-command pipeline** (write, then edit). It is **not** the same word as Crossfire’s internal **phases** (generation, review, synthesis inside each round).

Committed outputs for the announcement live alongside the prompts:

- `docs/launch-announcement-step-1-output.md` — final text from step 1
- `docs/launch-announcement-step-2-output.md` — final text from step 2

```bash
uv run crossfire run \
  --mode write \
  --instruction-file docs/launch-announcement-prompt.md \
  --context-file README.md \
  --num-generators 3 \
  --num-reviewers-per-candidate 3 \
  --num-rounds 5 \
  --output docs/launch-announcement-step-1-output.md
```

`--output` writes the final synthesis; the run still archives under `runs/<timestamp>/` (see README).

---

## Step 2 — `edit`: tighten the draft

**Why two modes:** `write` optimises for voice, structure, and energy; `edit` is ruthless about clarity, concision, and anti-waffle — same tool, different adversarial brief. That is a concrete way to run a second pass on your own announcement without one overloaded prompt.

Put the draft in **context** so the editing brief stays in **instruction** (instruction is never compressed; context may be trimmed if huge).

```bash
uv run crossfire run \
  --mode edit \
  --instruction-file docs/launch-announcement-tighten.md \
  --context-file docs/launch-announcement-step-1-output.md \
  --num-generators 1 \
  --num-reviewers-per-candidate 5 \
  --num-rounds 3 \
  --output docs/launch-announcement-step-2-output.md
```

Optional: `--no-enrich` on step 2 if the tighten brief is already explicit and you want less rewriting overhead.
