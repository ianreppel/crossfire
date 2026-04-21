# Crossfire: Automating the LLM Argument

What if Ralph Wiggum had friends?

I built Crossfire to automate the only reliable way to get strong LLM results: make them argue. Manually juggling browser tabs to have models critique each other works but is miserable. Crossfire runs the whole process for you.

The core problem is false confidence. A single LLM's answer sounds convincing, but without counterarguments, you can't gauge its strength. The solution is adversarial generation: multiple models create competing solutions while others attack them for weaknesses.

## How It Works: Generation, Review, Synthesis

Give Crossfire a task. It sends that prompt to multiple LLMs simultaneously to generate different approaches. Each candidate then faces multiple reviewer models hunting for factual errors, logic flaws, security holes, or other critical weaknesses. Finally, a synthesizer model takes all candidates and reviews, picks the best parts, and combines them into a refined result.

This generation → review → synthesis loop repeats for as many rounds as you specify.

The key insight? Fresh contexts prevent the groupthink that happens when you ask the same model to "improve" its own work. Each round starts clean, avoiding local optima. It's [multi-start evolutionary search](https://en.wikipedia.org/wiki/Evolutionary_algorithm) with delayed selection—the Ralph Wiggum method, but with actual friends providing feedback.

## Specialists, Not Generic Critics

The reviewers aren't bland critics. They're specialists with specific protocols:

* **Code Mode:** Paranoid senior engineers hunting for security flaws and missing edge cases.
* **Edit Mode:** Ruthless copyeditors slashing jargon and weak phrasing.
* **Write Mode:** A writers' group looking for plot holes and emotional authenticity.

The system finds material weaknesses, not bikeshed style.

## From Simple Queries to Complex Orchestration

For a straightforward research task:

```bash
uv run crossfire run --mode research --instruction "Compare error correction strategies for superconducting vs trapped-ion qubits"
```

For a complex workflow, like writing this announcement:

```bash
uv run crossfire run \
  --mode write \
  --instruction-file docs/launch-announcement-step-1-write-mode.md \
  --context-file README.md \
  --num-generators 5 \
  --num-reviewers-per-candidate 3 \
  --num-rounds 3 \
  --output docs/launch-announcement-step-1-output.md
```

Then refine it:

```bash
uv run crossfire run \
  --mode edit \
  --no-enrich \
  --instruction-file docs/launch-announcement-step-2-edit-mode.md \
  --context-file docs/launch-announcement-step-1-output.md \
  --num-generators 3 \
  --num-reviewers-per-candidate 3 \
  --num-rounds 3 \
  --output docs/launch-announcement-step-2-output.md
```

Within each round, generation and review happen in parallel. Crossfire stops early if reviewers find no material weaknesses, since further rounds won't add value.

_This post was written and edited by Crossfire (COST_PLACEHOLDER, TIME_PLACEHOLDER), and subsequently reviewed and tweaked by Ian._
