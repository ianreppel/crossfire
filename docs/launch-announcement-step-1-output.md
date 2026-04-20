What if Ralph Wiggum had friends?

I built Crossfire to automate the one good way I've found to get consistently strong results from LLMs: make them argue. Manually juggling five browser tabs to have GPT-4 critique Claude's code while Gemini points out flaws in both is powerful, but it's a miserable, soul-crushing workflow. Crossfire is the tool that runs the whole process for you.

The core problem is false confidence. When a single LLM gives you an answer, it sounds convincing. Without counterarguments or alternative perspectives, you have no way to gauge its strength. Manually simulating this adversarial process is effective but tedious. Crossfire automates it.

Here's how it works: you give Crossfire a task, and it sends that prompt to multiple LLMs simultaneously to generate different approaches. Each generated candidate gets shot down by multiple reviewer models that look for factual errors, logic flaws, security vulnerabilities, or whatever systematic weaknesses matter for your task. Then a synthesizer model takes all the candidates and reviews, picks the best parts, and combines them into a refined result. This generation → review → synthesis loop repeats for as many rounds as you specify.

The insight is that fresh contexts prevent the groupthink that happens when you ask the same model to "improve" its own work. Each round starts clean rather than getting trapped in local optima. It's multi-start evolutionary search with delayed selection—the Ralph Wiggum method, but with actual friends to provide feedback.

The reviewers aren't generic critics; they're specialists with specific protocols. In code mode, they act as paranoid senior engineers, hunting for security flaws and missing edge cases. In edit mode, they are ruthless copyeditors who slash jargon and weak phrasing. In write mode, they're a writers' group looking for plot holes and emotional authenticity. The system is designed to find material weaknesses, not to bikeshed style.

For a simple research query:

```bash
uv run crossfire run --mode research --instruction "Compare error correction strategies for superconducting vs trapped-ion qubits"
```

For complex tasks, you can orchestrate a larger process:

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

Then refine through a second pass:

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

Each round runs generation → review → synthesis sequentially, but within generation and review, model calls happen in parallel. Crossfire stops early when reviewers find no material weaknesses, since further rounds won't add value.

_This post was written and edited by Crossfire (COST_PLACEHOLDER, TIME_PLACEHOLDER), and subsequently reviewed and tweaked by Ian._