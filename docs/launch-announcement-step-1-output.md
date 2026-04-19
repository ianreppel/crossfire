# What if your draft could get critical reviews on demand?

You know the dance. You have a prompt—for code, for an essay, for analysis. You send it to one LLM. The output is... okay. A good starting point. So you open another tab, send it to a different model. That one has better structure but hallucinates a key detail. A third gets the detail right but writes like a corporate drone.

So you start cutting and pasting. Juggling tabs, merging ideas, fixing flaws. You become the human synthesizer, the quality control, the editor-in-chief for a team of inconsistent AI assistants. I found myself doing this constantly, manually pitting models against each other to triangulate a better result. It worked, but it was tedious, error-prone, and felt like a problem begging for automation.

That's why **Crossfire** exists.

## The Adversarial Loop

Crossfire is a multi-agent adversarial refinement orchestrator. In plain terms: it takes your instruction, sends it to several different LLMs to generate drafts, then sends each draft to another set of LLMs for critical review. Finally, it synthesizes the best parts of the reviewed drafts into a single, refined artifact.

The core loop is deceptively simple:

1. **Generate** — Create multiple independent attempts at the task
2. **Review** — Scrutinize each attempt from multiple adversarial perspectives  
3. **Synthesize** — Discard the weak, merge the strong, produce something better

This cycle repeats, with each round building on the synthesis of the last. Bad ideas get eliminated early so good ones can be rebuilt stronger. It's a way to escape the local maximum of a single model's thinking by introducing fresh, critical contexts at each step.

This multi-start evolutionary search—where multiple attempts compete and the best elements survive to the next generation—is what makes it work. It's an [improved Ralph Wiggum method](https://ianreppel.org/ralph-wiggum-as-a-degenerate-evolutionary-search/) that filters out the worst offenders through systematic build → break → rebuild cycles.

## Why Engineers Should Care

Crossfire automates the whole adversarial pipeline. Instead of trusting a single LLM's first attempt, it runs systematic pressure that catches flaws before they reach your final output. Each round applies fresh eyes so reviewers see new text with clear perspective rather than iterating endlessly on the same tired draft.

For engineers who want higher-quality, vetted outputs—code that compiles, prose that's clear, research that stands up to inspection—Crossfire gives you a systematic way to get there without juggling browser tabs or trusting a single pass.

## What It Looks Like

This blog post itself was generated using Crossfire's two-stage pipeline. First, a draft in `write` mode:

```bash
uv run crossfire run \
  --mode write \
  --instruction-file docs/launch-announcement-prompt.md \
  --context-file README.md \
  --num-generators 5 \
  --num-reviewers-per-candidate 3 \
  --num-rounds 3 \
  --output docs/launch-announcement-step-1-output.md
```

Then tightened in `edit` mode:

```bash
uv run crossfire run \
  --mode edit \
  --instruction-file docs/launch-announcement-tighten.md \
  --context-file docs/launch-announcement-step-1-output.md \
  --num-generators 3 \
  --num-reviewers-per-candidate 5 \
  --num-rounds 5 \
  --output docs/launch-announcement-step-2-output.md
```

For quick research questions, it's even simpler:

```bash
uv run crossfire run \
  --mode research \
  --instruction "Compare error correction strategies for superconducting vs trapped-ion qubits"
```

A typical run looks like this in your terminal:

```terminal
⠧ Round 1/3 | Phase: Generate | 5 candidates
⠧ Round 1/3 | Phase: Review   | 15 reviews  
⠧ Round 1/3 | Phase: Synthesize
✓ Round 1 complete. Weaknesses found: 7
⠧ Round 2/3 | Phase: Generate | 5 candidates
⠧ Round 2/3 | Phase: Review   | 15 reviews
⠧ Round 2/3 | Phase: Synthesize  
✓ Round 2 complete. Weaknesses found: 2
✓ Run complete. Early stop after 2 rounds. Output saved to runs/20250415_142356/output.md
```

Crossfire replaces the manual intuition of "let me try that again" with structured adversarial refinement. It turns the latent potential of having multiple LLMs at your disposal into a practical tool—not by making them agree, but by making them work through systematic build → break → rebuild cycles until what remains can withstand scrutiny.