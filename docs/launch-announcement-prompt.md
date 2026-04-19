# Step 1: Crossfire launch announcement — write mode

Write a blog post announcing Crossfire: what it is, why it exists, what problem it solves, why anyone might care, and the main adversarial loop in plain language.
You can use "shot down" once, then move on, e.g. build → break → rebuild loop.
Avoid extra combat imagery (no cage match, gauntlet, battlefield, slaughter, artillery), and no second gun pun.
DO NOT stack metaphors!

The post must open with a single strong one-liner that invites the reader to continue.

Voice: plain, focussed on engineers, but not bland.
Use "I" briefly for origin/intent, but keep Crossfire as the subject.
Never use "we" or corporate tone.

One short The Simpsons / Ralph Wiggum nod is enough (the context file (README) links the idea to Ralph Wiggum and to a piece on evolutionary search).
Use that link from the README, do not invent URLs!

If you need a framing hook, pick one and get to the product fast:
- What if Ralph Wiggum had friends?
- What if one is never enough?
- What if your draft could get critical reviews on demand?

You are free to use (parts) of the back story, but you can decide against it entirely.
Back story: I created Crossfire because I ended up pitting LLMs against each other to improve their outputs gradually, except I got tired of copy-pasting between five tabs while waiting for each to finish.
I often did so for research questions, editing documents, or writing small convenience scripts that had subtle bugs.
I still have to check every single reference, the logic of the code, orany claims made, but at least the worst offenders are filtered out most of the time with such an adversarial yet manual setup.
The problem fresh contexts solve is that each LLM looks at the text or code with fresh (robot) eyes rather than keeps noodling on it, believing each iteration is better because it worked on it for a while.
Code and text need to be clear to anyone looking upon it for the first time.

Include a short fake terminal snippet so readers see what a run looks like.
I'll swap in real output later.

The README context includes other example commands. 
Ignore those for this article.

The two fenced bash blocks below are the exact commands for this announcement pipeline. 
Copy these verbatim (same paths, flags, and numbers).

Step 1 — draft (`write` mode):

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

Step 2 — tighten (`edit` mode):

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

Also show one minimal one-off using `--instruction` (not `--instruction-file`), e.g. a single research question in quantum computing.
Keep that command short, so rely on defaults as much as possible.
Check the context file for defaults.
