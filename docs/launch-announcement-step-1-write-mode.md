# Step 1: Crossfire launch announcement — write mode

Write a blog post announcing Crossfire: what it is, why it exists, what problem it solves, and the main adversarial loop in plain language.
Weave the value proposition into the explanation of how it works.
Do not create a separate "why you should care" section.
You can use "shot down" once, then move on, e.g. build → break → rebuild loop.
Avoid extra combat imagery (no cage match, gauntlet, battlefield, slaughter, artillery), and no second gun pun.
DO NOT stack metaphors!

The post must open with a single short hook — a sentence, question, or quote — under 200 characters, alone on its first line.
Do not follow the hook with a "you know the feeling" scenario paragraph.
Go straight from hook to substance.

Voice: plain, focussed on engineers, but not bland.
Use "I" briefly for origin/intent, but keep Crossfire as the subject.
Never use "we" or corporate tone.

One short The Simpsons / Ralph Wiggum nod is enough (the context file (README) links the idea to Ralph Wiggum and to a piece on evolutionary search).
Use that link from the README.
Do not invent URLs!

If you need a framing hook, pick one and get to the product fast:
- What if Ralph Wiggum had friends?
- What if one is never enough?
- What if your draft could get critical reviews on demand?

You are free to use (parts) of the back story, but you can decide against it entirely.

Back story: I [plan for five versions](https://ianreppel.org/thoughts-on-writing/) of everything I write or code.
Why not let LLMs do some of that drafting and re-drafting?
I created Crossfire because I ended up pitting LLMs against each other to improve their outputs gradually, except I got tired of copy-pasting between five tabs while waiting for each to finish.
I still have to check every single reference, the logic of the code, or any claims made, but at least the worst offenders are filtered out most of the time with such an adversarial setup.
Fresh contexts are crucial: each LLM looks at the text or code with fresh (robot) eyes rather than keeps noodling on it, believing each iteration is better because it worked on it for a while.
But managing that manually means juggling five tabs and starting new chats after every round.

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
  --instruction-file docs/launch-announcement-step-1-write-mode.md \
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
  --no-enrich \
  --instruction-file docs/launch-announcement-step-2-edit-mode.md \
  --context-file docs/launch-announcement-step-1-output.md \
  --num-generators 3 \
  --num-reviewers-per-candidate 3 \
  --num-rounds 3 \
  --output docs/launch-announcement-step-2-output.md
```

Also show one minimal one-off using `--instruction` (not `--instruction-file`), e.g. a single research question in quantum computing.
That command must use only `--mode` and `--instruction`, nothing else.

End the post once, decisively.
No summary paragraph after the final code block.

Close with an italicized one-liner: "_This post was written and edited by Crossfire (COST_PLACEHOLDER, TIME_PLACEHOLDER), and subsequently reviewed and tweaked by Ian._"
Keep the placeholders exactly as shown.
Do not replace them with numbers.
