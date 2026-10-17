# Intro2DS — reproducibility and data-sensitivity study of virtual screening models

Each person takes **one published virtual-screening model** and produces three things with it:

1. its published evaluation numbers, reproduced from the released checkpoint;
2. the model **retrained from the released training data**;
3. the model **retrained on training-set variants** supplied here.

We are measuring how reproducible published numbers are from released artifacts, and how sensitive
they are to the exact composition of the training set. Step 2 is not a formality — it is the
baseline that step 3 is compared against, so a step-3 number means nothing without it.

## Three checkpoints, three reviews

Training runs cost real money and GPU time, so work is reviewed at three points. Each checkpoint is
one pull request. **A merged pull request is your approval to start the next checkpoint.**

| | checkpoint | what is in it | cost | why it is a gate |
|---|---|---|---|---|
| **A** | environment + reproduce + smoke train | `A1_env/`, `A2_reproduce/`, `A3_smoke/` | **under 1 GPU-hour total** | proves you can run the thing, reproduces the paper, and tells us what the real runs will cost |
| **B** | retrain on released data | `B_retrain_released/` | expensive | the baseline arm |
| **C** | retrain on the supplied variants | `C_retrain_variants/<variant_id>/` | expensive | the measurement |

Everything in checkpoint A is cheap on purpose. Submit all three A parts in **one** pull request.

**Do not start B or C before your previous pull request is merged.** If you are blocked waiting,
say so in the PR rather than starting the expensive run.

## Getting started

```bash
git clone https://github.com/kongwoang/Intro2DS.git
cd Intro2DS
cp -r submissions/_TEMPLATE submissions/<your-id>     # your id is at the top of your assignment
git checkout -b sub/<your-id>
```

Read these three, in order, before you touch a GPU:

1. your assignment in [`assignments/`](assignments/README.md) — what model, which files, exact deliverables
2. [`CONTRACT.md`](CONTRACT.md) — the rules; short, and every one exists because breaking it has cost time
3. [`report_schema.json`](report_schema.json) — the shape of each `summary.json`

## Questions, problems, and timing

**Open a GitHub issue in this repository.** That is the channel — not DM, not email — so that an
answer one person needs is visible to everyone else who will hit the same thing. Tag it with your
id (`01-drugclip`, and so on) in the title.

There is **no fixed deadline**, and we would rather have a correct checkpoint A than a fast one.
What matters is the order: finish A and get it merged before you start B, and B before C, because
the expensive runs are approved from what A tells us. If you are blocked, say so in an issue rather
than guessing or starting the next stage.

If a checkpoint-A reproduction fails the gate, that is where your work stops and it is a complete,
useful result. Say so and we will take it from there.

## When something is broken upstream

**Say so in an issue and stop.** Do not patch around it, do not guess, do not "make it work".

A documented "this does not reproduce, here is the evidence" is a genuine result of this study and
costs you nothing. A silently patched pipeline produces a number nobody can use, after spending the
GPU budget to get it.
