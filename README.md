# Intro2DS — reproducibility and data-sensitivity study of structure-based virtual screening

This repository coordinates one task across several people. Each person is assigned **one published
virtual-screening model** and does three things with it:

1. **Reproduce** the published evaluation numbers using the released checkpoint.
2. **Retrain** the model from the released training data, with the released configuration.
3. **Retrain** it again on one or more **training-set variants** supplied here.

The point of the study is to find out how sensitive each model's reported performance is to the
exact composition of its training set, and how reproducible the published numbers are from the
released artifacts. Step 2 is not a formality: it is the baseline that step 3 is compared against,
so a step-3 number is uninterpretable without it.

## Before you run anything expensive

Training runs cost money and GPU time, so **nothing past stage 3 starts without review**. Work
through the stages in order, open a pull request, and wait for it to be merged. A merged PR is the
approval record.

| stage | what | cost | gate |
|---|---|---|---|
| 1 | environment lock, `nvidia-smi`, import test | none | automated |
| 2 | score the **released checkpoint** on the published benchmarks | GPU-minutes | **hard gate** — must land within 10 % relative of the published headline metric |
| 3 | **smoke train**: 1 epoch on a small subset, log plus a checkpoint that loads | under 1 GPU-hour | reviewed by hand |
| 4 | full retrain on the **released** training data | expensive | starts only after stage 3 is merged |
| 5 | retrain on the supplied training-set variants | expensive | starts only after stage 4 lands |

Stages 1 to 3 need nothing from us except your assignment, so you can start today. The stage-5
variants arrive later.

## How to work

```bash
git clone https://github.com/kongwoang/Intro2DS.git
cd Intro2DS
cp -r submissions/_TEMPLATE submissions/<your-id>      # id is in your assignment
git checkout -b sub/<your-id>
# ... fill in one stage, commit, push, open a PR ...
```

Read [`CONTRACT.md`](CONTRACT.md) first. It is short, and the things it forbids are the things that
have actually wasted GPU time on this kind of work before.

Your assignment is in [`assignments/`](assignments/). Report format is
[`report_schema.json`](report_schema.json).

## Asking questions

If something in the upstream repository is broken, ambiguous, or undocumented, **say so in an issue
and stop**. Do not patch around it and do not guess. A documented "this does not reproduce and here
is the evidence" is a useful result for this study; a silently patched pipeline is not.
