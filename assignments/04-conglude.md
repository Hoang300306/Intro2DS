# Assignment 04-conglude — ConGLUDe

**Model:** ConGLUDe (ICML 2026)
**Upstream repository:** https://github.com/ml-jku/conglude
**Released artifacts:** training data on Zenodo (record 20354834); checkpoint under checkpoints/
**Benchmarks to evaluate on:** whatever the paper reports; list them in your report

You are one of several people each working on a different published model. You do not need to know
anything about the others, and nothing in this sheet depends on them.

## What this study is

We are measuring two things about published structure-based virtual screening models: how well the
reported numbers reproduce from the released artifacts, and how sensitive those numbers are to the
exact composition of the training set. Your job is to produce trustworthy numbers for ConGLUDe. The
interpretation is not your problem — accuracy and honest reporting are.

## Your directory

```
submissions/04-conglude/
```

Copy it from `submissions/_TEMPLATE`, work on branch `sub/04-conglude`, and open a pull request for each
stage. Read [`../CONTRACT.md`](../CONTRACT.md) before you start; it is short.

## Stage 1 — environment

Build the environment the repository asks for. Commit to `stage1_env/`:

- `environment.yml` or `requirements.txt` — the **resolved** versions, not the loose spec
- `nvidia-smi.txt` — the output, verbatim
- `import_test.txt` — output of importing the model package and printing `torch.__version__`,
  `torch.version.cuda`, `torch.cuda.is_available()`

Then produce `code/frozen.sha256`: a `sha256sum` listing of the files that compute metrics and load
benchmarks. For this repository that means the evaluation entry point and its metric computation. Err on the side of listing too many.
**Generate this once, at stage 1, and never regenerate it** — CI uses it to detect whether the
evaluation path moved under your feet.

Cost: no GPU time.

## Stage 2 — reproduce the published numbers  ← hard gate

Score the **released checkpoint** on the benchmarks above, using the repository's own evaluation
script, unmodified.

Then find the corresponding number **in the paper**, and record in `summary.json` under
`published_reference`: the metric, the value, where it is stated (table or section), and your
relative gap.

**The gate is 10 % relative on the headline metric.** Outside that, open an issue with your evidence
and stop — do not proceed to stage 3. A documented non-reproduction is a real result for this study
and is not a failure on your part.

Deliver: `stage2_reproduce/summary.json`, `stage2_reproduce/per_target.csv` (one row per target —
see the contract, this is not optional), `stage2_reproduce/log.txt`.

Cost: GPU-minutes.

## Stage 3 — smoke train  ← review gate

Train for **one epoch on a small subset** (a few hundred training groups is plenty). The point is
not a useful model; it is proof that the training path runs end to end on your machine, writes a
checkpoint, and that the checkpoint loads.

Deliver: `stage3_smoke/summary.json`, `stage3_smoke/log.txt`, and in `REPORT.md` the time for one
epoch plus your **extrapolated cost for the full run**. We budget the expensive stages from that
number, so think about it rather than guessing.

Cost: under one GPU-hour. **Stop here and wait for your pull request to be merged.**

## Stage 4 — full retrain on the released training data

Only after stage 3 is merged. Retrain from scratch on the **released** training data with the
released configuration, fixed seed, and the checkpoint selection rule the repository specifies.

This is the baseline the next stage is compared against, so it is half the measurement, not a
formality.

Deliver: `stage4_taskB/summary.json` (with `training_data.split_id = "released"`),
`per_target.csv`, `log.txt`, and the checkpoint's SHA256.

## Stage 5 — retrain on the supplied training-set variants

We will give you one or more training-set variants as drop-in replacements for the released training
files, with a manifest of SHA256 digests. For each one: retrain with **the same seed, the same
configuration and the same checkpoint rule as stage 4**, changing nothing but the input files.

Verify the digests of what you received and record them. CI checks them against the manifest.

Deliver one subdirectory per variant under `stage5_taskC/`, each with `summary.json`,
`per_target.csv` and `log.txt`.

## What makes this submission good

Per-target numbers, an empty or fully justified diff, one seed throughout, a stated checkpoint rule,
and an honest `Problems` section. Not a flattering number.
