# 04-conglude — ConGLUDe

| | |
|---|---|
| **Model** | ConGLUDe (ICML 2026) |
| **Upstream repository** | https://github.com/ml-jku/conglude |
| **Benchmarks** | the benchmarks the paper reports; list them explicitly in your report |
| **Your directory** | `submissions/04-conglude/` |
| **Your branch** | `sub/04-conglude` |

You are one of several people, each on a different model. Nothing here depends on anyone else.

## What this study is

We are measuring two things about published virtual-screening models: how well their reported
numbers reproduce from the released artifacts, and how sensitive those numbers are to the exact
composition of the training set. Your job is to produce **trustworthy numbers** for ConGLUDe.
Interpretation is not your problem; accuracy and honest reporting are.

## Released artifacts

Training data on Zenodo (record 20354834) — the archives `LB_train_val.zip` and `SB_train_val.zip`. A checkpoint is under `checkpoints/` in the repository.

## What we already checked for you

Nothing about the internal file layout is confirmed yet: the Zenodo archives are still being retrieved on our side. The repository and the Zenodo record are confirmed to exist, with the file list above.

## Notes specific to this model

**Your checkpoint C will be assigned later than the others**, because we are still preparing this corpus. Checkpoints A and B need nothing from us, so start on those now and do not wait. When you reach A1, tell us in your report which training files the repository actually reads — that is useful to us and it is the fastest way to get your C variants built correctly.

---

# Checkpoint A — environment, reproduction, smoke train

**One pull request containing all three parts.** Total cost: under one GPU-hour. Nothing expensive
happens until this is merged.

## A1 — environment  → `A1_env/`

Build the environment the repository asks for, then commit:

| file | content |
|---|---|
| `environment.yml` or `requirements.txt` | the **resolved** versions, not the loose spec |
| `nvidia-smi.txt` | the output, verbatim |
| `import_test.txt` | importing the model package, then printing `torch.__version__`, `torch.version.cuda`, `torch.cuda.is_available()` |
| `summary.json` | `stage` = `"A1_env"` |

Then produce `code/frozen.sha256` — a `sha256sum` listing of the files that compute metrics and load
benchmarks. For this repository that means the evaluation entry point and its metric computation. Err on the side of listing too many files.

**Generate it once, here, and never regenerate it.** CI uses it to detect whether the evaluation
path moved under your feet between checkpoints.

## A2 — reproduce the published numbers  → `A2_reproduce/`   ← hard gate

Score the **released checkpoint** on the benchmarks above, using the repository's own evaluation
script, unmodified.

Then find the matching number **in the paper** and record, in `summary.json` under
`published_reference`: the metric, its value, **which table or section it comes from**, and your
relative gap.

**The gate is 10 % relative on the headline metric.** Outside that: open an issue with your evidence
and stop. Do not go on to A3. A documented non-reproduction is a real result and is not your fault.

Deliver `summary.json`, `per_target.csv` (one row per target — see contract rule 3, this is not
optional) and `log.txt`.

## A3 — smoke train, and the cost estimate  → `A3_smoke/`

Train for **one epoch on a small subset** — a few hundred training groups is plenty. The point is
not a useful model. The point is proof that the training path runs end to end on your machine,
writes a checkpoint, and that the checkpoint loads again.

Deliver `summary.json`, `log.txt`, and in `REPORT.md`:

- how long one epoch took, on what subset
- **your extrapolated cost for checkpoint B, and for each variant in checkpoint C**
- what you extrapolated from

We approve the expensive runs from that estimate. A careless guess here gets your run either blocked
or wasted.

**Then stop. Open the pull request and wait for it to be merged.**

---

# Checkpoint B — retrain on the released training data  → `B_retrain_released/`

Only after checkpoint A is merged.

Retrain from scratch on the **released** training data, with the released configuration, a fixed
seed, and the checkpoint selection rule the repository specifies (contract rule 4).

This is the baseline that checkpoint C is compared against. It is half the measurement, not a
formality.

Deliver `summary.json` with `training_data.split_id = "released"`, `per_target.csv`, `log.txt`, and
the checkpoint's SHA256.

In `REPORT.md`, say how your retrained model compares to the released checkpoint you scored at A2.
They are usually **not** identical, and that difference is worth recording rather than hiding.

---

# Checkpoint C — retrain on the supplied variants  → `C_retrain_variants/<variant_id>/`

Only after checkpoint B is merged.

You will receive these variants:

- `conglude_ligandish`
- `conglude_ligandish_matched`

Each variant is a drop-in replacement for TO BE CONFIRMED — the exact training file names will be in the variant manifest before checkpoint C is assigned to you.

For every variant:

1. **Verify the digests first.** `sha256sum` each file you received and compare against
   `splits/MANIFEST.json`. CI checks the digests you record. A run on the wrong input is worse than
   no run, because it looks valid.
2. Retrain with **the same seed, configuration, checkpoint rule and code as checkpoint B**. The
   input file is the only thing that changes (contract rule 6).
3. Deliver `C_retrain_variants/<variant_id>/` containing `summary.json` (with `training_data` filled
   in), `per_target.csv` and `log.txt`.

---

## What makes a good submission

Per-target numbers. An empty or fully justified diff. One seed throughout. A stated checkpoint rule
with its source. An honest `Problems` section.

Not a flattering number.
