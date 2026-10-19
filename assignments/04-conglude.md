# 04-conglude — ConGLUDe

| | |
|---|---|
| **Model** | ConGLUDe (ICML 2026) |
| **Upstream repository** | https://github.com/ml-jku/conglude |
| **Benchmarks** | DUD-E and LIT-PCBA (`python download_data.py --dataset_name vs`). The repository also evaluates pocket-prediction and pocket-ranking sets; you are not asked for those, so do not download them |
| **Your directory** | `submissions/04-conglude/` |
| **Your branch** | `sub/04-conglude` |

You are one of several people, each on a different model. Nothing here depends on anyone else.

## Getting set up

```bash
git clone https://github.com/kongwoang/Intro2DS.git
cd Intro2DS
cp -r submissions/_TEMPLATE submissions/04-conglude
git checkout -b sub/04-conglude
```

**Read [`../CONTRACT.md`](../CONTRACT.md) before you touch a GPU.** It is ten short rules and every
one of them exists because breaking it has already cost somebody a run.

Everything you are told to run is in this repository: `tools/apply_variant.py` builds your variants,
and `scripts/validate_submission.py` is what CI runs on your pull request — run it yourself first,
it is quicker than waiting for CI.

Questions and problems go in a **GitHub issue in this repository**, titled with your id, not a DM:
an answer one person needs is usually one everybody needs.

## What this study is

We are measuring two things about published virtual-screening models: how well their reported
numbers reproduce from the released artifacts, and how sensitive those numbers are to the exact
composition of the training set. Your job is to produce **trustworthy numbers** for ConGLUDe.
Interpretation is not your problem; accuracy and honest reporting are.

## Released artifacts

Training data on Zenodo (record 20354834) — the archives `LB_train_val.zip` and `SB_train_val.zip`. A checkpoint is under `checkpoints/` in the repository.

## What we already checked for you

We hold `LB_train_val.zip` from Zenodo (7,855,192,840 bytes, md5 `cfd25fcc9c275b994560b0a6d0dd1147`, matching the record) and built the variants from the `raw/smiles_files` tree inside it. Note that `unzip` 6.00 cannot read this archive because it is zip64 and over 4 GB; use 7-Zip or Python's `zipfile`, and check the md5 rather than trusting `unzip -t`.

## Notes specific to this model

**This corpus needs one extra step, and getting it wrong would invalidate your comparison.** The release ships a `processed/` tree — `ligand_embeddings/descriptors.dat` at 1.5 GB and per-target graphs at 6.3 GB — which is *derived* from the raw SMILES files. A variant changes those SMILES, so `processed/` has to be regenerated from them.

That means you must **also regenerate `processed/` for your checkpoint-B run**. If checkpoint B uses the shipped features and checkpoint C uses freshly computed ones, any change in the feature code between the release and today becomes a difference between your two arms, and the comparison stops meaning anything. Regenerate both, from the same code, and say in your report that you did.

The README will tell you the opposite: *all downloaded datasets except `SB_train_val` are already processed*. That is true of the released data and false of yours. The order is download, then apply the variant to `raw/smiles_files`, then `python process_data.py --dataset_name LB_train_val`, then train — and check that the `processed/` files really changed. Skip the reprocessing and the run trains on the shipped features, which still contain every molecule the variant removed; it will finish cleanly and tell you the variant made no difference.

Budget for this in your A3 estimate — it is real cost on top of training, and it is one reason your A3 number matters.

Only the LB (ligand-based) training data is covered. If the paper's main results use the SB variant, say so at A2 and stop before C.

**`python eval.py` as shipped does not evaluate DUD-E or LIT-PCBA.** In `configs/datamodule/test_datasets/test_datasets.yaml` every dataset except `default_vs` is commented out, and `default_vs` is a generic placeholder pointing at a directory that is not a benchmark. Uncomment `dude` and `litpcba`, drop `default_vs` from the list, and commit that diff as `code/eval_setup.patch`. Run verbatim it would hand you confident numbers for the wrong data.

**Pass `logger=null` when you train.** `configs/train.yaml` defaults to `logger: wb`, so a training run will try to log to Weights & Biases and will upload the resolved config, variant name included, to whatever workspace is configured on the machine.

**The seed is a Hydra field**: `python train.py seed=0`. Note that `configs/trainer/default.yaml` sets `deterministic: false`, so two runs with the same seed are not bitwise identical. Do not try to fix that — record it as a limitation.

**The checkpoint rule is already unambiguous**: `save_top_k: 1`, `save_last: false`, `monitor: avg_val/virtual_screening/bedroc`, `mode: max`. Exactly one checkpoint survives and it is the best by that metric. State it with its source.

**Cost: `max_epochs: 500` with early stopping at `patience: 50`.** Report the per-epoch time and the epoch your B run actually stopped at. B and C may stop at different epochs; that is the model's own rule, so record it and do not override it.

**Only EF@5% is computed.** `model.py` builds `VirtualScreeningMetrics(ef_fractions=[0.05])`, so leave `ef_0.005`, `ef_0.01` and `ef_0.02` empty and do not add cutoffs. BEDROC is at alpha 80.5. The metric class writes a per-target CSV itself, so `per_target.csv` needs no patch — point at the file it produces.

The repository also ships agent command files under `.claude/`. If you use them, the evaluation path and the configs are still yours to keep fixed, and `code/frozen.sha256` is what will catch it if they are not.

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

**Name the metric first.** In `summary.json`, `published_reference.metric` is the metric **the paper
itself leads with** for that benchmark, and `published_reference.source` is the table or section it
appears in. Decide that before you compute your gap — the gate applies to the metric you named, so
picking it afterwards would just be choosing whichever number happens to pass.

**The gate is 10 % relative on that metric.** Outside it: open an issue with your evidence and stop.
Do not go on to A3. A documented non-reproduction is a real result and is not your fault.

Deliver `summary.json`, `per_target.csv` and `log.txt`.

`per_target.csv` has one row per benchmark target and is **not optional** (contract rule 3).
`target_id`, `n_actives` and `n_total` are always required. The metric columns — `auroc`, `bedroc`,
`ef_0.005`, `ef_0.01`, `ef_0.02`, `ef_0.05` — are required **wherever your model's own evaluation
computes them**; leave a column empty where it does not, and say which in `REPORT.md`. Do not invent
a metric your repository does not compute, and do not quietly substitute a different one.

**Also record the BEDROC alpha your code uses.** Implementations differ — 80.5 and 85 are both in
use — and one is not comparable with the other. Read it out of the code, do not assume it.

**And record the SHA256 of the checkpoint file you scored.** Some of these repositories load each
other's checkpoints without raising anything at all, so a mix-up produces numbers rather than an
error.

**Two kinds of edit to the evaluation path are allowed**, and only these two: adding a print or a
written row so that per-target numbers come out, and registering the benchmark you were asked to
score. Commit the diff as `code/eval_setup.patch`, list it in `REPORT.md`, and keep it byte-identical
across A2, B and C. Do not touch how a metric is computed, which molecules are scored, or the order
they are scored in — a patch that changes a number is a different experiment. If you think you need
any other change, open an issue first.

## A3 — smoke train, and the cost estimate  → `A3_smoke/`

Train for **one epoch on a small subset** — a few hundred training groups is plenty. The point is
not a useful model. The point is proof that the training path runs end to end on your machine,
writes a checkpoint, and that the checkpoint loads again.

Deliver `summary.json`, `log.txt`, and in `REPORT.md`:

- how long one epoch took, on what subset
- **your extrapolated cost for checkpoint B, and for each variant in checkpoint C**
- what you extrapolated from
- **what stops a run** — a fixed epoch count, or an early-stopping rule — and where that is
  configured. Several of these repositories set a high epoch cap and rely on early stopping, so
  multiplying your per-epoch time by the cap can overstate the cost many times over, while assuming
  an early stop is a guess. Say which you did.

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

Your variants replace the raw SMILES directory `LB_train_val/raw/smiles_files` of the ligand-based training corpus.

**You build them yourself from the released data you already have.** We do not send you training
data: a variant differs from the released file only in which records it keeps, so you get a small
spec saying which — tens of kilobytes instead of a gigabyte — and build it locally.

The exact commands for your variants:


**`conglude_ligandish`** — 1 file to build:

```bash
# replaces the released directory: LB_train_val/raw/smiles_files
python tools/apply_variant.py \
    splits/specs/conglude_ligandish/conglude_ligandish.json.gz \
    <your released LB_train_val/raw/smiles_files> \
    <where to write it>
```

**`conglude_ligandish_matched`** — 1 file to build:

```bash
# replaces the released directory: LB_train_val/raw/smiles_files
python tools/apply_variant.py \
    splits/specs/conglude_ligandish_matched/conglude_ligandish_matched.json.gz \
    <your released LB_train_val/raw/smiles_files> \
    <where to write it>
```

The script verifies your released input against the digest we recorded **before it writes
anything**, and verifies its own output afterwards. Both passing proves your download matches the
input the variant was built from and that the filter applied correctly.

It needs the `lmdb` package only for LMDB specs, which your model already requires.

If a check fails: **stop, do not train, open an issue with the message.** Usually it means your
released download differs from ours, which is itself worth knowing.

Then, for each variant:

1. Retrain with **the same seed, configuration, checkpoint rule and code as checkpoint B**. The input
   file is the only thing that changes (contract rule 6).
2. Record `split_id` and the digests the script printed in `summary.json`. CI compares them against
   `splits/MANIFEST.json`.
3. Deliver `C_retrain_variants/<variant_id>/` with `summary.json`, `per_target.csv` and `log.txt`.

---

## What makes a good submission

Per-target numbers. An empty or fully justified diff. One seed throughout. A stated checkpoint rule
with its source. An honest `Problems` section.

Not a flattering number.
