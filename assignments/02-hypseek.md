# 02-hypseek — HypSeek

| | |
|---|---|
| **Model** | HypSeek (AAAI 2026) |
| **Upstream repository** | https://github.com/jianhuiwemi/HypSeek |
| **Benchmarks** | DUD-E, DEKOIS 2.0, and the FEP sets the repository evaluates on |
| **Your directory** | `submissions/02-hypseek/` |
| **Your branch** | `sub/02-hypseek` |

You are one of several people, each on a different model. Nothing here depends on anyone else.

## Getting set up

```bash
git clone https://github.com/kongwoang/Intro2DS.git
cd Intro2DS
cp -r submissions/_TEMPLATE submissions/02-hypseek
git checkout -b sub/02-hypseek
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
composition of the training set. Your job is to produce **trustworthy numbers** for HypSeek.
Interpretation is not your problem; accuracy and honest reporting are.

## Released artifacts

The repository consumes the training corpus released by LigUnity (figshare 10.6084/m9.figshare.27966819, `train_label.zip` plus the blend LMDBs). A checkpoint is provided by the repository.

## What we already checked for you

Read from the repository's own code: `unimol/tasks/train_task.py` lines 514-515 load `train_label_pdbbind_seq.json` and `train_label_blend_seq_full.json` from `self.args.data`. Those are exactly the two files each variant supplies.

## Notes specific to this model

The training task also reads `test_datasets/dude.json`, `PCBA.json` and `dekois.json` and applies its own filter to the training assays, so **the test index files are required for training, not only for evaluation**. Leave that filter alone: it must run identically on the released data and on every variant, or checkpoint B and checkpoint C stop being comparable. `--protein-similarity-thres 1.0` in `train.sh` is part of that released configuration — do not change it.

**`train.sh` takes a mode, and you want `CASF`.** `bash train.sh CASF` validates on CASF-2016 and sets `--best-checkpoint-metric valid_bedroc --maximize-best-checkpoint-metric`, so the repository's own rule is best-by-`valid_bedroc`; score that and say so. `bash train.sh FEP` passes `--disable-validation` and sets no best-checkpoint metric, so it produces no best checkpoint at all — it is the affinity-ranking configuration, not the screening one.

**Two placeholders will stop you before anything runs**: `save_root="/save_root"` in `train.sh` is an absolute path at the filesystem root, which `mkdir -p` cannot create without root, and `DATA_ROOT=/path/test_datasets` in `test.sh` is not a real path. Set both to directories you own.

**`test.sh` runs `pip uninstall -y unimol` on every invocation.** That is deliberate on their side — it forces the local `unimol/` to win on `PYTHONPATH` — but it does mutate your environment, so your A1 `import_test.txt` may stop reproducing after you first evaluate. Note it in `REPORT.md` rather than treating it as a broken environment.

The evaluation prints per-target lines — `[DUD-E] <target> AUC=... BEDROC=... EF=[...]` — so `per_target.csv` needs no patch here. It also writes `saved_mols_embed.npy` and friends **per target**, which is tens of gigabytes over DUD-E and LIT-PCBA. Keep it out of git and budget the disk in A3.

The global batch is 96: four GPUs × `batch_size=24` × `update_freq=1`. If you have fewer GPUs, keep the product at 96 by raising `--update-freq` rather than training on fewer GPUs, which would change the contrastive batch.

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
benchmarks. For this repository that means `unimol/tasks/test_task.py`, the metric helpers it calls, and the test-dataset loader. Err on the side of listing too many files.

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

Your variants replace both `train_label_blend_seq_full.json` and `train_label_pdbbind_seq.json` in the training data directory.

**You build them yourself from the released data you already have.** We do not send you training
data: a variant differs from the released file only in which records it keeps, so you get a small
spec saying which — tens of kilobytes instead of a gigabyte — and build it locally.

The exact commands for your variants:


**`pocketaffdb_ligandish`** — 2 files to build (run the command once per file):

```bash
# replaces the released file: train_label_blend_seq_full.json
python tools/apply_variant.py \
    splits/specs/pocketaffdb_ligandish/pocketaffdb_ligandish__train_label_blend_seq_full.json.gz \
    <your released train_label_blend_seq_full.json> \
    <where to write it>
# replaces the released file: train_label_pdbbind_seq.json
python tools/apply_variant.py \
    splits/specs/pocketaffdb_ligandish/pocketaffdb_ligandish__train_label_pdbbind_seq.json.gz \
    <your released train_label_pdbbind_seq.json> \
    <where to write it>
```

**`pocketaffdb_ligandish_matched`** — 2 files to build (run the command once per file):

```bash
# replaces the released file: train_label_blend_seq_full.json
python tools/apply_variant.py \
    splits/specs/pocketaffdb_ligandish_matched/pocketaffdb_ligandish_matched__train_label_blend_seq_full.json.gz \
    <your released train_label_blend_seq_full.json> \
    <where to write it>
# replaces the released file: train_label_pdbbind_seq.json
python tools/apply_variant.py \
    splits/specs/pocketaffdb_ligandish_matched/pocketaffdb_ligandish_matched__train_label_pdbbind_seq.json.gz \
    <your released train_label_pdbbind_seq.json> \
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
