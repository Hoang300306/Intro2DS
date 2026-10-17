# 07-sprint — SPRINT

| | |
|---|---|
| **Model** | SPRINT (MLSB @ NeurIPS 2024) |
| **Upstream repository** | https://github.com/abhinadduri/panspecies-dti |
| **Benchmarks** | the DTI benchmarks the repository evaluates on (`--task`), and the Lit-PCBA target list in `targets.txt`; list in your report exactly what you ran |
| **Your directory** | `submissions/07-sprint/` |
| **Your branch** | `sub/07-sprint` |

You are one of several people, each on a different model. Nothing here depends on anyone else.

## Getting set up

```bash
git clone https://github.com/kongwoang/Intro2DS.git
cd Intro2DS
cp -r submissions/_TEMPLATE submissions/07-sprint
git checkout -b sub/07-sprint
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
composition of the training set. Your job is to produce **trustworthy numbers** for SPRINT.
Interpretation is not your problem; accuracy and honest reporting are.

## Released artifacts

The MERGED training corpus is fetched by `data/MERGED/huge_data/download.sh`. It unpacks to TSV files plus three `id_to_*.npy` lookup maps. Checkpoint links are in `checkpoints/README.md`.

## What we already checked for you

We hold the released corpus and confirmed the training files are TSVs of `ligand` (an integer id) and `aa_seq` (a UniProt accession), with SMILES resolved through `id_to_smiles.npy` (3,529,822 entries). The training files hold 574,754 positive and 64,361,902 negative rows over 312 and 289 targets. The variants were produced from those exact files, and the validation and test TSVs are unchanged.

## Notes specific to this model

**This is the largest corpus in the study: about 64.9 million training rows.** Your A3 cost extrapolation matters more here than for anyone else, because we will decide the scope of your B and C runs from it. If a full run is impractical, say so at A3 with your numbers and we will reduce the scope — that is a normal outcome, not a failure.

This model predicts binary interaction from sequence rather than ranking within a pocket, so the variants were built to preserve the **positive-to-negative ratio exactly** (identical to eight decimal places between the two variants). Do not resample or rebalance the classes: that would change the thing the variants hold fixed.

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
benchmarks. For this repository that means the evaluation path and its metric functions. Err on the side of listing too many files.

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

Your variants replace `merged_pos_uniq_train_rand.tsv` and `merged_neg_uniq_train_rand.tsv` in the MERGED data directory.

**You build them yourself from the released data you already have.** We do not send you training
data: a variant differs from the released file only in which records it keeps, so you get a small
spec saying which — tens of kilobytes instead of a gigabyte — and build it locally.

The exact commands for your variants:


**`sprint_merged_ligandish`** — 2 files to build (run the command once per file):

```bash
# replaces the released file: merged_neg_uniq_train_rand.tsv
python tools/apply_variant.py \
    splits/specs/sprint_merged_ligandish/sprint_merged_ligandish__merged_neg_uniq_train_rand.tsv.json.gz \
    <your released merged_neg_uniq_train_rand.tsv> \
    <where to write it>
# replaces the released file: merged_pos_uniq_train_rand.tsv
python tools/apply_variant.py \
    splits/specs/sprint_merged_ligandish/sprint_merged_ligandish__merged_pos_uniq_train_rand.tsv.json.gz \
    <your released merged_pos_uniq_train_rand.tsv> \
    <where to write it>
```

**`sprint_merged_ligandish_matched`** — 2 files to build (run the command once per file):

```bash
# replaces the released file: merged_neg_uniq_train_rand.tsv
python tools/apply_variant.py \
    splits/specs/sprint_merged_ligandish_matched/sprint_merged_ligandish_matched__merged_neg_uniq_train_rand.tsv.json.gz \
    <your released merged_neg_uniq_train_rand.tsv> \
    <where to write it>
# replaces the released file: merged_pos_uniq_train_rand.tsv
python tools/apply_variant.py \
    splits/specs/sprint_merged_ligandish_matched/sprint_merged_ligandish_matched__merged_pos_uniq_train_rand.tsv.json.gz \
    <your released merged_pos_uniq_train_rand.tsv> \
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
