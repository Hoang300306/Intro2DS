# 05-bindclip — BindCLIP

| | |
|---|---|
| **Model** | BindCLIP (KDD 2026) |
| **Upstream repository** | https://github.com/anjie-qiao/BindCLIP |
| **Benchmarks** | DUD-E, LIT-PCBA and FEP+ |
| **Your directory** | `submissions/05-bindclip/` |
| **Your branch** | `sub/05-bindclip` |

You are one of several people, each on a different model. Nothing here depends on anyone else.

## Getting set up

```bash
git clone https://github.com/kongwoang/Intro2DS.git
cd Intro2DS
cp -r submissions/_TEMPLATE submissions/05-bindclip
git checkout -b sub/05-bindclip
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
composition of the training set. Your job is to produce **trustworthy numbers** for BindCLIP.
Interpretation is not your problem; accuracy and honest reporting are.

## Released artifacts

Training data and checkpoints are on Google Drive, linked from the README. The training archives are `train_with_hardneg_by_random.7z` and `train_with_hardneg_by_unimol.7z`, each containing one LMDB.

## What we already checked for you

We hold both released archives and confirmed each LMDB has 66,164 records carrying `smi` and `pocket` plus the extra fields this model needs (`docked_coordinates`, `scaffold_fp`, `decoy_smi`, `decoy_mol`, `decoy_atoms`, `decoy_coordinates`, `decoy_scaffold_fp`). The variants were produced from those exact files and preserve every field untouched.

## Notes specific to this model

**The README says this model uses the same training split as a predecessor, but the file is not the same file** — each record additionally carries a binding pose and a negative molecule. Do not substitute another repository's LMDB; use the variants built for yours.

There are **two** released training versions (random and UniMol hard negatives) and we built variants for both. **Say in your A2 report which version the paper's main results use, and use that one throughout** — B and C must use the same version.

So although four variants are listed below, **you run only the two matching your chosen version**: two retrains at checkpoint C, not four. The other two exist so that whichever version you pick is already prepared; leave them alone and say in your report which pair you used. Only the primary ligand field was filtered when the variants were built; `decoy_smi` was left untouched, deliberately, so that your variants differ from the other corpora's in one way only.

**The README invites you to score both released checkpoints and keep whichever performs best for your use case. Do not do that here.** Choosing a checkpoint by its score on the benchmark you are reporting is how a number stops meaning anything. Pick the one the paper's main table used, decide that before you score, and if you score both, report both and say which is which.

**The README carries no numeric results table** — the paper is the only source for your `published_reference`, at `10.1145/3770855.3819040`. Do not read a value off a figure.

**RDKit 2022.09.5 is required**, and the README warns that another version may mean reprocessing the training data. Do not reprocess it: your variant is a filter over the released file, and the applier's digest check is what tells you whether the file you hold is the file the variant was built from.

**The effective batch is 192**, not its predecessor's 48: `bindclip.sh` sets `batch_size=48` with `update_freq=4` on one GPU. Keep it. The script's seed is 42 while the released checkpoint filenames mention other seeds, so set `weight_path` to the file you actually downloaded and keep your own training seed fixed at the script's value.

**This model's checkpoints load in its predecessor's inference code without complaint**, which the README presents as a feature. It also means a wrong `--path` yields numbers instead of an error, so the checkpoint SHA256 that A2 asks for is doing real work here. The evaluation inherits `test_dude`, which prints only the macro means, so `per_target.csv` needs the print-only patch described under A2.

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
benchmarks. For this repository that means the evaluation script and its metric functions. Err on the side of listing too many files.

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

Your variants replace the LMDB inside whichever training archive you use — `train_with_hardneg_by_random.lmdb` or `train_with_hardneg_by_unimol.lmdb`.

**You build them yourself from the released data you already have.** We do not send you training
data: a variant differs from the released file only in which records it keeps, so you get a small
spec saying which — tens of kilobytes instead of a gigabyte — and build it locally.

The exact commands for your variants:


**`bindclip_random_ligandish`** — 1 file to build:

```bash
# replaces the released file: train_with_hardneg_by_random.lmdb
python tools/apply_variant.py \
    splits/specs/bindclip_random_ligandish/bindclip_random_ligandish.json.gz \
    <your released train_with_hardneg_by_random.lmdb> \
    <where to write it>
```

**`bindclip_random_ligandish_matched`** — 1 file to build:

```bash
# replaces the released file: train_with_hardneg_by_random.lmdb
python tools/apply_variant.py \
    splits/specs/bindclip_random_ligandish_matched/bindclip_random_ligandish_matched.json.gz \
    <your released train_with_hardneg_by_random.lmdb> \
    <where to write it>
```

**`bindclip_unimol_ligandish`** — 1 file to build:

```bash
# replaces the released file: train_with_hardneg_by_unimol.lmdb
python tools/apply_variant.py \
    splits/specs/bindclip_unimol_ligandish/bindclip_unimol_ligandish.json.gz \
    <your released train_with_hardneg_by_unimol.lmdb> \
    <where to write it>
```

**`bindclip_unimol_ligandish_matched`** — 1 file to build:

```bash
# replaces the released file: train_with_hardneg_by_unimol.lmdb
python tools/apply_variant.py \
    splits/specs/bindclip_unimol_ligandish_matched/bindclip_unimol_ligandish_matched.json.gz \
    <your released train_with_hardneg_by_unimol.lmdb> \
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
