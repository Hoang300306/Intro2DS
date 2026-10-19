# 01-drugclip — DrugCLIP

| | |
|---|---|
| **Model** | DrugCLIP (NeurIPS 2023) |
| **Upstream repository** | https://github.com/bowen-gao/DrugCLIP |
| **Benchmarks** | DUD-E and LIT-PCBA |
| **Your directory** | `submissions/01-drugclip/` |
| **Your branch** | `sub/01-drugclip` |

You are one of several people, each on a different model. Nothing here depends on anyone else.

## Getting set up

```bash
git clone https://github.com/kongwoang/Intro2DS.git
cd Intro2DS
cp -r submissions/_TEMPLATE submissions/01-drugclip
git checkout -b sub/01-drugclip
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
composition of the training set. Your job is to produce **trustworthy numbers** for DrugCLIP.
Interpretation is not your problem; accuracy and honest reporting are.

## Released artifacts

Training data and a checkpoint are linked from the repository README (Google Drive). The training directory is `train_no_test_af/`, holding `train.lmdb`, `valid.lmdb`, `dict_mol.txt` and `dict_pkt.txt`.

## What we already checked for you

We hold the released `train_no_test_af/train.lmdb` and confirmed it contains 66,164 records, each a dict with `smi` (the ligand SMILES) and `pocket` (a four-character PDB code). The variants were produced from that exact file.

## Notes specific to this model

This model is the origin of a family of later methods, so its evaluation code is the one several other repositories inherit. Be especially careful not to modify it.

**`test_dude` prints only the macro means.** It loops over targets, collects `auc_list` and the EF dictionaries, and prints `np.mean` of each — the per-target values and the target names never reach the log. `per_target.csv` therefore needs the print-only patch described under A2. Print the target name inside the loop: the target order comes from `os.listdir`, which is filesystem order and not reproducible across machines, so position in the list is not an id.

For the same reason, make sure `./data/DUD-E/raw/all/` and `./data/lit_pcba/` contain **only** target directories. Both are read with `os.listdir` and every entry is treated as a target, so a stray `.DS_Store` or a leftover archive becomes a target and either crashes the run or lands in the mean.

**The checkpoint rule is best-by-`valid_bedroc`**: `drugclip.sh` passes `--best-checkpoint-metric valid_bedroc --maximize-best-checkpoint-metric`, and `test.sh` scores `checkpoint_best.pt`. State that and its source (contract rule 4).

**Both scripts hard-code a GPU index** — `CUDA_VISIBLE_DEVICES="1"` in `drugclip.sh`, and other indices elsewhere. Set it to a device you actually have; on a shared machine the wrong index is somebody else's job. The paths are relative, so run from the repository root. `TASK` in `test.sh` takes one benchmark at a time — run it once per benchmark.

The epoch budget is real: `epoch=200` with `--patience 2000`, so patience never fires and 200 epochs is what it will do. The global batch is 48 — `batch_size=48`, one GPU, `update_freq=1`. Keep it the same at B and C.

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
benchmarks. For this repository that means the benchmark loader and the metric functions (enrichment factor, BEDROC, AUROC) in the test/evaluation script. Err on the side of listing too many files.

**Generate it once, here, and never regenerate it.** CI uses it to detect whether the evaluation
path moved under your feet between checkpoints.

**If a pin no longer resolves**, move it to the nearest published release of the same line, record
it in `code/changes.diff` with its one-line justification, and carry on — a pinned pre-release that
has since been removed from PyPI is an index fact, not a change to the model. Do not loosen the pin
to a range, and do not spend hours compiling a source-only package that nothing in the repository
imports: check whether it is imported, and if it is not, drop it and say so. If the package is
imported, or you are unsure, open an issue instead of guessing.

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
written row so that per-target numbers come out (contract rule 3 says so already), and registering
the benchmark you were asked to score. They appear in `code/changes.diff` like any other edit, with
a one-line justification in `REPORT.md` (contract rule 2) — do not invent a second patch file, and
do put the diff's sha256 in `summary.json`, which is how CI holds it identical across A2, B and C.
Do not touch how a metric is computed, which molecules are scored, or the order they are scored in
— a patch that changes a number is a different experiment. If you think you need any other change,
open an issue first.

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

Your variants replace `train.lmdb` inside the released `train_no_test_af/` directory.

**You build them yourself from the released data you already have.** We do not send you training
data: a variant differs from the released file only in which records it keeps, so you get a small
spec saying which — tens of kilobytes instead of a gigabyte — and build it locally.

The exact commands for your variants:


**`drugclip_train_ligandish`** — 1 file to build:

```bash
# replaces the released file: train.lmdb
python tools/apply_variant.py \
    splits/specs/drugclip_train_ligandish/drugclip_train_ligandish.json.gz \
    <your released train.lmdb> \
    <where to write it>
```

**`drugclip_train_ligandish_matched`** — 1 file to build:

```bash
# replaces the released file: train.lmdb
python tools/apply_variant.py \
    splits/specs/drugclip_train_ligandish_matched/drugclip_train_ligandish_matched.json.gz \
    <your released train.lmdb> \
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
