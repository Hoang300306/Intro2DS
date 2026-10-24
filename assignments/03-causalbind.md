# 03-causalbind — CausalBind

| | |
|---|---|
| **Model** | CausalBind (NeurIPS 2026) |
| **Upstream repository** | https://github.com/lokali/CausalBind |
| **Benchmarks** | DUD-E, DEKOIS 2.0, LIT-PCBA and the FEP benchmarks |
| **Your directory** | `submissions/03-causalbind/` |
| **Your branch** | `sub/03-causalbind` |

You are one of several people, each on a different model. Nothing here depends on anyone else.

## Getting set up

```bash
git clone https://github.com/kongwoang/Intro2DS.git
cd Intro2DS
cp -r submissions/_TEMPLATE submissions/03-causalbind
git checkout -b sub/03-causalbind
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
composition of the training set. Your job is to produce **trustworthy numbers** for CausalBind.
Interpretation is not your problem; accuracy and honest reporting are.

## Released artifacts

`DATA.md` in the repository documents every input, its licence and where to get it. Training data comes from the LigUnity figshare corpus; several checkpoints are released.

## What we already checked for you

The `Expected Layout` section of `DATA.md` lists `data/train_label_blend_seq_full.json` and `data/train_label_pdbbind_seq.json`, which are exactly the two files each variant supplies.

## Notes specific to this model

**Read `DATA.md` before anything else** — it is unusually complete and will save you time. It also states that the training task reads the benchmark index files to filter training assays, so those are needed for training too.

**Two pins in `environment.yml` are already known to be bad, so do not spend time on them.** `hf-xet==1.1.5rc1` is not on PyPI — that release candidate became `1.1.5` on 2025-06-20 — and pip resolves the whole pip section before installing anything, so the one line fails the entire environment. Use `hf-xet==1.1.5`. It is the Xet transfer backend for `huggingface_hub`, which this repository calls only from `scripts/download_data.sh`, so it affects how bytes arrive and nothing else; no file in the repository imports it. `torchsort==0.1.10` is source-only on PyPI and will try to compile against your CUDA toolchain; nothing in the repository imports it either, so if that build fails, drop the line. Record both in `code/changes.diff`.

**Run the `sp` variant, and only `sp`.** The repository releases four — `sp`, `lr`, `emb`, `atom` — and `scripts/quick_start.sh` scores all four by default. `sp` is the variant the paper leads with (Theorem 2, first CausalBind row of Table 1) and the one the repository's own further analyses use; `atom` is described there as the interpretability variant. Use the same variant at A2, B and C: a number from one variant cannot be compared with a number from another, and doing all four would be twelve retrains instead of three. If your A3 estimate shows three `sp` runs do not fit your budget, say so in the pull request and propose `emb`, which is a smaller configuration — do not switch on your own.

**A session that dies does not cost you the run.** Uni-Core restores from `checkpoint_last.pt` in `--save-dir` on its own: `restore_file` defaults to that name, and because none of the `--reset-*` flags are set it brings back the optimizer, the LR scheduler and the dataloader position, not just the weights. So B and C can be run in chunks — as long as **`--save-dir` lives on storage that survives the session**, and you never delete `checkpoint_last.pt`. Say in your report how many times each arm was resumed.

**Score `checkpoint_last.pt`, not the best checkpoint.** `scripts/train.sh` passes `--keep-best-checkpoints 8`, so a best checkpoint will be sitting right next to it, but the paper and the README both report the last one. State that rule and its source in your report (contract rule 4).

**Name a DUD-E metric for the A2 gate.** DUD-E has 102 targets; LIT-PCBA has 15, and its EF@1% is volatile enough that the authors' own from-scratch run moved it by about a third while every DUD-E number stayed within 5 %. Score both benchmarks, but do not stake the 10 % gate on the small one. That same published comparison — the paper's numbers printed next to the repository's own from-scratch numbers — is also your yardstick at checkpoint B for how much drift is normal.

**If you do not have four 80 GB GPUs**, keep the global batch at 96: it is `N_GPU` × `--batch-size` × `--update-freq`, which the released configuration sets to 4 × 24 × 1. Lower the per-GPU batch and raise `--update-freq` to compensate. Do not just train on fewer GPUs — that changes the contrastive batch, which changes the model.

---

# Checkpoint A — environment, reproduction, smoke train

**One pull request containing all three parts.** Nothing expensive happens until this is merged.

The cost here is A2, and it scales with how many benchmark sets your model reports — a model with
six of them has cost 2.7 GPU-hours, so treat **a few GPU-hours** as normal and report what you
actually used. What checkpoint A must not do is approach the cost of a retrain. If your scoring
looks like it will run past roughly five GPU-hours, say so in the pull request before you spend it,
rather than after.

## A1 — environment  → `A1_env/`

Build the environment the repository asks for, then commit:

| file | content |
|---|---|
| `environment.yml` or `requirements.txt` | the **resolved** versions, not the loose spec |
| `nvidia-smi.txt` | the output, verbatim |
| `import_test.txt` | importing the model package, then printing `torch.__version__`, `torch.version.cuda`, `torch.cuda.is_available()` |
| `summary.json` | `stage` = `"A1_env"` |

Then produce `code/frozen.sha256` — a `sha256sum` listing of the files that compute metrics and load
benchmarks. For this repository that means the test task and its metric computation. Err on the side of listing too many files.

**Generate it once, here, and never regenerate it.** CI uses it to detect whether the evaluation
path moved under your feet between checkpoints. If your repository prints only macro means and you
need the per-target dump that contract rule 3 allows, **apply it before you generate this file** —
otherwise the checksum describes code that never scored anything.

### Verify your download now

Your variants are built from the released training file, and the applier checks your copy against
the digest we recorded before it writes anything. Do that check **here**, while it is free:

```bash
python tools/apply_variant.py --check-source <one of your specs> <your released training file>
```

It verifies and exits without writing. Put the digest it prints in `summary.json` under
`environment.source_digest_ok`, or quote it in `REPORT.md`.

Checking at checkpoint A rather than at C is the point: a release that differs from ours fails this
in minutes, instead of after you have paid for a full checkpoint-B run.

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

Mark it: set `published_reference.gate = true` on **exactly one** result. CI rejects zero and
rejects two. Every other benchmark you score still reports its `relative_gap_pct`, and none of them
is gated — so scoring more than you were asked to can never block you.

**The gate is 10 % relative on that metric.** Outside it: open an issue with your evidence and stop.
Do not go on to A3. A documented non-reproduction is a real result and is not your fault.

**If no published number is reachable from published artifacts at all** — the checkpoint the paper
scored was never released, or the benchmark files its evaluator reads were never released — then say
so in `published_reference.blocked` on the gated entry, with which artifact is missing and where you
looked. That entry then passes on the explanation instead of on a gap. Do not reconstruct the
missing artifact from a third-party source and compare that to the paper: a number from a rebuilt
benchmark is not a reproduction, and presenting it as one would be worse than reporting nothing.
Open an issue first either way.

Deliver `summary.json`, `per_target.csv` and `log.txt`.

`per_target.csv` has one row per benchmark target and is **not optional** (contract rule 3).
`target_id`, `n_actives` and `n_total` are always required. The metric columns — `auroc`, `bedroc`,
`ef_0.005`, `ef_0.01`, `ef_0.02`, `ef_0.05` — are required **wherever your model's own evaluation
computes them**; leave a column empty where it does not, and say which in `REPORT.md`. Do not invent
a metric your repository does not compute, and do not quietly substitute a different one.

**Also record the BEDROC alpha your code uses**, as `results[].bedroc_alpha`. Implementations
differ — 80.5 and 85 are both in use — and one is not comparable with the other. Read it out of the
code, do not assume it.

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

Your variants replace both `train_label_blend_seq_full.json` and `train_label_pdbbind_seq.json` in `data/`.

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
2. Record `split_id` in `summary.json`, and one `training_data.files` entry per file you built,
   each with `path` and `content_digest` — the full digest the script printed, not its first
   characters. CI compares every one against `splits/MANIFEST.json` and now rejects a missing
   entry, so list all of your files, not just one.
3. Deliver `C_retrain_variants/<variant_id>/` with `summary.json`, `per_target.csv` and `log.txt`.

---

## What makes a good submission

Per-target numbers. An empty or fully justified diff. One seed throughout. A stated checkpoint rule
with its source. An honest `Problems` section.

Not a flattering number.
