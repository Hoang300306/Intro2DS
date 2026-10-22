# 07-sprint — SPRINT

| | |
|---|---|
| **Model** | SPRINT (MLSB @ NeurIPS 2024) |
| **Upstream repository** | https://github.com/abhinadduri/panspecies-dti |
| **Benchmarks** | LIT-PCBA only, over the full target list (`--target-protein-id "all"`). The repository also evaluates DAVIS, BindingDB, BIOSNAP and the TDC leaderboard, but each needs a checkpoint trained on that task and only the MERGED/LIT-PCBA models are released, so those are out of scope — do not score them |
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

**For A3, a "group" here is a target, not a slice of rows.** Take a few dozen of the 312 targets that carry positives and keep every row belonging to them, so the positive-to-negative structure of a target survives. Download the full MERGED corpus even so — `--check-source` needs the whole file — and **leave the MMseqs2 filter at 0.9**: it is part of the path you are smoke-testing and it is cheap next to training.

**Checkpoint B cannot be resumed, and alone among these models that is a real constraint.** `train.py` calls `trainer.fit(model, datamodule=datamodule)` with no `ckpt_path`, so Lightning cannot restore the optimizer, the schedule or the epoch counter; `--checkpoint` only loads weights and then starts a fresh schedule from epoch 0, which is a different recipe. An interrupted run has to be started again.

**That does not make a full run expensive by itself.** The row count is large but the model is small, and the featurizers preload into an LMDB-backed store rather than running a protein language model per batch, so an epoch is a throughput question, not a given. Measure it: at `batch_size` 64 an epoch is on the order of a million steps, so **report steps per second and the wall clock of one real epoch on the full corpus**, and where your data sits — a network drive will bound you long before the GPU does. Then 15 epochs is arithmetic, and we decide from it.

If that arithmetic exceeds the uninterrupted time you can get, say so at A3 and propose passing `ckpt_path` to `trainer.fit` as a documented diff in `code/changes.diff`. It is a one-line change and it is defensible — but only if the **same patched code runs at B and at C**, and only if you report how many times each arm was interrupted, because a resume boundary is a place where two arms can stop being identical.

This model predicts binary interaction from sequence rather than ranking within a pocket, so the variants were built to preserve the **positive-to-negative ratio exactly** (identical to eight decimal places between the two variants). Do not resample or rebalance the classes: that would change the thing the variants hold fixed.

**The README's own SPRINT command omits `--task merged`, and the config default is `task: davis`.** Run verbatim it trains on DAVIS, your variant file is never opened, and nothing anywhere says so. Pass the task explicitly. `configs/saprot_agg_config.yaml` also defaults to `model_size: "small"` and `epochs: 250`, so the flags are not optional decoration: without them you train the wrong model for sixteen times the budget. The command is

```bash
ultrafast-train --exp-id LitPCBA --config configs/saprot_agg_config.yaml \
    --task merged --model-size large --epochs 15 --ship-model \
    --target-protein-id "all" --similarity-threshold 0.9 \
    --eval-pcba --no-wandb --replicate 0
```

**There is no evaluation-only entry point, so A2 needs care.** `--eval-pcba` adds a callback whose hook is `on_validation_epoch_end`, and nothing in the repository calls `trainer.validate()`. To score the released checkpoint you therefore start a run with `--checkpoint <the released file>` and read the first `[PCBA Eval - Epoch 0]` block the callback prints. Lightning runs a sanity-check validation **before** the first optimizer step, so that block should be the released weights and nothing else — but **verify it from your log** and say in `REPORT.md` whether any training step preceded it. If the first block only appears after an epoch of training, then the released checkpoint cannot be scored through this repository at all: open an issue, and A2 becomes a `blocked` gate. Keep it cheap with `--epochs 1` and leave `--ship-model` off at A2.

**The seed flag is `--replicate` (alias `--r`), not `--seed`.** `train.py` calls `torch.manual_seed(config.replicate)` and `np.random.seed(config.replicate)`; the default is 0. Pass it explicitly and keep it identical at B and C. `--no-wandb` matters too: without it the run logs to the project `ConPlexTrain` from the config.

**`--similarity-threshold 0.9` is the model's own MMseqs2 filter** and the released checkpoints are the 90 % models, so hold it at 0.9 everywhere. `mmseqs` must be on PATH and `data/MERGED/huge_data/id_to_sequence.npy` must be present; a failing MMseqs2 call raises rather than skipping the filter, so you will know.

**With `--task merged --ship-model`, three things change at once**: `save_top_k=-1`, so every epoch's checkpoint is kept and fifteen of them is a lot of disk; `limit_test_batches=0`, so the built-in test runs on nothing; and `ship_model.ckpt` is written at the end with the final weights. The released models are the shipped ones — score `ship_model.ckpt`, say so, and get your numbers from the `--eval-pcba` callback.

That callback prints, per target, AUROC, BEDROC and EF at 0.005, 0.01 and 0.05 — so `per_target.csv` needs no patch, `ef_0.02` stays empty, and **the BEDROC here is at alpha 85**, which is not the 80.5 most of this literature uses. Record the alpha. For `published_reference`, the released checkpoints correspond to Table 2 of arXiv 2411.15418.

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
