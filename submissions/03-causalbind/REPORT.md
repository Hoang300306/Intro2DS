# Report — CausalBind — 03-causalbind

## A1 — environment

Built on a single NVIDIA GeForce RTX 3060 (12 GB), which also drives the desktop session
(about 0.6 GB and some utilisation are taken by the display at idle). A1 and A2 run on this
machine; A3, B and C will run on rented A100 80GB GPUs, with the environment rebuilt from
`code/run.sh` and its lockfile diffed against `A1_env/environment.yml`.

The environment follows the CausalBind README: Miniforge (conda 26.7.2),
`conda env create -f environment.yml`, then Uni-Core with `python setup.py install`.
The pip step failed as shipped because `environment.yml` pins `hf-xet==1.1.5rc1`, a release
candidate that is no longer on PyPI; with the lead's approval (issue #8) the pin was changed
to `hf-xet==1.1.5`, the final release of that candidate. All other 142 pip pins resolved.
hf-xet is a download backend of huggingface_hub and is not imported by the CausalBind code.

Uni-Core is not pinned by the README; it was installed at
`ace6fae1c8479a9751f2bb1e1d6e4047427bc134` without `--enable-cuda-ext`, exactly as the README
does, so Uni-Core's fused kernels fall back to their PyTorch implementations (the
"fused_... is not installed" lines in `A1_env/import_test.txt`).

The lockfiles are exported from the live environment (`conda env export`,
`pip list --format=freeze`) rather than copied from the upstream file, because the installed
set differs from it: Uni-Core is added, pip replaces 12 conda-installed Jupyter dependencies
with its pinned versions, and Uni-Core's install fetched setuptools 84.0.0 as an egg, so the
exported `environment.yml` lists setuptools 84.0.0 although the active version is 75.8.0.

`code/frozen.sha256` covers `scripts/test.sh`, `scripts/variant_config.sh`, every tracked file
under `unimol/` and `vocab/` (45 files), taken at the upstream commit. The repository already
prints per-target metrics for all four benchmarks and saves per-target labels, so no
edit to the evaluation path is needed; `code/changes.diff` holds only the environment pin.

Wall clock about 2.0 h, including the wait for the decision on the hf-xet pin; no GPU compute.

## A2 — reproducing the published numbers

Gate metric, fixed before any scoring: **DUD-E EF@1%**, published value **47.69** for
CausalBind-SP (Table 1, p. 8 of arXiv:2610.07340; seed 1, last checkpoint, 102 targets,
confirmed as seed-1 last-checkpoint results by Table A4, p. 40). Gate: |gap| ≤ 10 %,
i.e. 42.92–52.46.

Why EF@1%: Table 1 lists AUROC first, but the only DUD-E claim in the text is that all
variants achieve "higher DUD-E AUROC and EF@1%" (p. 8), and the paper names BEDROC80.5 and
EF@1% as the early-enrichment metrics that matter more in practice (p. 8, p. 36). EF@1% is
the one metric both stated for DUD-E and in that group.

BEDROC alpha is 80.5 in the paper (p. 36) and in the code (`unimol/tasks/test_task.py`).
For scale: the paper's multi-seed mean for SP is 47.99 ± 0.39 (Table A16, n = 4), so seed
noise alone is about 1 %, well inside the gate.

### Result

The gate passes: DUD-E EF@1% is **47.688** against 47.69 published (−0.004 %), with all
102 targets scored. Every other number the paper prints for the benchmarks scored here agrees
to the precision it is printed at.

| Benchmark | Metric | Ours | Published | Source | Gap |
|---|---|---|---|---|---|
| **DUD-E (102), gate** | **EF@1%** | **47.688** | **47.69** | Table 1, p. 8 | **−0.004 %** |
| DUD-E (102) | AUROC | 0.9355 | 0.935 | Table 1, p. 8 | +0.05 % |
| DUD-E (102) | BEDROC80.5 | 0.7440 | 0.744 | Table 1, p. 8 | −0.01 % |
| DUD-E (102) | RE@0.5/1/2/5 % | 128.01 / 69.27 / 37.06 / 15.99 | 128.06 / 69.26 / 37.07 / 15.99 | Table A10 | ≤ 0.04 % |
| LIT-PCBA (15) | EF@1% | 8.559 | 8.56 | Table 1, p. 8 | −0.007 % |
| LIT-PCBA (15) | AUROC | 0.6395 | 0.639 | Table 1, p. 8 | +0.08 % |
| LIT-PCBA (15) | BEDROC80.5 | 0.0965 | 0.097 | Table 1, p. 8 | −0.56 % (rounding) |
| DEKOIS 2.0, 9-target OOD subset | AUROC | 0.8597 | 0.860 | Table A13, p. 47 | −0.04 % |
| DEKOIS 2.0, 9-target OOD subset | BEDROC80.5 | 0.6574 | 0.658 | Table A13, p. 47 | −0.08 % |
| DEKOIS 2.0, 9-target OOD subset | EF@1% | 23.002 | 23.00 | Table A13, p. 47 | +0.007 % |
| Merck-FEP-8 | Pearson r | 0.5978 | 0.598 | Table A11, p. 46 | −0.03 % |
| Merck-FEP-8 | Spearman ρ | 0.5474 | 0.548 | Table A11, p. 46 | −0.10 % |

In `summary.json` each benchmark carries one of these as its `published_reference`: EF@1%
for DUD-E (the gate) and LIT-PCBA, AUROC for the DEKOIS subset, Pearson r for Merck-FEP-8.
Only DUD-E is gated.

Scored but not published, so reported for reference only: all 81 DEKOIS targets, AUROC
0.9336, BEDROC80.5 0.7764, EF@1% 25.50; FEP JACS-8, Pearson r 0.454, Spearman ρ 0.484;
all 16 FEP targets, r 0.526, ρ 0.516.

### How it was run

- **Checkpoint:** `causalbind_sp.pt` from Hugging Face `LokaLi/CausalBind` at revision
  `000181dcf62caabdf3505cb3d1176720292faee9`, sha256
  `cc89bd6c3b04bcaa5477aa351db54410b248841b7a77b828e90c73b1a2b8abd4`, equal to the LFS sha256
  Hugging Face publishes. Rule: last checkpoint, seed 1, as the paper reports (Section 4, p. 8;
  Table A4, p. 40) and as the assignment sheet asks.
- **Pretrained backbones:** fetched by `scripts/download_data.sh`; their sha256 and the
  ESM-2 revision are in `summary.json` (`environment.notes`).
- **Benchmarks:**
  - DUD-E and LIT-PCBA were downloaded by hand from the LigUnity Google Drive; their md5s
    are in `summary.json`.
  - DEKOIS 2.0 came from `download_data.sh test` (figshare).
  - FEP came from LigUnity at `8ae35edb`, since the script does not fetch it (see Problems).
  - The index files (`dude.json`, `dekois.json`, …) are from LigUnity main at the same commit.
- **Scoring:** `bash scripts/test.sh sp <DUDE|PCBA|DEKOIS|FEP> checkpoints/causalbind_sp.pt
  results/A2_sp_<benchmark>`.
  - Nothing in the evaluation path was edited: the pre-flight step (A2.7 in `code/run.sh`)
    found the upstream commit, `code/changes.diff` and `code/frozen.sha256` intact before
    scoring, and they still are after it.
  - The settings are the script's own: fp16, `--seed 1`, batch 128 (64 inside the DUD-E and
    FEP loaders).
  - Each benchmark wrote to a fresh results directory, because the LIT-PCBA path reuses a
    `metrics.json` it finds there.
  - UniProt sequences were fetched live by the evaluator into `uniprot_fasta/`.
- **Source check:** the released training data passed `apply_variant.py --check-source`. The
  digests are in `summary.json` (`environment.source_digest_ok`).
- **Cost:** 1.48 GPU-hours on the RTX 3060, counting scoring only (downloads used no GPU):

  | Benchmark | Hours |
  |---|---|
  | DUD-E | 0.53 |
  | LIT-PCBA | 0.90 |
  | DEKOIS | 0.04 |
  | FEP | 0.002 |

  This is over the 1 GPU-hour budget for checkpoint A, before A3 is counted; recorded here.

### Per-target files

`code/make_per_target.py` builds `per_target_DUDE.csv`, `per_target_PCBA.csv` and
`per_target_DEKOIS.csv`:

- The metrics are read from the per-target lines the evaluator prints, not recomputed.
- `n_total` is the logged `num_mols`.
- `n_actives` is counted from the `saved_labels.npy` the evaluator writes.
- The evaluator prints AUROC and BEDROC to 4 decimals, so the means of those columns differ
  from the logged means in the 5th–6th decimal. EF is printed in full and its means match
  exactly.

`per_target_DEKOIS_OOD9.csv` is the 9 rows of Table A13's subset, filtered from the 81-target
file.

`code/make_per_target_fep.py` builds `per_target_FEP.csv` (16 targets) and
`per_target_FEP_Merck8.csv` from the table `test_fep` prints:

- FEP is scored as a regression, and the evaluator computes none of `auroc`, `bedroc` or
  `ef_*`, so those columns and `n_actives` are left empty.
- `n_total` is the number of ligands scored.
- Added columns: `group` (JACS/MERCK), `pearson_r`, `spearman_rho` and `r2`, at the
  5 decimals printed.
- The evaluator's own ACC/AUC columns are pairwise ranking metrics over the FEP edge files
  in `bench/`, which are not published, so they print NaN and are not reported.

`log.txt` is the four scoring logs, unedited, one after another.

## A3 — smoke train, and what the real runs will cost

Run on one rented A100-SXM4-80GB (Vast.ai), with the environment rebuilt from `code/run.sh`.

### Environment rebuilt on the A100

The rebuilt environment matches A1. `A3_smoke/lockfile_diff.txt` shows only two differences:

- the env prefix;
- the setuptools egg that Uni-Core's install fetches (84.2.1 here, 84.0.0 at A1).

Every pinned package came out at the same version. `import_test.txt` gives
`2.7.0+cu118 / 11.8 / True`. The pre-flight passed:

- upstream commit `01fe8e93`;
- `changes.diff` unchanged;
- `frozen.sha256` clean.

The downloads match what was used before:

- the backbone sha256s and the DEKOIS md5 equal A2;
- the eight training-data files match the md5s figshare publishes;
- `--check-source` gives the same two digests as A2.

One step had to differ from A1. The image ships the CUDA 11.8 toolkit, and with it torchsort's
source build failed (see Problems). `/usr/local/cuda` was moved aside for that one command, so
torchsort built its CPU extension only, as on the A1 machine, which has no toolkit.
`environment.yml` and `changes.diff` are unchanged.

### The smoke run

**Subset.** `code/make_smoke_subset.py` draws a seed-1 sample of 400 ChEMBL/BindingDB assays and
100 PDBbind pairs from the released label files:

- `train_label_blend_seq_full.json` sha256 `62a68509…c864`;
- `train_label_pdbbind_seq.json` sha256 `f913809d…22fe`.

`train_task.py` applies its usual FEP and test-target filters, and 278 assays survive. That gives
481 training items, which is 20 updates at batch 24.

**Command.** `code/smoke_train.sh` is `scripts/train.sh` with four changes:

- the data directory is an argument;
- `REPO_DIR` is fixed;
- one GPU;
- `--max-epoch 1`.

Every other flag is unchanged (batch 24, lr 1e-4, fp16, seed 1, CASF validation).

**Timings.** One epoch took 11 min 27 s end to end:

| Phase | Time |
|---|---|
| start-up and model build | 16 s |
| rebuild of the two index files (`cache/`, see Problems) | 10 min 34 s |
| 20 training updates | 20 s, i.e. about **1.0 s per update** |
| CASF validation | 4 s |
| saving and copying the checkpoint | 11 s |

**Checkpoint.** `checkpoint_last.pt` is sha256 `f3fba02c…d2ad`, 3.16 GB.

It loads again: scoring it with the unmodified evaluator on DEKOIS completed all 81 targets with
no `checkpoint/architecture mismatch` warning. The scores are those of an untrained model
(AUROC 0.498, BEDROC 0.097, EF@1% 3.32), as expected after 20 updates inside the warm-up. The
validation line prints `loss 0` because `three_hybrid_loss.py` sets the validation loss to 0
(line 176); it is not a failure.

### What stops a run

Training stops at a fixed epoch count, with no early stopping:

- `scripts/train.sh` passes `--max-epoch 50` and no `--patience`, and Uni-Core's default
  `--patience` is -1, i.e. off (`unicore/options.py:388`).
- The README and DATA.md both say 50 epochs, and the paper reports the epoch-50 checkpoint.

So the cost below is the per-epoch time multiplied by 50, not a guess at an early stop.

### Estimated cost of B and C

**Extrapolated from:** the measured 1.0 s per update (1 GPU, batch 24 per GPU, as in the real
runs), scaled to the real epoch size.

**Epoch size.** Applying `train_task.py`'s filters to the released labels leaves 18,316 assays
(the count DATA.md states). Cut into blocks of 32 ligands, plus the 16,744 PDBbind pairs, that is
45,163 items per epoch. At a global batch of 96 on 4 GPUs, that is **471 updates per epoch**.

**Per epoch.** 471 updates × 1.0–1.15 s, allowing 0–15 % for 4-GPU all-reduce, plus validation and
checkpointing, comes to **8.1–9.3 min per epoch**.

**Per run.** × 50 epochs = **6.8–7.7 h**. The README states 5–6 h on 4× A100 80GB, which
corresponds to 0.73–0.89 s per update. Twenty updates are a small sample that includes start-up
effects, so I take **5–8 h per run** as the range.

**Plan: 4× A100 80GB, as in the paper.**

| | Wall clock | Rented GPU-hours (4× A100) |
|---|---|---|
| Training, one 50-epoch run | 5–8 h | 20–32 |
| Machine set-up, once (environment, downloads, checks) | ~0.5 h | ~2 |
| Scoring each retrained checkpoint (DUD-E, LIT-PCBA, DEKOIS, FEP): `checkpoint_last.pt` is copied back and scored on the local RTX 3060, as at A2 (1.48 h there), while the next run trains | ~1.5 h local | 0 |
| **B** (set-up + one run) | **5.5–8.5 h** | **22–34** |
| **Each variant in C** (`pocketaffdb_ligandish`, `pocketaffdb_ligandish_matched`), on the same machine | **5–8 h** | **20–32** |
| **B + C** | **15.5–24.5 h** | **62–98** |

At the Vast.ai A100 prices seen here ($1.2–2 per GPU-hour), B + C is about **$75–195**.

Training is the whole of the rented cost: 23,550 updates (471 × 50). The range comes from the
per-update time:

- **Lower end, README:** 5–6 h per run, i.e. 0.73–0.89 s per update, gives 62–74 GPU-hours.
- **Upper end, A3:** 1.0–1.15 s per update gives up to 98 GPU-hours.

The A3 figure comes from only 20 updates, start-up included. It may also be slow because Uni-Core
is installed without its fused kernels, as the README does (the `fused_* is not installed` lines),
or because `--num-workers 0` prepares every batch (about 400 ligands and 24 pockets per GPU) on one
CPU core. Both are guesses, not measurements, so the range is left wide and narrowed at the start
of B (below).

Assumptions behind these numbers:

- **No index rebuild.** The released index files are placed in `data/cache/`, which skips the
  10.6-min rebuild at start-up. They are byte-identical to what `train_task.py` builds.
- **Disk.** The machine needs at least **250 GB**. Each checkpoint is 3.16 GB and `train.sh`
  keeps all 50 epochs plus up to 8 best, about 190 GB per run. After a run's `checkpoint_last.pt`
  has been copied back and hashed, its per-epoch checkpoints can be deleted before the next run.
  `--keep-last-epochs` itself is left unchanged.
- **Interruptions.** A checkpoint is written every epoch, so an interrupted session loses at most
  one epoch, about 9 min or 0.6 GPU-hours. Uni-Core resumes from `checkpoint_last.pt`.

### Measuring at the start of B

In the first ~300 updates of B (about 5 min after start-up), two things are measured.

**1. The real time per update on 4 GPUs.** It is read from the timestamps of the train log's
every-100-update lines.

**2. Whether the run is data-bound.** Two things are recorded:

- GPU utilisation every 5 s, with `nvidia-smi dmon -s u -d 5`;
- the CPU use of the four training processes, with `top`.

The cost is then recomputed and reported, and the run goes on or stops by this rule:

| Measured time per update | Run length | Action |
|---|---|---|
| ≤ 0.89 s | ≤ 6 h | Within the README figure; continue |
| 0.89–1.15 s | 6–8 h | Within this estimate; continue, and report the number |
| > 1.15 s | > 8 h | Stop before the end of the first epoch and report to the lead with the numbers. The run resumes from `checkpoint_last.pt`, so stopping costs little. |

If GPU utilisation stays well below full while each training process holds a CPU core at 100 %,
the run is waiting on data. In that case an issue proposing a larger `--num-workers` is opened,
with the measurements. Building Uni-Core with `--enable-cuda-ext` would be a separate proposal.
Neither is changed without the lead's approval, because both depart from the released
configuration.

This is the configuration the assignment asks for (`sp`, 4 GPUs, global batch 96, 50 epochs);
fewer GPUs or epochs would change the experiment. Whether 62–98 GPU-hours fits the budget is for
the lead to decide; if it does not, the assignment's fallback is `emb`.

### Cost of A3

The instance was rented from 07:24 to about 08:37 UTC, i.e. **1.2 GPU-hours** (1 GPU). Of that,
the GPU computed for under 6 minutes (training 20 s, reload scoring 4.7 min). The rest was:

- building the environment (twice, because of torchsort);
- downloads;
- the index rebuild.

Checkpoint A comes to about 2.7 GPU-hours in total (A2 1.48 + A3 1.2), over its 1 GPU-hour
budget.

## B — retrain on the released training data

Final numbers, wall clock, GPU hours. How the retrained model compares to the released checkpoint
you scored in A2 — these are usually **not** identical, and that difference is itself worth
recording rather than hiding.

## C — retrain on the supplied variants

One subsection per variant id. Numbers and cost. No interpretation needed from you: just say what
you ran and what came out.

## Changes to upstream code

- `environment.yml`, `hf-xet==1.1.5rc1` → `hf-xet==1.1.5`: the release candidate is no longer
  on PyPI, so the environment cannot be installed as shipped; approved by the lead in issue #8.

## Problems

- `environment.yml` pins `hf-xet==1.1.5rc1`, which is no longer on PyPI, so the pip step of
  `conda env create` fails; changed to `hf-xet==1.1.5` with the lead's approval (issue #8),
  recorded in `code/changes.diff`.
- The CausalBind README installs Uni-Core from its default branch without a pinned commit;
  used `ace6fae1c8479a9751f2bb1e1d6e4047427bc134`.
- `python setup.py install` for Uni-Core also fetched setuptools 84.0.0 as an egg; the active
  setuptools stays 75.8.0, but the exported `environment.yml` lists `setuptools==84.0.0`.
- The pip section of `environment.yml` replaces 12 conda-installed Jupyter dependencies
  (e.g. ipython 9.17.1 → 9.3.0) with its pinned versions.
- `scripts/download_data.sh` skips any file that is already non-empty, so an interrupted
  download is never resumed or replaced. The slow `DEKOIS_2.0x.zip` (figshare, about 200 KB/s)
  finished in one go, and its md5 was checked.
- `scripts/download_data.sh` does not fetch `test_datasets/FEP` (`fep_labels.json`,
  `lmdbs/`), which DATA.md names as the FEP source; it was taken from LigUnity at `8ae35edb`.
  The `bench/` CSVs that `test_fep_target` reads for ACC/AUC are not published, so ACC/AUC are
  NaN. Pearson r and Spearman ρ are unaffected, as the paper states (p. 46).
- `dude.json` maps CSF1R to P11362, which is FGFR1's UniProt ID (CSF1R is P07333), so CSF1R is
  scored with the FGFR1 sequence. Left unchanged, since the published numbers come from the
  same file.
- The DUD-E files count 22,800 actives over the 102 targets, against 22,886 in the DUD-E
  release. Scored as released.
- The released DEKOIS_2.0x LMDBs have fewer molecules than DEKOIS 2.0 defines
  (40 actives + 1,200 decoys = 1,240 per target) for 36 of the 81 targets:
  - `11betahsd1`, `17betahsd1` and `sars-hcov` lack 3, 1 and 1 actives;
  - the other 33 lack only decoys.

  Scored as released; the counts are in `per_target_DEKOIS.csv`.
- Table A13 also reports a 6-target scaffold-level DEKOIS subset but does not name the targets,
  so it is not compared.
- The paper's Table 2 caption says "five runs", while Table A16 gives n = 4 for the multi-seed
  SP numbers. The gate uses the seed-1 number of Table 1.
- The DUD-E log is missing its exit-code and end-time lines (those commands were not run). The
  run printed every mean and ended normally; the end time 00:31:22 is the log's last write.
- A2 used 1.48 GPU-hours, over the 1 GPU-hour budget for checkpoint A.
- `conda env create` failed on the rented machine. torchsort 0.1.10 is source-only, so pip
  builds it in an isolated env with the latest torch (CUDA 13.0). With the image's nvcc 11.8 on
  the path it also builds a CUDA kernel, and torch refuses the CUDA version mismatch. The A1
  machine has no toolkit, so there torchsort built only its CPU extension. `/usr/local/cuda` was
  moved aside for that one command, giving the same build; `environment.yml` and
  `code/changes.diff` are unchanged.
- Miniforge is installed from `releases/latest`, which moved from conda 26.7.2 (A1) to 26.9.4
  (A3). The environment itself is pinned and came out the same.
- `PairDataset` reads `DATA_DIR/cache/{mol_smi2idx,pocket_name2idx}_train_blend.json`, but
  `download_data.sh` saves the released copies in `data/`. Without them, the first run rebuilds
  both on one CPU core: 10.6 min here, before the first update. The rebuilt files are
  byte-identical to the released ones.
- A first version of `code/check_index_cache.py` looked records up by the key `str(idx)`, as the
  evaluator's `unimol` LMDBDataset does, and reported every sample as mismatched.
  `train_task.py` uses `unicore.data.LMDBDataset`, which indexes in LMDB key order. Corrected in
  commit `d834484`; the corrected check and the byte comparison both pass.
- Each training checkpoint is 3.16 GB, because it holds the optimizer state. With
  `--keep-last-epochs 50` and up to 8 best checkpoints, a 50-epoch run writes about 190 GB.
- Scoring the smoke checkpoint on DEKOIS took 4.7 min on the A100, against 2.7 min for the
  released checkpoint on the RTX 3060. Not investigated. Two candidate causes: loading the
  3.16 GB checkpoint, and the live UniProt fetch on a fresh machine.
- Checkpoint A used about 2.7 GPU-hours, over the 1 GPU-hour budget.
