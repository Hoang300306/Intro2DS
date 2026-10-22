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

What subset you trained on, how long one epoch took, and **your extrapolated cost for checkpoint B
and for each variant in C**. Say what you extrapolated from. We approve the expensive runs from this
number, so a careless guess here means either a blocked run or a wasted one.

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
