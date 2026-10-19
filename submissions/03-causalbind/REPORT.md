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
evaluation patch is needed and `code/eval_setup.patch` does not exist.

Wall clock about 2.0 h, including the wait for the decision on the hf-xet pin; no GPU compute.

## A2 — reproducing the published numbers

The published number you compared against, **with the table or section it comes from**, your number,
and the relative gap. If the gap is outside 10 %, say what you think is responsible and stop here —
that is a result, not a failure.

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

"environment.yml pins hf-xet==1.1.5rc1, which is no longer on PyPI, so the pip step of conda env create fails; changed to hf-xet==1.1.5 with the lead's approval (issue #8), recorded in code/changes.diff"
"The CausalBind README installs Uni-Core from its default branch without a pinned commit; used ace6fae1c8479a9751f2bb1e1d6e4047427bc134"
"python setup.py install for Uni-Core also fetched setuptools 84.0.0 as an egg; the active setuptools stays 75.8.0, but the exported environment.yml lists setuptools==84.0.0"
"The pip section of environment.yml replaces 12 conda-installed Jupyter dependencies (e.g. ipython 9.17.1 -> 9.3.0) with its pinned versions"
