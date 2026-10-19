#!/bin/bash
# Every command you ran, in order, verbatim. No placeholders, no "see above".
# Someone else must be able to paste this and get your result.
set -euo pipefail

# ---- A1: environment ----------------------------------------------------------------------
# Miniforge, interactive install with default answers (license yes, ~/miniforge3, init yes) -> conda 26.7.2
cd ~/Downloads
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda config --set auto_activate_base false
# environment.yml pins hf-xet==1.1.5rc1, which is no longer on PyPI; the lead approved hf-xet==1.1.5 (issue #8)
cd ~/CausalBind
sed -i 's/hf-xet==1.1.5rc1/hf-xet==1.1.5/' environment.yml
conda env create -f environment.yml
conda activate causalbind
# Uni-Core: not pinned by the CausalBind README; installed at the commit below, without --enable-cuda-ext (as in the README)
cd ~
git clone https://github.com/dptech-corp/Uni-Core.git
cd Uni-Core
git checkout ace6fae1c8479a9751f2bb1e1d6e4047427bc134
python setup.py install
# A1 deliverables
cd ~/Intro2DS/submissions/03-causalbind
conda env export > A1_env/environment.yml
pip list --format=freeze > A1_env/requirements.txt
nvidia-smi > A1_env/nvidia-smi.txt
cd ~/CausalBind
python -c "import unicore, unimol; import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())" > ~/Intro2DS/submissions/03-causalbind/A1_env/import_test.txt 2>&1
# frozen evaluation path: test scripts, the whole unimol/ package and vocab/, generated once
sha256sum scripts/test.sh scripts/variant_config.sh $(git ls-files unimol vocab) > ~/Intro2DS/submissions/03-causalbind/code/frozen.sha256
# record local changes to upstream (only the approved hf-xet pin)
git diff > ~/Intro2DS/submissions/03-causalbind/code/changes.diff
# ---- A2: score the released checkpoint ----------------------------------------------------

# ---- A3: smoke train (1 epoch, small subset) ----------------------------------------------

# ---- B: full retrain on the released training data ----------------------------------------

# ---- C: retrain on each supplied variant --------------------------------------------------
