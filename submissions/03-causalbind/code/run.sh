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
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate causalbind
cd ~/CausalBind
mkdir -p ~/causalbind_logs
# A2.1 record the unpinned upstreams the downloads read from
git ls-remote https://github.com/IDEA-XL/LigUnity refs/heads/main | tee ~/causalbind_logs/ligunity_main.txt

# A2.2 pretrained backbones (Uni-Mol v0.1, ESM-2 35M)
bash scripts/download_data.sh pretrain 2>&1 | tee ~/causalbind_logs/download_pretrain.log
sha256sum pretrain/*.pt | tee ~/causalbind_logs/pretrain_sha256.txt
head -1 pretrain/esm2_t12_35M_UR50D/.cache/huggingface/download/config.json.metadata | tee ~/causalbind_logs/esm2_revision.txt
# A2.3 benchmark index files (LigUnity main, recorded in A2.1), CASF, DEKOIS 2.0 (figshare 27967422)
# figshare is slow from here (about 170 KB/s, roughly 4 h for DEKOIS); the script skips any
# non-empty file, so if this is interrupted resume with: wget -c -O test_datasets/DEKOIS_2.0x.zip https://ndownloader.figshare.com/files/50994021
bash scripts/download_data.sh test 2>&1 | tee ~/causalbind_logs/download_test.log
md5sum test_datasets/DEKOIS_2.0x.zip
stat -c %s test_datasets/DEKOIS_2.0x.zip
ls test_datasets/DEKOIS_2.0x | wc -l

# A2.4 DUD-E and LIT-PCBA: downloaded by hand in a browser from the LigUnity Google Drive
# folder https://drive.google.com/drive/folders/1zW1MGpgunynFxTKXC2Q4RgWxZmg6CInV
# (dude.zip, pcba.zip), saved into ~/CausalBind/test_datasets/
cd ~/CausalBind/test_datasets
md5sum dude.zip pcba.zip | tee ~/causalbind_logs/drive_md5.txt
unzip -q pcba.zip
unzip -q dude.zip
mv data/protein/DUD-E/raw/all DUD-E && rm -r data
ls DUD-E | wc -l
ls lit_pcba | wc -l
cd ~/CausalBind
# A2.5 released CausalBind-SP checkpoint (Hugging Face LokaLi/CausalBind)
huggingface-cli download LokaLi/CausalBind causalbind_sp.pt --local-dir checkpoints
python -c "from huggingface_hub import model_info; print(model_info('LokaLi/CausalBind').sha)" | tee ~/causalbind_logs/hf_revision.txt
sha256sum checkpoints/causalbind_sp.pt | tee ~/causalbind_logs/ckpt_sha256.txt
python -c "from huggingface_hub import model_info; [print(s.rfilename, s.lfs.sha256) for s in model_info('LokaLi/CausalBind', files_metadata=True).siblings if s.rfilename == 'causalbind_sp.pt']"
# A2.6 released training labels only (figshare 27966819), to verify the source digest at
# checkpoint A; the training LMDBs are downloaded on the A100 machine
mkdir -p data
wget -nv -O data/train_label.zip https://ndownloader.figshare.com/files/50992179
md5sum data/train_label.zip
stat -c %s data/train_label.zip
(cd data && unzip -n -q train_label.zip)
S=~/Intro2DS/splits/specs/pocketaffdb_ligandish
python3 ~/Intro2DS/tools/apply_variant.py --check-source $S/pocketaffdb_ligandish__train_label_blend_seq_full.json.gz data/train_label_blend_seq_full.json 2>&1 | tee ~/causalbind_logs/check_source.txt
python3 ~/Intro2DS/tools/apply_variant.py --check-source $S/pocketaffdb_ligandish__train_label_pdbbind_seq.json.gz data/train_label_pdbbind_seq.json 2>&1 | tee -a ~/causalbind_logs/check_source.txt
# A2.7 pre-flight: upstream commit, the only diff, and the frozen evaluation path
git rev-parse HEAD
git diff | diff - ~/Intro2DS/submissions/03-causalbind/code/changes.diff && echo "diff OK"
sha256sum -c --quiet ~/Intro2DS/submissions/03-causalbind/code/frozen.sha256 && echo "frozen OK"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

# A2.8 score the released checkpoint on DUD-E (gated: EF@1%, the "ef 0.01 mean" line)
date | tee ~/causalbind_logs/A2_DUDE.log
bash scripts/test.sh sp DUDE checkpoints/causalbind_sp.pt results/A2_sp_DUDE 2>&1 | tee -a ~/causalbind_logs/A2_DUDE.log
echo "exit ${PIPESTATUS[0]}" | tee -a ~/causalbind_logs/A2_DUDE.log
date | tee -a ~/causalbind_logs/A2_DUDE.log
# A2.9 LIT-PCBA (reported, not gated); a fresh results dir, since test_pcba reuses any
# metrics.json it finds there and skips targets that raise
date | tee ~/causalbind_logs/A2_PCBA.log
bash scripts/test.sh sp PCBA checkpoints/causalbind_sp.pt results/A2_sp_PCBA 2>&1 | tee -a ~/causalbind_logs/A2_PCBA.log
echo "exit ${PIPESTATUS[0]}" | tee -a ~/causalbind_logs/A2_PCBA.log
date | tee -a ~/causalbind_logs/A2_PCBA.log
# A2.10 per-target CSVs, read back from the logs and saved_labels.npy (no metric is recomputed)
cd ~/Intro2DS/submissions/03-causalbind
mkdir -p A2_reproduce
python code/make_per_target.py DUD-E ~/causalbind_logs/A2_DUDE.log ~/CausalBind/results/A2_sp_DUDE/DUDE A2_reproduce/per_target_DUDE.csv
cd ~/CausalBind
# ---- A3: smoke train (1 epoch, small subset) ----------------------------------------------

# ---- B: full retrain on the released training data ----------------------------------------

# ---- C: retrain on each supplied variant --------------------------------------------------
