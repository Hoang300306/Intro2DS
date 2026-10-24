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
# A2.3b FEP benchmark (JACS-8 + Merck-FEP-8): download_data.sh does not fetch test_datasets/FEP,
# which DATA.md lists as the source; taken from LigUnity at the commit recorded in A2.1
git clone --filter=blob:none --no-checkout https://github.com/IDEA-XL/LigUnity.git ~/LigUnity
cd ~/LigUnity
git sparse-checkout set test_datasets/FEP
git checkout 8ae35edba5481aa13de8a5bac6b03c1c1860ef3c
git rev-parse HEAD
cp -r test_datasets/FEP ~/CausalBind/test_datasets/FEP
ls ~/CausalBind/test_datasets/FEP/lmdbs | wc -l
cd ~/CausalBind
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
python code/make_per_target.py PCBA ~/causalbind_logs/A2_PCBA.log ~/CausalBind/results/A2_sp_PCBA/PCBA A2_reproduce/per_target_PCBA.csv
cd ~/CausalBind
# A2.11 DEKOIS 2.0 (reported, not gated); all 81 targets are scored, the 9-target OOD subset of Table A13 is taken from them
date | tee ~/causalbind_logs/A2_DEKOIS.log
bash scripts/test.sh sp DEKOIS checkpoints/causalbind_sp.pt results/A2_sp_DEKOIS 2>&1 | tee -a ~/causalbind_logs/A2_DEKOIS.log
echo "exit ${PIPESTATUS[0]}" | tee -a ~/causalbind_logs/A2_DEKOIS.log
date | tee -a ~/causalbind_logs/A2_DEKOIS.log
# A2.12 DEKOIS per-target CSVs: all 81 targets, and the 9-target OOD subset of Table A13 (p. 47) filtered from it
cd ~/Intro2DS/submissions/03-causalbind
python code/make_per_target.py DEKOIS ~/causalbind_logs/A2_DEKOIS.log ~/CausalBind/results/A2_sp_DEKOIS/DEKOIS A2_reproduce/per_target_DEKOIS.csv
(head -1 A2_reproduce/per_target_DEKOIS.csv; grep -E "^(11betahsd1|ace2|adrb2|alr2|cox1|cyp2a6|inha|mmp2|tk)," A2_reproduce/per_target_DEKOIS.csv) > A2_reproduce/per_target_DEKOIS_OOD9.csv
cd ~/CausalBind
# A2.13 FEP (reported, not gated); Merck-FEP-8 Pearson r is compared with Table A11 (p. 46)
date | tee ~/causalbind_logs/A2_FEP.log
bash scripts/test.sh sp FEP checkpoints/causalbind_sp.pt results/A2_sp_FEP 2>&1 | tee -a ~/causalbind_logs/A2_FEP.log
echo "exit ${PIPESTATUS[0]}" | tee -a ~/causalbind_logs/A2_FEP.log
date | tee -a ~/causalbind_logs/A2_FEP.log
# A2.14 FEP per-target CSVs: all 16 targets, and the Merck-FEP-8 rows of Table A11 (p. 46)
cd ~/Intro2DS/submissions/03-causalbind
python code/make_per_target_fep.py ~/causalbind_logs/A2_FEP.log ~/CausalBind/results/A2_sp_FEP/FEP A2_reproduce/per_target_FEP.csv MERCK A2_reproduce/per_target_FEP_Merck8.csv
cd ~/CausalBind
# A2.15 log.txt: the four scoring logs, unedited, one after another
cd ~/Intro2DS/submissions/03-causalbind
for f in DUDE PCBA DEKOIS FEP; do echo "==== ~/causalbind_logs/A2_$f.log ===="; cat ~/causalbind_logs/A2_$f.log; done > A2_reproduce/log.txt
cd ~/CausalBind
# ---- A3: smoke train (1 epoch, small subset) ----------------------------------------------
# Vast.ai instance 57482916: 1x A100-SXM4-80GB, driver 535.230.02, 80 GB disk, image
# nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04, logged in as root; rented 07:24 UTC 2026-10-24.
# A3.0 (local) copy the submission repository to the instance (the branch is not pushed); the
# first rsync was checked before it had finished, so it was run again with -v
rsync -a -e "ssh -p 10970" ~/Intro2DS root@192.165.134.28:~/
rsync -av -e "ssh -p 10970" ~/Intro2DS root@192.165.134.28:~/
ssh -p 10970 root@192.165.134.28
# (instance) rsync keeps the local owner (uid 1000), and git refuses a repository root does not own
chown -R root:root ~/Intro2DS
cd ~/Intro2DS && git log --oneline -1 && git status --short | wc -l
ls submissions/03-causalbind/code

# A3.1 system packages the image lacks, then Miniforge non-interactively (latest -> conda 26.9.4;
# A1 had 26.7.2)
apt-get update
apt-get install -y git wget rsync unzip build-essential
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda --version
conda config --set auto_activate_base false

# A3.2 CausalBind at the upstream commit with the approved pin, the environment, Uni-Core (as A1)
git clone https://github.com/lokali/CausalBind.git ~/CausalBind
cd ~/CausalBind
git checkout 01fe8e93ca261f990e37ba3df7e7024cc5fa36dd
sed -i 's/hf-xet==1.1.5rc1/hf-xet==1.1.5/' environment.yml
# The first attempt failed in the pip step and is kept here as a comment:
#   conda env create -f environment.yml 2>&1 | tee ~/env_create.log
# torchsort==0.1.10 is source-only; pip builds it in an isolated env with the latest torch (CUDA
# 13.0), and with the image's nvcc 11.8 on the path it also builds its CUDA kernel, which torch
# refuses ("The detected CUDA version (11.8) mismatches ... (13.0)"). The A1 machine has no CUDA
# toolkit, so there torchsort built its CPU extension only. Hiding the toolkit for this one step
# gives the same build as A1 and leaves environment.yml and code/changes.diff unchanged.
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda env remove -n causalbind -y
mv /usr/local/cuda /usr/local/cuda.hidden
conda env create -f ~/CausalBind/environment.yml 2>&1 | tee ~/env_create.log
mv /usr/local/cuda.hidden /usr/local/cuda
conda activate causalbind
pip list | grep -i -E "^torch |torchsort|hf.xet"
cd ~
git clone https://github.com/dptech-corp/Uni-Core.git
cd Uni-Core
git checkout ace6fae1c8479a9751f2bb1e1d6e4047427bc134
python setup.py install 2>&1 | tee ~/unicore_install.log

# A3.3 lockfiles from this machine, diffed against A1 (diff exits 1 when they differ)
mkdir -p ~/causalbind_logs ~/Intro2DS/submissions/03-causalbind/A3_smoke
cd ~/Intro2DS/submissions/03-causalbind
conda env export > A3_smoke/environment.yml
pip list --format=freeze > A3_smoke/requirements.txt
nvidia-smi > A3_smoke/nvidia-smi.txt
diff A1_env/environment.yml A3_smoke/environment.yml > A3_smoke/lockfile_diff.txt || true
diff A1_env/requirements.txt A3_smoke/requirements.txt >> A3_smoke/lockfile_diff.txt || true
cd ~/CausalBind
python -c "import unicore, unimol; import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())" > ~/Intro2DS/submissions/03-causalbind/A3_smoke/import_test.txt 2>&1
cat ~/Intro2DS/submissions/03-causalbind/A3_smoke/lockfile_diff.txt
tail -3 ~/Intro2DS/submissions/03-causalbind/A3_smoke/import_test.txt

# A3.4 pre-flight: upstream commit, the only diff, and the frozen evaluation path
cd ~/CausalBind
git rev-parse HEAD
git diff | diff - ~/Intro2DS/submissions/03-causalbind/code/changes.diff && echo "diff OK"
sha256sum -c --quiet ~/Intro2DS/submissions/03-causalbind/code/frozen.sha256 && echo "frozen OK"

# A3.5 downloads: backbones, training data (figshare 27966819), test index files and DEKOIS;
# md5s compared with the figshare API (valid_lig.lmdb and valid_prot.lmdb are one file: each
# record holds both the ligand and the pocket)
bash scripts/download_data.sh pretrain 2>&1 | tee ~/causalbind_logs/A3_download_pretrain.log
sha256sum pretrain/*.pt | tee ~/causalbind_logs/A3_pretrain_sha256.txt
bash scripts/download_data.sh train 2>&1 | tee ~/causalbind_logs/A3_download_train.log
bash scripts/download_data.sh test 2>&1 | tee ~/causalbind_logs/A3_download_test.log
md5sum data/valid_label_seq.json data/valid_lig.lmdb data/valid_prot.lmdb data/train_label.zip data/pocket_name2idx_train_blend.json data/mol_smi2idx_train_blend.json data/train_prot_all_blend.lmdb data/train_lig_all_blend.lmdb | tee ~/causalbind_logs/A3_train_md5.txt
if cmp -s data/valid_lig.lmdb data/valid_prot.lmdb; then echo "valid_lig.lmdb and valid_prot.lmdb are identical"; else echo "valid_lig.lmdb and valid_prot.lmdb differ"; fi | tee -a ~/causalbind_logs/A3_train_md5.txt

# A3.6 source digest of the released labels on this machine (as A2.6)
S=~/Intro2DS/splits/specs/pocketaffdb_ligandish
python3 ~/Intro2DS/tools/apply_variant.py --check-source $S/pocketaffdb_ligandish__train_label_blend_seq_full.json.gz data/train_label_blend_seq_full.json 2>&1 | tee ~/causalbind_logs/A3_check_source.txt
python3 ~/Intro2DS/tools/apply_variant.py --check-source $S/pocketaffdb_ligandish__train_label_pdbbind_seq.json.gz data/train_label_pdbbind_seq.json 2>&1 | tee -a ~/causalbind_logs/A3_check_source.txt

# A3.7 the released index files against the LMDBs. This first ran with a version of
# check_index_cache.py that looked records up by the key str(idx) and reported every sample as
# mismatched; train_task.py reads with unicore.data.LMDBDataset, which indexes in LMDB key order.
# The corrected script is run in A3.11, after the smoke run had rebuilt both files.
C=~/Intro2DS/submissions/03-causalbind/code

# A3.8 smoke data directory, outside the repository: the subset labels, everything else linked;
# cache/ is left empty, so train_task.py rebuilds both index files itself
D=~/causalbind_smoke_data
mkdir -p $D
python $C/make_smoke_subset.py data $D 400 100 1 2>&1 | tee ~/causalbind_logs/A3_subset.txt
for f in train_lig_all_blend.lmdb train_prot_all_blend.lmdb valid_lig.lmdb valid_prot.lmdb valid_label_seq.json fep_assays.json fep_repeat_ligands_can.json fep_similar_ligands_0d5.json; do ln -s ~/CausalBind/data/$f $D/$f; done
ls -l $D

# A3.9 smoke train: one epoch, one GPU, seed 1 (the save root must not contain "no_similar")
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
date | tee ~/causalbind_logs/A3_smoke.log
bash $C/smoke_train.sh sp $D ~/causalbind_save/A3_smoke 1 2>&1 | tee -a ~/causalbind_logs/A3_smoke.log
echo "exit ${PIPESTATUS[0]}" | tee -a ~/causalbind_logs/A3_smoke.log
date | tee -a ~/causalbind_logs/A3_smoke.log

# A3.10 the checkpoint loads again: score it with the unmodified evaluator on DEKOIS; test.py
# loads with strict=False and warns "checkpoint/architecture mismatch" on any key difference
K=~/causalbind_save/A3_smoke/causalbind_sp_seed1/savedir/checkpoint_last.pt
ls -l ~/causalbind_save/A3_smoke/causalbind_sp_seed1/savedir
sha256sum $K | tee ~/causalbind_logs/A3_ckpt_sha256.txt
date | tee ~/causalbind_logs/A3_reload_DEKOIS.log
bash scripts/test.sh sp DEKOIS $K results/A3_smoke_DEKOIS 2>&1 | tee -a ~/causalbind_logs/A3_reload_DEKOIS.log
echo "exit ${PIPESTATUS[0]}" | tee -a ~/causalbind_logs/A3_reload_DEKOIS.log
date | tee -a ~/causalbind_logs/A3_reload_DEKOIS.log
grep -c "checkpoint/architecture mismatch" ~/causalbind_logs/A3_reload_DEKOIS.log || true

# A3.11 (local) the corrected check_index_cache.py (commit d834484) to the instance
rsync -av -e "ssh -p 10970" ~/Intro2DS/submissions/03-causalbind/code/check_index_cache.py root@192.165.134.28:~/Intro2DS/submissions/03-causalbind/code/
# (instance) the released index files against the LMDBs, and against the files train_task.py rebuilt
cd ~/CausalBind
python $C/check_index_cache.py data/train_lig_all_blend.lmdb data/mol_smi2idx_train_blend.json smi 2000 1 2>&1 | tee ~/causalbind_logs/A3_index_check.txt
python $C/check_index_cache.py data/train_prot_all_blend.lmdb data/pocket_name2idx_train_blend.json pocket 2000 1 2>&1 | tee -a ~/causalbind_logs/A3_index_check.txt
for f in mol_smi2idx_train_blend.json pocket_name2idx_train_blend.json; do cmp ~/causalbind_smoke_data/cache/$f data/$f && echo "$f: rebuilt by train_task.py, identical to the released file"; done | tee -a ~/causalbind_logs/A3_index_check.txt

# A3.12 log.txt and the environment logs, then (local) copy the results back; instance destroyed
cp ~/env_create.log ~/unicore_install.log ~/causalbind_logs/
mv ~/causalbind_logs/env_create.log ~/causalbind_logs/A3_env_create.log
mv ~/causalbind_logs/unicore_install.log ~/causalbind_logs/A3_unicore_install.log
cd ~/Intro2DS/submissions/03-causalbind
for f in A3_smoke A3_reload_DEKOIS; do echo "==== ~/causalbind_logs/$f.log ===="; cat ~/causalbind_logs/$f.log; done > A3_smoke/log.txt
ls -la A3_smoke ~/causalbind_logs
date -u
exit
rsync -av -e "ssh -p 10970" root@192.165.134.28:~/Intro2DS/submissions/03-causalbind/A3_smoke ~/Intro2DS/submissions/03-causalbind/
rsync -av -e "ssh -p 10970" root@192.165.134.28:~/causalbind_logs/ ~/causalbind_logs/

# ---- B: full retrain on the released training data ----------------------------------------

# ---- C: retrain on each supplied variant --------------------------------------------------
