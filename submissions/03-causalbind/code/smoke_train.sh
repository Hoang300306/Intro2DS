#!/usr/bin/env bash
# A3 smoke train: scripts/train.sh at upstream 01fe8e93 with only these changes -- the data
# directory is an argument, REPO_DIR is fixed (this file is not in the repository), one GPU by
# default, and --max-epoch 1 instead of 50. Every other flag is unchanged.
# Usage: bash smoke_train.sh VARIANT DATA_DIR SAVE_ROOT [SEED]
set -euo pipefail

variant="$1"
data_dir="$2"   # A3: the smoke subset directory instead of ${REPO_DIR}/data
save_root="$3"
seed="${4:-1}"

REPO_DIR="${HOME}/CausalBind"   # A3: this file lives outside the repository
source "${REPO_DIR}/scripts/variant_config.sh" "${variant}"

data_path="${data_dir}"
save_name="causalbind_${variant}_seed${seed}"
save_dir="${save_root}/${save_name}/savedir"
tmp_save_dir="${save_root}/${save_name}/tmp_save_dir"
tsb_dir="${save_root}/${save_name}/tsb_dir"
mkdir -p "${save_dir}" "${tmp_save_dir}" "${tsb_dir}" "${save_root}/train_log"

n_gpu="${N_GPU:-1}"   # A3: one GPU
MASTER_PORT="${MASTER_PORT:-10344}"
finetune_mol_model="${REPO_DIR}/pretrain/mol_pre_no_h_220816.pt"
finetune_pocket_model="${REPO_DIR}/pretrain/pocket_pre_220816.pt"

export OMP_NUM_THREADS=1
export NCCL_ASYNC_ERROR_HANDLING=1
UNICORE_TRAIN=$(which unicore-train)

torchrun --nproc-per-node=${n_gpu} --master-port=${MASTER_PORT} \
    ${UNICORE_TRAIN} ${data_path} \
    --user-dir "${REPO_DIR}/unimol" \
    --task train_task \
    --arch ${arch} \
    --loss three_hybrid_loss \
    --train-subset train --valid-subset valid \
    --valid-set CASF \
    --num-workers 0 --ddp-backend=c10d \
    --max-pocket-atoms 256 \
    ${model_flags} \
    --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-8 --clip-norm 1.0 \
    --lr-scheduler polynomial_decay --lr 1e-4 --warmup-ratio 0.06 --max-epoch 1 \
    --batch-size 24 --batch-size-valid 32 \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
    --update-freq 1 --seed ${seed} \
    --tensorboard-logdir ${tsb_dir} \
    --log-interval 100 --log-format simple \
    --validate-interval 1 \
    --all-gather-list-size 2048000 \
    --save-interval 1 \
    --save-dir ${save_dir} --tmp-save-dir ${tmp_save_dir} \
    --keep-best-checkpoints 8 --keep-last-epochs 50 \
    --find-unused-parameters \
    --finetune-pocket-model ${finetune_pocket_model} \
    --finetune-mol-model ${finetune_mol_model} \
    --max-lignum 16 \
    --learn-curv \
    --protein-similarity-thres 1.0 \
    --best-checkpoint-metric valid_bedroc --maximize-best-checkpoint-metric \
    2>&1 | tee "${save_root}/train_log/train_log_${save_name}.txt"
