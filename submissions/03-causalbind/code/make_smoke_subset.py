"""Cut the A3 smoke-train subset out of the released training labels.

Draws a fixed-seed random sample of records from each of the two released label files and
writes them, unchanged and in the order they appear in the released file, under the same file
names in OUT_DIR. train_task.py then applies its usual FEP and test-target filters to them, so
fewer groups than drawn reach training; the training log prints how many.

Usage: python make_smoke_subset.py DATA_DIR OUT_DIR N_BLEND N_PDBBIND SEED
  DATA_DIR  the directory holding the released train_label_*.json, e.g. ~/CausalBind/data
"""
import hashlib
import json
import random
import sys

data_dir, out_dir = sys.argv[1:3]
n = {"train_label_blend_seq_full.json": int(sys.argv[3]),
     "train_label_pdbbind_seq.json": int(sys.argv[4])}
seed = int(sys.argv[5])

for name, k in n.items():
    records = json.load(open(f"{data_dir}/{name}"))
    keep = sorted(random.Random(seed).sample(range(len(records)), k))
    path = f"{out_dir}/{name}"
    with open(path, "w") as f:
        json.dump([records[i] for i in keep], f)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    n_lig = sum(len(records[i]["ligands"]) for i in keep)
    print(f"{name}: {k} of {len(records)} records, {n_lig} ligands, sha256 {digest}")
