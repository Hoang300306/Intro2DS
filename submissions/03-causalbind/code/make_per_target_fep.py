"""Build per-target CSVs for FEP from a CausalBind test.py log.

test_fep prints one fixed-width row per target (TARGET 10, GRP 6, then R2, rho, r, ACC, AUC at
8 columns each, 5 decimals); the columns are cut by width, since a negative value runs into the
one before it. Nothing is recomputed. FEP is a regression benchmark, so n_actives and the
screening columns (auroc ... ef_0.05) stay empty; n_total is the number of ligands scored, the
length of the saved_labels.npy test_fep_target wrote for that target.

Usage: python make_per_target_fep.py LOG TARGET_DIR OUT_CSV [GROUP OUT_GROUP_CSV]
  TARGET_DIR  e.g. results/A2_sp_FEP/FEP
  GROUP       JACS or MERCK: also write the rows of that group alone to OUT_GROUP_CSV
"""
import csv
import sys

import numpy as np

log, target_dir, out = sys.argv[1:4]
group, out_group = (sys.argv[4], sys.argv[5]) if len(sys.argv) > 5 else (None, None)

rows, in_table = {}, False
for line in open(log):
    line = line.rstrip("\n").split("\r")[-1]
    if line.startswith("TARGET") and "GRP" in line:
        in_table = True
        continue
    if not in_table or line.startswith("---"):
        continue
    if not line.strip():
        break
    t, grp = line[:10].strip(), line[10:16].strip()
    r2, rho, r = (float(line[16 + 8 * i:24 + 8 * i]) for i in range(3))
    if t in rows:
        sys.exit(f"{t} is scored twice in {log}")
    n = len(np.load(f"{target_dir}/{t}/saved_labels.npy"))
    rows[t] = {"target_id": t, "n_actives": "", "n_total": n, "group": grp,
               "pearson_r": r, "spearman_rho": rho, "r2": r2}

cols = ["target_id", "n_actives", "n_total", "auroc", "bedroc",
        "ef_0.005", "ef_0.01", "ef_0.02", "ef_0.05", "group", "pearson_r", "spearman_rho", "r2"]


def write(path, keep):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols, restval="")
        w.writeheader()
        for t in sorted(keep):
            w.writerow(rows[t])
    print(f"{len(keep)} targets -> {path}")
    for c in ("pearson_r", "spearman_rho", "r2"):
        print(f"  mean {c} = {np.mean([rows[t][c] for t in keep])}")


write(out, list(rows))
if group:
    write(out_group, [t for t in rows if rows[t]["group"] == group])
