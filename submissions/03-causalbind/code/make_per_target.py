"""Build a per-target CSV (contract rule 3) from a CausalBind test.py log.

The metrics are read from the per-target lines test_task.py prints; nothing is recomputed.
n_total is the num_mols the same log prints, n_actives is the count of 1s in the
saved_labels.npy that test_task.py wrote for that target.

Usage: python make_per_target.py TAG LOG TARGET_DIR OUT_CSV
  TAG         log prefix: DUD-E, PCBA or DEKOIS
  TARGET_DIR  the directory holding one sub-directory per target, e.g. results/A2_sp_DUDE/DUDE
"""
import ast
import csv
import re
import sys

import numpy as np

tag, log, target_dir, out = sys.argv[1:5]
p = re.escape(tag)
start = re.compile(rf"\[{p}\] target=(\S+), num_mols=(\d+)")
done = re.compile(rf"\[{p}\] (\S+) AUC=([\d.]+), BEDROC=([\d.]+), EF=(\{{[^}}]*\}})")

n_mols, rows = {}, {}
for line in open(log):
    line = line.rstrip("\n").split("\r")[-1]  # drop tqdm progress written on the same line
    m = start.search(line)
    if m:
        n_mols[m[1]] = int(m[2])
        continue
    m = done.search(line)
    if m:
        t = m[1]
        if t in rows:
            sys.exit(f"{t} is scored twice in {log}")
        ef = ast.literal_eval(m[4])
        labels = np.load(f"{target_dir}/{t}/saved_labels.npy")
        if len(labels) != n_mols[t]:
            sys.exit(f"{t}: {len(labels)} labels but the log says num_mols={n_mols[t]}")
        rows[t] = [t, int((labels == 1).sum()), n_mols[t], m[2], m[3],
                   ef["0.005"], ef["0.01"], ef["0.02"], ef["0.05"]]

cols = ["target_id", "n_actives", "n_total", "auroc", "bedroc",
        "ef_0.005", "ef_0.01", "ef_0.02", "ef_0.05"]
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for t in sorted(rows):
        w.writerow(rows[t])

print(f"{len(rows)} targets -> {out}")
for i, c in enumerate(cols[3:], start=3):
    print(f"  mean {c} = {np.mean([float(r[i]) for r in rows.values()])}")
