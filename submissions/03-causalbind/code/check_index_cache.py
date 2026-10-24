"""Check that a released index file (smi -> idx or pocket name -> idx) matches an LMDB.

PairDataset reads DATA_DIR/cache/{mol_smi2idx,pocket_name2idx}_train_blend.json, and if it does
not find them rebuilds them by running every LMDB record through the dataset pipeline. The
released copies are downloaded into data/, not data/cache/. Before using them in place of the
rebuild, this checks a fixed-seed sample of entries.

train_task.py reads the training LMDBs with unicore.data.LMDBDataset, whose item idx is the
record at position idx in LMDB key order (self._keys[idx], keys sorted as bytes: '0', '1',
'10', ...). It is not the record stored under the key str(idx), which is how
unimol/data/lmdb_dataset.py (used by the evaluator) indexes. The check follows train_task.py.

Usage: python check_index_cache.py LMDB INDEX_JSON FIELD N SEED
  FIELD  smi for the ligand LMDB, pocket for the pocket LMDB
"""
import json
import pickle
import random
import sys

import lmdb

db, index_file, field, n, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
env = lmdb.open(db, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
with env.begin() as txn:
    keys = list(txn.cursor().iternext(values=False))
index = json.load(open(index_file))
print(f"{db}: {len(keys)} records; {index_file}: {len(index)} names, largest idx {max(index.values())}")
if max(index.values()) >= len(keys):
    sys.exit("the index points past the end of the LMDB")

bad = 0
with env.begin() as txn:
    for name in random.Random(seed).sample(sorted(index), min(n, len(index))):
        rec = pickle.loads(txn.get(keys[index[name]]))
        if rec.get(field) != name:
            bad += 1
            if bad <= 5:
                print(f"  mismatch: {name!r} -> idx {index[name]} holds {rec.get(field)!r}")
print(f"{min(n, len(index))} sampled, {bad} mismatched")
sys.exit(1 if bad else 0)
