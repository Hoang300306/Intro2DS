"""Build a training-set variant from the released training file and a variant spec.

You already have the released training file, because checkpoint B retrains on it. A variant differs
from it only in which records it keeps, so that is all a spec contains. This script applies one.

It checks the released file against the digest we recorded **before** writing anything, and checks
its own output against the digest we recorded afterwards. Both matching means your released download
is the same file we worked from and the filter applied correctly. A mismatch means stop and tell us —
it is far cheaper to find out here than after a training run.

Digests are over record *content*, sorted, not over file bytes: two LMDBs holding identical records
can differ byte-for-byte across library versions, and we do not want to send you chasing that.

    python tools/apply_variant.py <spec.json.gz> <released_file> <output_file>

For a spec covering several files (one per released file), pass them in the order the spec lists:

    python tools/apply_variant.py <spec.json.gz> --multi <released_dir> <output_dir>

Requires: Python 3.9+, and the `lmdb` package for LMDB specs (which the model already needs).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import pathlib
import sys


def content_digest_records(items) -> str:
    h = hashlib.sha256()
    for t in sorted(items):
        h.update(("\x1f".join("" if x is None else str(x) for x in t) + "\x1e").encode())
    return h.hexdigest()


_CLEANUP: list[pathlib.Path] = []


def die(msg: str) -> "None":
    for p in _CLEANUP:
        for q in (p, p.with_name(p.name + "-lock")):
            try:
                if q.exists():
                    q.unlink()
                    print(f"removed {q} so it cannot be mistaken for a usable file",
                          file=sys.stderr, flush=True)
            except OSError:
                pass
    print(f"\nSTOP: {msg}\n", file=sys.stderr, flush=True)
    print("Do not train on this. Open an issue with this message and the command you ran.",
          file=sys.stderr)
    raise SystemExit(2)


CHECK_ONLY = False


def source_verified(msg: str) -> None:
    """Every handler reports its source check here, so --check-source can stop at exactly the
    point where the download has been proven and nothing has been written yet."""
    print(f"source verified: {msg}", flush=True)
    if CHECK_ONLY:
        print("\n--check-source only: your released file matches the input this variant was built "
              "from. Nothing was written. Record this digest in your checkpoint-A report.",
              flush=True)
        raise SystemExit(0)


def load_spec(p: pathlib.Path) -> dict:
    raw = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
    return json.loads(raw)


def selected(spec: dict, n_total: int) -> list[int]:
    if spec["operation"] not in ("keep_indices", "drop_indices"):
        die(f"operation {spec['operation']!r} reached the index selector, which only understands "
            "keep_indices and drop_indices. That is a bug in apply_variant.py and not a problem "
            "with your file -- open an issue quoting this message.")
    idx = spec["indices"]
    if spec["operation"] == "keep_indices":
        keep = sorted(idx)
    elif spec["operation"] == "drop_indices":
        drop = set(idx)
        keep = [i for i in range(n_total) if i not in drop]
    else:
        die(f"unknown operation {spec['operation']!r}")
    bad = [i for i in idx if not 0 <= i < n_total]
    if bad:
        die(f"spec refers to record {bad[0]}, but the file has {n_total} records. "
            "This is the wrong source file for this spec.")
    return keep


# ---------------------------------------------------------------- lmdb
def do_lmdb(spec: dict, src: pathlib.Path, dst: pathlib.Path) -> None:
    try:
        import lmdb
    except ImportError:
        die("the `lmdb` package is required for this spec (pip install lmdb)")
    import pickle

    env = lmdb.open(str(src), subdir=False, readonly=True, lock=False, readahead=False)
    recs = []
    with env.begin() as txn:
        for _, v in txn.cursor():
            d = pickle.loads(v)
            recs.append((v, (d.get("smi"), str(d.get("pocket")))))
    n = len(recs)
    exp = spec["source"]
    want_fields = exp.get("record_fields")
    if want_fields is not None:
        with env.begin() as txn:
            first = pickle.loads(next(iter(txn.cursor()))[1])
        got_fields = sorted(first)
        if got_fields != sorted(want_fields):
            extra = sorted(set(got_fields) - set(want_fields))
            missing = sorted(set(want_fields) - set(got_fields))
            die("this file has the wrong record schema for the spec.\n"
                f"  unexpected fields: {extra or 'none'}\n"
                f"  missing fields:    {missing or 'none'}\n"
                "Two releases can share the same molecules and pockets while carrying different "
                "fields, so check you are using the training file for YOUR model.")
    if n != exp["n_records"]:
        die(f"your file has {n:,} records, the spec expects {exp['n_records']:,}. "
            "This is a different release of the training data.")
    got = content_digest_records([r[1] for r in recs])
    if got != exp["content_digest"]:
        die(f"source content digest mismatch.\n  expected {exp['content_digest']}\n  "
            f"got      {got}\nYour released training file differs from the one the variant was "
            "built from.")
    source_verified(f"{n:,} records, digest {got}")

    keep = selected(spec, n)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    _CLEANUP.append(dst)
    w = lmdb.open(str(dst), subdir=False, map_size=1 << 40)
    with w.begin(write=True) as txn:
        for newk, i in enumerate(keep):
            txn.put(str(newk).encode(), recs[i][0])
    w.sync(); w.close()

    out_dig = content_digest_records([recs[i][1] for i in keep])
    want = spec["expected_output"]
    if len(keep) != want["n_records"] or out_dig != want["content_digest"]:
        die(f"output does not match the spec.\n  records {len(keep):,} vs {want['n_records']:,}"
            f"\n  digest  {out_dig}\n          {want['content_digest']}")
    print(f"wrote {dst}  {len(keep):,} records  digest {out_dig}  VERIFIED", flush=True)


# ---------------------------------------------------------------- json label lists
def do_json_labels(spec: dict, src: pathlib.Path, dst: pathlib.Path) -> None:
    data = json.load(open(src))
    ident = [(r.get("assay_id") or r.get("uniprot") or "", len(r.get("ligands") or []))
             for r in data]
    exp = spec["source"]
    if len(data) != exp["n_records"]:
        die(f"your file has {len(data):,} entries, the spec expects {exp['n_records']:,}")
    got = content_digest_records(ident)
    if got != exp["content_digest"]:
        die(f"source content digest mismatch\n  expected {exp['content_digest']}\n  got      {got}")
    source_verified(f"{len(data):,} entries, digest {got}")

    # per-entry ligand keep lists; an entry absent from the map is dropped entirely
    keepmap = {int(k): v for k, v in spec["ligand_keep"].items()}
    out = []
    for i, rec in enumerate(data):
        if i not in keepmap:
            continue
        r = dict(rec)
        ligs = r.get("ligands") or []
        r["ligands"] = [ligs[j] for j in keepmap[i]]
        out.append(r)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _CLEANUP.append(dst)
    json.dump(out, open(dst, "w"))
    rows = sum(len(r.get("ligands") or []) for r in out)
    want = spec["expected_output"]
    if len(out) != want["n_records"] or rows != want["n_rows"]:
        die(f"output mismatch: {len(out):,}/{rows:,} entries/rows vs "
            f"{want['n_records']:,}/{want['n_rows']:,}")
    out_dig = content_digest_records(
        [(r.get("assay_id") or r.get("uniprot") or "", len(r.get("ligands") or [])) for r in out])
    print(f"wrote {dst}  {len(out):,} entries / {rows:,} rows  digest {out_dig}  VERIFIED",
          flush=True)


# ---------------------------------------------------------------- tsv / csv
def do_table(spec: dict, src: pathlib.Path, dst: pathlib.Path) -> None:
    sep = "\t" if spec.get("separator", "\t") == "\t" else ","
    with open(src, newline="") as f:
        rd = csv.reader(f, delimiter=sep)
        header = next(rd)
        rows = [tuple(r) for r in rd]
    if len(rows) > 5_000_000:
        print(f"note: {len(rows):,} rows held in memory; this needs several GB of RAM and some "
              "minutes. If it is killed, say so in an issue rather than trimming the file.",
              flush=True)
    exp = spec["source"]
    if len(rows) != exp["n_records"]:
        die(f"your file has {len(rows):,} rows, the spec expects {exp['n_records']:,}")
    got = content_digest_records(rows)
    if got != exp["content_digest"]:
        die(f"source content digest mismatch\n  expected {exp['content_digest']}\n  got      {got}")
    source_verified(f"{len(rows):,} rows, digest {got}")

    if spec["operation"] == "drop_column_values":
        col = spec["column"]
        if col not in header:
            die(f"the spec selects rows by the column {col!r}, which this file's header does not "
                f"have (it has: {', '.join(header)}). This is the wrong source file for the spec.")
        ci = header.index(col)
        drop = {str(v) for v in spec["values"]}
        keep = [i for i, r in enumerate(rows) if r[ci] not in drop]
    else:
        keep = selected(spec, len(rows))
    dst.parent.mkdir(parents=True, exist_ok=True)
    _CLEANUP.append(dst)
    with open(dst, "w", newline="") as f:
        w = csv.writer(f, delimiter=sep)
        w.writerow(header)
        for i in keep:
            w.writerow(rows[i])
    out_dig = content_digest_records([rows[i] for i in keep])
    want = spec["expected_output"]
    if len(keep) != want["n_records"] or out_dig != want["content_digest"]:
        die(f"output mismatch\n  rows   {len(keep):,} vs {want['n_records']:,}"
            f"\n  digest {out_dig}\n         {want['content_digest']}")
    print(f"wrote {dst}  {len(keep):,} rows  digest {out_dig}  VERIFIED", flush=True)



def _lines_digest(lines) -> str:
    """Digest a file's lines in order. Counts alone are not enough: a size-matched control has the
    same per-file counts as the policy by construction, so a count-based digest would accept one in
    place of the other."""
    h = hashlib.sha256()
    for x in lines:
        h.update(x.encode()); h.update(b"\n")
    return h.hexdigest()


def pick(n: int, k: int, key: str, seed: int) -> list:
    """Deterministic k-of-n: indices ordered by sha256(key:index:seed), first k, sorted.

    Used instead of shipping an index list where the list would be tens of megabytes. It depends on
    nothing but hashlib, so it gives the same answer on any platform and any Python version.
    """
    if k >= n:
        return list(range(n))
    order = sorted(range(n), key=lambda i: hashlib.sha256(
        f"{key}:{i}:{seed}".encode()).digest())
    return sorted(order[:k])


# ---------------------------------------------------------------- per-target SMILES directories
def do_smiles_dir(spec: dict, src: pathlib.Path, dst: pathlib.Path) -> None:
    """One directory per target, each holding plain-text SMILES files, one molecule per line."""
    content = {}
    for d in sorted(x for x in src.iterdir() if x.is_dir()):
        for f in sorted(x for x in d.iterdir() if x.is_file()):
            lines = [x.strip() for x in f.read_text().splitlines() if x.strip()]
            content[f"{d.name}/{f.name}"] = lines
    exp = spec["source"]
    if len(content) != exp["n_files"]:
        die(f"your tree has {len(content):,} files, the spec expects {exp['n_files']:,}")
    tot = sum(len(v) for v in content.values())
    if tot != exp["n_records"]:
        die(f"your tree has {tot:,} lines, the spec expects {exp['n_records']:,}")
    got = content_digest_records([(k, _lines_digest(v)) for k, v in content.items()])
    if got != exp["content_digest"]:
        die(f"source content digest mismatch\n  expected {exp['content_digest']}\n  got      {got}")
    source_verified(f"{len(content):,} files, {tot:,} lines, digest {got}")

    op = spec["operation"]
    out_lines = {}
    for key, lines in content.items():
        if op == "drop_smiles_lines":
            drop = set(spec["values"])
            keep = [i for i, x in enumerate(lines) if x not in drop]
        elif op == "deterministic_subsample":
            k = spec["n_keep"].get(key)
            if k is None:
                die(f"spec has no keep count for {key}")
            keep = pick(len(lines), k, key, spec["seed"])
        else:
            die(f"unsupported operation {op!r} for a smiles directory")
        t, f = key.split("/", 1)
        d = dst / t
        d.mkdir(parents=True, exist_ok=True)
        _CLEANUP.append(d / f)
        (d / f).write_text("\n".join(lines[i] for i in keep) + ("\n" if keep else ""))
        out_lines[key] = [lines[i] for i in keep]

    want = spec["expected_output"]
    out_dig = content_digest_records([(k, _lines_digest(v)) for k, v in out_lines.items()])
    n_out = sum(len(v) for v in out_lines.values())
    if n_out != want["n_records"] or out_dig != want["content_digest"]:
        die(f"output mismatch\n  lines  {n_out:,} vs {want['n_records']:,}"
            f"\n  digest {out_dig}\n         {want['content_digest']}")
    print(f"wrote {dst}  {len(out_lines):,} files / {n_out:,} lines  "
          f"digest {out_dig}  VERIFIED", flush=True)
    print("\nNOTE: this corpus ships a `processed/` tree derived from these SMILES. Delete it so "
          "your code regenerates it, and regenerate it for your released-data run too -- otherwise "
          "one arm uses shipped features and the other freshly computed ones.", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?",
                    help="omit together with --check-source to verify the input only")
    ap.add_argument("--check-source", action="store_true",
                    help="verify your released file against the spec and exit without writing")
    a = ap.parse_args()
    global CHECK_ONLY
    CHECK_ONLY = a.check_source
    if not CHECK_ONLY and a.dst is None:
        die("give an output path, or pass --check-source to only verify the input")
    spec = load_spec(pathlib.Path(a.spec))
    print(f"variant {spec['variant_id']}  format {spec['format']}", flush=True)
    fn = {"lmdb": do_lmdb, "json_labels": do_json_labels, "tsv": do_table, "csv": do_table,
          "smiles_dir": do_smiles_dir}.get(spec["format"])
    if fn is None:
        die(f"unsupported format {spec['format']!r}")
    fn(spec, pathlib.Path(a.src), pathlib.Path(a.dst) if a.dst else pathlib.Path(os.devnull))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
