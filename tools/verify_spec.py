"""Verify a spec against the real corpus file it was built from, writing nothing.

This is the audit tool, not something an assignee needs: it requires the released corpus, so it
cannot run in CI. Use it whenever a spec is built or changed, from a machine that holds the data.

The bug it exists to find: the spec builder and apply_variant.py compute the recorded digest
independently, so a spec can carry a digest no version of the applier could ever reproduce. That
happened to six of the sixteen specs -- the two pocketaffdb pdbbind files and all four SPRINT
files. A valid selection with a bad digest is a digest bug on our side, not a bad download, so the
three verdicts are reported separately:

  SOURCE  the recorded source digest against what the applier recomputes
  COUNT   the recorded record count against the file
  OUTPUT  the recorded output digest against the replayed selection

COUNT passing while SOURCE fails is the signature of a digest bug. All three failing, or COUNT
failing, means the file really is a different release.

Usage: python tools/verify_spec.py <dir holding apply_variant.py> <spec.json.gz> <source path>
                                   [--stream-tsv]

--stream-tsv sorts the serialised records on disk instead of holding them in memory, for a file
too large to fit (the SPRINT negative set is 64.4 million rows). It produces the same digest: \x1f
sorts below any printable byte, so byte order over the serialised form equals Python order over
the tuples. Both paths were run against the same file and agree.
"""
import gzip, importlib.util, json, os, pathlib, subprocess, sys, tempfile

AP_DIR, SPEC, SRC = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
STREAM = "--stream-tsv" in sys.argv
s = importlib.util.spec_from_file_location("ap", pathlib.Path(AP_DIR) / "apply_variant.py")
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
cd, spec = m.content_digest_records, json.loads(gzip.decompress(SPEC.read_bytes()))
exp, want, fmt = spec["source"], spec["expected_output"], spec["format"]
R = {"spec": SPEC.name, "variant": spec["variant_id"], "format": fmt, "src": str(SRC)}


def sel(n):
    """The selection, replayed. Mirrors selected() but reports instead of exiting."""
    op = spec["operation"]
    if op in ("keep_indices", "drop_indices"):
        idx = spec["indices"]
        bad = [i for i in idx if not 0 <= i < n]
        if bad:
            return None, f"index {bad[0]} out of range for {n} records"
        return (sorted(idx) if op == "keep_indices"
                else [i for i in range(n) if i not in set(idx)]), None
    return None, f"operation {op} needs the handler, not the index selector"


def stream_digest_counted(rows_iter):
    """stream_digest, and the number of records it saw."""
    n = [0]

    def counting():
        for r in rows_iter:
            n[0] += 1
            yield r
    return n[0] if False else (lambda d: (n[0], d))(stream_digest(counting()))


def stream_digest(rows_iter):
    """content_digest_records for a file too large to hold: sort the serialised records on disk.
    \x1f sorts below any printable byte, so byte order over the serialised form equals Python's
    order over the tuples."""
    import hashlib
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".ser") as f:
        tmp = f.name
        for t in rows_iter:
            f.write("\x1f".join("" if x is None else str(x) for x in t) + "\x1e\n")
    env = dict(os.environ, LC_ALL="C")
    srt = tmp + ".sorted"
    subprocess.run(["sort", "-o", srt, tmp], check=True, env=env)
    h = hashlib.sha256()
    with open(srt) as f:
        for line in f:
            h.update(line[:-1].encode())
    os.unlink(tmp); os.unlink(srt)
    return h.hexdigest()


try:
    if fmt == "lmdb":
        import lmdb, pickle
        env = lmdb.open(str(SRC), subdir=False, readonly=True, lock=False, readahead=False)
        ident, fields = [], None
        with env.begin() as txn:
            for _, v in txn.cursor():
                d = pickle.loads(v)
                if fields is None:
                    fields = sorted(d)
                ident.append((d.get("smi"), str(d.get("pocket"))))
        R["n"], R["fields_ok"] = len(ident), (fields == sorted(exp.get("record_fields") or fields))
        R["source_digest"] = cd(ident)
        keep, err = sel(len(ident))
        R["select_error"] = err
        if keep is not None:
            R["n_out"], R["output_digest"] = len(keep), cd([ident[i] for i in keep])
    elif fmt == "json_labels":
        data = json.load(open(SRC))
        R["n"] = len(data)
        R["source_digest"] = cd([m.json_ident(r) for r in data])
        km = {int(k): v for k, v in spec["ligand_keep"].items()}
        out, bad = [], 0
        for i, rec in enumerate(data):
            if i not in km:
                continue
            ligs = rec.get("ligands") or []
            if any(j >= len(ligs) for j in km[i]):
                bad += 1; continue
            out.append({**rec, "ligands": [ligs[j] for j in km[i]]})
        R["select_error"] = f"{bad} entries have out-of-range ligand indices" if bad else None
        R["n_out"] = len(out)
        R["n_rows_out"] = sum(len(r["ligands"]) for r in out)
        R["output_digest"] = cd([m.json_ident(r) for r in out])
    elif fmt in ("tsv", "csv"):
        import csv as _csv
        sep = "\t" if spec.get("separator", "\t") == "\t" else ","
        with open(SRC, newline="") as f:
            rd = _csv.reader(f, delimiter=sep)
            header = next(rd)
            if STREAM:
                def rows():
                    with open(SRC, newline="") as g:
                        r2 = _csv.reader(g, delimiter=sep); next(r2)
                        for row in r2:
                            yield tuple(row)
                R["n"], R["source_digest"] = stream_digest_counted(rows())
                op = spec["operation"]
                if op == "drop_column_values":
                    if spec["column"] not in header:
                        R["select_error"] = f"column {spec['column']} not in header {header}"
                    else:
                        ci, drop = header.index(spec["column"]), {str(v) for v in spec["values"]}
                        R["select_error"] = None
                        R["n_out"], R["output_digest"] = stream_digest_counted(
                            r for r in rows() if r[ci] not in drop)
                elif op in ("drop_indices", "keep_indices"):
                    idx = set(spec["indices"])
                    hit = ((lambda i: i not in idx) if op == "drop_indices"
                           else (lambda i: i in idx))
                    R["select_error"] = None
                    R["n_out"], R["output_digest"] = stream_digest_counted(
                        r for i, r in enumerate(rows()) if hit(i))
                else:
                    R["select_error"] = f"operation {op} not streamable"
            else:
                rows = [tuple(r) for r in rd]
                R["n"] = len(rows)
                R["source_digest"] = cd(rows)
                if spec["operation"] == "drop_column_values":
                    ci = header.index(spec["column"]) if spec["column"] in header else None
                    if ci is None:
                        R["select_error"] = f"column {spec['column']} not in header {header}"
                        keep = None
                    else:
                        drop = {str(v) for v in spec["values"]}
                        keep, R["select_error"] = [i for i, r in enumerate(rows)
                                                   if r[ci] not in drop], None
                else:
                    keep, R["select_error"] = sel(len(rows))
                if keep is not None:
                    R["n_out"], R["output_digest"] = len(keep), cd([rows[i] for i in keep])
    elif fmt == "smiles_dir":
        # mirror do_smiles_dir exactly: two levels, blank lines dropped, per-file line digest
        content = {}
        for d in sorted(x for x in pathlib.Path(SRC).iterdir() if x.is_dir()):
            for f in sorted(x for x in d.iterdir() if x.is_file()):
                content[f"{d.name}/{f.name}"] = [x.strip() for x in f.read_text().splitlines()
                                                 if x.strip()]
        R["n_files"] = len(content)
        R["expect_n_files"] = exp.get("n_files")
        R["n"] = sum(len(v) for v in content.values())
        R["source_digest"] = cd([(k, m._lines_digest(v)) for k, v in content.items()])
        op = spec["operation"]
        out_lines, err = {}, None
        for key, lines in content.items():
            if op == "drop_smiles_lines":
                drop = set(spec["values"])
                keep = [i for i, x in enumerate(lines) if x not in drop]
            elif op == "deterministic_subsample":
                k = spec["n_keep"].get(key)
                if k is None:
                    err = f"spec has no keep count for {key}"; break
                keep = m.pick(len(lines), k, key, spec["seed"])
            else:
                err = f"unsupported operation {op}"; break
            out_lines[key] = [lines[i] for i in keep]
        R["select_error"] = err
        if err is None:
            R["n_out"] = sum(len(v) for v in out_lines.values())
            R["output_digest"] = cd([(k, m._lines_digest(v)) for k, v in out_lines.items()])
    else:
        R["select_error"] = f"unknown format {fmt}"
except Exception as e:
    R["exception"] = f"{type(e).__name__}: {e}"

R["expect_n"] = exp.get("n_records")
R["expect_source_digest"] = exp.get("content_digest")
R["expect_n_out"] = want.get("n_records")
R["expect_output_digest"] = want.get("content_digest")
R["SOURCE"] = ("PASS" if R.get("source_digest") == R["expect_source_digest"]
               else "FAIL" if R.get("source_digest") else "?")
R["COUNT"] = "PASS" if R.get("n") == R["expect_n"] else "FAIL"
R["OUTPUT"] = ("PASS" if R.get("output_digest") == R["expect_output_digest"]
               else "FAIL" if R.get("output_digest") else "?")
print(json.dumps(R))
