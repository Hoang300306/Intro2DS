"""Regression tests for tools/apply_variant.py.

The column-value path is the one the two SPRINT policy specs need; it used to raise
KeyError 'indices' right after verifying the source, so that variant could not be built.
--check-source is what moves the download check to checkpoint A, before a B run is paid for.

Run: python scripts/test_applier.py
"""
import gzip, json, pathlib, shutil, subprocess, sys, tempfile, importlib.util
T = pathlib.Path(tempfile.mkdtemp(prefix="apply_variant_test_"))
AP = str(pathlib.Path(__file__).resolve().parent.parent / "tools" / "apply_variant.py")
spec_ = importlib.util.spec_from_file_location("ap", AP); m = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(m)

rows = [("1", "P1"), ("2", "P2"), ("3", "P1"), ("4", "P3"), ("5", "P2"), ("6", "P1")]
src = T / "in.tsv"
src.write_text("ligand\taa_seq\n" + "".join(f"{a}\t{b}\n" for a, b in rows))
keep = [r for r in rows if r[0] not in {"2", "5"}]

spec = {"variant_id": "t_policy", "format": "tsv", "operation": "drop_column_values",
        "separator": "\t", "column": "ligand", "values": [2, 5],
        "source": {"file": "in.tsv", "n_records": len(rows),
                   "content_digest": m.content_digest_records(rows)},
        "expected_output": {"n_records": len(keep),
                            "content_digest": m.content_digest_records(keep)}}
sp = T / "spec.json.gz"
sp.write_bytes(gzip.compress(json.dumps(spec).encode()))

def run(args):
    r = subprocess.run([sys.executable, "-I", AP, *args], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()

print("--- 1. drop_column_values"); rc, o = run([str(sp), str(src), str(T / "out.tsv")])
print("rc", rc); print(o)
got = (T / "out.tsv").read_text().splitlines()
assert rc == 0 and len(got) == 5 and got[0] == "ligand\taa_seq", got
assert [l.split("\t")[0] for l in got[1:]] == ["1", "3", "4", "6"], got

print("\n--- 2. --check-source, no dst (must not write)")
out2 = T / "must_not_exist.tsv"
rc, o = run([str(sp), str(src), "--check-source"])
print("rc", rc); print(o)
assert rc == 0 and not out2.exists()

print("\n--- 3. wrong column name")
bad = dict(spec); bad["column"] = "ligand_id"
bp = T / "bad.json.gz"; bp.write_bytes(gzip.compress(json.dumps(bad).encode()))
rc, o = run([str(bp), str(src), str(T / "x.tsv")])
print("rc", rc); print(o[-300:])
assert rc == 2 and "header does not" in o and not (T / "x.tsv").exists()

print("\n--- 4. missing dst without --check-source")
rc, o = run([str(sp), str(src)])
print("rc", rc); print(o[-200:])
assert rc == 2 and "--check-source" in o

print("\n--- 5. index spec still works")
spec2 = {"variant_id": "t_matched", "format": "tsv", "operation": "drop_indices",
         "separator": "\t", "indices": [1, 4],
         "source": spec["source"], "expected_output": spec["expected_output"]}
p2 = T / "spec2.json.gz"; p2.write_bytes(gzip.compress(json.dumps(spec2).encode()))
rc, o = run([str(p2), str(src), str(T / "out2.tsv")])
print("rc", rc); print(o)
assert rc == 0 and (T / "out2.tsv").read_text() == (T / "out.tsv").read_text()
shutil.rmtree(T, ignore_errors=True)
print("\nALL TSV TESTS PASS")
