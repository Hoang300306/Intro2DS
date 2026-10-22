"""Regression tests for validate_submission.py.

Each case is something the validator used to get wrong: it accepted a submission with no
per-target CSV and no log as long as a gap of 0 was declared, it raised KeyError on every
checkpoint-C summary because it read the manifest with the wrong field names, and it applied
the 10 % gate to every benchmark instead of the one the sheet asked to be named.

Run: python scripts/test_validator.py
"""
import json, pathlib, shutil, subprocess, sys
R = pathlib.Path(__file__).resolve().parent.parent
S = R / "submissions" / "_zz_test"
V = [sys.executable, "-I", str(R / "scripts" / "validate_submission.py")]

def run():
    r = subprocess.run(V + [str(S)], capture_output=True, text=True, cwd=R)
    return r.returncode, (r.stdout + r.stderr)

def fresh():
    shutil.rmtree(S, ignore_errors=True)
    for st in ("A1_env", "A2_reproduce", "A3_smoke"):
        (S / st).mkdir(parents=True)
    c = S / "code"; c.mkdir()
    (c / "UPSTREAM.txt").write_text("https://example.invalid/x\ncommit 0123456789abcdef0123456789abcdef01234567\n")
    (c / "run.sh").write_text("echo hi\n")
    (c / "changes.diff").write_text("")
    (c / "frozen.sha256").write_text("deadbeef  unimol/tasks/x.py\n")
    (S / "REPORT.md").write_text("# r\n")

def summ(stage, **kw):
    d = {"stage": stage, "model": "m", "run": {"seed": 1, "checkpoint_rule": "best"}}
    d.update(kw); return d

print("=" * 70, "\nCASE 1 (Codex): no CSV, no log, only relative_gap_pct=0\n", "=" * 70)
fresh()
for st in ("A1_env", "A3_smoke"):
    (S / st / "summary.json").write_text(json.dumps(summ(st)))
(S / "A2_reproduce" / "summary.json").write_text(json.dumps(summ(
    "A2_reproduce", results=[{"benchmark": "DUD-E", "n_targets": 102, "per_target_csv": "",
                              "metrics": {"auroc": 0.8},
                              "published_reference": {"metric": "auroc", "value": 0.8,
                                                      "source": "Table 1",
                                                      "relative_gap_pct": 0}}])))
rc, out = run()
print(out.strip()[:1400])
assert rc != 0, "an empty submission still passes"
for must in ("per_target_csv is empty", "log.txt is missing", "gate = true"):
    assert must in out, f"missing complaint: {must}"
print("\n-> now rejected, for the right three reasons")

print("\n" + "=" * 70, "\nCASE 2 (Codex): checkpoint C raised KeyError 'path'\n", "=" * 70)
man = json.load(open(R / "splits" / "MANIFEST.json"))
vid = "drugclip_train_ligandish"
f0 = man[vid]["files"][0]
fresh()
for st in ("A1_env", "A3_smoke"):
    (S / st / "summary.json").write_text(json.dumps(summ(st)))
(S / "A2_reproduce" / "summary.json").write_text(json.dumps(summ(
    "A2_reproduce", results=[{"benchmark": "DUD-E", "n_targets": 1, "per_target_csv": "pt.csv",
                              "metrics": {}, "published_reference": {
                                  "metric": "auroc", "value": 0.8, "source": "T1",
                                  "relative_gap_pct": 1.0, "gate": True}}])))
hdr = ",".join(["target_id", "n_actives", "n_total", "auroc", "bedroc",
                "ef_0.005", "ef_0.01", "ef_0.02", "ef_0.05"])
csv_rows = hdr + "\n" + "\n".join(f"t{i},5,100,0.8,0.5,1,2,3,4" for i in range(102)) + "\n"
for st in ("A2_reproduce", "B_retrain_released", "C_retrain_variants"):
    (S / st).mkdir(exist_ok=True)
    (S / st / "log.txt").write_text("ran\n")
(S / "A2_reproduce" / "pt.csv").write_text(csv_rows)
(S / "A2_reproduce" / "summary.json").write_text(json.dumps(summ(
    "A2_reproduce", results=[{"benchmark": "DUD-E", "n_targets": 102, "per_target_csv": "pt.csv",
                              "metrics": {}, "published_reference": {
                                  "metric": "auroc", "value": 0.8, "source": "T1",
                                  "relative_gap_pct": 1.0, "gate": True}}])))
(S / "B_retrain_released" / "pt.csv").write_text(csv_rows)
(S / "B_retrain_released" / "summary.json").write_text(json.dumps(summ(
    "B_retrain_released", training_data={"split_id": "released"},
    results=[{"benchmark": "DUD-E", "n_targets": 102, "per_target_csv": "pt.csv", "metrics": {}}])))
cdir = S / "C_retrain_variants" / vid; cdir.mkdir(parents=True)
(cdir / "log.txt").write_text("ran\n"); (cdir / "pt.csv").write_text(csv_rows)
(cdir / "summary.json").write_text(json.dumps(summ(
    "C_retrain_variants",
    training_data={"split_id": vid,
                   "files": [{"path": f0["replaces_file"], "content_digest": "WRONG" * 12}]},
    results=[{"benchmark": "DUD-E", "n_targets": 102, "per_target_csv": "pt.csv", "metrics": {}}])))
rc, out = run()
print(out.strip()[:1200])
assert "KeyError" not in out and "Traceback" not in out, "still crashing"
assert "does not match the published manifest" in out, "digest comparison did not run"
print("\n-> no crash; the wrong digest is caught")

print("\nCASE 3: the same submission with the real digest must pass that check")
(cdir / "summary.json").write_text(json.dumps(summ(
    "C_retrain_variants",
    training_data={"split_id": vid,
                   "files": [{"path": f0["replaces_file"],
                              "content_digest": f0["content_digest"]}]},
    results=[{"benchmark": "DUD-E", "n_targets": 102, "per_target_csv": "pt.csv", "metrics": {}}])))
rc, out = run()
print(out.strip()[:800])
assert "does not match the published manifest" not in out and "missing a verified entry" not in out
print("\n-> accepted")
shutil.rmtree(S, ignore_errors=True)
print("\nALL VALIDATOR TESTS PASS")


print("\n" + "=" * 70, "\nCASE 4: a gate blocked by an unpublished artifact passes on its reason\n", "=" * 70)
fresh()
for st in ("A1_env", "A3_smoke"):
    (S / st / "summary.json").write_text(json.dumps(summ(st)))
hdr2 = ",".join(["target_id", "n_actives", "n_total", "auroc", "bedroc",
                 "ef_0.005", "ef_0.01", "ef_0.02", "ef_0.05"])
csv2 = hdr2 + "\n" + "\n".join(f"t{i},5,100,0.8,0.5,1,2,3,4" for i in range(102)) + "\n"
(S / "A2_reproduce" / "pt.csv").write_text(csv2)
(S / "A2_reproduce" / "log.txt").write_text("ran\n")
REASON = ("the checkpoint the paper scored was never released and the benchmark CSVs its "
          "evaluator reads were never released either; searched the repository releases and "
          "every related HuggingFace repository")


def a2(ref):
    (S / "A2_reproduce" / "summary.json").write_text(json.dumps(summ(
        "A2_reproduce", results=[{"benchmark": "DUD-E", "n_targets": 102,
                                  "per_target_csv": "pt.csv", "metrics": {},
                                  "published_reference": ref}])))
    return run()


rc, out = a2({"metric": "ef_0.01", "value": 51.44, "source": "Table 2",
              "gate": True, "blocked": REASON})
assert "relative_gap_pct is" not in out, out
assert "gate = true" not in out, out
assert "blocked needs a real explanation" not in out, out
print("-> accepted with no gap at all, because the reason is on the record")

rc, out = a2({"metric": "ef_0.01", "value": 51.44, "source": "Table 2",
              "gate": True, "blocked": "not available"})
assert "blocked needs a real explanation" in out, out
print("-> a one-line excuse is rejected")

rc, out = a2({"metric": "ef_0.01", "value": 51.44, "source": "Table 2", "gate": True})
assert "relative_gap_pct is" in out, out
print("-> and omitting both is still a complaint")
shutil.rmtree(S, ignore_errors=True)
print("\nALL BLOCKED-GATE TESTS PASS")
