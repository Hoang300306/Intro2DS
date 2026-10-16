"""Validate one submission directory. Run by CI on every pull request.

This catches the mechanical failures so that human review can spend its time on the things that
actually need judgement -- whether the diff is justified and whether the reproduction is close
enough. Everything it checks is something that, left unchecked, makes an expensive run unusable
rather than merely wrong.

  python scripts/validate_submission.py submissions/01-drugclip
  python scripts/validate_submission.py --all
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCHEMA = json.load(open(ROOT / "report_schema.json"))
PER_TARGET_COLS = ["target_id", "n_actives", "n_total", "auroc", "bedroc",
                   "ef_0.005", "ef_0.01", "ef_0.02", "ef_0.05"]
SCORING = {"A2_reproduce", "B_retrain_released", "C_retrain_variants"}
ORDER = ["A1_env", "A2_reproduce", "A3_smoke", "B_retrain_released", "C_retrain_variants"]
# the three review points; everything in CHECKPOINTS["A"] lands in one pull request
CHECKPOINTS = {"A": ["A1_env", "A2_reproduce", "A3_smoke"],
               "B": ["B_retrain_released"],
               "C": ["C_retrain_variants"]}


class Problems(list):
    def add(self, where: str, msg: str) -> None:
        self.append(f"{where}: {msg}")


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_required(obj, schema, where, pr: Problems, path="") -> None:
    """Minimal draft-07 subset: required, type, enum, additionalProperties."""
    t = schema.get("type")
    if t == "object":
        if not isinstance(obj, dict):
            pr.add(where, f"{path or 'root'} should be an object"); return
        for k in schema.get("required", []):
            if k not in obj:
                pr.add(where, f"missing required field `{(path + '.' + k).lstrip('.')}`")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for k in obj:
                if k not in props:
                    pr.add(where, f"unexpected field `{(path + '.' + k).lstrip('.')}`")
        for k, v in obj.items():
            if k in props:
                check_required(v, props[k], where, pr, f"{path}.{k}" if path else k)
    elif t == "array":
        if not isinstance(obj, list):
            pr.add(where, f"{path} should be an array"); return
        for i, v in enumerate(obj):
            if "items" in schema:
                check_required(v, schema["items"], where, pr, f"{path}[{i}]")
    elif "enum" in schema:
        if obj not in schema["enum"]:
            pr.add(where, f"{path} = {obj!r} is not one of {schema['enum']}")
    elif t == "integer":
        if not isinstance(obj, int) or isinstance(obj, bool):
            pr.add(where, f"{path} should be an integer")
    elif t == "number":
        if not isinstance(obj, (int, float)) or isinstance(obj, bool):
            pr.add(where, f"{path} should be a number")
    elif t == "string":
        if not isinstance(obj, str):
            pr.add(where, f"{path} should be a string")
        elif schema.get("minLength") and len(obj) < schema["minLength"]:
            pr.add(where, f"{path} is shorter than {schema['minLength']}")
    elif t == "boolean":
        if not isinstance(obj, bool):
            pr.add(where, f"{path} should be a boolean")


def validate_per_target(p: pathlib.Path, pr: Problems, where: str, min_rows: int = 5) -> None:
    try:
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        pr.add(where, f"{p.name} will not parse as CSV: {e}"); return
    if not rows:
        pr.add(where, f"{p.name} has no data rows"); return
    missing = [c for c in PER_TARGET_COLS if c not in rows[0]]
    if missing:
        pr.add(where, f"{p.name} is missing columns {missing}")
    if len(rows) < min_rows:
        pr.add(where, f"{p.name} has only {len(rows)} rows; a benchmark has more targets than that")
    ids = [r.get("target_id") for r in rows]
    if len(set(ids)) != len(ids):
        pr.add(where, f"{p.name} repeats a target_id")
    for r in rows[:]:
        for c in ("auroc", "bedroc"):
            try:
                v = float(r[c])
            except (KeyError, TypeError, ValueError):
                continue
            if not (0.0 <= v <= 1.0):
                pr.add(where, f"{p.name} target {r.get('target_id')}: {c}={v} is outside [0,1]")
                break


def validate_code(sub: pathlib.Path, pr: Problems) -> str | None:
    c = sub / "code"
    where = "code"
    up = c / "UPSTREAM.txt"
    if not up.exists():
        pr.add(where, "code/UPSTREAM.txt is missing"); return None
    txt = up.read_text()
    if "commit" not in txt.lower():
        pr.add(where, "UPSTREAM.txt does not state a commit SHA")
    if "<" in txt and ">" in txt:
        pr.add(where, "UPSTREAM.txt still contains <placeholders>")
    if not (c / "run.sh").exists():
        pr.add(where, "code/run.sh is missing")
    elif "<" in (c / "run.sh").read_text() and ">" in (c / "run.sh").read_text():
        pr.add(where, "code/run.sh still contains <placeholders>; it must be verbatim")
    d = c / "changes.diff"
    if not d.exists():
        pr.add(where, "code/changes.diff is missing (an empty file is the right answer when you "
                      "changed nothing)")
        return None
    return sha256(d)


def validate(sub: pathlib.Path) -> Problems:
    pr = Problems()
    if not sub.is_dir():
        pr.add(str(sub), "not a directory"); return pr
    if not (sub / "REPORT.md").exists():
        pr.add(sub.name, "REPORT.md is missing")
    diff_sha = validate_code(sub, pr)

    # frozen-file fingerprint must be committed at stage 1 and never change afterwards
    frozen = sub / "code" / "frozen.sha256"
    if not frozen.exists():
        pr.add("code", "code/frozen.sha256 is missing; generate it at checkpoint A from the list "
                       "in your assignment and never regenerate it")

    def filled(d: pathlib.Path) -> bool:
        return d.is_dir() and any(not f.name.startswith(".") for f in d.iterdir())

    present = [s for s in ORDER if filled(sub / s)]
    if not present:
        pr.add(sub.name, "no checkpoint has any content yet")
    # stages must be filled in order -- no starting an expensive stage early
    for i, s in enumerate(ORDER):
        if s in present:
            for earlier in ORDER[:i]:
                if earlier not in present:
                    pr.add(s, f"filled before `{earlier}`; parts run in order")

    # C holds one subdirectory per variant, so expand it into those subdirectories
    units: list[tuple[str, pathlib.Path]] = []
    for s in present:
        if s == "C_retrain_variants":
            subs = [x for x in sorted((sub / s).iterdir())
                    if x.is_dir() and not x.name.startswith(".")]
            if not subs:
                pr.add(s, "no variant subdirectory; C_retrain_variants/<variant_id>/ per variant")
            units += [(s, x) for x in subs]
        else:
            units.append((s, sub / s))

    seeds, rules = set(), set()
    for s, d in units:
        sfile = d / "summary.json"
        if not sfile.exists():
            pr.add(f"{s}/{d.name}" if s == "C_retrain_variants" else s,
                   "summary.json is missing"); continue
        try:
            summ = json.load(open(sfile))
        except Exception as e:
            pr.add(s, f"summary.json will not parse: {e}"); continue
        check_required(summ, SCHEMA, s, pr)
        if summ.get("stage") != s:
            pr.add(s, f"summary.json says stage={summ.get('stage')!r}")
        code = summ.get("code") or {}
        if diff_sha and code.get("changes_diff_sha256") not in (None, diff_sha):
            pr.add(s, "code.changes_diff_sha256 does not match code/changes.diff")
        if (code.get("changes_justified") is False):
            pr.add(s, "changes_justified is false; justify each hunk in REPORT.md")
        run = summ.get("run") or {}
        if "seed" in run:
            seeds.add(run["seed"])
        if run.get("checkpoint_rule"):
            rules.add(run["checkpoint_rule"])

        if s in SCORING:
            res = summ.get("results") or []
            if not res:
                pr.add(s, "results is empty")
            for r in res:
                ptc = r.get("per_target_csv")
                if not ptc:
                    continue
                q = pathlib.Path(ptc)
                p = q if q.is_absolute() else d / q
                if not p.exists():
                    pr.add(s, f"per_target_csv `{ptc}` does not exist")
                else:
                    validate_per_target(p, pr, s)
            if s == "A2_reproduce":
                for r in res:
                    ref = r.get("published_reference") or {}
                    if "relative_gap_pct" not in ref:
                        pr.add(s, f"{r.get('benchmark')}: published_reference.relative_gap_pct is "
                                  "required at stage 2 -- state the paper's number, where it comes "
                                  "from, and your gap")
                    elif abs(ref["relative_gap_pct"]) > 10:
                        pr.add(s, f"{r.get('benchmark')}: reproduction is "
                                  f"{ref['relative_gap_pct']:+.1f} % from the published value, "
                                  "outside the 10 % gate -- open an issue instead of continuing")
            if s == "C_retrain_variants":
                td = summ.get("training_data") or {}
                if not td.get("split_id"):
                    pr.add(s, "training_data.split_id is required")
                if not td.get("files"):
                    pr.add(s, "training_data.files must list each split file with its sha256")
                man = ROOT / "splits" / "MANIFEST.json"
                if man.exists() and td.get("split_id"):
                    try:
                        pub = json.load(open(man)).get(td["split_id"])
                    except Exception:
                        pub = None
                    if pub:
                        want = {f["path"]: f["sha256"] for f in pub.get("files", [])}
                        for f in td["files"]:
                            k = pathlib.Path(f["path"]).name
                            hit = next((v for p2, v in want.items()
                                        if pathlib.Path(p2).name == k), None)
                            if hit and hit != f.get("sha256"):
                                pr.add(s, f"{k}: sha256 does not match the published manifest -- "
                                          "this run used different data")
                    elif td["split_id"] != "released":
                        pr.add(s, f"split_id `{td['split_id']}` is not in splits/MANIFEST.json")

    if len(seeds) > 1 and {"B_retrain_released", "C_retrain_variants"} & set(present):
        pr.add(sub.name, f"more than one seed across stages ({sorted(seeds)}); checkpoints B "
                         "and C must share a seed")
    if len(rules) > 1:
        pr.add(sub.name, f"inconsistent checkpoint_rule across stages ({sorted(rules)})")
    return pr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    subs = ([p for p in sorted((ROOT / "submissions").iterdir())
             if p.is_dir() and not p.name.startswith("_")]
            if a.all else [pathlib.Path(p) for p in a.paths])
    if not subs:
        print("nothing to validate (no submissions yet)"); return 0
    bad = 0
    for s in subs:
        pr = validate(s)
        if pr:
            bad += 1
            print(f"\n✗ {s.name} — {len(pr)} problem(s)")
            for p in pr:
                print(f"    {p}")
        else:
            print(f"✓ {s.name}")
    print()
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
