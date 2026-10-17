"""Pre-flight the handoff: every claim in an assignment must be satisfiable from the repository.

Run before sending the sheets out. It catches the class of error that would otherwise surface on
somebody else's machine, after they had already downloaded a corpus.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
man = json.load(open(ROOT / "splits" / "MANIFEST.json"))
problems: list[str] = []

# 1. every variant an assignment names must exist in the manifest, and vice versa
named: dict[str, set[str]] = {}
for sheet in sorted((ROOT / "assignments").glob("[0-9]*.md")):
    aid = sheet.stem
    body = sheet.read_text()
    block = body.split("# Checkpoint C")[-1]
    # sheets name variants as a bolded heading over their generated commands, and an
    # unpublished one still as a bullet, so accept both spellings
    ids = set(re.findall(r"^\*\*`([a-z0-9_]+)`\*\*", block, re.M))
    ids |= set(re.findall(r"^- `([a-z0-9_]+)`", block, re.M))
    named[aid] = ids
    for v in sorted(ids):
        if v not in man:
            problems.append(f"{aid}: names variant `{v}`, which is not in MANIFEST.json")
        elif aid not in man[v]["applies_to"]:
            problems.append(f"{aid}: names `{v}`, but the manifest assigns it to "
                            f"{man[v]['applies_to']}")
for v, e in sorted(man.items()):
    for who in e["applies_to"]:
        if v not in named.get(who, set()):
            problems.append(f"{v}: manifest assigns it to {who}, whose sheet does not name it")

# 2. every spec the manifest points at must exist, be readable, and agree with the manifest
for v, e in sorted(man.items()):
    for f in e["files"]:
        p = ROOT / f["spec"]
        if not p.exists():
            problems.append(f"{v}: spec {f['spec']} is missing"); continue
        try:
            d = json.loads(gzip.decompress(p.read_bytes()))
        except Exception as ex:
            problems.append(f"{v}: spec {p.name} will not decode ({ex})"); continue
        if d["variant_id"] != v:
            problems.append(f"{v}: spec says variant_id={d['variant_id']!r}")
        for k in ("content_digest", "source_content_digest"):
            got = (d["expected_output"] if k == "content_digest" else d["source"])["content_digest"]
            if f[k] != got:
                problems.append(f"{v}/{p.name}: manifest {k} disagrees with the spec")

# 3. a policy and its matched control must not share an output digest -- if they do, the check
#    cannot tell them apart, which is the whole point of having it
for v in sorted(man):
    if v.endswith("_matched"):
        continue
    mate = v + "_matched"
    if mate not in man:
        problems.append(f"{v}: has no matched control in the manifest"); continue
    a = {f["replaces_file"]: f["content_digest"] for f in man[v]["files"]}
    b = {f["replaces_file"]: f["content_digest"] for f in man[mate]["files"]}
    for k in a:
        if k in b and a[k] == b[k]:
            problems.append(f"{v} and {mate} share an output digest for {k}: the verification "
                            "cannot distinguish them")

# 4. the applier must support every format the specs use
fmts = set()
for v, e in man.items():
    for f in e["files"]:
        fmts.add(f["format"])
src = (ROOT / "tools" / "apply_variant.py").read_text()
for fmt in sorted(fmts):
    if f'"{fmt}"' not in src:
        problems.append(f"apply_variant.py has no handler for format `{fmt}`")

print(f"{len(man)} variants, {sum(len(e['files']) for e in man.values())} specs, "
      f"formats {sorted(fmts)}")
for a in sorted(named):
    print(f"  {a:14s} -> {', '.join(sorted(named[a])) or '(none named)'}")
if problems:
    print(f"\n{len(problems)} problem(s):")
    for x in problems:
        print(f"  ✗ {x}")
    sys.exit(1)
print("\nhandoff is consistent")
