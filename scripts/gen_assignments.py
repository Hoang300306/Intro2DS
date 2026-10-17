"""Generate the seven assignment sheets from one table.

Each sheet is self-contained: the assignee needs nothing from the others and nothing from us beyond
the sheet, the contract and the variant files. Deliberately says nothing about why the variants
differ from each other -- the framing given, reproducibility and sensitivity of reported performance
to training-set composition, is accurate and sufficient, and a person who does not know the
hypothesis cannot tune toward it.

Only facts that were actually checked go in `verified`. Anything unchecked is marked as such so the
assignee knows to confirm it before relying on it.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAN = json.load(open(ROOT / "splits" / "MANIFEST.json"))


def variant_commands(aid: str, variants: list[str]) -> str:
    """The exact applier invocations for this person, one per spec.

    Generated rather than written by hand because several variants cover more than one released
    file, and a single example command would leave somebody applying half of their variant.
    """
    out = []
    for v in variants:
        e = MAN.get(v)
        if e is None:
            out.append(f"- `{v}` — **not yet published**; we will tell you when it is.")
            continue
        n = len(e["files"])
        out.append(f"\n**`{v}`** — {n} file{'s' if n > 1 else ''} to build"
                   + (" (run the command once per file)" if n > 1 else "") + ":\n")
        out.append("```bash")
        for f in e["files"]:
            noun = "directory" if f["format"] == "smiles_dir" else "file"
            out.append(f"# replaces the released {noun}: {f['replaces_file']}")
            out.append(f"python tools/apply_variant.py \\")
            out.append(f"    {f['spec']} \\")
            out.append(f"    <your released {f['replaces_file']}> \\")
            out.append(f"    <where to write it>")
        out.append("```")
    return "\n".join(out)

M = [
 dict(id="01-drugclip", name="DrugCLIP", venue="NeurIPS 2023",
   repo="https://github.com/bowen-gao/DrugCLIP",
   artifacts="Training data and a checkpoint are linked from the repository README (Google Drive). "
             "The training directory is `train_no_test_af/`, holding `train.lmdb`, `valid.lmdb`, "
             "`dict_mol.txt` and `dict_pkt.txt`.",
   benchmarks="DUD-E and LIT-PCBA",
   replaces="`train.lmdb` inside the released `train_no_test_af/` directory",
   variants=["drugclip_train_ligandish", "drugclip_train_ligandish_matched"],
   verified="We hold the released `train_no_test_af/train.lmdb` and confirmed it contains 66,164 "
            "records, each a dict with `smi` (the ligand SMILES) and `pocket` (a four-character PDB "
            "code). The variants were produced from that exact file.",
   frozen="the benchmark loader and the metric functions (enrichment factor, BEDROC, AUROC) in the "
          "test/evaluation script",
   notes="This model is the origin of a family of later methods, so its evaluation code is the one "
         "several other repositories inherit. Be especially careful not to modify it."),

 dict(id="02-hypseek", name="HypSeek", venue="AAAI 2026",
   repo="https://github.com/jianhuiwemi/HypSeek",
   artifacts="The repository consumes the training corpus released by LigUnity "
             "(figshare 10.6084/m9.figshare.27966819, `train_label.zip` plus the blend LMDBs). A "
             "checkpoint is provided by the repository.",
   benchmarks="DUD-E, DEKOIS 2.0, and the FEP sets the repository evaluates on",
   replaces="both `train_label_blend_seq_full.json` and `train_label_pdbbind_seq.json` in the "
            "training data directory",
   variants=["pocketaffdb_ligandish", "pocketaffdb_ligandish_matched"],
   verified="Read from the repository's own code: `unimol/tasks/train_task.py` lines 514-515 load "
            "`train_label_pdbbind_seq.json` and `train_label_blend_seq_full.json` from "
            "`self.args.data`. Those are exactly the two files each variant supplies.",
   frozen="`unimol/tasks/test_task.py`, the metric helpers it calls, and the test-dataset loader",
   notes="The training task also reads `test_datasets/dude.json`, `PCBA.json` and `dekois.json` and "
         "applies its own filter to the training assays, so **the test index files are required for "
         "training, not only for evaluation**. Leave that filter alone: it must run identically on "
         "the released data and on every variant, or checkpoint B and checkpoint C stop being "
         "comparable."),

 dict(id="03-causalbind", name="CausalBind", venue="NeurIPS 2026",
   repo="https://github.com/lokali/CausalBind",
   artifacts="`DATA.md` in the repository documents every input, its licence and where to get it. "
             "Training data comes from the LigUnity figshare corpus; several checkpoints are "
             "released.",
   benchmarks="DUD-E, DEKOIS 2.0, LIT-PCBA and the FEP benchmarks",
   replaces="both `train_label_blend_seq_full.json` and `train_label_pdbbind_seq.json` in `data/`",
   variants=["pocketaffdb_ligandish", "pocketaffdb_ligandish_matched"],
   verified="The `Expected Layout` section of `DATA.md` lists `data/train_label_blend_seq_full.json` "
            "and `data/train_label_pdbbind_seq.json`, which are exactly the two files each variant "
            "supplies.",
   frozen="the test task and its metric computation",
   notes="**Read `DATA.md` before anything else** — it is unusually complete and will save you time. "
         "It also states that the training task reads the benchmark index files to filter training "
         "assays, so those are needed for training too. Several checkpoints are released; say in "
         "your report which one you scored in A2 and why."),

 dict(id="04-conglude", name="ConGLUDe", venue="ICML 2026",
   repo="https://github.com/ml-jku/conglude",
   artifacts="Training data on Zenodo (record 20354834) — the archives `LB_train_val.zip` and "
             "`SB_train_val.zip`. A checkpoint is under `checkpoints/` in the repository.",
   benchmarks="the benchmarks the paper reports; list them explicitly in your report",
   replaces="TO BE CONFIRMED — the exact training file names will be in the variant manifest before "
            "checkpoint C is assigned to you",
   variants=["conglude_ligandish", "conglude_ligandish_matched"],
   verified="Nothing about the internal file layout is confirmed yet: the Zenodo archives are still "
            "being retrieved on our side. The repository and the Zenodo record are confirmed to "
            "exist, with the file list above.",
   frozen="the evaluation entry point and its metric computation",
   notes="**This corpus needs one extra step, and getting it wrong would invalidate your "
         "comparison.** The release ships a `processed/` tree — `ligand_embeddings/descriptors.dat` "
         "at 1.5 GB and per-target graphs at 6.3 GB — which is *derived* from the raw SMILES files. "
         "A variant changes those SMILES, so `processed/` has to be regenerated from them.\n\n"
         "That means you must **also regenerate `processed/` for your checkpoint-B run**. If "
         "checkpoint B uses the shipped features and checkpoint C uses freshly computed ones, any "
         "change in the feature code between the release and today becomes a difference between "
         "your two arms, and the comparison stops meaning anything. Regenerate both, from the same "
         "code, and say in your report that you did.\n\n"
         "Budget for this in your A3 estimate — it is real cost on top of training, and it is one "
         "reason your A3 number matters.\n\n"
         "Only the LB (ligand-based) training data is covered. If the paper's main results use the "
         "SB variant, say so at A2 and stop before C."),

 dict(id="05-bindclip", name="BindCLIP", venue="KDD 2026",
   repo="https://github.com/anjie-qiao/BindCLIP",
   artifacts="Training data and checkpoints are on Google Drive, linked from the README. The "
             "training archives are `train_with_hardneg_by_random.7z` and "
             "`train_with_hardneg_by_unimol.7z`, each containing one LMDB.",
   benchmarks="DUD-E, LIT-PCBA and FEP+",
   replaces="the LMDB inside whichever training archive you use — "
            "`train_with_hardneg_by_random.lmdb` or `train_with_hardneg_by_unimol.lmdb`",
   variants=["bindclip_random_ligandish", "bindclip_random_ligandish_matched",
             "bindclip_unimol_ligandish", "bindclip_unimol_ligandish_matched"],
   verified="We hold both released archives and confirmed each LMDB has 66,164 records carrying "
            "`smi` and `pocket` plus the extra fields this model needs "
            "(`docked_coordinates`, `scaffold_fp`, `decoy_smi`, `decoy_mol`, `decoy_atoms`, "
            "`decoy_coordinates`, `decoy_scaffold_fp`). The variants were produced from those exact "
            "files and preserve every field untouched.",
   frozen="the evaluation script and its metric functions",
   notes="**The README says this model uses the same training split as a predecessor, but the file "
         "is not the same file** — each record additionally carries a binding pose and a negative "
         "molecule. Do not substitute another repository's LMDB; use the variants built for yours.\n\n"
         "There are **two** released training versions (random and UniMol hard negatives) and we "
         "built variants for both. **Say in your A2 report which version the paper's main results "
         "use, and use that one throughout** — B and C must use the same version.\n\n"
         "So although four variants are listed below, **you run only the two matching your chosen "
         "version**: two retrains at checkpoint C, not four. The other two exist so that whichever "
         "version you pick is already prepared; leave them alone and say in your report which pair "
         "you used. Only the primary "
         "ligand field was filtered when the variants were built; `decoy_smi` was left untouched, "
         "deliberately, so that your variants differ from the other corpora's in one way only."),

 dict(id="06-drughash", name="DrugHash", venue="AAAI 2025",
   repo="https://github.com/nwod-edispu/DrugHash",
   artifacts="The repository README gives a Google Drive checkpoint and states that training uses "
             "PDBBind complexes with LMDB files placed under `train_no_test_af/` — the same layout "
             "and the same released data as its predecessor.",
   benchmarks="DUD-E and LIT-PCBA",
   replaces="`train.lmdb` inside `train_no_test_af/`",
   variants=["drugclip_train_ligandish", "drugclip_train_ligandish_matched"],
   verified="The README's Data section says to place the LMDB files under `train_no_test_af/`, which "
            "is the predecessor's released training directory. We hold that `train.lmdb` (66,164 "
            "records with `smi` and `pocket`) and built the variants from it.",
   frozen="the retrieval evaluation and its metric code",
   notes="This model learns binary hash codes and ranks by Hamming distance, so its metric code "
         "differs from its predecessor's even though the data is shared. Freeze **its** evaluation "
         "path, not the predecessor's.\n\n"
         "The README mentions optional HomoAug augmentation. **Do not use it** unless the paper's "
         "main results do; if they do, use it identically at B and C and say so."),

 dict(id="07-sprint", name="SPRINT", venue="MLSB @ NeurIPS 2024",
   repo="https://github.com/abhinadduri/panspecies-dti",
   artifacts="The MERGED training corpus is fetched by `data/MERGED/huge_data/download.sh`. It "
             "unpacks to TSV files plus three `id_to_*.npy` lookup maps. Checkpoint links are in "
             "`checkpoints/README.md`.",
   benchmarks="the DTI benchmarks the repository evaluates on (`--task`), and the Lit-PCBA target "
              "list in `targets.txt`; list in your report exactly what you ran",
   replaces="`merged_pos_uniq_train_rand.tsv` and `merged_neg_uniq_train_rand.tsv` in the MERGED "
            "data directory",
   variants=["sprint_merged_ligandish", "sprint_merged_ligandish_matched"],
   verified="We hold the released corpus and confirmed the training files are TSVs of `ligand` (an "
            "integer id) and `aa_seq` (a UniProt accession), with SMILES resolved through "
            "`id_to_smiles.npy` (3,529,822 entries). The training files hold 574,754 positive and "
            "64,361,902 negative rows over 312 and 289 targets. The variants were produced from "
            "those exact files, and the validation and test TSVs are unchanged.",
   frozen="the evaluation path and its metric functions",
   notes="**This is the largest corpus in the study: about 64.9 million training rows.** Your A3 "
         "cost extrapolation matters more here than for anyone else, because we will decide the "
         "scope of your B and C runs from it. If a full run is impractical, say so at A3 with your "
         "numbers and we will reduce the scope — that is a normal outcome, not a failure.\n\n"
         "This model predicts binary interaction from sequence rather than ranking within a pocket, "
         "so the variants were built to preserve the **positive-to-negative ratio exactly** "
         "(identical to eight decimal places between the two variants). Do not resample or rebalance "
         "the classes: that would change the thing the variants hold fixed."),
]

TPL = """# {id} — {name}

| | |
|---|---|
| **Model** | {name} ({venue}) |
| **Upstream repository** | {repo} |
| **Benchmarks** | {benchmarks} |
| **Your directory** | `submissions/{id}/` |
| **Your branch** | `sub/{id}` |

You are one of several people, each on a different model. Nothing here depends on anyone else.

## Getting set up

```bash
git clone https://github.com/kongwoang/Intro2DS.git
cd Intro2DS
cp -r submissions/_TEMPLATE submissions/{id}
git checkout -b sub/{id}
```

**Read [`../CONTRACT.md`](../CONTRACT.md) before you touch a GPU.** It is ten short rules and every
one of them exists because breaking it has already cost somebody a run.

Everything you are told to run is in this repository: `tools/apply_variant.py` builds your variants,
and `scripts/validate_submission.py` is what CI runs on your pull request — run it yourself first,
it is quicker than waiting for CI.

Questions and problems go in a **GitHub issue in this repository**, titled with your id, not a DM:
an answer one person needs is usually one everybody needs.

## What this study is

We are measuring two things about published virtual-screening models: how well their reported
numbers reproduce from the released artifacts, and how sensitive those numbers are to the exact
composition of the training set. Your job is to produce **trustworthy numbers** for {name}.
Interpretation is not your problem; accuracy and honest reporting are.

## Released artifacts

{artifacts}

## What we already checked for you

{verified}

## Notes specific to this model

{notes}

---

# Checkpoint A — environment, reproduction, smoke train

**One pull request containing all three parts.** Total cost: under one GPU-hour. Nothing expensive
happens until this is merged.

## A1 — environment  → `A1_env/`

Build the environment the repository asks for, then commit:

| file | content |
|---|---|
| `environment.yml` or `requirements.txt` | the **resolved** versions, not the loose spec |
| `nvidia-smi.txt` | the output, verbatim |
| `import_test.txt` | importing the model package, then printing `torch.__version__`, `torch.version.cuda`, `torch.cuda.is_available()` |
| `summary.json` | `stage` = `"A1_env"` |

Then produce `code/frozen.sha256` — a `sha256sum` listing of the files that compute metrics and load
benchmarks. For this repository that means {frozen}. Err on the side of listing too many files.

**Generate it once, here, and never regenerate it.** CI uses it to detect whether the evaluation
path moved under your feet between checkpoints.

## A2 — reproduce the published numbers  → `A2_reproduce/`   ← hard gate

Score the **released checkpoint** on the benchmarks above, using the repository's own evaluation
script, unmodified.

Then find the matching number **in the paper** and record, in `summary.json` under
`published_reference`: the metric, its value, **which table or section it comes from**, and your
relative gap.

**Name the metric first.** In `summary.json`, `published_reference.metric` is the metric **the paper
itself leads with** for that benchmark, and `published_reference.source` is the table or section it
appears in. Decide that before you compute your gap — the gate applies to the metric you named, so
picking it afterwards would just be choosing whichever number happens to pass.

**The gate is 10 % relative on that metric.** Outside it: open an issue with your evidence and stop.
Do not go on to A3. A documented non-reproduction is a real result and is not your fault.

Deliver `summary.json`, `per_target.csv` and `log.txt`.

`per_target.csv` has one row per benchmark target and is **not optional** (contract rule 3).
`target_id`, `n_actives` and `n_total` are always required. The metric columns — `auroc`, `bedroc`,
`ef_0.005`, `ef_0.01`, `ef_0.02`, `ef_0.05` — are required **wherever your model's own evaluation
computes them**; leave a column empty where it does not, and say which in `REPORT.md`. Do not invent
a metric your repository does not compute, and do not quietly substitute a different one.

## A3 — smoke train, and the cost estimate  → `A3_smoke/`

Train for **one epoch on a small subset** — a few hundred training groups is plenty. The point is
not a useful model. The point is proof that the training path runs end to end on your machine,
writes a checkpoint, and that the checkpoint loads again.

Deliver `summary.json`, `log.txt`, and in `REPORT.md`:

- how long one epoch took, on what subset
- **your extrapolated cost for checkpoint B, and for each variant in checkpoint C**
- what you extrapolated from

We approve the expensive runs from that estimate. A careless guess here gets your run either blocked
or wasted.

**Then stop. Open the pull request and wait for it to be merged.**

---

# Checkpoint B — retrain on the released training data  → `B_retrain_released/`

Only after checkpoint A is merged.

Retrain from scratch on the **released** training data, with the released configuration, a fixed
seed, and the checkpoint selection rule the repository specifies (contract rule 4).

This is the baseline that checkpoint C is compared against. It is half the measurement, not a
formality.

Deliver `summary.json` with `training_data.split_id = "released"`, `per_target.csv`, `log.txt`, and
the checkpoint's SHA256.

In `REPORT.md`, say how your retrained model compares to the released checkpoint you scored at A2.
They are usually **not** identical, and that difference is worth recording rather than hiding.

---

# Checkpoint C — retrain on the supplied variants  → `C_retrain_variants/<variant_id>/`

Only after checkpoint B is merged.

Your variants replace {replaces}.

**You build them yourself from the released data you already have.** We do not send you training
data: a variant differs from the released file only in which records it keeps, so you get a small
spec saying which — tens of kilobytes instead of a gigabyte — and build it locally.

The exact commands for your variants:

{variantcmds}

The script verifies your released input against the digest we recorded **before it writes
anything**, and verifies its own output afterwards. Both passing proves your download matches the
input the variant was built from and that the filter applied correctly.

It needs the `lmdb` package only for LMDB specs, which your model already requires.

If a check fails: **stop, do not train, open an issue with the message.** Usually it means your
released download differs from ours, which is itself worth knowing.

Then, for each variant:

1. Retrain with **the same seed, configuration, checkpoint rule and code as checkpoint B**. The input
   file is the only thing that changes (contract rule 6).
2. Record `split_id` and the digests the script printed in `summary.json`. CI compares them against
   `splits/MANIFEST.json`.
3. Deliver `C_retrain_variants/<variant_id>/` with `summary.json`, `per_target.csv` and `log.txt`.

---

## What makes a good submission

Per-target numbers. An empty or fully justified diff. One seed throughout. A stated checkpoint rule
with its source. An honest `Problems` section.

Not a flattering number.
"""

out = ROOT / "assignments"
out.mkdir(exist_ok=True)
for m in M:
    vl = "\n".join(f"- `{v}`" for v in m["variants"])
    (out / f"{m['id']}.md").write_text(TPL.format(
        variantlist=vl, variantcmds=variant_commands(m["id"], m["variants"]), **m))

idx = ["# Assignments", "",
       "One model per person. Each sheet is self-contained — read yours, the "
       "[contract](../CONTRACT.md), and [`report_schema.json`](../report_schema.json).", "",
       "| id | model | venue | variants replace |", "|---|---|---|---|"]
for m in M:
    idx.append(f"| [`{m['id']}`]({m['id']}.md) | {m['name']} | {m['venue']} | "
               f"{m['replaces'].split(' in ')[0].split(' inside ')[0]} |")
idx += ["", "## Why some of you share variant files", "",
        "Several of these models train on the same released corpus, so they receive the same "
        "variant files. That is not redundancy: the point is to see the same training-set change "
        "through different architectures. Others train on their own corpus and get variants built "
        "from it.", "",
        "Everything in checkpoint A needs nothing from us, so **nobody has to wait** to start."]
(out / "README.md").write_text("\n".join(idx) + "\n")
print(f"wrote {len(M)} assignment sheets + index")
