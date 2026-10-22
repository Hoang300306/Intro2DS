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
         "several other repositories inherit. Be especially careful not to modify it.\n\n"
         "**A session that dies does not cost you the run.** Uni-Core restores from "
         "`checkpoint_last.pt` in `--save-dir` on its own: `restore_file` defaults to that name, "
         "and because none of the `--reset-*` flags are set it brings back the optimizer, the LR "
         "scheduler and the dataloader position, not just the weights. So B and C can be run in "
         "chunks — as long as **`--save-dir` lives on storage that survives the session**, and you "
         "never delete `checkpoint_last.pt`. Say in your report how many times each arm was "
         "resumed. If you are on one GPU instead of four, that is what makes the long wall clock "
         "survivable.\n\n"
         "**`test_dude` prints only the macro means.** It loops over targets, collects `auc_list` "
         "and the EF dictionaries, and prints `np.mean` of each — the per-target values and the "
         "target names never reach the log. `per_target.csv` therefore needs the print-only patch "
         "described under A2. Print the target name inside the loop: the target order comes from "
         "`os.listdir`, which is filesystem order and not reproducible across machines, so position "
         "in the list is not an id.\n\n"
         "For the same reason, make sure `./data/DUD-E/raw/all/` and `./data/lit_pcba/` contain "
         "**only** target directories. Both are read with `os.listdir` and every entry is treated "
         "as a target, so a stray `.DS_Store` or a leftover archive becomes a target and either "
         "crashes the run or lands in the mean.\n\n"
         "**The checkpoint rule is best-by-`valid_bedroc`**: `drugclip.sh` passes "
         "`--best-checkpoint-metric valid_bedroc --maximize-best-checkpoint-metric`, and `test.sh` "
         "scores `checkpoint_best.pt`. State that and its source (contract rule 4).\n\n"
         "**Both scripts hard-code a GPU index** — `CUDA_VISIBLE_DEVICES=\"1\"` in `drugclip.sh`, "
         "and other indices elsewhere. Set it to a device you actually have; on a shared machine the "
         "wrong index is somebody else's job. The paths are relative, so run from the repository "
         "root. `TASK` in `test.sh` takes one benchmark at a time — run it once per benchmark.\n\n"
         "The epoch budget is real: `epoch=200` with `--patience 2000`, so patience never fires and "
         "200 epochs is what it will do. The global batch is 48 — `batch_size=48`, one GPU, "
         "`update_freq=1`. Keep it the same at B and C."),

 dict(id="02-hypseek", name="HypSeek", venue="AAAI 2026",
   repo="https://github.com/jianhuiwemi/HypSeek",
   artifacts="The repository consumes the training corpus released by LigUnity "
             "(figshare 10.6084/m9.figshare.27966819, `train_label.zip` plus the blend LMDBs). "
             "**It publishes no checkpoint of its own and links none** — see the notes below for "
             "what exists and what to do about it. The processed benchmarks come from LigUnity "
             "too: Dekois 2.0 from figshare 10.6084/m9.figshare.27967422 (`DEKOIS_2.0x.zip`, "
             "2,599,906,193 bytes, md5 `bc334226eb8dbdedf00473ac7d2e9b10`), and DUD-E with "
             "LIT-PCBA from the DrugCLIP Google Drive release, folder "
             "`1zW1MGpgunynFxTKXC2Q4RgWxZmg6CInV`. The benchmark index JSONs alone are not "
             "enough; you need these processed trees.",
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
         "comparable. `--protein-similarity-thres 1.0` in `train.sh` is part of that released "
         "configuration — do not change it.\n\n"
         "**Two pins in `environment.yml` are already known to be bad, so do not spend time on "
         "them.** `hf-xet==1.1.5rc1` is not on PyPI — that release candidate became `1.1.5` on "
         "2025-06-20 — and pip resolves the whole pip section before installing anything, so the "
         "one line fails the entire environment. Use `hf-xet==1.1.5`, or drop it: it is a transfer "
         "backend for `huggingface_hub`, which this repository never calls. `torchsort==0.1.10` is "
         "source-only on PyPI and will try to compile against your CUDA toolchain; nothing here "
         "imports it either, so if that build fails, drop the line. Record both in "
         "`code/changes.diff`.\n\n"
         "**The screening checkpoint the README names does not exist publicly, and that is a "
         "result, not your problem.** `README` tells you to score DUD-E and LIT-PCBA with "
         "`checkpoint_avg_41-50_vs.pt` and FEP with `checkpoint_avg_41-50_rk.pt`, links neither, "
         "and the repository has no releases. Of the two, only `checkpoint_avg_41-50_rk.pt` was "
         "ever published, on `huggingface.co/THU-ATOM/HypSeek-AIRDD`; we checked that repository "
         "and the LigUnity ones, and there is no `_vs` checkpoint anywhere public. **Do not "
         "substitute one.** Not the ranking checkpoint for screening, and not LigUnity\'s "
         "`fengb/LigUnity_VS` weights, which are a different architecture.\n\n"
         "**The affinity-ranking side is blocked too**, so do not go looking there either. "
         "`test_fep_target` reads `bench/fep/<target>/result_dG.csv` and "
         "`bench/jacs_set/<target>_edges.csv` (or `bench/merck/...`) before it computes anything, "
         "and neither HypSeek nor LigUnity publishes those files. The LMDBs and `FEP/"
         "fep_labels.json` are published and do match each other; the CSVs the evaluator needs "
         "are not. **Do not rebuild them** from Uni-FEP-Benchmarks or the Schrodinger set: the "
         "ligand SMILES do not correspond one-to-one, and a correlation computed on a rebuilt "
         "benchmark is not a reproduction of the published one.\n\n"
         "So **A2 for this model records a blocked reproduction**, which is the result. Score the "
         "published `checkpoint_avg_41-50_rk.pt` with the unmodified evaluator on DUD-E, DEKOIS "
         "and LIT-PCBA — those three read only published LMDBs plus the index JSONs, so they do "
         "run — deliver `per_target.csv` for each, and put the paper\'s number and your gap in "
         "`published_reference` as usual. On the gated entry set `gate = true` together with "
         "`blocked`, explaining that the screening checkpoint was never released and the FEP "
         "benchmark CSVs were never released, and where you looked. The gate passes on that. What "
         "it buys us is what the 10 % rule was protecting: proof that the environment and the "
         "evaluation path run end to end and produce non-degenerate per-target metrics, before "
         "anyone pays for a retrain. The numbers themselves are not comparable with the paper — "
         "its DUD-E early enrichment of 51.44 comes from the checkpoint that does not exist — and "
         "`REPORT.md` must say so in those words.\n\n"
         "Checkpoints B and C are unaffected, and this is why they matter more here than "
         "elsewhere: they train from scratch, they never needed a released checkpoint, and the "
         "comparison between them is internal to your own runs.\n\n"
         "One thing to watch in that evaluator: `test_dude` wraps each target in "
         "`except ValueError` and **skips it with only a warning** when a pocket comes back empty. "
         "A macro mean over 99 targets looks exactly like one over 102. Count the targets that "
         "actually scored, put that number in `n_targets`, and grep your log for `Skipping`.\n\n"
         "**`train.sh` takes a mode, and you want `CASF`.** `bash train.sh CASF` validates on "
         "CASF-2016 and sets `--best-checkpoint-metric valid_bedroc --maximize-best-checkpoint-"
         "metric`, so the repository's own rule is best-by-`valid_bedroc`; score that and say so. "
         "`bash train.sh FEP` passes `--disable-validation` and sets no best-checkpoint metric, so "
         "it produces no best checkpoint at all — it is the affinity-ranking configuration, not the "
         "screening one.\n\n"
         "**A session that dies does not cost you the run.** Uni-Core restores from "
         "`checkpoint_last.pt` in `--save-dir` on its own: `restore_file` defaults to that name, "
         "and because none of the `--reset-*` flags are set it brings back the optimizer, the LR "
         "scheduler and the dataloader position, not just the weights. So B and C can be run in "
         "chunks — as long as **`--save-dir` lives on storage that survives the session**, and you "
         "never delete `checkpoint_last.pt`. Say in your report how many times each arm was "
         "resumed. If you are on one GPU instead of four, that is what makes the long wall clock "
         "survivable.\n\n"
         "**Two placeholders will stop you before anything runs**: `save_root=\"/save_root\"` in "
         "`train.sh` is an absolute path at the filesystem root, which `mkdir -p` cannot create "
         "without root, and `DATA_ROOT=/path/test_datasets` in `test.sh` is not a real path. Set "
         "both to directories you own.\n\n"
         "**`test.sh` runs `pip uninstall -y unimol` on every invocation.** That is deliberate on "
         "their side — it forces the local `unimol/` to win on `PYTHONPATH` — but it does mutate "
         "your environment, so your A1 `import_test.txt` may stop reproducing after you first "
         "evaluate. Note it in `REPORT.md` rather than treating it as a broken environment.\n\n"
         "The evaluation prints per-target lines — `[DUD-E] <target> AUC=... BEDROC=... EF=[...]` — "
         "so `per_target.csv` needs no patch here. It also writes `saved_mols_embed.npy` and "
         "friends **per target**, which is tens of gigabytes over DUD-E and LIT-PCBA. Keep it out "
         "of git and budget the disk in A3.\n\n"
         "The global batch is 96: four GPUs × `batch_size=24` × `update_freq=1`. If you have fewer "
         "GPUs, keep the product at 96 by raising `--update-freq` rather than training on fewer "
         "GPUs, which would change the contrastive batch."),

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
         "assays, so those are needed for training too.\n\n"
         "**Two pins in `environment.yml` are already known to be bad, so do not spend time on "
         "them.** `hf-xet==1.1.5rc1` is not on PyPI — that release candidate became `1.1.5` on "
         "2025-06-20 — and pip resolves the whole pip section before installing anything, so the "
         "one line fails the entire environment. Use `hf-xet==1.1.5`. It is the Xet transfer "
         "backend for `huggingface_hub`, which this repository calls only from "
         "`scripts/download_data.sh`, so it affects how bytes arrive and nothing else; no file in "
         "the repository imports it. `torchsort==0.1.10` is source-only on PyPI and will try to "
         "compile against your CUDA toolchain; nothing in the repository imports it either, so if "
         "that build fails, drop the line. Record both in `code/changes.diff`.\n\n"
         "**Run the `sp` variant, and only `sp`.** The repository releases four — `sp`, `lr`, `emb`, "
         "`atom` — and `scripts/quick_start.sh` scores all four by default. `sp` is the variant the "
         "paper leads with (Theorem 2, first CausalBind row of Table 1) and the one the "
         "repository's own further analyses use; `atom` is described there as the interpretability "
         "variant. Use the same variant at A2, B and C: a number from one variant cannot be "
         "compared with a number from another, and doing all four would be twelve retrains instead "
         "of three. If your A3 estimate shows three `sp` runs do not fit your budget, say so in "
         "the pull request and propose `emb`, which is a smaller configuration — do not switch on "
         "your own.\n\n"
         "**A session that dies does not cost you the run.** Uni-Core restores from "
         "`checkpoint_last.pt` in `--save-dir` on its own: `restore_file` defaults to that name, "
         "and because none of the `--reset-*` flags are set it brings back the optimizer, the LR "
         "scheduler and the dataloader position, not just the weights. So B and C can be run in "
         "chunks — as long as **`--save-dir` lives on storage that survives the session**, and you "
         "never delete `checkpoint_last.pt`. Say in your report how many times each arm was "
         "resumed.\n\n"
         "**Score `checkpoint_last.pt`, not the best checkpoint.** `scripts/train.sh` passes "
         "`--keep-best-checkpoints 8`, so a best checkpoint will be sitting right next to it, but "
         "the paper and the README both report the last one. State that rule and its source in "
         "your report (contract rule 4).\n\n"
         "**Name a DUD-E metric for the A2 gate.** DUD-E has 102 targets; LIT-PCBA has 15, and its "
         "EF@1% is volatile enough that the authors' own from-scratch run moved it by about a "
         "third while every DUD-E number stayed within 5 %. Score both benchmarks, but do not "
         "stake the 10 % gate on the small one. That same published comparison — the paper's "
         "numbers printed next to the repository's own from-scratch numbers — is also your "
         "yardstick at checkpoint B for how much drift is normal.\n\n"
         "**If you do not have four 80 GB GPUs**, keep the global batch at 96: it is `N_GPU` × "
         "`--batch-size` × `--update-freq`, which the released configuration sets to 4 × 24 × 1. "
         "Lower the per-GPU batch and raise `--update-freq` to compensate. Do not just train on "
         "fewer GPUs — that changes the contrastive batch, which changes the model."),

 dict(id="04-conglude", name="ConGLUDe", venue="ICML 2026",
   repo="https://github.com/ml-jku/conglude",
   artifacts="Training data on Zenodo (record 20354834) — the archives `LB_train_val.zip` and "
             "`SB_train_val.zip`. A checkpoint is under `checkpoints/` in the repository.",
   benchmarks="DUD-E and LIT-PCBA (`python download_data.py --dataset_name vs`). The repository "
              "also evaluates pocket-prediction and pocket-ranking sets; you are not asked for "
              "those, so do not download them",
   replaces="the raw SMILES directory `LB_train_val/raw/smiles_files` of the ligand-based training "
            "corpus",
   variants=["conglude_ligandish", "conglude_ligandish_matched"],
   verified="We hold `LB_train_val.zip` from Zenodo (7,855,192,840 bytes, md5 "
            "`cfd25fcc9c275b994560b0a6d0dd1147`, matching the record) and built the variants from "
            "the `raw/smiles_files` tree inside it. Note that `unzip` 6.00 cannot read this archive "
            "because it is zip64 and over 4 GB; use 7-Zip or Python's `zipfile`, and check the md5 "
            "rather than trusting `unzip -t`.",
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
         "The README will tell you the opposite: *all downloaded datasets except `SB_train_val` are "
         "already processed*. That is true of the released data and false of yours. The order is "
         "download, then apply the variant to `raw/smiles_files`, then "
         "`python process_data.py --dataset_name LB_train_val`, then train — and check that the "
         "`processed/` files really changed. Skip the reprocessing and the run trains on the "
         "shipped features, which still contain every molecule the variant removed; it will finish "
         "cleanly and tell you the variant made no difference.\n\n"
         "Budget for this in your A3 estimate — it is real cost on top of training, and it is one "
         "reason your A3 number matters.\n\n"
         "Only the LB (ligand-based) training data is covered. If the paper's main results use the "
         "SB variant, say so at A2 and stop before C.\n\n"
         "**`python eval.py` as shipped does not evaluate DUD-E or LIT-PCBA.** In "
         "`configs/datamodule/test_datasets/test_datasets.yaml` every dataset except `default_vs` "
         "is commented out, and `default_vs` is a generic placeholder pointing at a directory that "
         "is not a benchmark. Uncomment `dude` and `litpcba` and drop `default_vs` from the list; "
         "that edit belongs in `code/changes.diff` with its one-line justification. Run verbatim it "
         "would hand you confident numbers for the wrong data.\n\n"
         "**A session that dies does not cost you the run**, but you have to ask for the resume: "
         "`python train.py checkpoint_name=<name>` continues from a saved checkpoint, as the README "
         "documents. Keep `checkpoints/ConGLUDe/` on storage that survives the session, and say in "
         "your report how many times each arm was resumed.\n\n"
         "**Pass `logger=null` when you train.** `configs/train.yaml` defaults to `logger: wb`, so "
         "a training run will try to log to Weights & Biases and will upload the resolved config, "
         "variant name included, to whatever workspace is configured on the machine.\n\n"
         "**The seed is a Hydra field**: `python train.py seed=0`. Note that "
         "`configs/trainer/default.yaml` sets `deterministic: false`, so two runs with the same "
         "seed are not bitwise identical. Do not try to fix that — record it as a limitation.\n\n"
         "**The checkpoint rule is already unambiguous**: `save_top_k: 1`, `save_last: false`, "
         "`monitor: avg_val/virtual_screening/bedroc`, `mode: max`. Exactly one checkpoint survives "
         "and it is the best by that metric. State it with its source.\n\n"
         "**Cost: `max_epochs: 500` with early stopping at `patience: 50`.** Report the per-epoch "
         "time and the epoch your B run actually stopped at. B and C may stop at different epochs; "
         "that is the model's own rule, so record it and do not override it.\n\n"
         "**Only EF@5% is computed.** `model.py` builds `VirtualScreeningMetrics(ef_fractions="
         "[0.05])`, so leave `ef_0.005`, `ef_0.01` and `ef_0.02` empty and do not add cutoffs. "
         "BEDROC is at alpha 80.5. The metric class writes a per-target CSV itself, so "
         "`per_target.csv` needs no patch — point at the file it produces.\n\n"
         "The repository also ships agent command files under `.claude/`. If you use them, the "
         "evaluation path and the configs are still yours to keep fixed, and `code/frozen.sha256` "
         "is what will catch it if they are not."),

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
            "`smi`, `pocket`, `docked_coordinates` and `scaffold_fp`. **The two differ in their "
            "negative-molecule fields**, so check against the right list: the random archive has "
            "15 fields with `decoy_smi`, `decoy_mol`, `decoy_atoms`, `decoy_coordinates` and "
            "`decoy_scaffold_fp`; the UniMol archive has 16, named `hard_decoy_*` and carrying an "
            "extra `hard_is_random`. The applier checks the field names before it writes, so the "
            "wrong archive is rejected rather than silently filtered. The variants were produced "
            "from those exact files and preserve every field untouched.",
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
         "deliberately, so that your variants differ from the other corpora's in one way only.\n\n"
         "**A session that dies does not cost you the run.** Uni-Core restores from "
         "`checkpoint_last.pt` in `--save-dir` on its own: `restore_file` defaults to that name, "
         "and because none of the `--reset-*` flags are set it brings back the optimizer, the LR "
         "scheduler and the dataloader position, not just the weights. So B and C can be run in "
         "chunks — as long as **`--save-dir` lives on storage that survives the session**, and you "
         "never delete `checkpoint_last.pt`. Say in your report how many times each arm was "
         "resumed.\n\n"
         "**The README invites you to score both released checkpoints and keep whichever performs "
         "best for your use case. Do not do that here.** Choosing a checkpoint by its score on the "
         "benchmark you are reporting is how a number stops meaning anything. Pick the one the "
         "paper's main table used, decide that before you score, and if you score both, report "
         "both and say which is which.\n\n"
         "**The README carries no numeric results table** — the paper is the only source for your "
         "`published_reference`, at `10.1145/3770855.3819040`. Do not read a value off a figure.\n\n"
         "**RDKit 2022.09.5 is required**, and the README warns that another version may mean "
         "reprocessing the training data. Do not reprocess it: your variant is a filter over the "
         "released file, and the applier's digest check is what tells you whether the file you hold "
         "is the file the variant was built from.\n\n"
         "**The effective batch is 192**, not its predecessor's 48: `bindclip.sh` sets "
         "`batch_size=48` with `update_freq=4` on one GPU. Keep it. The script's seed is 42 while "
         "the released checkpoint filenames mention other seeds, so set `weight_path` to the file "
         "you actually downloaded and keep your own training seed fixed at the script's value.\n\n"
         "**This model's checkpoints load in its predecessor's inference code without complaint**, "
         "which the README presents as a feature. It also means a wrong `--path` yields numbers "
         "instead of an error, so the checkpoint SHA256 that A2 asks for is doing real work here. "
         "The evaluation inherits `test_dude`, which prints only the macro means, so "
         "`per_target.csv` needs the print-only patch described under A2."),

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
         "main results do; if they do, use it identically at B and C and say so.\n\n"
         "**A session that dies does not cost you the run.** Uni-Core restores from "
         "`checkpoint_last.pt` in `--save-dir` on its own: `restore_file` defaults to that name, "
         "and because none of the `--reset-*` flags are set it brings back the optimizer, the LR "
         "scheduler and the dataloader position, not just the weights. So B and C can be run in "
         "chunks — as long as **`--save-dir` lives on storage that survives the session**, and you "
         "never delete `checkpoint_last.pt`. Say in your report how many times each arm was "
         "resumed.\n\n"
         "**The README has no numbers — its results are images.** `published_reference` must come "
         "from the paper (AAAI 2025, 39:17041-17049, or arXiv 2407.19790). Do not read a value off "
         "`figures/accuracy.png`.\n\n"
         "**`test.sh` and `retrieval.sh` ship with the authors' own absolute paths** "
         "(`/home/hanj/...`). Replace them. Note also that the LIT-PCBA directory the code reads is "
         "`./data/lib-pcba/raw/lit_pcba/` — that spelling is in the source, so create it exactly, "
         "and keep the benchmark directories free of stray files, since every entry `os.listdir` "
         "returns is treated as a target.\n\n"
         "One released checkpoint, `checkpoint_best.pt` under `savedir_drughash/`, and the training "
         "script's `--best-checkpoint-metric valid_bedroc --maximize-best-checkpoint-metric` agrees "
         "with it: the rule is best-by-`valid_bedroc`. Ignore the `savedir_10_0_s1/` path in "
         "`retrieval.sh`; it is not the directory the README tells you to use.\n\n"
         "The evaluation inherits `test_dude`, which prints only the macro means, so "
         "`per_target.csv` needs the print-only patch described under A2. The benchmark data and the "
         "Uni-Mol pretrained weights come from the predecessor's release — if another Uni-Mol model "
         "is already set up on this machine, one copy of those is enough. The global batch is 48; "
         "keep it the same at B and C."),

 dict(id="07-sprint", name="SPRINT", venue="MLSB @ NeurIPS 2024",
   repo="https://github.com/abhinadduri/panspecies-dti",
   artifacts="The MERGED training corpus is fetched by `data/MERGED/huge_data/download.sh`. It "
             "unpacks to TSV files plus three `id_to_*.npy` lookup maps. Checkpoint links are in "
             "`checkpoints/README.md`.",
   benchmarks="LIT-PCBA only, over the full target list (`--target-protein-id \"all\"`). The "
              "repository also evaluates DAVIS, BindingDB, BIOSNAP and the TDC leaderboard, but "
              "each needs a checkpoint trained on that task and only the MERGED/LIT-PCBA models "
              "are released, so those are out of scope — do not score them",
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
         "**For A3, a \"group\" here is a target, not a slice of rows.** Take a few dozen of the "
         "312 targets that carry positives and keep every row belonging to them, so the "
         "positive-to-negative structure of a target survives. Download the full MERGED corpus "
         "even so — `--check-source` needs the whole file — and **leave the MMseqs2 filter at "
         "0.9**: it is part of the path you are smoke-testing and it is cheap next to training.\n\n"
         "**Checkpoint B cannot be resumed, and alone among these models that is a real "
         "constraint.** `train.py` calls `trainer.fit(model, datamodule=datamodule)` with no "
         "`ckpt_path`, so Lightning cannot restore the optimizer, the schedule or the epoch "
         "counter; `--checkpoint` only loads weights and then starts a fresh schedule from epoch "
         "0, which is a different recipe. An interrupted run has to be started again.\n\n"
         "**That does not make a full run expensive by itself.** The row count is large but the "
         "model is small, and the featurizers preload into an LMDB-backed store rather than "
         "running a protein language model per batch, so an epoch is a throughput question, not a "
         "given. Measure it: at `batch_size` 64 an epoch is on the order of a million steps, so "
         "**report steps per second and the wall clock of one real epoch on the full corpus**, and "
         "where your data sits — a network drive will bound you long before the GPU does. Then "
         "15 epochs is arithmetic, and we decide from it.\n\n"
         "If that arithmetic exceeds the uninterrupted time you can get, say so at A3 and propose "
         "passing `ckpt_path` to `trainer.fit` as a documented diff in `code/changes.diff`. It is "
         "a one-line change and it is defensible — but only if the **same patched code runs at B "
         "and at C**, and only if you report how many times each arm was interrupted, because a "
         "resume boundary is a place where two arms can stop being identical.\n\n"
         "This model predicts binary interaction from sequence rather than ranking within a pocket, "
         "so the variants were built to preserve the **positive-to-negative ratio exactly** "
         "(identical to eight decimal places between the two variants). Do not resample or rebalance "
         "the classes: that would change the thing the variants hold fixed.\n\n"
         "**The README's own SPRINT command omits `--task merged`, and the config default is "
         "`task: davis`.** Run verbatim it trains on DAVIS, your variant file is never opened, and "
         "nothing anywhere says so. Pass the task explicitly. `configs/saprot_agg_config.yaml` also "
         "defaults to `model_size: \"small\"` and `epochs: 250`, so the flags are not optional "
         "decoration: without them you train the wrong model for sixteen times the budget. The "
         "command is\n\n"
         "```bash\n"
         "ultrafast-train --exp-id LitPCBA --config configs/saprot_agg_config.yaml \\\n"
         "    --task merged --model-size large --epochs 15 --ship-model \\\n"
         "    --target-protein-id \"all\" --similarity-threshold 0.9 \\\n"
         "    --eval-pcba --no-wandb --replicate 0\n"
         "```\n\n"
         "**There is no evaluation-only entry point, so A2 needs care.** `--eval-pcba` adds a "
         "callback whose hook is `on_validation_epoch_end`, and nothing in the repository calls "
         "`trainer.validate()`. To score the released checkpoint you therefore start a run with "
         "`--checkpoint <the released file>` and read the first `[PCBA Eval - Epoch 0]` block the "
         "callback prints. Lightning runs a sanity-check validation **before** the first optimizer "
         "step, so that block should be the released weights and nothing else — but **verify it "
         "from your log** and say in `REPORT.md` whether any training step preceded it. If the "
         "first block only appears after an epoch of training, then the released checkpoint cannot "
         "be scored through this repository at all: open an issue, and A2 becomes a `blocked` "
         "gate. Keep it cheap with `--epochs 1` and leave `--ship-model` off at A2.\n\n"
         "**The seed flag is `--replicate` (alias `--r`), not `--seed`.** `train.py` calls "
         "`torch.manual_seed(config.replicate)` and `np.random.seed(config.replicate)`; the default "
         "is 0. Pass it explicitly and keep it identical at B and C. `--no-wandb` matters too: "
         "without it the run logs to the project `ConPlexTrain` from the config.\n\n"
         "**`--similarity-threshold 0.9` is the model's own MMseqs2 filter** and the released "
         "checkpoints are the 90 % models, so hold it at 0.9 everywhere. `mmseqs` must be on PATH "
         "and `data/MERGED/huge_data/id_to_sequence.npy` must be present; a failing MMseqs2 call "
         "raises rather than skipping the filter, so you will know.\n\n"
         "**With `--task merged --ship-model`, three things change at once**: `save_top_k=-1`, so "
         "every epoch's checkpoint is kept and fifteen of them is a lot of disk; "
         "`limit_test_batches=0`, so the built-in test runs on nothing; and `ship_model.ckpt` is "
         "written at the end with the final weights. The released models are the shipped ones — "
         "score `ship_model.ckpt`, say so, and get your numbers from the `--eval-pcba` callback.\n\n"
         "That callback prints, per target, AUROC, BEDROC and EF at 0.005, 0.01 and 0.05 — so "
         "`per_target.csv` needs no patch, `ef_0.02` stays empty, and **the BEDROC here is at alpha "
         "85**, which is not the 80.5 most of this literature uses. Record the alpha. For "
         "`published_reference`, the released checkpoints correspond to Table 2 of arXiv "
         "2411.15418."),
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
path moved under your feet between checkpoints. If your repository prints only macro means and you
need the per-target dump that contract rule 3 allows, **apply it before you generate this file** —
otherwise the checksum describes code that never scored anything.

### Verify your download now

Your variants are built from the released training file, and the applier checks your copy against
the digest we recorded before it writes anything. Do that check **here**, while it is free:

```bash
python tools/apply_variant.py --check-source <one of your specs> <your released training file>
```

It verifies and exits without writing. Put the digest it prints in `summary.json` under
`environment.source_digest_ok`, or quote it in `REPORT.md`.

Checking at checkpoint A rather than at C is the point: a release that differs from ours fails this
in minutes, instead of after you have paid for a full checkpoint-B run.

**If a pin no longer resolves**, move it to the nearest published release of the same line, record
it in `code/changes.diff` with its one-line justification, and carry on — a pinned pre-release that
has since been removed from PyPI is an index fact, not a change to the model. Do not loosen the pin
to a range, and do not spend hours compiling a source-only package that nothing in the repository
imports: check whether it is imported, and if it is not, drop it and say so. If the package is
imported, or you are unsure, open an issue instead of guessing.

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

Mark it: set `published_reference.gate = true` on **exactly one** result. CI rejects zero and
rejects two. Every other benchmark you score still reports its `relative_gap_pct`, and none of them
is gated — so scoring more than you were asked to can never block you.

**The gate is 10 % relative on that metric.** Outside it: open an issue with your evidence and stop.
Do not go on to A3. A documented non-reproduction is a real result and is not your fault.

**If no published number is reachable from published artifacts at all** — the checkpoint the paper
scored was never released, or the benchmark files its evaluator reads were never released — then say
so in `published_reference.blocked` on the gated entry, with which artifact is missing and where you
looked. That entry then passes on the explanation instead of on a gap. Do not reconstruct the
missing artifact from a third-party source and compare that to the paper: a number from a rebuilt
benchmark is not a reproduction, and presenting it as one would be worse than reporting nothing.
Open an issue first either way.

Deliver `summary.json`, `per_target.csv` and `log.txt`.

`per_target.csv` has one row per benchmark target and is **not optional** (contract rule 3).
`target_id`, `n_actives` and `n_total` are always required. The metric columns — `auroc`, `bedroc`,
`ef_0.005`, `ef_0.01`, `ef_0.02`, `ef_0.05` — are required **wherever your model's own evaluation
computes them**; leave a column empty where it does not, and say which in `REPORT.md`. Do not invent
a metric your repository does not compute, and do not quietly substitute a different one.

**Also record the BEDROC alpha your code uses**, as `results[].bedroc_alpha`. Implementations
differ — 80.5 and 85 are both in use — and one is not comparable with the other. Read it out of the
code, do not assume it.

**And record the SHA256 of the checkpoint file you scored.** Some of these repositories load each
other's checkpoints without raising anything at all, so a mix-up produces numbers rather than an
error.

**Two kinds of edit to the evaluation path are allowed**, and only these two: adding a print or a
written row so that per-target numbers come out (contract rule 3 says so already), and registering
the benchmark you were asked to score. They appear in `code/changes.diff` like any other edit, with
a one-line justification in `REPORT.md` (contract rule 2) — do not invent a second patch file, and
do put the diff's sha256 in `summary.json`, which is how CI holds it identical across A2, B and C.
Do not touch how a metric is computed, which molecules are scored, or the order they are scored in
— a patch that changes a number is a different experiment. If you think you need any other change,
open an issue first.

## A3 — smoke train, and the cost estimate  → `A3_smoke/`

Train for **one epoch on a small subset** — a few hundred training groups is plenty. The point is
not a useful model. The point is proof that the training path runs end to end on your machine,
writes a checkpoint, and that the checkpoint loads again.

Deliver `summary.json`, `log.txt`, and in `REPORT.md`:

- how long one epoch took, on what subset
- **your extrapolated cost for checkpoint B, and for each variant in checkpoint C**
- what you extrapolated from
- **what stops a run** — a fixed epoch count, or an early-stopping rule — and where that is
  configured. Several of these repositories set a high epoch cap and rely on early stopping, so
  multiplying your per-epoch time by the cap can overstate the cost many times over, while assuming
  an early stop is a guess. Say which you did.

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
2. Record `split_id` in `summary.json`, and one `training_data.files` entry per file you built,
   each with `path` and `content_digest` — the full digest the script printed, not its first
   characters. CI compares every one against `splits/MANIFEST.json` and now rejects a missing
   entry, so list all of your files, not just one.
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
