# The contract

Ten rules. Every one of them exists because breaking it has already cost somebody GPU time.

## 1. Do not modify the evaluation path

Your assignment names the files that compute metrics and load benchmarks. At checkpoint A you
produce `code/frozen.sha256`, a checksum listing of those files, and **CI fails if it changes
afterwards**. Seven people's numbers are only comparable if the metric code is identical.

If you think a metric is wrong: open an issue. Do not fix it.

## 2. Submit the upstream commit and your diff, not a copy of the repository

In `code/`:

| file | what |
|---|---|
| `UPSTREAM.txt` | the repository URL and the **exact commit SHA** you used |
| `changes.diff` | `git diff` against that commit — **ideally empty**; create the file even when empty |
| `run.sh` | the verbatim commands you ran, in order, no placeholders |
| `frozen.sha256` | generated once at checkpoint A, never regenerated |

Every non-empty hunk in `changes.diff` needs a one-line justification in `REPORT.md`.

A diff can be reviewed in minutes. Seven vendored repositories cannot.

## 3. Report per-target scores, never just the mean

Every scoring run writes `per_target.csv`, one row per benchmark target:

```
target_id,n_actives,n_total,auroc,bedroc,ef_0.005,ef_0.01,ef_0.02,ef_0.05
```

A mean cannot be analysed afterwards, and recovering per-target numbers costs the entire run again.

If the upstream script only prints means, add a dump of the per-target values. That is the one diff
we expect to see, and it is allowed.

## 4. State your checkpoint selection rule

Models differ: some evaluate the best epoch, some an average of the last several. Put the rule in
`summary.json` as `checkpoint_rule`, taken from what the repository or paper specifies, and give the
source in `checkpoint_rule_source`.

**If upstream is ambiguous, report both** `best` and `avg_last10`, as two runs.

Comparing a best-epoch checkpoint against an averaged one compares two recipes, not two training
sets. This has silently wasted whole runs.

## 5. One seed, fixed, recorded

One seed per run, in `summary.json`. **Never change the seed between checkpoint B and checkpoint C**
— that difference is the entire measurement.

## 6. Change exactly one thing at checkpoint C

Same seed, same configuration, same checkpoint rule, same code as checkpoint B. The only difference
is the input file. If you need to change anything else, stop and open an issue.

## 7. Build the variants yourself, and let the script verify them

You are not sent a copy of the training data. You already have the released training file — checkpoint
B retrains on it — and a variant differs from it only in which records it keeps. So you get a small
**spec** saying which, and you build the variant locally:

```bash
python tools/apply_variant.py splits/specs/<variant_id>/<spec>.json.gz \
       <your released training file> <output file>
```

The script checks your released file against the digest we recorded **before writing anything**, and
checks its own output afterwards. Both passing means your download is the same file the variant was
built from *and* the filter applied correctly — which being handed a finished file could never show.

If either check fails it stops and tells you. **Do not train on a file that failed a check**; open an
issue with the message. It is far cheaper to find out there than after a training run.

Record `split_id` and the digests the script prints in `summary.json`. CI compares them against
`splits/MANIFEST.json`.

Digests are over record **content**, sorted — not over file bytes. Two LMDBs holding identical
records can differ byte-for-byte across library versions, and we are not sending you after that.

## 8. Do not edit a script while a job is running

A long job re-reads its own scripts. Editing one mid-run has produced runs whose output columns
change halfway through and cannot be combined. Finish the run, then edit.

## 9. Beware pipelines in job scripts

`set -o pipefail` plus a pipe into `head` turns a successful command into exit 141 on SIGPIPE. A
driver script reading that exit code marks every job failed and leaves the GPU idle. This cost six
GPU-hours on this project. Capture output whole, then slice it.

## 10. Log what it cost

`wall_clock_hours` and `gpu_hours` in every `summary.json`. We budget the later checkpoints from
your checkpoint-A numbers, so make the A3 extrapolation a real estimate rather than a guess.

---

## What gets a submission sent back

- a changed file listed in `code/frozen.sha256`
- any change to metric or benchmark-loading code
- a missing or malformed `per_target.csv`
- an unstated or inconsistent `checkpoint_rule`
- a different seed between B and C
- checkpoint-C digests that do not match `splits/MANIFEST.json`
- starting B or C before the previous pull request was merged
