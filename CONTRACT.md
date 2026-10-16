# The contract

Short, and every rule here exists because breaking it has wasted real GPU time.

## 1. Do not modify the evaluation path

Your assignment lists the files that compute metrics and load benchmarks, with their checksums, in
`FROZEN.md`. **CI fails if any of them changes.** Seven people's numbers only mean something next to
each other if the metric code is identical. If you believe a metric is wrong, open an issue; do not
fix it.

## 2. Submit the upstream commit and your diff, not a copy of the repository

In `code/`:

- `UPSTREAM.txt` — the upstream repository URL and the exact commit SHA you used.
- `changes.diff` — `git diff` against that commit. **Ideally empty.** Every non-empty hunk needs a
  one-line justification in `REPORT.md`.
- `run.sh` — the verbatim commands you ran, in order, with no placeholders.

Reviewing a diff takes minutes. Reviewing a vendored repository does not, times seven.

## 3. Report per-target scores, not just the mean

Every scoring stage produces `per_target.csv` with one row per benchmark target:

```
target_id,n_actives,n_total,auroc,bedroc,ef_0.005,ef_0.01,ef_0.02,ef_0.05
```

A mean alone cannot be analysed afterwards, and re-running to recover per-target numbers costs the
whole run again. If the upstream evaluation script only prints means, add a dump of the per-target
values — that counts as an allowed diff, and it is the one diff we expect to see.

## 4. State your checkpoint selection rule

Models differ: some evaluate the single best epoch, some an average of the last several. Put the
rule in `summary.json` as `checkpoint_rule`, using whatever the upstream repository or the paper
specifies. **If it is ambiguous, report both** `best` and `avg_last10` as separate runs. Comparing a
best-epoch checkpoint against an averaged one compares two recipes, not two training sets.

## 5. Fix and record the seed

One seed per run, recorded in `summary.json`. Never change the seed between stage 4 and stage 5 —
that is the comparison the whole study rests on.

## 6. Do not edit a script while a job is running

A long job re-reads its scripts. Editing one mid-run has produced runs whose output columns change
halfway through and cannot be combined. Finish, then edit.

## 7. Verify the data you trained on

For stage 5, `summary.json` must carry `split_id` and the SHA256 of each file you were given. CI
compares them against the published manifest. A run on the wrong input is worse than no run,
because it looks valid.

## 8. Beware shell pipelines in job scripts

`set -o pipefail` plus a pipe into `head` turns a successful command into exit 141 on SIGPIPE, and a
queue driver reading that exit code will mark every job failed and idle the GPU. Capture output
whole, then slice it.

## 9. Log what it cost

`summary.json` carries `wall_clock_hours` and `gpu_hours`. We are budgeting the later stages from
your numbers.

## What gets a submission rejected

- a changed file in `FROZEN.md`
- a metric or benchmark-loading change
- missing `per_target.csv`
- an unstated or inconsistent `checkpoint_rule`
- a stage-5 run whose split digests do not match the manifest
- starting stage 4 or 5 before the stage-3 PR is merged
