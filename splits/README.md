# Training-set variants

Variants are published as **GitHub Release assets**, not committed here: they are large JSON files
and git is the wrong place for them. This directory holds only `MANIFEST.json`, which is the
authoritative record of what each variant contains.

For each variant id, the manifest lists every file with its SHA256, row count and group count.
Before training, verify what you received:

```bash
sha256sum <file>          # compare against MANIFEST.json
```

CI checks the digests you record in `stage5_taskC/*/summary.json` against this manifest, so a run on
the wrong input is caught rather than silently included.

Variants are grouped by corpus. You will be pointed at the ids that apply to your model; ignore the
rest.
