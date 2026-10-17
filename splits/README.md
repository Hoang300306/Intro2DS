# Training-set variants

**No training data is distributed here.** You already download the released training file for your
model — checkpoint B retrains on it — and a variant differs from it only in which records it keeps.
So this directory holds small **specs** saying which, and you build the variant locally.

```
splits/
  MANIFEST.json          what each variant is, which released file it replaces, and its digests
  specs/<variant_id>/    the spec(s) for that variant, a few tens of KB each
```

To build one:

```bash
python tools/apply_variant.py splits/specs/<variant_id>/<spec>.json.gz \
       <your released training file> <output file>
```

The script verifies your released file against the digest in the manifest **before writing
anything**, and verifies its own output afterwards. Both passing proves two things at once: your
download is the same file the variant was built from, and the filter applied correctly. Being handed
a finished file could only ever have shown the second.

If either check fails, **stop and open an issue**. Do not train on it.

## Why digests are over content, not bytes

Two LMDBs holding identical records can differ byte-for-byte across library versions, so a byte
digest would produce mismatches that mean nothing. The digests here are over the sorted record
contents instead, which is stable across versions and is what actually matters.

## Which variants apply to you

Your assignment names them. Several models train on the same released corpus and so share specs;
others have their own. Ignore the ones not listed in your sheet.

## The two groups of variants are not quite the same intervention

Worth stating plainly, because it changes what a comparison between people means.

The `pocketaffdb_*` variants are the **exact files** from the study they came from, reproduced here
rather than regenerated, so that a result on them is directly comparable to the measurement already
made on that corpus. The selection there is target-aware: whether a given molecule is removed
depends on which target the assay measures.

Every other corpus gets the **same molecule set applied flatly** — a molecule is removed wherever it
appears. That is the natural generalisation to a corpus with a different structure, and it is what
makes the variants well defined for corpora that are not organised by assay at all.

So: two people on `pocketaffdb` are running the same intervention as the original measurement and as
each other, which makes their numbers directly comparable. People on other corpora are running the
flat version, which answers a closely related question rather than the identical one. Neither is
wrong; they are just not interchangeable, and nobody should present them as a single number.
