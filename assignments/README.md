# Assignments

One model per person. Each sheet is self-contained — read yours, the [contract](../CONTRACT.md), and [`report_schema.json`](../report_schema.json).

| id | model | venue | variants replace |
|---|---|---|---|
| [`01-drugclip`](01-drugclip.md) | DrugCLIP | NeurIPS 2023 | `train.lmdb` |
| [`02-hypseek`](02-hypseek.md) | HypSeek | AAAI 2026 | both `train_label_blend_seq_full.json` and `train_label_pdbbind_seq.json` |
| [`03-causalbind`](03-causalbind.md) | CausalBind | NeurIPS 2026 | both `train_label_blend_seq_full.json` and `train_label_pdbbind_seq.json` |
| [`04-conglude`](04-conglude.md) | ConGLUDe | ICML 2026 | TO BE CONFIRMED — the exact training file names will be |
| [`05-bindclip`](05-bindclip.md) | BindCLIP | KDD 2026 | the LMDB |
| [`06-drughash`](06-drughash.md) | DrugHash | AAAI 2025 | `train.lmdb` |
| [`07-sprint`](07-sprint.md) | SPRINT | MLSB @ NeurIPS 2024 | `merged_pos_uniq_train_rand.tsv` and `merged_neg_uniq_train_rand.tsv` |

## Why some of you share variant files

Several of these models train on the same released corpus, so they receive the same variant files. That is not redundancy: the point is to see the same training-set change through different architectures. Others train on their own corpus and get variants built from it.

Everything in checkpoint A needs nothing from us, so **nobody has to wait** to start.
