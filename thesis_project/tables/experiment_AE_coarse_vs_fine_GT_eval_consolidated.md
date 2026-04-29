# Experiments A–E: Coarse vs Fine GT — consolidated evaluation tables
Generated for drafting (e.g. thesis §4.4 Re-evaluation under Fine GT). Source snapshots: `thesis_project/tables/expA` … `expE`, files `*_coarse.csv` and `*_fine.csv`.
**Fine GT** labels: hierarchical `Section : Subtopic` in `ai_video_rbk/annotations_fine/` (see `thesis_project/src/generate_fine_gt_from_vtt.py`).
---
## Summary: best macro-F1 row per experiment
| Exp | Best F1 (coarse) | Best setting (coarse) | Best F1 (fine) | Best setting (fine) | Same best setting? |
| --- | --- | --- | --- | --- | --- |
| A | 0.112 | A2, seconds, w=10 | 0.3702 | A1, sentences, w=3 | no |
| B | 0.1234 | window_size=3 | 0.3702 | window_size=3 | yes |
| C | 0.1234 | Threshold + Local minima + Min-distance | 0.3702 | Threshold + Local minima + Min-distance | yes |
| D | 0.1698 | Marker | 0.3702 | Baseline | no |
| E | 0.1446 | E2 | 0.4083 | E2 | yes |
_“Same best setting?” compares a short signature (method/window/rule/model/E-setting); fine GT can change which row wins even when the grid is unchanged._
---
## Experiment A
- **Best (coarse GT):** F1=0.112 — `A2, seconds, w=10`
- **Best (fine GT):** F1=0.3702 — `A1, sentences, w=3`
### A — full table, **coarse** GT
_File:_ `thesis_project/tables/expA/expA_summary_table_coarse.csv`
| Method | Representation | Window Unit | Window Size | Threshold | Local Minima | Min-Distance | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | Sentence-based | sentences | 5 | 0.55 | Yes | 20s | 0.0812 | 0.1765 | 0.11 |
| A2 | Time-based | seconds | 10 | 0.55 | Yes | 20s | 0.0713 | 0.2644 | 0.112 |
### A — full table, **fine** GT
_File:_ `thesis_project/tables/expA/expA_summary_table_fine.csv`
| Method | Representation | Window Unit | Window Size | Threshold | Local Minima | Min-Distance | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | Sentence-based | sentences | 3 | 0.55 | Yes | 20s | 0.2481 | 0.7354 | 0.3702 |
| A1 | Sentence-based | sentences | 5 | 0.55 | Yes | 20s | 0.3323 | 0.198 | 0.2456 |
| A1 | Sentence-based | sentences | 7 | 0.55 | Yes | 20s | 0.3333 | 0.0353 | 0.0637 |
| A2 | Time-based | seconds | 10 | 0.55 | Yes | 20s | 0.3417 | 0.3694 | 0.3523 |
| A2 | Time-based | seconds | 15 | 0.55 | Yes | 20s | 0.3313 | 0.3745 | 0.3495 |
| A2 | Time-based | seconds | 5 | 0.55 | Yes | 20s | 0.3385 | 0.3562 | 0.3469 |

---
## Experiment B
- **Best (coarse GT):** F1=0.1234 — `window_size=3`
- **Best (fine GT):** F1=0.3702 — `window_size=3`
### B — full table, **coarse** GT
_File:_ `thesis_project/tables/expB/expB_window_size_summary_coarse.csv`
| Window Size | Precision | Recall | F1 | Predicted Boundaries | Interpretation |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0641 | 1.0 | 0.1205 | 154.0 | very sensitive / noisy |
| 3 | 0.0678 | 0.7079 | 0.1234 | 102.75 | local context |
| 5 | 0.0812 | 0.1765 | 0.11 | 21.5 | balanced baseline |
| 7 | 0.0833 | 0.025 | 0.0384 | 3.0 | smoother context |
| 10 | 0.125 | 0.025 | 0.0417 | 1.25 | overly broad context |
### B — full table, **fine** GT
_File:_ `thesis_project/tables/expB/expB_window_size_summary_fine.csv`
| Window Size | Precision | Recall | F1 | Predicted Boundaries | Interpretation |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.2221 | 0.9924 | 0.3622 | 154.0 | very sensitive / noisy |
| 3 | 0.2481 | 0.7354 | 0.3702 | 102.75 | local context |
| 5 | 0.3323 | 0.198 | 0.2456 | 21.5 | balanced baseline |
| 7 | 0.3333 | 0.0353 | 0.0637 | 3.0 | smoother context |
| 10 | 0.0 | 0.0 | 0.0 | 1.25 | overly broad context |

---
## Experiment C
- **Best (coarse GT):** F1=0.1234 — `Threshold + Local minima + Min-distance`
- **Best (fine GT):** F1=0.3702 — `Threshold + Local minima + Min-distance`
### C — full table, **coarse** GT
_File:_ `thesis_project/tables/expC/expC_rule_summary_coarse.csv`
| Rule | Precision | Recall | F1 | Predicted Boundaries | Interpretation |
| --- | --- | --- | --- | --- | --- |
| Threshold only | 0.0474 | 0.7634 | 0.0891 | 161.25 | most sensitive / highest noise risk |
| Threshold + Local minima | 0.0483 | 0.7079 | 0.0903 | 145.25 | noise-suppressed local dips |
| Threshold + Local minima + Min-distance | 0.0678 | 0.7079 | 0.1234 | 102.75 | redundancy-reduced practical segmentation |
### C — full table, **fine** GT
_File:_ `thesis_project/tables/expC/expC_rule_summary_fine.csv`
| Rule | Precision | Recall | F1 | Predicted Boundaries | Interpretation |
| --- | --- | --- | --- | --- | --- |
| Threshold only | 0.169 | 0.7814 | 0.2771 | 161.25 | most sensitive / highest noise risk |
| Threshold + Local minima | 0.1875 | 0.7814 | 0.3016 | 145.25 | noise-suppressed local dips |
| Threshold + Local minima + Min-distance | 0.2481 | 0.7354 | 0.3702 | 102.75 | redundancy-reduced practical segmentation |

---
## Experiment D
- **Best (coarse GT):** F1=0.1698 — `Marker`
- **Best (fine GT):** F1=0.3702 — `Baseline`
### D — full table, **coarse** GT
_File:_ `thesis_project/tables/expD/expD_model_comparison_coarse.csv`
| Model | Semantic | Marker | Filler | Slide | Precision | Recall | F1 | Predicted Boundaries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | ✓ | × | × | × | 0.0678 | 0.7079 | 0.1234 | 102.75 |
| Marker | × | ✓ | × | × | 0.1058 | 0.4358 | 0.1698 | 40.75 |
| Filler | × | × | ✓ | × | 0.0655 | 0.8134 | 0.1212 | 122.0 |
| Slide | × | × | × | ✓ | 0.1027 | 0.121 | 0.1057 | 13.0 |
| +Marker | ✓ | ✓ | × | × | 0.0725 | 0.8639 | 0.1336 | 117.5 |
| +Filler | ✓ | × | ✓ | × | 0.057 | 0.9194 | 0.1072 | 158.0 |
| +Slide | ✓ | × | × | ✓ | 0.071 | 0.7606 | 0.1295 | 106.75 |
| All | ✓ | ✓ | ✓ | ✓ | 0.0568 | 0.9722 | 0.1073 | 167.75 |
### D — full table, **fine** GT
_File:_ `thesis_project/tables/expD/expD_model_comparison_fine.csv`
| Model | Semantic | Marker | Filler | Slide | Precision | Recall | F1 | Predicted Boundaries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | ✓ | × | × | × | 0.2481 | 0.7354 | 0.3702 | 102.75 |
| Marker | × | ✓ | × | × | 0.2806 | 0.3362 | 0.302 | 40.75 |
| Filler | × | × | ✓ | × | 0.2249 | 0.8068 | 0.3484 | 122.0 |
| Slide | × | × | × | ✓ | 0.2054 | 0.0867 | 0.1212 | 13.0 |
| +Marker | ✓ | ✓ | × | × | 0.2389 | 0.8055 | 0.3674 | 117.5 |
| +Filler | ✓ | × | ✓ | × | 0.2042 | 0.9356 | 0.3337 | 158.0 |
| +Slide | ✓ | × | × | ✓ | 0.2405 | 0.7405 | 0.3623 | 106.75 |
| All | ✓ | ✓ | ✓ | ✓ | 0.1951 | 0.9481 | 0.3221 | 167.75 |

---
## Experiment E
- **Best (coarse GT):** F1=0.1446 — `E2`
- **Best (fine GT):** F1=0.4083 — `E2`
### E — full table, **coarse** GT
_File:_ `thesis_project/tables/expE/expE_pruning_summary_coarse.csv`
| Setting | Threshold | Local Minima | Min-distance | Prominence | Precision | Recall | F1 | Predicted Boundaries | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 0.55 | Yes | 20s | - | 0.0678 | 0.7079 | 0.1234 | 102.75 | baseline pruning |
| E2 | 0.55 | Yes | 30s | - | 0.081 | 0.6801 | 0.1446 | 82.75 | fewer close duplicates |
| E3 | 0.55 | Yes | 45s | - | 0.0729 | 0.4684 | 0.1259 | 62.0 | stronger suppression |
| E4 | 0.55 | Yes | 60s | - | 0.0524 | 0.2719 | 0.0876 | 50.25 | very strict spacing |
### E — full table, **fine** GT
_File:_ `thesis_project/tables/expE/expE_pruning_summary_fine.csv`
| Setting | Threshold | Local Minima | Min-distance | Prominence | Precision | Recall | F1 | Predicted Boundaries | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 0.55 | Yes | 20s | - | 0.2481 | 0.7354 | 0.3702 | 102.75 | baseline pruning |
| E2 | 0.55 | Yes | 30s | - | 0.2905 | 0.6923 | 0.4083 | 82.75 | fewer close duplicates |
| E3 | 0.55 | Yes | 45s | - | 0.3082 | 0.5586 | 0.3954 | 62.0 | stronger suppression |
| E4 | 0.55 | Yes | 60s | - | 0.3229 | 0.472 | 0.3821 | 50.25 | very strict spacing |

---
