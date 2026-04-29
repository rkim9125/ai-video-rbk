# Experiment Notes — Experiments A through E

All runs vary **one factor at a time** on the same corpus (lectures 1–4, WebVTT transcripts, shared ground truth and tolerance). See `experiment_master_table.csv`, `experiment_report.html`, and per-experiment `*_summary.csv` for full numbers.

---

## Experiment A — Window Representation (sentence vs. time)

### Rationale
Lecture segmentation depends strongly on **how transcript context is grouped**. ASR captions have unreliable sentence boundaries, so we compare **sentence-based sliding windows** vs. **fixed-duration (10 s) windows** under identical embedding, thresholding, and evaluation.

### Setup
- **A1:** Sentence-based, window size **5**, threshold **0.55**, local minima on, min-distance **20 s**, tolerance **±30 s**.
- **A2:** Time-based **10 s** windows; all other settings matched.
- Outputs: `thesis_project/results/expA_representation/`, `tables/expA_summary_table.csv`, `expA_lecture_level_table.csv`.

### Results (macro average, four lectures)
| Method | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| A1 Sentence w5 | 0.0812 | 0.1765 | **0.11** |
| A2 Time 10 s | 0.0713 | 0.2644 | **0.112** |

### Interpretation
- **F1 is essentially tied**; neither representation dominates on this setup.
- Time-based shows **higher recall and lower precision** → tends toward **more predictions, less precision**.
- Later work can keep **sentence-based** as the default while stating that **representation alone does not fix performance**.

---

## Experiment B — Sentence window size

### Rationale
Short, choppy captions make **small windows noisy** and **large windows** overly smooth. We sweep window size only to study the sensitivity–stability trade-off.

### Setup
- Sentence representation; threshold **0.55**; local minima + min-distance **20 s**; other pipeline settings fixed.
- **Window sizes:** 1, 3, 5, 7, 10 sentences.
- Outputs: `results/expB_window_size/`, `tables/expB_window_size_summary.csv`.

### Results (macro)
| ws | Precision | Recall | F1 | Avg pred / lecture |
|----|-----------|--------|-----|---------------------|
| 1 | 0.0641 | 1.000 | 0.1205 | 154.0 |
| 3 | 0.0678 | 0.7079 | **0.1234** | 102.75 |
| 5 | 0.0812 | 0.1765 | 0.11 | 21.5 |
| 7 | 0.0833 | 0.025 | 0.0384 | 3.0 |
| 10 | 0.125 | 0.025 | 0.0417 | 1.25 |

### Interpretation
- **w = 1:** Near-max recall but **prediction explosion** → severe over-segmentation.
- **w = 3:** Reasonable F1 and moderate prediction count → **reference setting** for Experiments C and E.
- **w ≥ 7:** Too few predictions → **recall collapse**; very large context is a poor fit for this pipeline.
- Window size **directly controls boundary density**; discuss together with **alignment to GT granularity**.

---

## Experiment C — Boundary decision rules

### Rationale
Similarity **scoring** and **selection rules** are distinct. We compare threshold-only, +local minima, and +min-distance to see how stricter post-processing changes FP/TP balance.

### Setup
- **Fixed:** sentence, **window size 3**, threshold **0.55**, min-distance **20 s** (disabled where the rule variant requires), tolerance **±30 s**.
- **Rules:** (1) threshold only (2) + local minima (3) + min-distance filter.
- Outputs: `results/expC_boundary_rule/`, `tables/expC_rule_summary.csv`.

### Results (macro)
| Rule | Precision | Recall | F1 | Avg pred / lecture |
|------|-----------|--------|-----|---------------------|
| Threshold only | 0.0474 | 0.7634 | 0.0891 | 161.25 |
| + Local minima | 0.0483 | 0.7079 | 0.0903 | 145.25 |
| + Min-distance | **0.0678** | 0.7079 | **0.1234** | 102.75 |

### Interpretation
- Stricter rules → **fewer predictions**, **higher precision and F1** (for this configuration).
- Supports the claim that **part of the failure is decision logic**, not embeddings alone.
- Absolute F1 remains low → **signal strength and GT definition** still limit performance.

---

## Experiment D — Structural / textual cues (markers, fillers, slides)

### Rationale
Pure semantic dips struggle on **administrative segments, Q&A, and example-heavy teaching**. We test **discourse markers, filler density, and slide transitions from video** alone and combined with semantics.

### Setup
- Baseline semantic pipeline aligned with Experiment C (see `run_experiment_d.py` for exact flags).
- Variants: marker-only, filler-only, slide-only, semantic + each cue, all cues.
- Slides: `thesis_project/data/slide_transitions/` (e.g. ffmpeg scene detection).
- Outputs: `results/expD_structural/`, `tables/expD_model_comparison.csv`.

### Results (macro)
| Model | F1 | Avg pred / lecture | Notes |
|-------|-----|---------------------|--------|
| Baseline | 0.1234 | 102.75 | Semantic only |
| **Marker** | **0.1698** | 40.75 | Best F1 in table |
| Filler | 0.1212 | 122.0 | High recall, low precision |
| Slide | 0.1057 | 13.0 | Very low recall |
| +Marker | 0.1336 | 117.5 | Fusion underperforms marker-only here |
| +Filler | 0.1072 | 158.0 | |
| +Slide | 0.1295 | 106.75 | |
| All | 0.1073 | 167.75 | Too many predictions; F1 drops |

### Interpretation
- **Discourse markers** can act as **coarse anchors** and improve F1.
- Slide detection may **under-trigger** depending on video and threshold; treat as **auxiliary** unless retuned.
- **Fusion is not universally helpful**; **which cues to combine and how** matters.

---

## Experiment E — Min-distance pruning (prediction count control)

### Rationale
Test whether **candidate explosion** explains poor F1 by sweeping **min-distance** only: 20 → 30 → 45 → 60 s.

### Setup
- **Fixed:** sentence **w3**, threshold **0.55**, local minima on, min-distance filter on.
- **Varied:** min-distance **20, 30, 45, 60** s.
- Outputs: `results/expE_prediction_pruning/`, `tables/expE_pruning_summary.csv`, `expE_pruning_lecture_f1.csv`.

### Results (macro)
| Setting | Min-dist | Precision | Recall | F1 | Avg pred / lecture |
|---------|----------|-----------|--------|-----|---------------------|
| E1 | 20 s | 0.0678 | 0.7079 | 0.1234 | 102.75 |
| **E2** | **30 s** | **0.0810** | 0.6801 | **0.1446** | 82.75 |
| E3 | 45 s | 0.0729 | 0.4684 | 0.1259 | 62.0 |
| E4 | 60 s | 0.0524 | 0.2719 | 0.0876 | 50.25 |

### Interpretation
- **20 → 30 s:** Best F1 with fewer predictions → **over-segmentation is a real failure mode**.
- **Beyond 30 s:** Recall collapses; **spacing alone hits a ceiling** → motivates granularity-aware follow-up analysis.
- Experiment E **isolates excess prediction count**; remaining error is tied to **GT scale, sensitivity, and structural cues**.

---
