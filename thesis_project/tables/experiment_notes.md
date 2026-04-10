# Experiment Notes — Experiments A through E (+ F1)

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
- Absolute F1 remains low → **signal strength and GT definition** still limit performance; motivates D and error analysis.

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
- **Discourse markers** can act as **coarse anchors** and improve F1 → aligns with **Experiment F (marker-gated)** motivation.
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
- **Beyond 30 s:** Recall collapses; **spacing alone hits a ceiling** → motivates **granularity-aligned detection (Experiment F)**.
- Experiment E **isolates excess prediction count**; remaining error is tied to **GT scale, sensitivity, and structural cues**.

---

## Experiment F1 — Marker-gated semantic detection (implemented)

### Rationale
Reduce sensitivity to fine-grained semantic dips alone: **discourse markers** propose *where* to look; **semantic** (local minima below a confirmation threshold) must agree within **±Δ s**. Final **min-distance 30 s** matches the Experiment E best spacing.

### Fixed baseline (F shared)
- Sentence windows, **w = 3**, **min-distance 30 s** after gating, evaluation **±30 s**.
- Similarities computed **once per lecture** (cached under `results/expF1_marker_gated/_similarity_cache/`).

### Procedure
1. **Semantic pool:** all adjacent-window dips that are **local minima** and **similarity &lt; confirm_threshold** (no min-distance before gating).
2. **Markers:** same regex set as Experiment D on VTT subtitle starts; **20 s** deduplication between marker hits.
3. **Gate:** keep a dip iff **∃ marker** with **|t_marker − t_dip| ≤ Δ**.
4. **Post-filter:** **30 s** min-distance (stronger dip wins when overlapping).

### Swept parameters
| Parameter | Values |
|-----------|--------|
| Marker window **±Δ** | 5, 10, 15 s |
| Semantic confirm threshold | 0.55, 0.60 (*similarity &lt; T*; **lower T = stricter**) |

### Macro results (four lectures, `expF1_marker_gated_summary.csv`)
| Setting | ±Δ (s) | Confirm | F1 | Avg pred / lecture |
|---------|--------|---------|-----|-------------------|
| F1_pm5_c055_md30 | 5 | 0.55 | 0.0921 | 12.0 |
| F1_pm5_c06_md30 | 5 | 0.60 | 0.1069 | 18.5 |
| F1_pm10_c055_md30 | 10 | 0.55 | 0.1290 | 20.5 |
| F1_pm10_c06_md30 | 10 | 0.60 | 0.1562 | 27.0 |
| F1_pm15_c055_md30 | 15 | 0.55 | 0.1530 | 24.5 |
| **F1_pm15_c06_md30** | **15** | **0.60** | **0.1802** | **30.25** |

### Interpretation
- **Best F1 in this sweep** beats **semantic-only baseline (0.1234)** and **Experiment E best (0.1446)**, and edges **marker-only from D (0.1698)** — with a **clear mechanism** (marker window + semantic confirmation).
- Wider **Δ** and **looser** confirm threshold (0.60) add recall; **tight Δ + 0.55** yields very few predictions (high precision potential but low recall).
- Run script: `python3 thesis_project/src/run_experiment_f1.py` (optional: `--marker-windows`, `--confirm-thresholds`).

### Lecture-level vs. baseline (semantic E: `sentence_w3_t055_md30`)
Macro gains **do not** hold uniformly. Compare:
- `tables/expF1_marker_gated_lecture_f1.csv` — per-lecture F1 for baseline and each F1 setting.
- `tables/expF1_marker_gated_lecture_delta_vs_baseline.csv` — ΔF1 vs baseline.
- `tables/expF1_marker_gated_improvement_counts.csv` — how many lectures improved / worsened per setting.

**Takeaway:** For **F1_pm15_c06_md30** (best macro), **3 lectures improve** and **lecture4 is worse** (F1 0.0625 vs baseline 0.1124). **No grid setting improves all four lectures** in this sweep; interpret macro F1 as **average behavior**, not guaranteed per-lecture gains.

---

## Experiment F2 — Prominence filtering on the F1-best marker gate

### Motivation (after F1)
F1 showed that **discourse markers work as coarse anchors**, but **false positives** remain: many predictions align with **fine-grained explanation shifts** rather than section-level GT. F2 **fixes** the F1-best configuration (**±15 s**, confirm **similarity &lt; 0.60**, **min-distance 30 s**, sentence **w=3**) and adds a **single extra knob**: minimum **similarity prominence** for each semantic dip (method 1: `min(left_avg − dip, right_avg − dip)` with **context span = 1**). Order: **local minimum + confirm threshold → prominence filter → marker gate → min-distance**.

### Formal experiment sentence
Experiment F1 showed that discourse markers are effective as coarse anchors, but the resulting system still produces false positives due to fine-grained semantic fluctuations. Experiment F2 therefore kept the best F1 setting fixed and tested whether prominence-based filtering of semantic dips can further suppress shallow explanation-level transitions and improve coarse boundary alignment.

### Swept parameter
| Parameter | Values |
|-----------|--------|
| **prominence_min** | 0.00 (no extra filter; matches F1 best), 0.02, 0.04, 0.06, 0.08 |

### Macro results (`expF2_prominence_summary.csv`)
| prominence_min | F1 | Avg pred / lecture | Precision | Recall |
|----------------|-----|-------------------|-----------|--------|
| 0.00 | **0.1802** | **30.25** | 0.1187 | 0.3803 |
| 0.02 | 0.1709 | 29.25 | 0.1130 | 0.3553 |
| 0.04 | 0.1611 | 28.75 | 0.1073 | 0.3275 |
| 0.06 | 0.1432 | 27.50 | 0.0970 | 0.2770 |
| 0.08 | 0.1444 | 26.75 | 0.0990 | 0.2720 |

### Interpretation
- On this grid, **macro F1 monotonically decreases** as prominence_min increases: the **F1-best system remains F2 with prominence_min = 0** (numerically identical to **F1_pm15_c06_md30**).
- **Lecture4** stays **weak** across the sweep (F1 ~0.0625 at prom=0; slight F1 bump to **0.0645** for prom ≥ 0.02 with **no change in TP** in the printed runs — a **tiny** precision-side effect). This supports treating lecture4’s gap as **unlikely to be solved by prominence alone**; **marker coverage / cue strength** (and later **slide** or other cues) remain plausible explanations.
- Counts vs **prom0**: see `expF2_prominence_lecture_improvement_vs_prom0.csv` (e.g. prom **0.02** improves **3** lectures and worsens **1** vs prom0, but **macro** still drops because the worsened lecture’s loss dominates the average).

### Artifacts and command
- Summary: `tables/expF2_prominence_summary.csv`
- Lecture F1 matrix: `tables/expF2_prominence_lecture_f1.csv`
- vs prom0 counts: `tables/expF2_prominence_lecture_improvement_vs_prom0.csv`
- Per-setting outputs: `results/expF2_prominence/<setting>/`
- Run: `python3 thesis_project/src/run_experiment_f2.py` (optional: `--prominence-mins`, `--context-span`).

---

## Experiment F3 — Slide transitions as auxiliary structure (after F1-best, no F2 prominence)

### Motivation
F2 showed that **tightening semantic dips** (prominence) did not help. F3 **does not** further restrict semantics; it **complements** the F1-best pipeline with **slide transition times** (`thesis_project/data/slide_transitions/lecture*_slides.txt`, same source as Experiment D).

### Fixed (F1 best)
Sentence **w=3**, semantic confirm **similarity &lt; 0.60**, local minima, marker **±15 s**, **min-distance 30 s**, same marker regex + dedupe as F1. **Prominence is off.**

### Decision rules
- **OR (`F3_OR_*`):** `marker_near AND (semantic_confirm OR slide_near)` — implemented as **marker-gated semantic dips** (same as F1) **plus** auxiliary boundaries at **each slide time** `s` with `marker_near(s)` (at `t=s`, `slide_near(t)` holds for any positive window). Then **min-distance 30 s** (keep stronger dip = lower similarity when ties occur).
- **AND (`F3_AND_sw20`):** `marker_near AND semantic_confirm AND slide_near` — keep only marker-gated semantic dips whose boundary time is within **±20 s** of **some** slide time (precision-oriented; very strict).

### Formal experiment sentence
In Experiment F3, the marker-gated framework is extended by incorporating slide transitions as an additional structural cue. Instead of further restricting semantic dips, the goal is to complement marker-based anchors with slide-based signals where marker coverage is thin. The primary decision rule is marker presence combined with either semantic confirmation or slide proximity.

### Swept parameter (thesis grid)
| Setting | Rule | Slide window (±s) |
|---------|------|-------------------|
| F3_OR_sw20_pm15_c06_md30 | OR | 20 |
| F3_OR_sw30_pm15_c06_md30 | OR | 30 |
| F3_AND_sw20_pm15_c06_md30 | AND | 20 |

**Note on OR and slide window:** With the OR implementation above, **F3_OR_sw20** and **F3_OR_sw30** produce **identical** predictions: auxiliary candidates are placed **at** slide timestamps, so `slide_near` is satisfied for any window &gt; 0. The two rows document the planned grid; the meaningful OR vs AND comparison is **OR vs AND**.

### Macro results (`expF3_slide_summary.csv`)
| Setting | F1 | Avg pred / lecture | Precision | Recall |
|---------|-----|-------------------|-----------|--------|
| F3_OR_sw20 / sw30 | **0.1935** | 30.75 | 0.1275 | 0.4081 |
| F3_AND_sw20 | 0.0000 | 2.25 | 0.0000 | 0.0000 |

### Interpretation
- **OR** improves **macro F1** vs F1-best (**0.1802 → 0.1935**) with similar prediction density (~**30.8** vs ~30.3 per lecture). **Lecture4** F1 rises from **~0.0625** (F1-best) to **0.1212** — slide-assisted recall at marker-adjacent slide times helps this lecture in particular.
- **AND** is **too strict** here: very few boundaries survive (semantic dip must align with a slide within ±20 s), yielding **near-zero** macro F1 — useful as a **sanity / precision** check, not as a production rule.

### Artifacts and command
- Summary: `tables/expF3_slide_summary.csv`
- Lecture F1: `tables/expF3_slide_lecture_f1.csv`
- Δ vs F1-best macro reference: `tables/expF3_slide_lecture_delta_vs_f1best.csv`
- Outputs: `results/expF3_slide/<setting>/`
- Run: `python3 thesis_project/src/run_experiment_f3.py` (`--slides-dir`, `--lectures` optional).

---

## GT protocol rerun (coarse vs hierarchical fine GT)

To check whether section-level GT underestimates clip-oriented detection quality, Experiments **F1–F3** were rerun on two annotation protocols:
- **Coarse GT:** `ai_video_rbk/annotations_corrected` (section-level boundaries)
- **Fine GT:** `ai_video_rbk/annotations_fine` (hierarchical labels: `Section : Subtopic`)

### Best-setting comparison
| Experiment | Coarse best setting | Coarse best F1 | Fine best setting | Fine best F1 | Δ (Fine - Coarse) |
|------------|---------------------|----------------|-------------------|--------------|-------------------|
| F1 | `F1_pm15_c06_md30` | 0.1802 | `F1_pm15_c06_md30` | 0.3114 | +0.1312 |
| F2 | `F2_pm15_c06_p00_md30_s1` | 0.1802 | `F2_pm15_c06_p00_md30_s1` | 0.3114 | +0.1312 |
| F3 | `F3_OR_sw20_pm15_c06_md30` (=`sw30`) | 0.1935 | `F3_OR_sw20_pm15_c06_md30` (=`sw30`) | 0.3088 | +0.1153 |

### Interpretation (important)
- Fine GT substantially raises measured precision/F1 for F1–F3, consistent with the hypothesis that many former "false positives" are actually valid **clip-worthy fine transitions** under a finer annotation protocol.
- Under fine GT, **F1/F2 best (0.3114)** slightly exceeds **F3 OR best (0.3088)**; this suggests slide cues are still useful structurally, but are not automatically superior when the semantic-marker pair already aligns well with fine-grained labels.
- F2 still peaks at `prominence=0.00`, confirming the earlier conclusion: additional prominence filtering does not improve this pipeline on this dataset.

### Comparison artifacts
- Coarse snapshots: `tables/expF/expF1_marker_gated_summary_coarse.csv`, `tables/expF/expF2_prominence_summary_coarse.csv`, `tables/expF/expF3_slide_summary_coarse.csv`
- Fine snapshots: `tables/expF/expF1_marker_gated_summary_fine.csv`, `tables/expF/expF2_prominence_summary_fine.csv`, `tables/expF/expF3_slide_summary_fine.csv`
- Side-by-side HTML: `tables/expF/expF_gt_comparison.html`

---

## Full rerun on fine GT (A-F)

All experiments were rerun using `ai_video_rbk/annotations_fine` to align evaluation with clip-oriented, hierarchical boundaries (`Section : Subtopic`).

### Best setting per experiment (fine GT)
| Experiment | Best setting | Precision | Recall | F1 | Avg pred / lecture |
|------------|--------------|-----------|--------|----|--------------------|
| A | `A2` (time-based, 10s) | 0.0713 | 0.2644 | 0.1120 | - |
| B | `window_size=3` | 0.2481 | 0.7354 | 0.3702 | 102.75 |
| C | `Threshold + Local minima + Min-distance` | 0.2481 | 0.7354 | 0.3702 | 102.75 |
| D | `Baseline (semantic only)` | 0.2481 | 0.7354 | 0.3702 | 102.75 |
| E | `E2 (min-distance=30s)` | 0.2905 | 0.6923 | **0.4083** | 82.75 |
| F1 | `F1_pm15_c06_md30` | 0.3381 | 0.2939 | 0.3114 | 30.25 |
| F2 | `F2_pm15_c06_p00_md30_s1` | 0.3381 | 0.2939 | 0.3114 | 30.25 |
| F3 | `F3_OR_sw20_pm15_c06_md30` (=`sw30`) | 0.3316 | 0.2939 | 0.3088 | 30.75 |

### Key observations
- Under fine GT, **E2 (30s pruning)** gives the highest macro F1 among A-F (**0.4083**), indicating that redundancy control remains highly effective.
- F-branch methods (F1/F2/F3) become substantially stronger than under coarse GT, but in this run they do not surpass E2 macro F1.
- F2 still peaks at `prominence=0.00`, reinforcing that added prominence filtering does not help on this dataset/settings.

### Updated reports
- Main report (A-F): `tables/experiment_report.html`
- F coarse-vs-fine comparison report: `tables/expF/expF_gt_comparison.html`

---

## Cross-cutting summary (A–E, + F1, + F2, + F3)

1. **Representation (A) and window size (B)** did not yield large F1 gains alone; **rules (C) and pruning (E)** most directly control **prediction density**.
2. Among **structural cues (D)**, **markers** most convincingly raise F1 on this dataset.
3. **Follow-up manual analysis** (separate artifacts): many `spurious_far` cases resemble **fine-grained explanation shifts** rather than meaningless noise, motivating **coarse-aware detection (Experiment F)**.
4. **F1** shows that **marker-primary + semantic validation** can improve F1 while keeping an interpretable design.
5. **F2** (prominence on top of F1-best): on the tested grid, **no macro gain** vs F1-best; supports treating remaining errors as **not fully fixable by dip sharpness alone** and motivates **F3** (e.g. slide / richer cues), especially for **lecture4**.
6. **F3** (slides + F1-best): **OR** rule improves **macro F1** and **lecture4** vs F1-best under coarse GT; **AND** is a strict ablation with near-zero F1.
7. Re-running F1–F3 on **hierarchical fine GT** materially changes absolute metrics (higher F1), showing that GT granularity is not a minor detail but a core evaluation design choice for clip-oriented segmentation.

---

## Artifact index

| Experiment | Summary table | Result directory |
|------------|---------------|------------------|
| A | `expA_summary_table.csv` | `results/expA_representation/` |
| B | `expB_window_size_summary.csv` | `results/expB_window_size/` |
| C | `expC_rule_summary.csv` | `results/expC_boundary_rule/` |
| D | `expD_model_comparison.csv` | `results/expD_structural/` |
| E | `expE_pruning_summary.csv` | `results/expE_prediction_pruning/` |
| F1 | `expF1_marker_gated_summary.csv` (+ lecture tables) and coarse/fine snapshots in `expF/` | `results/expF1_marker_gated/` |
| F2 | `expF2_prominence_summary.csv` (+ lecture tables) and coarse/fine snapshots in `expF/` | `results/expF2_prominence/` |
| F3 | `expF3_slide_summary.csv` (+ lecture tables) and coarse/fine snapshots in `expF/` | `results/expF3_slide/` |

Combined HTML report: `tables/experiment_report.html`  
Experiment F proposal: `tables/experiment_F_coarse_aware_proposal.md`

---

## Next steps (experiments and writing)

- **Writing:** Report **F3 OR** as the strongest **rule-based** combination on this dataset; discuss **F3 AND** as an intentionally **over-constrained** variant; cite **lecture4** gains and remaining **limitation** language if ceiling persists.
- **Optional follow-ups:** Refine slide timestamps, align slide–speech lag, or relax AND with a larger slide window — outside the current single-factor grids.
- **Thesis / report:** Condense tables where possible; repeatedly state **single-factor control** for clarity.
- **Limitations:** GT is **coarse (section-level)** while the model fires on **finer semantic shifts**; pair with **error-analysis evidence** for a stronger discussion.
