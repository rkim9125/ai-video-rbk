# Experiment F — Coarse-Aware Boundary Detection

### Brief for advisor meetings (Experiment E → error analysis → F)

---

## 0. One-line narrative

**Experiment E** showed that tightening min-distance improves F1 only up to a point. **Alignment and error analysis** suggest the model is less “randomly wrong” than **over-sensitive to fine-grained, still-meaningful shifts** relative to **coarse GT**. **Experiment F** realigns detection toward **coarse, structurally cued** transitions.

---

## 1. Motivation — why Experiment F

### 1.1 From Experiment E
- A **min-distance sweep (20 / 30 / 45 / 60 s)** reduces prediction count.
- **30 s** gives the best macro F1 in our sweep, yet absolute F1 remains modest.
- So **candidate explosion** is **not the only** explanation.

### 1.2 From follow-on analysis

**Quantitative (alignment)**  
- A large share of false positives are **`spurious_far`** (nearest GT **> 120 s**), on the order of **~60–72%** of FP by lecture in our bucket definitions — many predictions are **far in time** from any GT timestamp.

**Qualitative (manual coding, lecture 1)**  
- For the **25 farthest `spurious_far`** predictions (VTT snippets, fixed pipeline: md30, w3, t0.55), tags (1–5 scheme):

| Tag (short) | Share | Meaning |
|-------------|-------|---------|
| Example / explanation shift | **~60%** | Same broad topic; different example or explanatory thread |
| Potential subtopic boundary | **~16%** | Possible larger shift; we do **not** claim syllabus-level truth |
| Minor discourse shift | **~16%** | Q&A, admin, tone / participant change |
| Transcript / disfluency noise | **~8%** | Broken ASR, fillers, insertions |
| Completely meaningless | **~0%** | No case classified as empty / random |

*Note: These proportions are a **sample** (lecture 1, top-25 farthest `spurious_far`); they are **not** a formal estimate of global FP composition.*

### 1.3 Motivation (one paragraph)

The model responds to **meaning transitions finer than coarse GT boundaries**, not only to noise. The next step is therefore not **only** “add more features,” but **match detector sensitivity to task granularity**.

---

## 2. Key insight

- In this sample, most `spurious_far` cases look like **fine-grained meaningful shifts**, not pure noise.
- The pipeline is **over-sensitive to local similarity fluctuation**.
- **Over-generation** and **sensitivity–GT mismatch** coexist.

---

## 3. Research question

**Can we improve segmentation by reducing sensitivity to fine-grained fluctuations and aligning detection with coarse, topic-level (or structurally cued) boundaries?**

---

## 4. Hypothesis

If boundary decisions are **constrained** toward **coarse / structurally salient** transitions rather than every semantic dip, **false positives fall, precision rises**, and **F1 improves**, possibly with some **recall** cost.

---

## 5. Experiment design — Experiment F

### Shared
- **Metrics:** Precision, Recall, F1, prediction count (same tolerance as before).
- **Comparisons:** Prediction count vs. GT count; FP reduction vs. pre-F baselines.

---

### F1 — Marker-gated semantic detection

**Idea:** Use **discourse markers** as **primary triggers**; use semantic dips only to **validate within ±Δ**.

**Procedure**
1. Detect marker times `t` (e.g. now, let’s move on, next, today we… — reuse Experiment D patterns).
2. For each `t`, **accept** a boundary only if a **semantic candidate** (from threshold + local minima + min-distance pipeline) falls inside **±Δ s**; otherwise **reject**.

**Sweep (example)**

| Parameter | Values |
|-----------|--------|
| Marker window Δ | ±5 s, ±10 s, ±15 s |
| Semantic threshold | 0.55, 0.60 |
| Min-distance | **30 s fixed** (near-optimal in E) |

**Expectation:** Higher precision; recall may dip; F1 up if coarse anchoring helps.

---

### F2 — Prominence-based semantic filtering

**Idea:** Keep only **deep** dips relative to local context, not every local minimum.

**Implementation sketch**
- Define **similarity prominence** (e.g. difference between local minimum and neighboring peaks / local mean).
- Keep dips with **prominence > τ** or **top-k** globally per lecture.

**Sweep (example)**

| Parameter | Values |
|-----------|--------|
| Prominence threshold τ | 0.02, 0.05, 0.08 |
| Window size | **3 fixed** |
| Min-distance | **30 s fixed** |

**Expectation:** Suppress shallow dips (example-level shifts); retain stronger shifts → higher precision.

---

### F3 — Prediction count alignment (optional diagnostic)

**Idea:** Keep only **top-N** candidates per lecture to **match GT scale** — tests how much metrics move if count is forced down.

**Sweep**

| Parameter | Values |
|-----------|--------|
| N (per lecture) | 10, 15, 20 |

**Purpose:** **Diagnostic** ablation for granularity / over-prediction; **not** necessarily the final deployed model.

---

## 6. Evaluation plan

- Same as before: **P / R / F1**, **prediction count**.
- Recommended additions:
  - Per-lecture / per-setting **prediction count vs. GT count**.
  - If feasible: shift in FP buckets (`redundant_near_gt` / `offset` / `spurious_far`) vs. pre-F.

---

## 7. Expected contribution

Shows that improvement is not **only** “more features,” but **aligning sensitivity with task granularity**, linked by **E + error analysis + F** in one story: **why it failed → what we changed**.

---

## 8. Short English pitch (memorizable)

> From error analysis, most false positives in our `spurious_far` sample are not random noise but **fine-grained explanation shifts** that do not match our **coarse** ground truth. So instead of only adding features, I want to **reduce sensitivity to small fluctuations** and emphasize **coarse, structurally cued** transitions. That motivates **marker-gated semantic validation** and **prominence-based filtering** in Experiment F.

---

## 9. One-sentence summary

**Experiment F moves the objective from “detect every shift” to “detect coarse, meaningful boundaries.”**

---

## 10. Current status

| Item | Status |
|------|--------|
| Experiment design (A–E and F draft) | Solid |
| Alignment, FP typing, manual coding | Strengthens narrative |
| **Remaining work** | Implement and run F1–F3; tables; short interpretation |

---

## 11. Evidence files (reproducibility)

- Experiment E: `thesis_project/tables/expE_pruning_summary.csv`
- Alignment analysis: `thesis_project/tables/analysis_alignment/<lecture>/`
- Spurious export: `thesis_project/tables/spurious_far_review/<lecture>/`
- Manual tags (lecture 1, top 25): `thesis_project/tables/spurious_far_review/lecture1/spurious_far_top25_manual_tags.csv`

---

*Document version: advisor-ready link from Experiment E through error analysis to Experiment F.*
