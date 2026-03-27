# Experiment Notes

## Experiment A

### Purpose
Compare sentence-based and time-based transcript windows for semantic boundary detection.

### Controlled Variables
- Threshold: `0.55`
- Local minima: `on`
- Minimum distance between boundaries: `20s`
- Evaluation tolerance: `+-30s`
- Lecture set: same lectures across both methods

### Independent Variable
Window representation.

### Expected Outcome
Sentence-based windows may preserve semantic coherence better, while time-based windows may be more robust to STT segmentation noise and uneven sentence lengths.

### Run Log Template
- Date:
- Setting (`sentence_w5_t055_d20` or `time_10s_t055_d20`):
- Lectures:
- Notes:

## Experiment B - Window Size
Placeholder for window-size sensitivity runs.

## Experiment C - Boundary Rule
Placeholder for threshold/local-minimum/min-distance ablation runs.

## Experiment D - Structural
Placeholder for multimodal or structural cue integration.
