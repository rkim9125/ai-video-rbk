## Overview
This project aims to automatically segment engineering lecture videos into meaningful topic-based clips.

We explore a baseline approach using silence detection and analyze its limitations, followed by a planned semantic-based segmentation approach.

## Dataset

Lecture recordings are not included in this repository due to size and data usage restrictions.

The dataset consists of multiple engineering lecture videos used for experimentation.

To reproduce the experiments, users should provide their own lecture video files in the following structure:

silence_baseline/data/
  lecture1.mp4
  lecture2.mp4
  ...

## Pipeline

1. Extract audio from lecture videos
2. Detect silence regions
3. Predict segmentation boundaries
4. Compare with ground truth annotations

## Results

Silence-based segmentation performed poorly on continuous lecture data, as lectures contain minimal pauses between topics.

This highlights the need for semantic-based segmentation approaches.

## Future Work

We plan to implement a semantic segmentation approach using transcript embeddings to detect topic shifts.