# Corrected Ground Truth (v1)

These corrected annotation files keep the original topic labels and only fix
objective timestamp alignment issues found during VTT cross-checking.

## What was corrected

- `lecture1_boundaries.txt`: first boundary `00:00:12` -> `00:00:13`
- `lecture3_boundaries.txt`: first boundary `00:00:12` -> `00:00:13`
- `lecture4_boundaries.txt`: first boundary `00:00:12` -> `00:00:13`

Reason: the first GT boundary was slightly earlier than the first subtitle cue
start in the corresponding `.vtt` file (about 0.5-0.6s).

## What was not changed

- Topic titles
- All other timestamps
- `lecture2_boundaries.txt` (already within VTT time range)
