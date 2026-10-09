# Running a study training session on Kaggle

All training happens on Kaggle, **accelerator = GPU T4 x2**. Do not use the P100 option: our pinned
PyTorch is a CUDA 13 build, and CUDA 13 dropped support for the P100's GPU generation.

Each executor uses **one account**. The run split is in [`../RUNS.md`](../RUNS.md).

## First session on an account

A machine that has not passed `smoke.py` does not produce reported numbers (INVARIANTS.md section 1.2).

```python
# 1. Pinned environment. Internet must be ON for this cell.
!pip install -q -r /kaggle/working/study/requirements.txt

# 2. Dataset, fingerprinted BEFORE Ultralytics converts it (conversion deletes the raw annotations).
!python /kaggle/working/study/scripts/fingerprint_visdrone.py

# 3. Proof the environment is sound: conversion counts, invariants, area-split metrics.
!python /kaggle/working/study/scripts/smoke.py

# 4. Batch sizes and speed for this machine class. Once per account is enough.
!python /kaggle/working/study/kaggle/probe_kaggle.py
```

Keep `verification/smoke_results.json` and `verification/kaggle_probe.json` from this session: they are the
account's record.

## Every training session

```python
!python /kaggle/working/study/scripts/studytrain.py \
    --model yolo11s.pt --imgsz 640 --seed 0 --device 0 --batch-per-gpu 16
```

- Use **Save Version -> Save & Run All (Commit)**, not an interactive session. A committed run keeps going
  without a browser and saves `/kaggle/working` as the version's output.
- A session is capped at about 12 hours. A run that needs longer is continued in the next session: attach the
  previous version's output as an input, copy the run directory into place, and add `--resume`.
- Two runs can share a session, one per GPU (`--device 0` and `--device 1` in two cells run as separate
  processes), or one run can use both GPUs (`--device 0,1`). The probe's `pairing` block says which is better
  here; whichever is used, `--batch-per-gpu` stays the same.

## Two rules that protect the results

1. **Mirror the checkpoints.** Copy `runs/baseline/<name>/weights/last.pt` and the run record off Kaggle after
   each session. No single account should hold the only copy of a run.
2. **Re-fingerprint after any re-download.** A different dataset copy is a different experiment. The SHA-256 of
   each zip is recorded in INVARIANTS.md section 2; a mismatch means stop.
