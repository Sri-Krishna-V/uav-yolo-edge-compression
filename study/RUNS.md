# Baseline run plan

Resolved 2026-10-09 by the map ticket *Baseline training recipe*. Recipe: [`cfg/train_baseline.yaml`](cfg/train_baseline.yaml).
Launcher: [`scripts/studytrain.py`](scripts/studytrain.py). Every run's record lands in `verification/runs/<name>.json`.

**Two executors, both on Kaggle, one account each (~30 GPU-h per week each).** The split is by model family, so
each person's runs stay on one account and one machine class — which is what makes the seed-noise numbers valid.

## The runs

`A` = matrix resolution (640), `B` = resolution study. Seed 0 is the **reference baseline**; seeds 1 and 2 are the
labelled noise runs. Order within each owner is top to bottom.

| # | Owner | Model | imgsz | Seed | Run name | Why it exists |
|---|---|---|---|---|---|---|
| A1 | srikr | YOLO11s | 640 | 0 | `yolo11s-640-s0` | **Reference baseline. The critical path**: *Confirm the small-object effect* needs only this |
| A2 | teammate | YOLO26s | 640 | 0 | `yolo26s-640-s0` | Reference baseline, second family |
| A3 | srikr | YOLO11s | 640 | 1 | `yolo11s-640-s1` | Noise run |
| A4 | srikr | YOLO11s | 640 | 2 | `yolo11s-640-s2` | Noise run |
| A5 | teammate | YOLO26s | 640 | 1 | `yolo26s-640-s1` | Noise run |
| A6 | teammate | YOLO26s | 640 | 2 | `yolo26s-640-s2` | Noise run |
| B1 | srikr | YOLO11s | 960 | 0 | `yolo11s-960-s0` | Resolution study |
| B2 | teammate | YOLO26s | 960 | 0 | `yolo26s-960-s0` | Resolution study |
| B3 | srikr | YOLO11s | 1280 | 0 | `yolo11s-1280-s0` | Resolution study. **Cut first if the schedule slips** |
| B4 | teammate | YOLO26s | 1280 | 0 | `yolo26s-1280-s0` | Resolution study. **Cut first if the schedule slips** |

Teacher runs (YOLO11m, YOLO26m) inherit this same recipe but are not scheduled: *Distillation method* decides
whether they happen at all.

## Order of work, and why

1. **A1 and A2 first.** Everything downstream waits on a trained baseline, and A1 alone unblocks the ticket that
   decides whether the paper's central claim survives.
2. **A3-A6 next.** They are the cheapest runs in the study and they produce the noise floor that three other
   tickets need before their results can be called real.
3. **B1, B2** for the resolution table.
4. **B3, B4 last**, because 1280 costs roughly 4x a 640 run for one table row.

## Cost, and the cut order

Rough estimates until `kaggle/probe_kaggle.py` reports (a 640 run is ~5 h on one T4, 960 ~2.25x, 1280 ~4x):

| Group | Runs | Rough GPU-hours |
|---|---|---|
| A1-A2 | 2 | ~10 |
| A3-A6 | 4 | ~20 |
| B1-B2 | 2 | ~24 |
| B3-B4 | 2 | ~42 |

Two accounts supply ~60 GPU-h per week, so the whole baseline set is roughly one to one-and-a-half weeks of
quota — before any compression run. **Cut order if the schedule slips: B3/B4 first, then B1/B2, then A5/A6.**
Never cut A1-A4: they carry the critical path and the noise floor.

## Running one

```bash
python scripts/studytrain.py --model yolo11s.pt --imgsz 640 --seed 0 --device 0 --batch-per-gpu <from probe>
python scripts/studytrain.py --model yolo11s.pt --imgsz 640 --seed 0 --device 0 --batch-per-gpu <same> --resume
```

- `batch-per-gpu` is held constant for a given imgsz across every machine and both executors. Ultralytics splits
  `batch` across GPUs and does not sync BatchNorm, so the per-GPU batch is the comparable quantity.
- A run that will not fit in one Kaggle session is continued with `--resume` in the next session. `last.pt` is
  written every epoch, so a killed session costs at most one epoch.
- `studytrain.py` evaluates on `val` and `test-dev` through `studyeval` when training ends, and writes the record.

## Provisional schedule (one month from 2026-10-09)

*Final matrix and schedule* owns the real schedule; this is the training half of it, so the GPU never idles.

| Week | Training (Kaggle, both executors) | In parallel, no GPU needed |
|---|---|---|
| 1 (Oct 9-15) | A1, A2, then A3-A6 | *Precision and export matrix*, *Measurement protocol*, *YOLO26 pruning feasibility*, Jetson access (teammate), power meter |
| 2 (Oct 16-22) | B1, B2; first compression runs once *Confirm the small-object effect* reports | *Scale-aware pruning recipe*, Pi 5 and phone harness |
| 3 (Oct 23-29) | Compression and recovery runs | Device benchmarks, including Jetson |
| 4 (Oct 30-Nov 5) | Overflow and repeats only | Writing |
| Buffer (Nov 6-9) | No new runs | Writing, deck revision |
