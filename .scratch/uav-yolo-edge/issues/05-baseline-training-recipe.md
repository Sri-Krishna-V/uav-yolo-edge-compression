# Baseline training recipe

Type: grilling
Status: resolved
Blocked by: 15
Assignee: srikr

## Question

Lock the exact baseline training configuration for YOLO11s and YOLO26s on VisDrone, decide whether **resolution is a study variable or a fixed constant**, and allocate runs to machines.

The VisDrone research resolved most of the sub-questions and turned the first one into a scope decision.

### The live decision: is imgsz a constant or an axis?

Resolution is the single biggest lever on VisDrone - **bigger than model scale**. Measured, identical recipe, COCO-pretrained:

| Model | imgsz | mAP50 | mAP50-95 |
|---|---|---|---|
| yolov8s | 640 | 0.404 | 0.234 (approx) |
| **yolov8s** | **960** | **0.496** | **0.308** |
| yolov8l | 640 | - | 0.278 |

`yolov8s@960` **beats** `yolov8l@640` at about a third of the parameters. 640->960 buys ~+9.2 mAP50; 640->1024/1280 buys +8 to +12 across eight models.

**And it is unpublished ground:** no paper ablates 640 / 960 / 1280 for one s-scale model under a fixed recipe. That table is cheap and would be a real contribution.

So decide:

1. **Fix imgsz at one value** (simplest, preserves GPU-hours for the pruning sweep), **or promote it to a study axis** (adds a genuine second contribution, costs GPU-hours that *Scale-aware pruning recipe* also wants). These compete for the same budget - decide deliberately, not by drift.
2. If fixed: the recommendation is **960 as the headline config** (best mAP-per-GPU-hour on 6 GB) with **640 as the edge-realistic config**, since 1280 on a 6 GB 4050 at batch 2-4 is brutally slow and changes gradient-noise characteristics versus the Kaggle and Colab runs.
3. Remember imgsz propagates to **every deployment benchmark**: a 960-input model on a Pi 5 CPU is far slower than a 640 one. Whether any Pi configuration is real-time at all partly rides on this.

### Settled by research - carry through, do not re-litigate

- **Budget GPU-hours to resolution, not epochs.** 100->600 epochs buys ~+4 mAP50 at 6x cost; 640->960 buys ~+9.
- **Class handling**: the Ultralytics converter is safe. `score==0` is exactly `category in {0, 11}`, so ignored regions and `others` are dropped correctly.
- **Splits, `max_det`, version pins**: frozen in *Freeze the experimental invariants*.
- **Expected baseline**: test-dev 31-35 mAP50 / 18-21 mAP50-95 at 640. Anything near that is a SUCCESS, not a failure.

### Still to decide

4. **Batch size** under 6 GB VRAM at the chosen imgsz, and whether AMP or gradient accumulation is needed.
5. **Epoch count** and early-stopping policy, given the resolution-over-epochs finding.
6. **Augmentation** - mosaic, close_mosaic, mixup, perspective. Note `mixup=0.4` measurably helped in the published sweep.
7. **Machine allocation** - which of laptop / Kaggle (30 h/wk) / Colab runs which job, with a checkpoint and resume strategy, because a 5-hour run on a disconnecting free tier without resume is a wasted day.

The output is a config file, not prose: an Ultralytics-compatible training config that every later run inherits from.

## Answer

Resolved 2026-10-09. Deliverables: `study/cfg/train_baseline.yaml` (the recipe), `study/scripts/studytrain.py` (launcher, which also runs `studyeval` on val and test-dev and writes `verification/runs/<name>.json`), `study/RUNS.md` (the 10-run plan, owners, order, cut order), `study/kaggle/` (session guide and the machine probe). `INVARIANTS.md` §7 records the parts later tickets must not drift from.

**1. imgsz is an axis for baselines only; the matrix resolution is 640.** Carrying two resolutions through the whole matrix would have doubled every compressed cell, the pruning sweep and every deployment benchmark. Instead: baselines at 640/960/1280 give the accuracy-vs-latency table nobody has published for an s-scale model under a fixed recipe, and **every compressed configuration runs at 640** — the edge-realistic size, directly comparable to published rows, and ~2.25× cheaper per pruning cell. Ultralytics' own Pi 5 table gives YOLO26n/NCNN **67 ms @640**; scaled by FLOPs, YOLO26s is roughly **200–250 ms @640 (~4–5 FPS)** and ~2 FPS at 960. New terms in `CONTEXT.md`: *input resolution*, *matrix resolution*, *resolution study*, *reference baseline*.

**2. Three traps found by reading the 8.4.149 source, all silent, all now closed in the config.**

- **`optimizer=auto` is not one optimizer.** It picks MuSGD above 10,000 optimizer iterations and AdamW below (`trainer.py:1145`), which on VisDrone is the boundary at ~99 epochs. A 100-epoch baseline would have trained with MuSGD and its 30–50 epoch pruning-recovery runs with AdamW at lr 7.1e-4, with nothing in the logs flagging it. **SGD is pinned** (lr0 0.01, momentum 0.937, linear schedule), and every later run inherits it.
- **Gradient accumulation is already on.** `nbs=64` accumulates to ~64 images per update whatever `batch` is (`trainer.py:310`), and weight decay is rescaled with it. So `batch` only moves BatchNorm statistics and speed. Combined with no SyncBatchNorm and `batch` being split across GPUs (`trainer.py:293`), the quantity to hold constant across machines is the **per-GPU batch**, not the total. Autobatch (`batch=-1`) is banned: it would pick a different value per machine.
- **`cache='ram'` is non-deterministic** (`data/base.py:156`). Left at `false`; `'disk'` is the deterministic option if dataloading proves to be the bottleneck.

**3. Schedule and augmentation.** 100 epochs, early stopping disabled (`patience=10000`), because 100 matches the closest prior work (arXiv 2605.24831 Table 5 → YOLO26s 32.5/18.2 test-dev) and early stopping truncates the LR schedule and makes compute per run unequal. Stock Ultralytics augmentation, written out explicitly so an upstream default change cannot move our numbers. No mixup: the only VisDrone evidence is one run on a different model (+1.4 test mAP50, inside noise), and `copy_paste` is segment/OBB-only in 8.4.149, so the research document's copy_paste debate does not apply to boxes. `workers` is **pinned at 4 and recorded**, because worker count changes the augmentation RNG stream.

**4. Machine allocation: Kaggle only, T4 ×2, two executors, one account each.** The laptop is out — the 4050's 6 GB plus a Windows paging-file failure killed every probe above 640/batch 8 (YOLO11s @640 batch 8 fit at 2.8 GB, ~44 img/s). Kaggle's **P100 cannot run the pinned stack**: torch 2.14 is a CUDA 13 build and CUDA 13 dropped Pascal. Sessions are ~12 h and `last.pt` is written every epoch, so a killed session costs one epoch and `--resume` continues it. Runs are split by model family — srikr takes the YOLO11s line, the teammate the YOLO26s line — which keeps each model's seed-noise comparison on one account and one machine class. **One account per person**, because Kaggle ties GPU time to phone verification specifically to stop quota farming, and a blocked account would take its checkpoints with it; checkpoints are mirrored off Kaggle after every session.

**5. Noise floor: 3 seeds per study model at 640, plus bootstrap resampling on every reported evaluation.** Distillation's expected +0.6 mAP, the +1.0 mAP50 YOLO26-vs-YOLOv8 gap, and every ΔAP-small verdict are all unjudgeable without it, and these are the cheapest runs in the study. Seeds 1 and 2 are a **labelled exception** to the `seed: 0` invariant. Bootstrap is free: it re-scores the saved predictions over resampled test images, but it measures only test-set noise, so it does not replace the seed runs.

**Open, deliberately not decided here:** the exact per-GPU batch per resolution, and whether two GPUs are better used as one DDP run or two concurrent runs. Both are measurements, not decisions — `study/kaggle/probe_kaggle.py` reports them on the first Kaggle session, and neither changes any result, only wall-clock. The default until it reports is two concurrent runs, one per GPU.
