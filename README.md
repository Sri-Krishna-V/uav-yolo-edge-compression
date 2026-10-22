# UAV YOLO edge compression

A controlled study of what model compression costs **small objects** in UAV imagery, and what it buys on real edge
hardware. Two detectors (YOLO11s, YOLO26s), one dataset (VisDrone2019-DET), four platforms (Raspberry Pi 5, Jetson
Nano, a Galaxy M34 phone, laptop CPU), and every accuracy claim reported with COCO-style area splits
(AP-small / AP-medium / AP-large) rather than aggregate mAP alone.

**Status: training starts now.** The experimental invariants and the baseline training recipe are frozen; compression,
export and measurement decisions are still open and tracked as issues.

## The claim

Compression recipes are calibrated on COCO, where objects are large. VisDrone is the opposite: ~68% of instances are
under 32x32 px and images hold ~70 objects. The hypothesis is that quantization and pruning cost **AP-small**
disproportionately, and that the damage can be mapped per pyramid level and repaired.

The weaker claim is the floor: even if the effect is small, the benchmark and the deployment guidelines stand.

Prior art is cited up front, not discovered by a reviewer: [I-YOLOv11n](papers/) states the
pruning-harms-small-objects hypothesis on UAV imagery. What is unoccupied is the **measurement** — a controlled
per-level sensitivity map at matched parameters *and matched measured device latency*.

## Three rules this repo enforces

1. **Latency, never FLOPs.** TensorRT issue #3884 records 20% structured pruning of YOLOv8s buying *zero* FPS on a
   Jetson Nano. Every compression claim is reported in measured device latency; FLOPs and parameters are descriptive
   columns only.
2. **One evaluator.** Every reported number comes from `study/scripts/studyeval.py`, never `model.val()`, which
   silently drops the area splits, mixes two PR integrators, and scores at 100 detections per image on a dataset whose
   images hold up to 461. See [`study/INVARIANTS.md`](study/INVARIANTS.md) section 4.
3. **One recipe.** Every training run inherits [`study/cfg/train_baseline.yaml`](study/cfg/train_baseline.yaml).
   Notably `optimizer` is pinned: Ultralytics' `auto` picks a *different optimizer* above and below 10,000 iterations,
   which would have given baselines and pruning-recovery runs different optimizers with nothing in the logs.

## Layout

| Path | What it holds |
|---|---|
| [`study/INVARIANTS.md`](study/INVARIANTS.md) | Every frozen setting with its justification. Read this first |
| [`study/cfg/`](study/cfg) | `invariants.yaml` (shared/train/eval) and `train_baseline.yaml` (the recipe) |
| [`study/scripts/`](study/scripts) | `studyenv` (environment guard), `studyeval` (the only sanctioned evaluator), `studytrain` (launcher), `fingerprint_visdrone`, `smoke` |
| [`study/kaggle/`](study/kaggle) | The team guide to training on Kaggle, its wizard, the session runner, the run table and the machine probe |
| [`study/RUNS.md`](study/RUNS.md) | The baseline run plan: 10 runs, owners, order, cut order |
| [`study/verification/`](study/verification) | Per-machine smoke records, dataset fingerprint, per-run result records |
| [`GLOSSARY.md`](GLOSSARY.md) | Glossary. *Cell*, *configuration*, *platform* vs *runtime*, *matrix resolution* |
| [`.scratch/uav-yolo-edge/`](.scratch/uav-yolo-edge) | The decision map and its research findings, with sources |

## Reproducing a run

Training runs on Kaggle (accelerator **T4 x2**; the P100 option cannot run a CUDA 13 build of PyTorch).
**Start with the team guide, [`study/kaggle/README.md`](study/kaggle/README.md)**, and its wizard:

```bash
bash study/kaggle/wizard.sh        # once per person: Kaggle account -> smoke test -> first training session
bash study/kaggle/wizard.sh next   # after every Kaggle session
```

What one Kaggle session runs, step by step (any machine with the pinned environment can run the same):

```bash
python study/scripts/fingerprint_visdrone.py --download --convert   # fetch, SHA-256-check and convert VisDrone
python study/scripts/smoke.py --tag kaggle-<name>                   # this machine may now produce reported numbers
python study/scripts/studytrain.py --model yolo11s.pt --imgsz 640 --seed 0 --device 0 --batch-per-gpu <runs.yaml>
```

A machine without a passing `smoke.py` record does not produce numbers that go in the paper.

## How the work is tracked

Decisions are issues. The **map** issue carries the destination, the frozen context and every decision made so far;
each decision is a child issue, closed with its answer. Open issues are the questions still unanswered.

## Licence

Code is AGPL-3.0, following [Ultralytics](https://github.com/ultralytics/ultralytics), on which it depends.
The PDFs in `papers/` are CC BY 4.0 and belong to their authors.
