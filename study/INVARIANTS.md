# Experimental invariants

Frozen 2026-09-12 by the map ticket *Freeze the experimental invariants*. Every training run and every accuracy number in the study inherits these. **Changing any of them after the first baseline run invalidates every cell measured before the change**, so re-run those cells or leave the value alone.

Machine-readable source: [`cfg/invariants.yaml`](cfg/invariants.yaml). Enforcement lives in [`scripts/studyenv.py`](scripts/studyenv.py) (environment, settings, version floor) and [`scripts/studyeval.py`](scripts/studyeval.py) (the only sanctioned evaluation).

```python
import studyenv   # always first: must precede any ultralytics import
import studyeval

model.train(data=studyenv.DATA, ..., **studyenv.TRAIN_ARGS)
result = studyeval.evaluate("runs/.../best.pt", split="test", imgsz=IMGSZ)   # split="test" is test-dev
```

---

## 1. Environment

| Package | Pin | Why |
|---|---|---|
| `ultralytics` | **==8.4.149** (hard floor 8.4.133) | AP-small/medium/large for custom datasets arrived in 8.4.133 (PR #25981). Below that the metrics simply do not exist, and `studyenv.py` exits. Pinned exactly, not as a floor, because `studyeval.py` patches `detect/val.py::coco_evaluate`, and that patch was verified against this exact source. |
| `faster-coco-eval` | **==1.8.0** | Computes the area splits. Ultralytics lists it only as an *extra* and would `pip install` it mid-run. `YOLO_AUTOINSTALL=false` makes a missing copy a hard failure instead of a silent change to the environment. |
| `torch` / `torchvision` | **2.14.0 / 0.29.0** (laptop: `+cu130`) | Same version on every machine. The CUDA build may differ per machine and is recorded per machine (see §1.2). |
| Python | 3.12.6 (laptop) | Recorded, not enforced; Kaggle/Colab images vary. |

Full resolved laptop set: [`requirements.lock.txt`](requirements.lock.txt). Portable pins: [`requirements.txt`](requirements.txt).

### 1.1 Settings isolation

On this laptop, the machine-global Ultralytics `settings.json` points `datasets_dir`, `runs_dir` and `weights_dir` at a different project (`Self/Projects/Courtroom-Video-Analyzer-Agent`). If nothing overrides it, study datasets and runs land there. `studyenv.py` sets `YOLO_CONFIG_DIR=study/.ultralytics` and writes study-local directories (plus `sync=False`) **before** Ultralytics is imported, because Ultralytics reads those paths once at import. Always `import studyenv` first.

### 1.2 Per-machine record

Run `scripts/smoke.py` once on every machine that trains or evaluates (laptop, Kaggle, Colab). The `environment` block of `verification/smoke_results.json` is that machine's record: torch CUDA build, cuDNN, GPU, driver. A machine without a passing smoke run does not produce reported numbers.

### 1.3 Export toolchains: **not frozen here**

NCNN, LiteRT/TFLite, TensorRT/JetPack and ONNX Runtime versions depend on *Precision and export matrix*, which is still open. The rule is frozen now: **pin each platform's toolchain at its first export, and add a row here before any benchmark number from that platform is recorded.**

| Platform | Runtime | Toolchain versions | Pinned on |
|---|---|---|---|
| *(first export on each platform)* | | | |

---

## 2. Dataset

VisDrone2019-DET, from the Ultralytics mirror `https://github.com/ultralytics/assets/releases/download/v0.0.0/`, **downloaded 2026-09-12 (14:49 UTC)**. Fingerprinted by [`scripts/fingerprint_visdrone.py`](scripts/fingerprint_visdrone.py) → [`verification/visdrone_fingerprint.json`](verification/visdrone_fingerprint.json).

| Split | Zip SHA-256 | Images | Raw rows | Kept by converter | Ignored (cat 0) | Others (cat 11) | Objects/img mean / max |
|---|---|---|---|---|---|---|---|
| train | `86a77eba93137bfc16e4993860de9245b0675c0dba0d3ab98fb458699e256f84` | 6,471 | 353,550 | 343,205 | 8,813 | 1,532 | 53.0 / 902 |
| val | `abeea063037e5d20398837deb11084e652402a34ddf4f207bdf541a6f2a35ef9` | 548 | 40,169 | 38,759 | 1,378 | 32 | 70.7 / 317 |
| test-dev | `b28a94b06dfd9e36ce77ff8155fb82b9d2c030f198a76105933bd56c6ea6a68d` | 1,610 | 77,547 | 75,102 | 2,180 | 265 | 46.6 / 461 |

- **Converter verified on our own download.** `score == 0` is exactly `category ∈ {0, 11}` on every row of every split, with zero exceptions. Image counts, ignored/others counts and test-dev per-class counts match an independent download made during the VisDrone research. The smoke run also checks the converted YOLO labels row for row against the raw zips.
- **Any re-download** (a new machine, Kaggle): run `fingerprint_visdrone.py` *before* the first Ultralytics use. The conversion deletes the raw annotations. A SHA-256 mismatch means different data, so stop.
- The test-challenge split is not downloaded and never used.

---

## 3. Split discipline

| Split | Use |
|---|---|
| `train` (6,471) | training only |
| `val` (548) | model selection, tuning, ablations, early stopping (Ultralytics picks `best.pt` on it) |
| `test-dev` (1,610) | **headline numbers**. In Ultralytics it is `split="test"`. |

- **Report both columns**, val and test-dev, labelled. The gap is a consistent ~6 mAP50, so quoting val against someone else's test-dev flatters us by ~6 points.
- **Never tune on test-dev.** No hyperparameter, threshold, epoch count or checkpoint choice may look at it.
- **Never report test-challenge.** Its labels are withheld and the server has been dormant since 2024.

---

## 4. Evaluation

| Setting | Value | Why |
|---|---|---|
| `max_det` | **500** | The official VisDrone protocol uses 500. The default is worse than a low cap: left at 300, `DetectionValidator._check_max_det` **silently raises it to the observed maximum of whatever split is loaded**, so training validation would run at 902 (train), val at 317 and test-dev at 461. Three protocols in one paper. Pinned to 500 it only warns. It never binds on GT count at eval time (val max 317, test-dev max 461). |
| `iou` | 0.7 | NMS IoU. Measurably moves VisDrone mAP (0.6 vs 0.7 was worth +0.9 val mAP50 in a published sweep), so it is held constant and NMS tuning is never confounded with compression. |
| `conf` | 0.001 | Ultralytics' val default, made explicit. |
| `nms` | `null` | See §5. |
| `save_json` | `true` | Without it the area splits are not computed. |
| `rect` | `true` | What `model.val()` uses. |
| `batch` | 16 | Rectangular batching pads each batch to its largest shape, so batch size moves mAP slightly. Constant across cells. If a config does not fit, evaluate on a larger GPU; do not lower the batch. |
| COCO `maxDets` | **[1, 10, 100, 500]** | See trap 3 below. |
| precision | FP32 (`quantize` unset) | Accuracy of `.pt` checkpoints. Accuracy of quantized artifacts belongs to *Precision and export matrix*. |

### 4.1 Three traps in the stock Ultralytics path, all silent

Verified against the ultralytics 8.4.149 and faster-coco-eval 1.8.0 source. `studyeval.evaluate()` fixes all three. **Never report a number from `model.val()`.**

1. **`model.val()` drops the area splits.** It returns `validator.metrics`, whose `results_dict` holds only the native keys. `metrics/mAP_small|medium|large(B)` exist only in the stats dict the validator returns, which `model.val()` discards.
2. **The headline and the area splits come from different integrators.** For a custom dataset, faster-coco-eval's overall AP replaces `metrics/mAP50-95(B)` only for COCO/LVIS (`detect/val.py` L609). On VisDrone the headline stays on Ultralytics' trapezoidal integrator, which reads up to 3.4% high versus pycocotools, while the area splits use COCO's 101-point interpolation.
3. **The area splits are computed at 100 detections per image.** faster-coco-eval defaults to `maxDets=[1, 10, 100]` and computes *every* AP at `maxDets[-1]` (`core/cocoeval.py` L684, L504-510), whatever `max_det` is. **19.5% of val images and 7.9% of test-dev images hold more than 100 objects.** `studyeval` sets `[1, 10, 100, 500]`, matching the official `AP@[.50:.95 | maxDets=500]` and `AR{1,10,100,500}`.

Also: **per-epoch training validation never computes area splits** (`engine/validator.py` returns from the training branch before `eval_json`). `best.pt` is selected on Ultralytics' native val fitness. That counts as model selection on val, which is allowed.

### 4.2 Reporting rule

**Every accuracy number in the paper is a `coco/*` stat from `studyeval.evaluate()`**: `coco/AP_all`, `coco/AP_50`, `coco/AP_75`, `coco/AP_small`, `coco/AP_medium`, `coco/AP_large`, `coco/AR_500`. All come from one integrator at maxDets=500. Never `metrics/mAP50-95(B)`, and never the two families in one table.

### 4.3 Protocol caveat: goes in the paper, verbatim or close

> Accuracy is computed with faster-coco-eval (COCO-style 101-point interpolation, maxDets = 500, COCO area ranges in native image pixels) over the 10 evaluated VisDrone classes. It is **not** the official VisDrone toolkit (`evalDET.m`) metric and should not be compared directly with numbers produced by that toolkit: the official protocol discards detections falling ≥50% inside ignored regions (ours counts them as false positives), weights class APs by image frequency (ours is a macro average), and integrates precision-recall with all-point VOC interpolation. All three differences bias our numbers low relative to the official toolkit.

Never put Ultralytics-pipeline numbers and official-toolkit numbers in one table without this caveat.

### 4.4 Exported artifacts (ONNX, TensorRT, NCNN, LiteRT)

`studyeval.evaluate()` accepts an exported artifact, but for any static (non-dynamic) export the validator **overrides two eval invariants**: it forces `rect=False` (square letterbox) and takes `batch` and `imgsz` from the export metadata, usually batch 1 (`engine/validator.py` L200-204, L222-223). The recorded `validator_args` show the override.

So **never compare a quantized artifact against the `.pt` checkpoint**, because that confounds letterboxing with quantization. The reference for any quantized artifact is an **FP32 (or FP16 where FP32 is unavailable) export of the same format, at the same `imgsz`, evaluated the same way**. Accuracy damage is `artifact − reference-export`, never `artifact − .pt`.

---

## 5. Which YOLO26 head is scored: `nms: null`

`nms: null` means external NMS on the **one-to-many** head, for both families. YOLO11 has only that head. For YOLO26, the NMS-free one-to-one head is scored only when `nms=False`.

- **Why:** the Ultralytics exporter disables the one-to-one head on every INT8 edge path in this matrix (see *YOLO26 reality check*). Accuracy is measured on the head that actually gets deployed, and YOLO11 and YOLO26 are compared under the same postprocessing.
- **Cheap to revisit:** `nms` does not affect training, which keeps both heads, only which head is scored. A later decision to headline the one-to-one head needs re-evaluation, not re-training.
- **One-to-one head numbers** may appear only as separately labelled rows (`studyeval.evaluate(..., nms=False)` records `end2end_head_scored: true`).

---

## 6. Reproducibility

- `seed: 0`, `deterministic: true`. These are Ultralytics' defaults, made explicit. Ultralytics calls `torch.use_deterministic_algorithms(True, warn_only=True)`, so runs are *not* guaranteed bit-exact across GPUs or machines. Run-to-run variance is the job of the statistical treatment still in the map's fog, not of these flags.
- Record for every run: the commit or copy of `cfg/invariants.yaml`, the machine's smoke record, and the `studyeval` output JSON (it carries `validator_args`, so any deviation is visible).

### 6.1 The one labelled exception to `seed: 0`

The **noise runs** — seeds 1 and 2 of each study model at the matrix resolution — deliberately break `seed: 0`. They exist to measure run-to-run training noise, which three decisions depend on (distillation's expected +0.6 mAP, the +1.0 mAP50 YOLO26-vs-YOLOv8 gap, and every ΔAP-small verdict). They are never reported as study cells, and `seed: 0` remains the invariant for every cell. Noise runs for one model stay on **one machine class and one account**, or they measure hardware differences instead of seed noise.

Bootstrap resampling of test-dev images (over the saved `predictions.json`) is reported alongside, and costs no GPU time. It captures only test-set sampling noise, so it does not replace the seed runs.

---

## 7. Training recipe

Frozen 2026-10-09 by the map ticket *Baseline training recipe*. Machine-readable source: [`cfg/train_baseline.yaml`](cfg/train_baseline.yaml), applied by [`scripts/studytrain.py`](scripts/studytrain.py). The run plan is [`RUNS.md`](RUNS.md). **Every training run in the study inherits this file** — baselines, pruning-recovery runs, distillation runs, and teachers if they happen.

| Setting | Value | Why |
|---|---|---|
| `optimizer` | **SGD** (lr0 0.01, momentum 0.937, linear schedule) | **Never `auto`.** `auto` selects MuSGD above 10,000 optimizer iterations and AdamW below (`trainer.py:1145`); on VisDrone that boundary is ~99 epochs, so a 100-epoch baseline would get MuSGD and a 30–50 epoch recovery run AdamW at lr 7.1e-4, silently. |
| `epochs` / `patience` | 100 / 10000 (early stopping off) | 100 matches the closest prior work, so our baselines line up with a published row. Early stopping truncates the LR schedule and makes compute per run unequal across cells meant to be matched. |
| augmentation | Ultralytics defaults, written out | No mixup (one run on a different model, inside noise); `copy_paste` is segment/OBB-only in 8.4.149 and does nothing for boxes. Written out so an upstream default change cannot move our numbers. |
| `workers` | **4, pinned and recorded** | Worker count changes the augmentation RNG stream, so runs with different worker counts are not strictly comparable. |
| `cache` | `false` | `cache='ram'` is non-deterministic (`data/base.py:156`); `'disk'` is the deterministic alternative if dataloading is the bottleneck. |
| batch | **per-GPU batch is the constant**, never `batch=-1` | `nbs=64` accumulates to ~64 images per update whatever `batch` is (`trainer.py:310`), so `batch` only moves BatchNorm statistics and speed. Ultralytics splits `batch` across GPUs (`trainer.py:293`) and never syncs BatchNorm, so one-GPU and two-GPU runs match only at equal **per-GPU** batch. Autobatch would pick a different value per machine. |
| input resolution | **matrix resolution 640** for every compressed configuration | Baselines also run at 960 and 1280, as the resolution study only. |

### 7.1 Training machines

**Kaggle only, accelerator T4 ×2.** The P100 option cannot run this environment: torch 2.14 is a CUDA 13 build and CUDA 13 dropped Pascal support. The RTX 4050 laptop does not train. Each executor uses one account; `scripts/smoke.py` and `kaggle/probe_kaggle.py` run once per account before it produces numbers, and both records are kept. Checkpoints are mirrored off Kaggle after every session.

---

## Verification

`scripts/smoke.py` passed on 2026-09-12, laptop (`Mr-W-R-A-T-H`: Windows 11, Python 3.12.6, torch 2.14.0+cu130, cuDNN 92400, RTX 4050 Laptop 6 GB, driver 616.64), **0 failures**. Full record: [`verification/smoke_results.json`](verification/smoke_results.json).

| Check | Result |
|---|---|
| Raw-zip fingerprint vs independent research download | images, ignored/others rows, test-dev per-class counts: **exact match** |
| Converted YOLO labels vs raw zips, per class, per split | **exact match** (train 343,205 / val 38,759 / test-dev 75,102 rows; class ids ⊆ 0-9) |
| Shared + train invariants reached the trainer | yes, YOLO11s and YOLO26s |
| `studyeval` on val and test-dev: `metrics/mAP_small\|medium\|large(B)` + 15 `coco/*` stats present and finite | yes, all 4 evaluations |
| faster-coco-eval ran at `maxDets=[1, 10, 100, 500]` | yes, all 4 |
| Every eval invariant reached the validator (`max_det`, `iou`, `conf`, `nms`, `save_json`, `rect`, `batch`) | yes, all 4 |
| Headline evaluation scored the one-to-many head | yes (`end2end_head_scored: false`) for both families |
| `nms=False` probe selects YOLO26's one-to-one head | yes (`end2end_head_scored: true`) |

Values from the smoke models (1 epoch on 10% of train) are meaningless as accuracy, but they demonstrate two of the traps on real output:

- **Integrators differ.** YOLO26s val: `metrics/mAP50-95(B)` = 0.0506 vs `coco/AP_all` = 0.0521. Same predictions, two numbers.
- **The 100-detection cap binds even on a weak model.** YOLO26s val: `coco/AR_100` = 0.1821 vs `coco/AR_500` = 0.1852. A trained model emits more confident detections per dense image, so the gap should grow.

Smoke artifacts: `runs/smoke/`. The raw zips (~1.9 GB) are kept in `datasets/VisDrone/`, so the fingerprint can be re-checked.
