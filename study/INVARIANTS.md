# Experimental invariants

Frozen 2026-09-12 by the map ticket *Freeze the experimental invariants*. Every training run and every accuracy number in the study inherits these. **Changing any of them after the first baseline run invalidates every cell measured before the change**, so re-run those cells or leave the value alone.

Machine-readable source: [`cfg/invariants.yaml`](cfg/invariants.yaml). Enforcement lives in [`scripts/studyenv.py`](scripts/studyenv.py) (environment, settings, version floor) and [`scripts/studyeval.py`](scripts/studyeval.py) (the only sanctioned evaluation); the laptop's memory cap in [`scripts/studycap.py`](scripts/studycap.py) (§1.4).

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

Run `scripts/smoke.py` once on every machine that trains or evaluates (laptop, Kaggle, Colab). The `environment` block of its record is that machine's record: torch CUDA build, cuDNN, GPU, driver. The laptop's is `verification/smoke_results.json`; every other machine passes `--tag` and gets its own file, e.g. `verification/smoke_results-kaggle-srikr.json`, so no record overwrites another. A machine without a passing smoke run does not produce reported numbers.

### 1.3 Export toolchains: **not frozen here**

NCNN, LiteRT/TFLite, TensorRT/JetPack and ONNX Runtime versions depend on *Precision and export matrix*, which is still open. The rule is frozen now: **pin each platform's toolchain at its first export, and add a row here before any benchmark number from that platform is recorded.**

| Platform | Runtime | Toolchain versions | Pinned on |
|---|---|---|---|
| Laptop CPU (accuracy only, *Confirm the small-object effect*) | ONNX Runtime static INT8; OpenVINO + NNCF INT8 | onnx 1.23.2, onnxruntime 1.31.0, onnxslim 0.1.98; openvino 2026.4.1, nncf 3.4.0 | 2026-10-15 |
| Jetson Orin Nano, JetPack 6 only (latency only; RAM TBD) | TensorRT FP16 and INT8 engines, built on the board by `studyexport.py` | JetPack TBD (L4T R36.TBD), TensorRT TBD (10.3.x expected), CUDA TBD, cuDNN TBD; torch 2.10.0 / torchvision 0.25.0 (Jetson wheels), numpy 1.26.4, ultralytics 8.4.149, onnx 1.23.2, onnxslim 0.1.98; power mode TBD (highest nvpmodel mode, `jetson_clocks` on) | TBD (first export) |
| *(first export on each platform)* | | | |

The Jetson's torch is not §1's 2.14.0: the board only exports and benchmarks, and NVIDIA's Jetson wheels exist only for older versions. Fill the TBDs from `scripts/jetson_probe.py` and the first engine's `manifest.json` (its `engine` block). The engines score the one-to-many head (`nms=None`) at both precisions; the one-to-one head this TensorRT would keep (§5) is the probe's "YOLO26 head" line.

### 1.4 The laptop rule: heavy steps run under a memory cap

Decided 22 October 2026 by the map ticket *Final matrix and schedule* (#10); built by #20. The laptop has 15 GB of RAM and often only 2–4 GB free. On 19 October an uncapped job grew to 25.8 GB and paged it for 3.5 h: on Windows a Python `MemoryError` arrives only at the system commit limit, after the page file has grown, so nothing inside a job fails fast.

**The rule: on the laptop, every heavy step runs in a child process under a hard memory cap and a timeout, and a step that goes over either is stopped, never waited out.** The heavy steps are scoring's ground-truth build, its COCO evaluation, and the calibration of every INT8 export.

**The code enforces it; nobody applies it by hand.** `scripts/studycap.py` starts the step suspended, puts it in a Windows Job Object whose memory limit covers it and every process it starts, then lets it run. The kernel refuses any commit past the cap, and the parent stops the whole job at once. The parent also stops it at the timeout. The cap is the only memory limit, so it protects the laptop only while it is below what is free: close other programs before a step that needs most of it. A stopped step exits 1 with a message that names the step and the limit it crossed. A step that stays inside its limits keeps its own output and exit code.

| Script | What runs capped | Unit |
|---|---|---|
| `studyscore.py` | each run's scoring: its ground-truth build (`studyeval.py ground-truth`) and its COCO evaluation | one run |
| `studyexport.py` | each INT8 variant's export, calibration included | one variant; a stopped variant is reported failed and the others go on |

- **Limits**: 4 GB and 60 minutes per step. Change them with `--mem-cap-gb` / `--timeout-min` on either script, or `STUDY_MEM_CAP_GB` / `STUDY_TIMEOUT_MIN`. INT8 export at 960 and 1280 needs 4–5 GB: raise the cap for those, with the laptop otherwise idle.
- **Off Windows the cap does nothing**: on the Pis, the Jetson and Kaggle the steps run as before, with no cap and no timeout. `scripts/tests/test_studycap.py` proves both sides: a step over its cap or its timeout is stopped at once, and the same step off Windows runs to the end with its exit code.
- **Checked on the laptop, 23 October 2026**: scoring a 50-image A1 ONNX Runtime FP32 run went through the cap, rebuilding the 640 test-dev ground truth byte-identical to the cached one and writing the same `score.json` as the uncapped path. At `--mem-cap-gb 0.3` and `--timeout-min 0.02` the scorer stopped with exit 1 and said why. A1's ONNX Runtime INT8 export at 640 went through the cap in 116 s, and its graph and weights are identical to the 21 October export behind #9's parity numbers (only the export date in the metadata differs). At `--mem-cap-gb 1` the export was stopped after 11 s and reported failed.

---

## 2. Dataset

VisDrone2019-DET, from the Ultralytics mirror `https://github.com/ultralytics/assets/releases/download/v0.0.0/`, **downloaded 2026-09-12 (14:49 UTC)**. Fingerprinted by [`scripts/fingerprint_visdrone.py`](scripts/fingerprint_visdrone.py) → [`verification/visdrone_fingerprint.json`](verification/visdrone_fingerprint.json).

| Split | Zip SHA-256 | Images | Raw rows | Kept by converter | Ignored (cat 0) | Others (cat 11) | Objects/img mean / max |
|---|---|---|---|---|---|---|---|
| train | `86a77eba93137bfc16e4993860de9245b0675c0dba0d3ab98fb458699e256f84` | 6,471 | 353,550 | 343,205 | 8,813 | 1,532 | 53.0 / 902 |
| val | `abeea063037e5d20398837deb11084e652402a34ddf4f207bdf541a6f2a35ef9` | 548 | 40,169 | 38,759 | 1,378 | 32 | 70.7 / 317 |
| test-dev | `b28a94b06dfd9e36ce77ff8155fb82b9d2c030f198a76105933bd56c6ea6a68d` | 1,610 | 77,547 | 75,102 | 2,180 | 265 | 46.6 / 461 |

- **Converter verified on our own download.** `score == 0` is exactly `category ∈ {0, 11}` on every row of every split, with zero exceptions. Image counts, ignored/others counts and test-dev per-class counts match an independent download made during the VisDrone research. The smoke run also checks the converted YOLO labels row for row against the raw zips.
- **Any re-download** (a new machine, Kaggle): run `fingerprint_visdrone.py` *before* the first Ultralytics use. The conversion deletes the raw annotations. A SHA-256 mismatch means different data, so stop: the script checks the three hashes above and fails on a mismatch. `--download --convert` fetches the zips, fingerprints them, converts once and checks every label row; Kaggle's `session.py setup` runs exactly that every session. The fingerprint record carries the machine's host name, and the conversion check refuses a record from another machine (the repo ships the laptop's).
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
| `conf` | 0.001 | Ultralytics' val default, made explicit: the **evaluation threshold**. Latency on a platform is timed at the deployment threshold, 0.25 (§4.5). |
| `nms` | `null` | See §5. |
| `save_json` | `true` | Without it the area splits are not computed. |
| `rect` | `true` | What `model.val()` uses. |
| `batch` | 16 | Rectangular batching pads each batch to its largest shape, so batch size moves mAP slightly. Constant across cells. If a config does not fit, evaluate on a larger GPU; do not lower the batch. |
| COCO `maxDets` | **[1, 10, 100, 500]** | See trap 3 below. |
| precision | FP32 (`quantize` unset) | Accuracy of `.pt` checkpoints. The precision of exported artifacts on a platform follows §8. |

### 4.1 Three traps in the stock Ultralytics path, all silent

Verified against the ultralytics 8.4.149 and faster-coco-eval 1.8.0 source. `studyeval.evaluate()` fixes all three. **Never report a number from `model.val()`.**

1. **`model.val()` drops the area splits.** It returns `validator.metrics`, whose `results_dict` holds only the native keys. `metrics/mAP_small|medium|large(B)` exist only in the stats dict the validator returns, which `model.val()` discards.
2. **The headline and the area splits come from different integrators.** For a custom dataset, faster-coco-eval's overall AP replaces `metrics/mAP50-95(B)` only for COCO/LVIS (`detect/val.py` L609). On VisDrone the headline stays on Ultralytics' trapezoidal integrator, which reads up to 3.4% high versus pycocotools, while the area splits use COCO's 101-point interpolation.
3. **The area splits are computed at 100 detections per image.** faster-coco-eval defaults to `maxDets=[1, 10, 100]` and computes *every* AP at `maxDets[-1]` (`core/cocoeval.py` L684, L504-510), whatever `max_det` is. The cap applies **per image per category** (`core/cocoeval.py` L255-269, as in pycocotools' `evaluateImg`), not per image. 19.5% of val images and 7.9% of test-dev images hold more than 100 objects in total, but only **1.1% of val and 2.6% of test-dev images hold more than 100 objects of one class**. That is where the cap is guaranteed to cost recall. Measured on our runs, the effect is 0.33–0.53 points of AR (`AR_500` − `AR_100`, test-dev). It is a correctness fix, not a headline effect. `studyeval` sets `[1, 10, 100, 500]`, matching the official `AP@[.50:.95 | maxDets=500]` and `AR{1,10,100,500}`.

Also: **per-epoch training validation never computes area splits** (`engine/validator.py` returns from the training branch before `eval_json`). `best.pt` is selected on Ultralytics' native val fitness. That counts as model selection on val, which is allowed.

### 4.2 Reporting rule

**Every accuracy number in the paper is a `coco/*` stat from `studyeval.evaluate()`**: `coco/AP_all`, `coco/AP_50`, `coco/AP_75`, `coco/AP_small`, `coco/AP_medium`, `coco/AP_large`, `coco/AR_500`. All come from one integrator at maxDets=500. Never `metrics/mAP50-95(B)`, and never the two families in one table.

### 4.3 Protocol caveat: goes in the paper, verbatim or close

> Accuracy is computed with faster-coco-eval (COCO-style 101-point interpolation, maxDets = 500, COCO area ranges in native image pixels) over the 10 evaluated VisDrone classes. It is **not** the official VisDrone toolkit (`evalDET.m`) metric and should not be compared directly with numbers produced by that toolkit: the official protocol discards detections falling ≥50% inside ignored regions (ours counts them as false positives), weights class APs by image frequency (ours is a macro average), and integrates precision-recall with all-point VOC interpolation. All three differences bias our numbers low relative to the official toolkit.

Never put Ultralytics-pipeline numbers and official-toolkit numbers in one table without this caveat.

### 4.4 Exported artifacts (ONNX, TensorRT, NCNN, LiteRT)

`studyeval.evaluate()` accepts an exported artifact, but for any static (non-dynamic) export the validator **overrides two eval invariants**: it forces `rect=False` (square letterbox) and takes `batch` and `imgsz` from the export metadata, usually batch 1 (`engine/validator.py` L200-204, L222-223). The recorded `validator_args` show the override.

So **never compare a quantized artifact against the `.pt` checkpoint**, because that confounds letterboxing with quantization. The reference for any quantized artifact is an **FP32 (or FP16 where FP32 is unavailable) export of the same format, at the same `imgsz`, evaluated the same way**. Accuracy damage is `artifact − reference-export`, never `artifact − .pt`.

### 4.5 Platform runs: two confidence thresholds, and the scorer

Decided 22 October 2026 by the map ticket *Measurement protocol* (#8); built by #22. The terms are defined in [`GLOSSARY.md`](../GLOSSARY.md).

| Threshold | Value | Used for |
|---|---|---|
| **evaluation threshold** | **0.001** (`shared.conf`) | accuracy: every detection a platform saves, and so every accuracy number |
| **deployment threshold** | **0.25** (`platform.deploy_conf`) | **latency**: what a deployed detector runs. At 0.001, NMS handles up to 30,000 candidates, so post-processing looks more costly than on a real drone |

- **Two post-processing passes per image.** `studybench.py` runs post-processing twice on the same model output and times each pass on its own: at 0.001 (`t_post_ms`, detections saved) and at 0.25 (`t_post_deploy_ms` and `n_dets_deploy`, timed only, not saved). The 0.25 pass goes first on odd image indices, so a warm cache favours neither. The NMS-free head gets the same two passes. The 0.25 pass gives exactly the 0.001 pass's detections that score above 0.25, and it never changes the saved ones (`scripts/tests/test_deploy_threshold.py`).
- **What the second pass costs** (laptop, 23 October 2026, median per image): on the one-to-many head (A1 at 640, full test-dev, 4 threads) the 0.25 pass takes 1.15 ms (ONNX Runtime FP32) and 1.16 ms (INT8), against 15.5 and 16.0 ms for the 0.001 pass, so it adds under 1% to a 170–260 ms image. On the NMS-free head (YOLO26s COCO weights, 50 images) it takes 0.11–0.16 ms. Adding it did not change the 0.001 pass's own time (100 images: 18.7 → 19.1 ms FP32, 19.5 → 19.4 ms INT8).
- **Latency means `e2e_deploy`**: pre + net + the 0.25 pass, its median over the run. Every compression budget and every equal-latency comparison uses it. `meta.json` holds it in `timing_ms` beside `post_deploy`, and records `deploy_conf`. `t_total_ms` (pre + net + post at 0.001) stays in the record and is never called latency. A post-processing time is always stated with its threshold. Result B shows both.
- **The scorer for platform runs is `scripts/studyscore.py`**: torch-free, with `studyeval.py`'s ground truth and COCO settings. It refuses a run whose `conf`, `iou`, `max_det` or `deploy_conf` disagrees with `cfg/invariants.yaml`. `score.json` reports latency at both thresholds and names `e2e_deploy` as the latency.
- **On one condition: the parity check.** Whenever `studyscore.py` or `studyeval.py` changes, A1 (`yolo11s-640-s0`) at 640 runs through `studybench.py` on the laptop with ONNX Runtime FP32 and INT8 over the full test-dev, and `studyscore.py` must give exactly #9's numbers: **18.76 / 9.04** (FP32) and **16.84 / 7.74** (INT8), AP_all / AP-small. Until it does, no platform run is scored with the changed code.
  - **Run the INT8 half at `--threads 8`.** #9 evaluated through Ultralytics, which leaves ONNX Runtime at one thread per physical core, 8 on the laptop, and ONNX Runtime's INT8 detections change with the thread count (FP32's did not move across 1–16 threads; `studybench.py`'s docstring). At the platform runs' 4 threads, the same INT8 artifact gives **16.86 / 7.65**: that is the laptop reference for a 4-thread INT8 cell, not a parity failure.
  - Last passed: 23 October 2026, with #22's two passes in the bench. FP32 at 4 threads gave 18.76 / 9.04 and INT8 at 8 threads 16.84 / 7.74. Over the first 100 test-dev images, the bench before and after #22 wrote byte-identical `predictions.json` for both.

### 4.6 Platform runs: energy, on the Pi 5 only

Decided 22 October 2026 by the map ticket *Measurement protocol* (#8), from *Power measurement capability* (#7); built by #26. The **rail sum** is defined in [`GLOSSARY.md`](../GLOSSARY.md).

- **Where:** the Pi 5 only, and only for the cells of the equal-latency pairs. The other platforms get no energy numbers. Joules per frame is compared within the Pi 5 only, never across platforms: each platform's power reading has a different boundary (#7).
- **What is read:** the PMIC rail sum, Σ V_i·I_i over the rails `vcgencmd pmic_read_adc` reports both for, every 0.5 s. It leaves out the USB ports, the fan and everything else fed straight from 5 V, and the PMIC's conversion losses. It is always labelled **"rail sum, not wall power"**. The only published comparison, one hobbyist's board, puts it 18–31 % below input power.
- **Never in a latency pass.** Energy runs are a session of their own: a plan with `"mode": "energy"`, run by `studysession.py`. Latency sessions never start the sampler; a Pi 5's setup checks read the PMIC once, before any run. An energy run saves no per-frame time and no detections, and `studyscore.py` refuses it.
- **One energy run:** after the start gate, a 1-minute idle reading, then 3 minutes of the cell with the sampler on. The idle window and the run window are recorded next to each other, and every reading is kept, with its window, in the run's `power.csv` (`idle`, `between`, `run` or `after`; `bench` for a run that crashed and has no window). Each frame is what a deployed detector does: pre + net + post at the deployment threshold, over the first 100 test-dev images. They are decoded before the run and cycled in order, so reading files is not in the energy. The model load and the 20 warm-up images happen between the two windows and are in neither. Inside the run window, `vcgencmd` runs at the start, every 50 frames and at the end, twice each time: the throttle flags, and the CPU clock (#28). Those calls are in the energy, the same in every run.
  - **J per frame** = the run's mean rail sum × its duration / its frames.
  - **Net of idle** = (run mean − idle mean) × duration / frames. Both are recorded.
- **The sampler's cost on FPS**, measured once per energy session. Right after the plan's first cell, that cell runs once more with the same rest minute and the same 3 minutes, but with the sampler off. `session.json` records the FPS with and without the sampler, the difference, and which of the two runs throttled. It is one pair of runs: #7 saw FPS move about 7 % between two identical laptop runs, so a difference of a few percent is within the noise. That bounds the sampler's cost; it does not measure it.
- The latency session's other rules apply unchanged: the start gate, the shuffled order, 4 threads, a re-run after a crash or a throttle (the sampler-off run's included). There is no drift check, since no latency comes from the session. A run whose windows hold fewer than 10 readings has no energy, and counts as a crash.

### 4.7 Platform runs: the Pi 5's cooling and clock lock

Decided by #29: the cooling on 25 October 2026 (recorded on #8), the clock lock by its pilot on 26 October 2026
([`verification/pi5_pilot_record.md`](verification/pi5_pilot_record.md)).

- **Cooling:** a fan with a 4-pin plug in the Pi 5's FAN socket, on the firmware's default fan curve. Each session records the fan's rpm (#28).
- **The clock lock: `arm_freq=2000` in `/boot/firmware/config.txt`, for every Pi 5 run.** At the stock 2400 MHz, the pilot's full cell settled at 81 °C and hit the soft temperature limit (`get_throttled` bit 3), with the clock dipping to 2256–2366 MHz. At 2000 MHz it held 75–77 °C (peak 79.3 °C) with no throttle bit. The lock costs about 9 % of latency (ONNX Runtime FP32, A1 at 640: 355.7 → 388.5 ms).
- **Every Pi 5 number is at 2000 MHz,** and the paper says so. A Pi 5 session whose `session.json` records another `cooling.arm_freq` is not used.
- **The start gate is unchanged** (below 50 °C, or 3 minutes). With the fan, the Pi idles at about 50 °C, so 9 of the first 15 gates opened at the timeout. Whether the rule changes is open on #8.

---

## 5. Which YOLO26 head is scored: `nms: null`

`nms: null` means external NMS on the **one-to-many** head, for both families. YOLO11 has only that head. For YOLO26, the NMS-free one-to-one head is scored only when `nms=False`.

- **Why:** the one-to-many head is what most edge paths run. The Ultralytics exporter disables the one-to-one head for NCNN at any precision, LiteRT `int8`/`w8a16`, TensorRT <8.5 at any precision (an original Jetson Nano on JetPack 4.x), and TensorRT 10.3.x INT8 on JetPack 6 (`engine/exporter.py` L682-713; see *YOLO26 reality check*). **ONNX Runtime and OpenVINO exports keep the one-to-one head, INT8 included.** Whether that head survives INT8 accurately is unmeasured. Since 8.4.142 (PR #26066), predict, val and export all default to one-to-many + NMS unless `nms=False` is passed, so `nms: null` is also the library default. Accuracy is measured on the head that actually gets deployed, and YOLO11 and YOLO26 are compared under the same postprocessing.
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

**Kaggle only, accelerator T4 ×2.** The P100 option cannot run this environment: torch 2.14 is a CUDA 13 build and CUDA 13 dropped Pascal support. The RTX 4050 laptop does not train. Each executor uses one account. `scripts/smoke.py` runs once per account before it produces numbers; `kaggle/probe_kaggle.py` runs once for the study (the machine class is the same on both accounts) and its per-imgsz batch and pairing verdict are frozen in `kaggle/runs.yaml`. Both records are kept. Checkpoints are mirrored off Kaggle after every session. The procedure is [`kaggle/README.md`](kaggle/README.md).

---

## 8. Precision

Decided 22 October 2026 by the map ticket *Precision and export matrix*. The term is defined in [`GLOSSARY.md`](../GLOSSARY.md).

**The precision rule.** A `baseline` cell runs at its runtime's default precision on that platform, with no compression step from us. `quantized` means INT8 post-training quantization with calibration (QAT is out of the current matrix). An FP16 cell is a baseline wherever FP16 is the runtime's default (NCNN on the Pi 5, TensorRT on the Jetson). It is never a quantized cell. Precision is part of a cell's experimental condition, so every result table gives it a column of its own.

**The OpenVINO FP32 rule.** OpenVINO runs an FP32 model in true FP32 on every platform. `studybench.py` compiles with `INFERENCE_PRECISION_HINT=f32` and `EXECUTION_MODE_HINT=ACCURACY` everywhere. Without them the CPU plugin lowers an FP32 model to f16 on ARM and to bf16 on an x86 CPU with AVX-512 BF16. The laptop's Ryzen 7 8845HS (Zen 4) is such a CPU, so before this rule its "FP32" OpenVINO runs computed in bf16. That includes the #9 OpenVINO FP32 reference, which gets scored again at true FP32. On the laptop's x86 CPU, a quantized OpenVINO model keeps its INT8 layers under these hints, and only its float layers (the Detect head) stay in f32. On the Pi 5's ARM CPU it does not: OpenVINO 2026.4.1 compiles the INT8 model with no u8/i8 layer (45 convolutions in f16, 43 in f32), and the precision guard stops the run. **So the Pi 5 has no OpenVINO INT8 cell** (decided 28 October 2026, #38); its quantized cells are ONNX Runtime INT8.

**Recorded, not named.** Every `studybench.py` run writes the precision its runtime really computes in to `meta.json` (`precision`), and writes the evidence and its source to `runtime_info` (`precision_from`). The precision is never read from the artifact's name. `studyscore.py` copies it into `score.json`, and `studysession.py` into `summary.csv`.

| Runtime | Precision | Taken from |
|---|---|---|
| ONNX Runtime (CPU) | FP32, or INT8 | INT8 when the graph has quantize/dequantize or integer operators. The CPU provider never lowers float arithmetic. |
| OpenVINO (CPU) | FP32, or INT8 (x86 only; none on the Pi 5, #38) | The runtime precision of the compiled model's convolution and matmul layers. INT8 when any of them runs on u8/i8. A BF16 or FP16 verdict means the FP32 rule failed. |
| NCNN (CPU) | FP16 where its defaults give FP16 arithmetic on that CPU, else FP32 | ncnn's options and its CPU probe. ncnn computes in FP16 only when `use_fp16_storage` and `use_fp16_arithmetic` are both on (the defaults) and `cpu_support_arm_asimdhp()` finds ARMv8.2 FP16 arithmetic. That probe reports it only under a 64-bit OS. So FP16 on the Pi 5's Cortex-A76, and FP32 on the Pi 4B's Cortex-A72 and on x86. Those two store at most a few layers' weights in fp16. |
| TensorRT (Jetson GPU) | FP16, or INT8 | An INT8 engine is built with detailed profiling, so the engine inspector lists each layer's formats. INT8 when any layer runs on Int8, and a header that disagrees is noted. An FP16 engine lists layer names only, so its Ultralytics header (`args.quantize`) is the only evidence. FP16 lets TensorRT pick FP16 kernels; it still runs a layer in FP32 when that is faster. |

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
