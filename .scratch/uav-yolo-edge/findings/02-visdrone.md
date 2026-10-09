# Findings: VisDrone2019-DET training practice for s-scale YOLO

Ticket: `issues/02-visdrone-training-practice.md`
Researched: 2026-09-12
Ultralytics version current at time of research: **8.4.149** (PyPI). Key facts below depend on **>= 8.4.133**.

---

## TL;DR — the five things that change the plan

1. **The deck's numbers are COCO, and the gap is enormous.** `47.8` is exactly `YOLO26s` **COCO val** mAP50-95 (e2e column) and `40.1` is `YOLO26n` COCO val mAP50-95 (e2e). On VisDrone the same model class lands at **~18 mAP50-95 on test-dev** and **~23–24 on val**, at imgsz 640. Budget for roughly **40% of the COCO number**.
2. **Resolution is the single biggest lever, bigger than model scale or any architecture paper's contribution.** YOLOv8s 640→960 buys **+9.2 mAP50** on val; that is more than 640→YOLOv8**l** (+4.9). Multiple independent sources agree on **+8 to +12 mAP50 for 640→1024/1280**.
3. **AP-small / AP-medium / AP-large DOES come for free** — but only in Ultralytics **>= 8.4.133**, only with `save_json=True`, and *not* from the official VisDrone MATLAB toolkit, which never emits area splits. Details in Q5. This is the single most load-bearing finding for the paper's central claim.
4. **val and test-dev differ by ~6 mAP50 points, consistently.** Papers report both; you must say which. test-dev labels **are public** and the dataset authors explicitly recommend reporting on test-dev.
5. **The Ultralytics converter is correct, but its *evaluation* is not official-toolkit-comparable.** The conversion is safe (verified by counting every annotation row — see Q3.2). The mismatch is at eval time: Ultralytics has no ignored-region concept, defaults `max_det=300` where the official protocol uses 500, and averages classes differently. All three bias Ultralytics mAP **low**. Set `max_det=500` and footnote the protocol.

---

## Q2 (PRIORITY) — Realistic VisDrone baselines for s-scale YOLO

### 2.1 First, the correction: where the deck's numbers come from

From [docs.ultralytics.com/models/yolo26](https://docs.ultralytics.com/models/yolo26/) — **COCO val2017**, imgsz 640:

| Model | COCO mAPval 50-95 | COCO mAPval 50-95 (e2e) | Params (M) | FLOPs (B) |
|---|---|---|---|---|
| YOLO26n | 40.9 | **40.1** | 2.4 | 5.5 |
| YOLO26s | 48.6 | **47.8** | 9.5 | 20.9 |
| YOLO26m | 53.1 | 52.5 | 20.4 | 68.4 |

From [docs.ultralytics.com/models/yolo11](https://docs.ultralytics.com/models/yolo11/) — **COCO val2017**, imgsz 640:

| Model | COCO mAPval 50-95 | Params (M) | FLOPs (B) |
|---|---|---|---|
| YOLO11n | 39.5 | 2.6 | 6.5 |
| **YOLO11s** | **47.0** | 9.4 | 21.6 |
| YOLO11m | 51.5 | 20.1 | 68.1 |

**The deck's `40.1 / 47.8` pair is verbatim the YOLO26n / YOLO26s COCO e2e column.** These are COCO-80-class numbers on COCO val2017. They have nothing to do with VisDrone. Ultralytics publishes **no** VisDrone baseline for any model — the [VisDrone dataset docs page](https://docs.ultralytics.com/datasets/detect/visdrone/) contains no accuracy figures at all.

### 2.2 What s-scale YOLO actually gets on VisDrone

**VisDrone2019-DET-val (548 images), imgsz 640:**

| Model | mAP50 | mAP50-95 | Epochs | Source |
|---|---|---|---|---|
| YOLOv8s | 40.4 | 24.0 | Ultralytics default | [xuanandsix/VisDrone-yolov8](https://github.com/xuanandsix/VisDrone-yolov8) (A40) |
| YOLOv8s | 38.7 | 23.3 | 100 | [DroneScan-YOLO, arXiv 2604.13278](https://arxiv.org/html/2604.13278v1) Table 6 |
| YOLOv8s | 39.1 | 23.4 | 300 | [RTUAV-YOLO, PMC12608591](https://pmc.ncbi.nlm.nih.gov/articles/PMC12608591/) Table 1 |
| YOLOv8s | 43.0 | 26.0 | **600** | [SL-YOLO, arXiv 2411.11477v3](https://arxiv.org/html/2411.11477v3) |
| **YOLO11s** | **39.4** | **23.6** | 300 | [RTUAV-YOLO, PMC12608591](https://pmc.ncbi.nlm.nih.gov/articles/PMC12608591/) Table 1 |
| **YOLO11s** | **38.2** | **22.7** | 200 | [YOLO11s-UAV, PMC12942582](https://pmc.ncbi.nlm.nih.gov/articles/PMC12942582/) |
| YOLO11n | 33.1 | 19.3 | 300 | RTUAV-YOLO Table 1 |

**VisDrone2019-DET-test-dev (1,610 images), imgsz 640:**

| Model | mAP50 | mAP50-95 | Epochs | Source |
|---|---|---|---|---|
| YOLOv8s | 32.0 | 18.4 | default | [xuanandsix](https://github.com/xuanandsix/VisDrone-yolov8) |
| YOLOv8s | 31.5 | 17.9 | 100 | [arXiv 2605.24831](https://arxiv.org/html/2605.24831) Table 5 |
| YOLOv8s | 35.2 | — | 600 | [SL-YOLO](https://arxiv.org/html/2411.11477v3) |
| **YOLO26s** | **32.5** | **18.2** | 100 | [arXiv 2605.24831](https://arxiv.org/html/2605.24831) Table 5 |
| YOLO26n | 26.2 | 14.2 | 100 | same |
| YOLO26m | 36.7 | 21.2 | 100 | same |
| YOLO11s | 31.4 | — | 200 | [PMC12942582](https://pmc.ncbi.nlm.nih.gov/articles/PMC12942582/) |
| YOLO11s | 32.3 | 18.5 | 300 | [dronefreak/visdrone-yolov11s](https://huggingface.co/dronefreak/visdrone-yolov11s) (split stated as "test") |

### 2.3 The single most relevant source: YOLO26 vs YOLOv8 on VisDrone

[arXiv 2605.24831](https://arxiv.org/html/2605.24831), **Table 5, VisDrone test set**, imgsz 640, batch 16, 100 epochs, Ultralytics defaults, RTX 6000 Ada:

| Model | Scale | Prec. | Rec. | F1 | mAP50 | mAP50:95 |
|---|---|---|---|---|---|---|
| YOLOv8 | n | 0.398 | 0.304 | 0.345 | 0.268 | 0.149 |
| YOLOv8 | **s** | 0.443 | 0.355 | 0.394 | **0.315** | **0.179** |
| YOLOv8 | m | 0.498 | 0.367 | 0.423 | 0.345 | 0.200 |
| YOLOv8 | l | 0.508 | 0.393 | 0.443 | 0.367 | 0.214 |
| YOLOv8 | x | 0.517 | 0.395 | 0.448 | 0.368 | 0.214 |
| YOLO26 | n | 0.388 | 0.307 | 0.343 | 0.262 | 0.142 |
| YOLO26 | **s** | 0.454 | 0.363 | 0.403 | **0.325** | **0.182** |
| YOLO26 | m | 0.507 | 0.395 | 0.444 | 0.367 | 0.212 |
| YOLO26 | l | 0.517 | 0.398 | 0.449 | 0.372 | 0.215 |
| YOLO26 | x | 0.520 | 0.411 | 0.459 | 0.383 | 0.224 |

Efficiency, same paper (Table 6): YOLOv8s 11.17 M params / 28.69 GFLOPs / 5.55 ms GPU / 33.87 ms CPU; **YOLO26s 10.01 M / 22.58 GFLOPs / 7.15 ms GPU / 46.60 ms CPU**.

Two things worth flagging for the plan:
- **YOLO26s beats YOLOv8s on VisDrone by only +1.0 mAP50 / +0.3 mAP50-95.** The scale step (s→m) is worth ~+4 mAP50, i.e. 4× more than the architecture step. Do not build the paper's narrative on YOLO26-vs-YOLO11 accuracy deltas; they will be inside run-to-run noise.
- **YOLO26s is *slower* than YOLOv8s in this paper's measurements** (7.15 vs 5.55 ms GPU; 46.60 vs 33.87 ms CPU) despite fewer params and FLOPs. That is the opposite of the marketing and is directly relevant to an edge-latency paper. Treat as one data point; measure it yourself.

### 2.4 Expected range to reset expectations on

For **YOLO11s / YOLO26s, COCO-pretrained, imgsz 640, 100–300 epochs, Ultralytics defaults**:

| Split | mAP50 | mAP50-95 |
|---|---|---|
| val (548) | **38–43** | **23–26** |
| test-dev (1,610) | **31–35** | **18–21** |
| val @ 960–1280 | **48–55** | **30–36** |

The COCO→VisDrone drop for the same s-scale model is roughly **47 → 18 mAP50-95 on test-dev** (−29 points, ~62% relative). This is normal and expected: VisDrone is 10 dense classes of tiny aerial objects, ~70 boxes/image, ~68% of instances below 32×32 px.

### 2.5 One source I would not cite

An agent-derived YOLO11s val baseline of **44.6 / 29.4** for [MASF-YOLO (arXiv 2504.18136)](https://arxiv.org/pdf/2504.18136) was obtained by *subtracting the abstract's claimed +4.6 / +3.5 deltas* from the MASF-YOLO-s result — not read off a table. It is ~5 points above every directly-read YOLO11s val figure (38.2, 39.4). Do not use it without opening the paper's Table I directly.

Sources also disagree at the low end: [MDPI Appl. Sci. 16(7):3559](https://www.mdpi.com/2076-3417/16/7/3559) reportedly gives YOLOv8l = 15.9% mAP50 on VisDrone, ~3× below every other source. The article and PDF both returned HTTP 403; its absolutes are excluded here.

---

## Q1 — Input resolution

### 1.1 What published work uses

640 dominates the *improved-YOLO* literature (everyone is comparing architecture deltas at fixed resolution, not chasing peak mAP). 1280–1536 is what competition/SOTA entries use. Ultralytics' own VisDrone docs example is `epochs=100, imgsz=640`, but the same page explicitly advises against it for this dataset.

### 1.2 The resolution ablations that exist

**(a) YOLOv8s, 640 → 960, identical recipe, COCO-pretrained, Ultralytics defaults** ([xuanandsix/VisDrone-yolov8](https://github.com/xuanandsix/VisDrone-yolov8), single A40):

| Model | imgsz | Other | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 |
|---|---|---|---|---|---|---|
| yolov8s | 640 | default | 0.404 | 0.240 | 0.320 | 0.184 |
| yolov8l | 640 | default | 0.453 | 0.278 | 0.361 | 0.213 |
| **yolov8s** | **960** | default | **0.496** | **0.308** | **0.396** | **0.234** |
| yolov8s | 960 | multi-scale | 0.492 | 0.304 | 0.399 | 0.236 |
| yolov8s-p2 | 960 | default | 0.519 | 0.321 | 0.417 | 0.244 |
| yolov8s-p2 | 960 | mixup=0.4 | 0.522 | 0.325 | 0.431 | 0.255 |
| yolov8s-p2 | 960 | mixup=0.4, val iou=0.6 | **0.531** | **0.327** | **0.437** | **0.256** |

**Read the first three rows together.** `yolov8s @960` (0.308 mAP50-95) **beats** `yolov8l @640` (0.278) — at ~1/3 the parameters. On VisDrone, resolution beats capacity. That framing is worth a figure in the paper.

**(b) 640 → 1024, eight models, one recipe** ([Sensors 2023, MFP-YOLO, PMC10346989](https://pmc.ncbi.nlm.nih.gov/articles/PMC10346989/), Table 5). Caveat: **trained from scratch, no pretrained weights**, 300 epochs, batch 8 — absolutes are low, deltas are the point:

| Model | val mAP50 @640 | @1024 | Δ | test mAP50 @640 | @1024 | Δ |
|---|---|---|---|---|---|---|
| YOLOv5s | 33.1 | 45.1 | **+12.0** | 27.8 | 38.0 | +10.2 |
| YOLOv8n | 33.4 | 43.4 | +10.0 | 27.2 | 35.9 | +8.7 |
| YOLOX-s | 32.6 | 40.3 | +7.7 | 27.7 | 34.1 | +6.4 |
| YOLOv7-tiny | 35.4 | 45.2 | +9.8 | 30.3 | 38.1 | +7.8 |
| YOLOv3 | 39.2 | 51.9 | +12.7 | 32.6 | 43.1 | +10.5 |
| YOLOv5x | 40.8 | 53.2 | +12.4 | 34.2 | 43.9 | +9.7 |
| MFP-YOLO | 46.0 | 53.7 | +7.7 | 35.8 | 44.1 | +8.3 |

**(c) 1280 → 640: −9 mAP50 points**, user-reported on the official Ultralytics forum; Ultralytics staff (BurhanQ) response: *"Given the size of the objects are quite small in the VisDrone dataset, that's not terribly surprising"*, suggesting 800/960 as middle ground. [community.ultralytics.com thread 1614](https://community.ultralytics.com/t/standard-epochs-and-imgsz-for-training-yolo11-yolov12-on-visdrone-dataset/1614)

**(d) Why, mechanically** ([DroneScan-YOLO](https://arxiv.org/html/2604.13278v1)): *"At 640px, an 8×8 pixel object produces a 1×1 activation on P3 (stride 8). At 1280px, the same object produces a 2×2 activation on P3, and 4×4 on the new P2 head (stride 4)."*

**Converged rule of thumb: +8 to +12 mAP50 for 640 → 1024/1280 on VisDrone.** Larger than almost any architecture paper's claimed contribution (typically +4 to +9 at fixed 640).

### 1.3 SAHI / sliced inference

Standard on VisDrone, and the gains are large — but it is an inference-time-only lever with a multiplicative latency cost.

[SAHI, Akyon et al., ICIP 2022](https://arxiv.org/abs/2202.06934), VisDrone val, MMDetection detectors (FI = full-image finetune, SF = sliced finetune, PO = fused full+sliced prediction):

| Setup | AP50 | AP50 small | AP50 medium | AP50 large |
|---|---|---|---|---|
| FCOS+FI | 25.8 | 14.2 | 39.6 | 45.1 |
| FCOS+SAHI+PO | 29.0 (+3.2) | 18.9 | 41.5 | 46.4 |
| FCOS+SF+SAHI+FI+PO | **38.5 (+12.7)** | 25.9 | 55.4 | 59.8 |
| VFNet+FI | 28.8 | 16.8 | 44.0 | 47.5 |
| VFNet+SF+SAHI+FI+PO | **42.2 (+13.4)** | 29.6 | 59.2 | 63.3 |
| TOOD+FI | 29.4 | 18.1 | 44.1 | 50.0 |
| TOOD+SF+SAHI+FI+PO | **43.5 (+14.1)** | 31.7 | 59.8 | 65.4 |

[Keleş et al., arXiv 2203.04799](https://arxiv.org/abs/2203.04799) — YOLO-specific, VisDrone test-dev, COCO-pretrained, trained at 1088, sliced at 640 windows / 0.1 overlap:
- Best: **YOLOv5-L, crop-trained + sliced inference = 48.8 AP50 on test-dev**.
- Sliced inference raised AP50 in **all 8** experiments. YOLOv5-S +4.1 AP50; YOLOv5-L crop-trained +10.8 AP50.
- **Large-object AP50 *drops* when slicing** (buses/trucks cut across tiles). Known failure mode.
- Null result: crop-training + *standard* full-image inference is **worse** than standard/standard. Sliced training only pays with sliced inference.

Cost, [ASAHI arXiv 2604.19233](https://arxiv.org/html/2604.19233v1) (VisDrone val, TPH-YOLOv5, 512px slices, 15% overlap): TPH full inference 34.2 mAP50 → +SAHI **55.1 mAP50 at 4.67 img/s** — i.e. ~6–12 forward passes per frame.

**Recommendation for this study:** on a Pi 5 / Jetson Nano target, training at 960–1280 (one forward pass) will almost certainly beat 640 + 6-tile SAHI on the accuracy-per-millisecond curve. SAHI belongs in related work, or as one clearly-labelled "upper bound at any latency" row — not in the main matrix.

### 1.4 VRAM on 6 GB — no official table exists

Searched Ultralytics docs (`usage/cfg`, `guides/model-training-tips`), `autobatch.py` source, and issues #17488, #18105, #18849, #19488, #19917, #21730. What is documented:

- [`autobatch.py`](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/utils/autobatch.py): `batch=-1` targets **60% of total CUDA memory**; on GPUs **< 16 GB it only profiles batch ∈ {1, 2, 4, 8, 16}**. So on a 4050 6GB it will never propose > 16.
- `amp=True` is default and is the main memory saver. Activation memory scales ∝ imgsz² × batch, so 640→1280 is a **4× activation increase** at fixed batch.
- Measured anchors (all on larger cards): YOLO11s @640 batch 40 on RTX 4090 → ~7.2 GB ([#19917](https://github.com/ultralytics/ultralytics/issues/19917)); YOLOv8x @1280 needed batch=4 on a 24 GB A5000 ([#18849](https://github.com/ultralytics/ultralytics/issues/18849)); DroneScan-YOLO used **batch=4 @1280 on a 16 GB 4090 Laptop**; TPH-YOLOv5 @1536 forced to **batch=2**.
- Also relevant: *"YOLO11 uses double sized GPU memory"* vs YOLOv5 at the same imgsz ([#17488](https://github.com/ultralytics/ultralytics/issues/17488)).

**Extrapolation (not a cited measurement) for YOLO11s/YOLO26s on 6 GB:** ~batch 8–16 @640, ~batch 4 @1024, ~batch 2–4 @1280. Verify with `batch=-1` and a 20–30 epoch sanity run. The trainer auto-halves batch and retries up to 3× on a first-epoch OOM.

**Planning consequence:** imgsz 1280 on the 4050 at batch 2–4 will be brutally slow and will change gradient-noise characteristics vs. the Kaggle/Colab runs. If resolution is a study variable, pin it to **960** as the headline config (best mAP-per-GPU-hour on 6 GB) and treat 640 as the "edge-realistic" config.

---

## Q3 — Annotation handling (ignored regions, `others`)

### 3.1 What the official format says

Official annotation format, 8 comma-separated fields per line ([VisDrone2018-DET-toolkit README](https://github.com/VisDrone/VisDrone2018-DET-toolkit)):

| # | Field | Meaning |
|---|---|---|
| 1 | `bbox_left` | x of top-left corner |
| 2 | `bbox_top` | y of top-left corner |
| 3 | `bbox_width` | width in px |
| 4 | `bbox_height` | height in px |
| 5 | **`score`** | **GT: 1 or 0.** 1 = considered in evaluation; **0 = ignored**. (−1 in detection results) |
| 6 | `object_category` | 0 = ignored regions, 1 = pedestrian, 2 = people, 3 = bicycle, 4 = car, 5 = van, 6 = truck, 7 = tricycle, 8 = awning-tricycle, 9 = bus, 10 = motor, **11 = others** |
| 7 | `truncation` | GT: 0 (none) or 1 (1%–50%); −1 in detections |
| 8 | `occlusion` | GT: 0 (none), 1 (1%–50%), 2 (50%–100%); −1 in detections |

The toolkit README states the rule plainly:

> **"The detections in the ignored regions or labeled as 'others' will be not considered in evaluation."**

So both category 0 and category 11 are outside the 10-class evaluated set, by design.

### 3.2 What Ultralytics' converter actually does

Verbatim from [`ultralytics/cfg/datasets/VisDrone.yaml`](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/VisDrone.yaml) (`visdrone2yolo`, current `main`):

```python
with open(f, encoding="utf-8") as file:
    for row in [x.split(",") for x in file.read().strip().splitlines()]:
        if row[4] != "0":  # Skip ignored regions
            x, y, w, h = map(int, row[:4])
            cls = int(row[5]) - 1
            # Convert to YOLO format
            x_center, y_center = (x + w / 2) * dw, (y + h / 2) * dh
            w_norm, h_norm = w * dw, h * dh
            lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
```

Read this carefully — it is the pitfall the ticket is asking about:

- **The filter is on `row[4]`, the `score` column — NOT on `row[5]`, the `object_category`.** There is no explicit `if cls == 0` or `if cls == 10` check anywhere.
- `cls = int(row[5]) - 1` maps category 1→0 … 10→9. It would also map category **0 → −1** and category **11 → 10**.
- `names:` declares exactly 10 classes (0–9). A surviving category-0 or category-11 row therefore produces an **out-of-range label** (`-1` or `10`).

The [Ultralytics docs](https://docs.ultralytics.com/datasets/detect/visdrone/) assert this works out: the converter *"skip[s] regions marked as ignored (which also excludes the unused 'others' category)"*. That claim is true **only if** every category-0 and category-11 row in the official GT carries `score = 0` — which the official spec implies but the code does not enforce.

**This was checked empirically.** All three splits were downloaded from the Ultralytics mirror and `score` (col 5) was cross-tabulated against `object_category` (col 6) over every annotation row:

| Split | Images | score=0, cat=0 | score=0, cat=11 | score=0 with cat 1-10 | score=1 with cat 0 or 11 |
|---|---|---|---|---|---|
| train | 6,471 | 8,813 | 1,532 | **0** | **0** |
| val | 548 | 1,378 | 32 | **0** | **0** |
| test-dev | 1,610 | 2,180 | 265 | **0** | **0** |

**In VisDrone2019-DET, `score == 0` is exactly equivalent to `object_category in {0, 11}`, with zero exceptions.** Therefore:

- Filtering on `row[4] == "0"` and filtering on `category in {0, 11}` are **the same filter** on this dataset. The `score` column carries no information beyond the category.
- The theoretical out-of-range labels (`cls = 10` from category 11, `cls = -1` from category 0) **never occur** with the official data. The Ultralytics converter is correct, despite looking fragile.

Still worth a 5-second sanity check after download (costs nothing, catches a corrupted mirror):

```bash
cut -d' ' -f1 datasets/VisDrone/labels/train/*.txt | sort -u   # expect exactly 0..9
```

The legacy [yolov5 `data/VisDrone.yaml`](https://raw.githubusercontent.com/ultralytics/yolov5/master/data/VisDrone.yaml) uses identical logic, so YOLOv5-era VisDrone numbers are on the same footing.

### 3.3 What the official toolkit does, and why your numbers will not match it

The conversion is fine. **The mismatch is at evaluation time, and it is structural, not a bug.**

**Ignored regions - the official rule.** [`utils/dropObjectsInIgr.m`](https://raw.githubusercontent.com/VisDrone/VisDrone2018-DET-toolkit/master/utils/dropObjectsInIgr.m) rasterizes every category-0 box into a binary mask, builds an integral image, then deletes - from **both** GT and detections - any box with **>= 50% of its own area** inside the mask:

```matlab
idxIgr = gt(:, 6) == 0;
igrRegion = max(1,gt(idxIgr, 1:4));
...
igrVal = tl + br - tr - bl;
if(igrVal/(h*w)<0.5)
    idxLeftDet = cat(1, idxLeftDet, i);   % keep
end
```

This is a **50%-of-detection-area rule, not an IoU rule**. A detection inside an ignore region is neither TP nor FP - it simply vanishes.

**`others` - dropped by omission.** `dropObjectsInIgr` does *not* filter category 11 (`gt(:,6) ~= 0` keeps it). The discard happens in [`utils/calcAccuracy.m`](https://raw.githubusercontent.com/VisDrone/VisDrone2018-DET-toolkit/master/utils/calcAccuracy.m), whose class loop is hard-coded `for idClass = 1:10`. Category-11 GT and detections are never selected - never an FN, never a match target. Consequence: a box predicted as `car` landing on an *others* object is still a **plain false positive** (unless it also sits in an ignore region).

**The five deltas between Ultralytics and `evalDET.m`:**

| # | Axis | Official toolkit | Ultralytics | Bias |
|---|---|---|---|---|
| 1 | **Ignored regions** | detections >=50% inside an ignore region are **deleted** | no ignore concept - those GT boxes were dropped at conversion, so detections there are **false positives** | **Ultralytics low** |
| 2 | **maxDets** | **500** (`evalDET.m`) | `max_det: 300` ([`default.yaml`](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/default.yaml) L57) | **Ultralytics low** - caps recall at ~70 obj/img |
| 3 | **Class averaging** | image-frequency-**weighted** across classes | unweighted mean over `ap_class_index` | either way; large, class counts are skewed (train: 144,867 `car` vs 3,246 `awning-tricycle`) |
| 4 | **PR integration** | VOC all-points ([`VOCap.m`](https://raw.githubusercontent.com/VisDrone/VisDrone2018-DET-toolkit/master/utils/VOCap.m)) | 101-point interp + **trapezoidal** sum | **Ultralytics high**, up to 3.4% |
| 5 | **Resolution / NMS** | no resize imposed; toolkit removed its own NMS in v1.0.3 | `imgsz: 640`, NMS at `iou: 0.7` | context-dependent |

On (3): `calcAccuracy.m` accumulates `evalClass` inside the per-image loop and never resets it, so `AP(evalClass,:)` indexes rows **with repetition, once per image containing that class**. The official "AP" is therefore an image-frequency-weighted mean over classes, not a macro-average. (Code reading only; no VisDrone document acknowledges it.)

On (4), the two apparently conflicting sources reconcile exactly. [`ultralytics/utils/metrics.py` `compute_ap`](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/utils/metrics.py):

```python
method = "interp"  # methods: 'continuous', 'interp'
if method == "interp":
    x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
    func = np.trapezoid if checks.check_version(np.__version__, ">=2.0") else np.trapz
    ap = func(np.interp(x, mrec, mpre), x)  # integrate
```

Ultralytics interpolates onto COCO's 101 points but then **integrates trapezoidally**, where pycocotools takes the plain **mean** of the 101 precisions. Hence [Roboflow100-VL (arXiv 2505.20612), Appendix B](https://arxiv.org/html/2505.20612v2): *"Ultralytics uses a trapezoidal sum, which inflates model performance by as much 3.4% compared to pycocotools."*

**How big is the ignored-region effect?** Bounded, but concentrated where it hurts:

| Split | Images with >=1 ignored region | Share |
|---|---|---|
| train | 2,648 / 6,471 | 40.9% |
| val | **340 / 548** | **62.0%** |
| test-dev | 533 / 1,610 | 33.1% |

Ignored regions cover only **~1.0% of val pixel area**, but they sit precisely over dense unlabelled crowds and vehicle parks - exactly where a good detector fires most. The *direction* (Ultralytics penalises, official does not, so **Ultralytics AP is biased low**) is solid; the *magnitude* is unquantified. No published A/B of identical predictions scored both ways exists - searched Ultralytics issues #17536, #18031, #4047, #6876, #12758 and yolov5 [#2340](https://github.com/ultralytics/yolov5/issues/2340) (*"Ignore regions"*, asks exactly this, **closed stale with no maintainer answer**).

**Bottom line:** an Ultralytics `mAP50-95` on VisDrone is **not** the official `AP@[0.50:0.95 | maxDets=500]`. Never put them in one table without a caveat. At minimum set **`max_det=500`**. If you want official-comparable numbers, dump predictions as `<x>,<y>,<w>,<h>,<score>,<category+1>,-1,-1` and run `evalDET.m` - a cheap rigour win, since the closest prior work ([arXiv 2605.24831](https://arxiv.org/html/2605.24831v1)) reports **Ultralytics-pipeline numbers only**.

---

## Q4 - Split discipline

VisDrone2019-DET splits: **train 6,471 / val 548 / test-dev 1,610 / test-challenge 1,580**.

### 4.1 test-dev labels ARE public; test-challenge labels are not

From the official [VisDrone-Dataset README](https://github.com/VisDrone/VisDrone-Dataset):

> "Note that the bounding box annotations of test-dev are avalialbe. Researchers can use test-dev to publish papers. **testset-challenge is used for VisDrone2020 Challenge and the annotations is unavailable.**"

| Split | Size | Images | Annotations |
|---|---|---|---|
| trainset | 1.44 GB | 6,471 | yes |
| valset | 0.07 GB | 548 | yes |
| **testset-dev** | 0.28 GB | 1,610 | **yes** - README marks the link "(GT avalialbe)" |
| testset-challenge | 0.28 GB | 1,580 | **withheld** |

Verified by download: the Ultralytics mirror of test-dev contains **1,610 annotation `.txt` files with real labels** (first row of the first file: `685,463,110,65,1,4,0,0` - a `car`, score=1). Full test-dev class distribution: 21,006 pedestrian / 6,376 people / 1,302 bicycle / 28,074 car / 5,771 van / 2,659 truck / 530 tricycle / 599 awning-tricycle / 2,940 bus / 5,845 motor, plus 2,180 ignored regions and 265 others.

Caveat: the **toolkit** README predates the test-dev release and still says only "training and validation" annotations are public. That text is **stale**; the dataset repo and challenge site supersede it.

Ultralytics' yaml already maps test-dev to the `test:` key (`splits = {..., "VisDrone2019-DET-test-dev": "test"}`), so `model.val(split="test")` gives the test-dev number with no extra work. The test-challenge download URL is present but commented out.

### 4.2 Which split do papers report?

**Both, inconsistently - the single biggest comparability trap in the VisDrone literature.**

| Paper | Split | Evidence |
|---|---|---|
| SOD-YOLO (2025) | **val** | Table 1 caption: *"Comparison of different object detectors on VisDrone2019-DET-val."* [arXiv 2507.12727](https://arxiv.org/html/2507.12727v1) |
| TPH-YOLOv5 (ICCVW 2021) | **test-challenge** | *"On DET-test-challenge dataset, the AP result of TPH-YOLOv5 are 39.18%"* [arXiv 2108.11539](https://arxiv.org/abs/2108.11539) |
| YOLO26 vs YOLOv8 (2026) | **val AND test-dev**, scored with Ultralytics | *"All models were evaluated on the validation and test splits using the built-in evaluation pipeline provided by the Ultralytics YOLO framework."* [arXiv 2605.24831](https://arxiv.org/html/2605.24831v1) |
| VisDrone-DET challenge papers (2018/2019) | **test-challenge** | ECCV'18 / ICCV'19 workshops |

Pattern: **challenge entries report test-challenge; academic improved-YOLO papers mostly report val**; a minority also give test-dev.

**The val-to-test-dev gap is large and consistent: ~6 mAP50 points, ~5 mAP50-95 points.** e.g. YOLOv8s 40.4 val -> 32.0 test ([xuanandsix](https://github.com/xuanandsix/VisDrone-yolov8)); YOLO11s 38.2 val -> 31.4 test ([PMC12942582](https://pmc.ncbi.nlm.nih.gov/articles/PMC12942582/)). Quote a val number against someone's test-dev number and you will look ~6 points better than you are.

Not verified: the exact split wording for CEASC ([arXiv 2303.14488](https://arxiv.org/abs/2303.14488)), UFPMP-Det ([arXiv 2112.10415](https://arxiv.org/abs/2112.10415)) and MASF-YOLO ([arXiv 2504.18136](https://arxiv.org/pdf/2504.18136)) - full-text extraction failed. Do not cite those three's splits from this document.

### 4.3 Evaluation server: up, but dormant - and you do not need it

- `aiskyeye.com` responds HTTP 200 on `/`, `/home/`, `/evaluate/`, `/faq/` (checked 2026-09-12).
- **test-challenge is time-boxed.** [Test Guidelines](https://aiskyeye.com/evaluate/test-guidelines-2022/): *"The test-challenge server will only remain open during the competition."*
- **test-dev does not need the server.** [FAQ](https://aiskyeye.com/faq/): *"You can evaluate the test-dev result on your local machine. The ground truth of test-dev set is available."*
- **Dormancy signals:** newest announced event is *"Vision Meets Drones 2024"* (July 2024); the [leaderboard](https://aiskyeye.com/leaderboard/) lists 2020-2024 only, no 2025/2026, and the 2024 Object Detection tab shows "Coming soon". The DET toolkit's last code commit is **2019-04-28**; the dataset repo's last commit is **2023-09-24**.

**Do not plan on submitting to the server.** Treat test-challenge as unavailable.

### 4.4 Which split for the headline number

**Headline on `test-dev` (1,610). Use `val` only as the tuning/ablation split.**

1. **The dataset authors say so.** [Test Guidelines](https://aiskyeye.com/evaluate/test-guidelines-2022/): *"Test-dev: The test-dev split is the default test data for general evaluation. **We recommend the authors to generally report the results on test-dev for fair comparison.**"* and *"We encourage the participants to use the validation set to debug their algorithms."*
2. **Hygiene.** Ultralytics selects checkpoints and early-stops on the `val:` key. A number reported on the same 548 images you selected on is optimistically biased. test-dev is 3x larger and untouched.
3. **Comparability.** test-dev is what the official challenge-era baselines were computed on. `val` numbers in the literature are a grab-bag of incompatible pipelines.
4. **It is free.** `model.val(split="test")` already points at test-dev.

**Recipe:** train on `train`, model-select on `val`, headline on `test-dev`. Report both columns. Never tune on test-dev. Never report test-challenge. Set `max_det=500`. Reporting an official-toolkit number alongside the Ultralytics one is a cheap rigour win over the closest prior work.

---

## Q5 (PRIORITY) — Small-object evaluation: do AP-small/medium/large come for free?

**Short answer: from the official VisDrone toolkit, NO. From Ultralytics >= 8.4.133 with `save_json=True`, YES — and with the correct COCO area thresholds applied in native image coordinates.**

### 5.1 The official VisDrone toolkit does NOT give area splits

[`evalDET.m`](https://github.com/VisDrone/VisDrone2018-DET-toolkit) computes exactly seven metrics, over all categories, at IoU 0.50:0.05:0.95, maxDets 500:

`AP`, `AP50`, `AP75`, `AR1`, `AR10`, `AR100`, `AR500`

There is **no** area-range parameter and **no** AP-small/medium/large output. VisDrone's official protocol is COCO-*style* in its IoU sweep, but it deliberately substitutes the AR@{1,10,100,500} family (reflecting up to ~500 objects/image) for COCO's area splits.

**Consequence:** every paper that reports APs/APm/APl on VisDrone did so by converting VisDrone to COCO JSON and running the COCO API themselves — not by running the official toolkit. e.g. [arXiv 2606.09081](https://arxiv.org/html/2606.09081) states data was *"converted to COCO format"* with *"COCO-style area ranges"*.

### 5.2 Ultralytics DOES give them, as of 8.4.133

This is recent enough that most blog posts, forum answers, and the long-standing [issue #6043](https://github.com/ultralytics/ultralytics/issues/6043) (*"How can I get map_small, map_medium, and map_large?"*, **closed as not planned**) are now **out of date**. The Ultralytics `val` docs page still does not mention it either.

**Release**: [ultralytics v8.4.133](https://newreleases.io/project/github/ultralytics/ultralytics/release/v8.4.133) — *"Custom detection datasets can now report small-, medium-, and large-object mAP when using `save_json=True`"*, PR **#25981** by @fcakyon, *"building temporary COCO-format annotations internally while preserving existing native metrics and prediction files"*, applying *"during training validation, final-model validation, and standalone validation."* Latest PyPI release is **8.4.149**, so this is available now.

**Mechanism**, from [`ultralytics/models/yolo/detect/val.py`](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/models/yolo/detect/val.py) on `main`:

```python
# init_metrics
self.is_custom_json = self.args.save_json and self.args.task == "detect" and not (self.is_coco or self.is_lvis)
self.gdict = getattr(self, "gdict", None) if self.is_custom_json else None
self.build_gdict = self.is_custom_json and self.gdict is None
...
if self.build_gdict:
    self.gdict = {"images": [], "annotations": [], "categories": [{"id": x} for x in self.class_map]}
```

```python
# coco_evaluate
if self.args.save_json and len(self.jdict) and (self.is_coco or self.is_lvis or self.gdict):
    LOGGER.info("\nEvaluating faster-coco-eval mAP...")
    ...
    check_requirements("faster-coco-eval>=1.6.7")
    from faster_coco_eval import COCO, COCOeval_faster
```

```python
stats["metrics/mAP_small(B)"]  = val.stats_as_dict["AP_small"]
stats["metrics/mAP_medium(B)"] = val.stats_as_dict["AP_medium"]
stats["metrics/mAP_large(B)"]  = val.stats_as_dict["AP_large"]
```

VisDrone is not COCO and not LVIS, so `is_custom_json` is True the moment you pass `save_json=True`, and the whole path fires. **No dataset conversion, no manual pycocotools script, no COCO JSON on disk.**

### 5.3 The areas are computed correctly (this is the part that could have been wrong)

Ground truth boxes written into the temporary COCO dict are scaled **back to the original image size** before the area is computed:

```python
boxes = ops.xyxy2ltwh(
    ops.scale_boxes(pbatch["imgsz"], pbatch["bboxes"].clone(),
                    pbatch["ori_shape"], pbatch["ratio_pad"])
).tolist()
...
"area": b[2] * b[3],
```

and predictions are likewise scaled before serialization:

```python
if self.args.save_json or self.args.save_txt:
    predn_scaled = self.scale_preds(predn, pbatch)
if self.args.save_json:
    self.pred_to_json(predn_scaled, pbatch)
```

`ops.scale_boxes(imgsz → ori_shape)` undoes the letterbox. So the area is `w × h` in **native VisDrone pixels**, and `faster-coco-eval` applies the standard COCO ranges — small `[0², 32²)`, medium `[32², 96²)`, large `[96², 1e5²)` — in that same native frame.

**This matters a lot.** Had areas been measured in the letterboxed 640×640 frame, a 1360×765 VisDrone image downscaled ~2.1× would push nearly everything into the "small" bucket and your numbers would be incomparable to every published APs. They are not; they are directly comparable.

(One incidental detail the implementation handles: VisDrone filenames like `0000006_00159_d_0000001.jpg` are non-numeric, so `pred_to_json` stores the stem string as `image_id`; `eval_json` then re-maps prediction `image_id`s onto the generated `gdict["images"]` ids via `pred_counts`. Non-numeric filenames are not a problem.)

### 5.4 The one caveat you must handle in the paper

You will now have **two mAP50-95 numbers for the same run**:
- `metrics/mAP50-95(B)` — Ultralytics' native computation, **trapezoidal PR integration**
- the faster-coco-eval `AP_all` — **101-point interpolation**, COCO-standard

These differ, and per [Roboflow100-VL Appendix B](https://arxiv.org/html/2505.20612v2), *"Ultralytics uses a trapezoidal sum, which inflates model performance by as much 3.4% compared to pycocotools."* The `mAP_small/medium/large` values come from the COCO-style path.

**Do not mix them.** Pick the faster-coco-eval / COCO-style family for every number in the paper (overall AP, AP50, AP75, APs, APm, APl) so the headline mAP and the area splits are computed by the same integrator. Say so in the methods section. `save_json=True` gives you both, so this is purely a reporting-discipline decision.

### 5.5 Why the area splits will show what you need them to

VisDrone is overwhelmingly small-object:

- **68.54% of VisDrone val objects are COCO-small (< 32×32 px); 30.81% are "very small" (< 16×16 px).** [arXiv 2606.09081](https://arxiv.org/html/2606.09081)
- *"68% of annotated instances occupy less than 32×32 pixels."* [DroneScan-YOLO, arXiv 2604.13278](https://arxiv.org/html/2604.13278v1)
- The 548 val images contain **38,759 boxes — ~70 objects per image.** [Ultralytics VisDrone docs](https://docs.ultralytics.com/datasets/detect/visdrone/)

And the area-split spread on VisDrone is already dramatic *before* any compression. [EdgeYOLO (arXiv 2302.07483)](https://ar5iv.labs.arxiv.org/html/2302.07483), Table 2, **VisDrone2019-DET-val, 640×640**:

| Model | AP | AP50 | AP75 | **APs** | **APm** | **APl** |
|---|---|---|---|---|---|---|
| YOLOv5-X | 22.6 | 38.6 | — | — | — | — |
| Faster-RCNN + ResNeXt101 | 22.6 | 40.2 | — | — | — | — |
| Cascade-RCNN + ResNeXt101 | 24.4 | 41.2 | — | — | — | — |
| YOLOX-X | 25.8 | 43.2 | 26.2 | **15.9** | **38.0** | **52.4** |
| EdgeYOLO | 26.4 | 44.8 | 26.2 | **16.3** | **38.7** | **53.1** |

**APl is ~3.3× APs at 640.** That baseline spread is exactly the headroom the paper's central claim needs: if PTQ/pruning/distillation degrade small objects disproportionately, the APs column has ~16 points to lose and will move visibly, while APl (~53) will barely budge. Report `ΔAPs / ΔAP` as the degradation-asymmetry statistic.

Also note [SAHI's](https://arxiv.org/abs/2202.06934) table in §1.3 is itself evidence that the APs/APm/APl decomposition on VisDrone is sensitive and well-behaved — it separates cleanly under an intervention that specifically targets small objects.

### 5.6 Practical recipe

```python
from ultralytics import YOLO
m = YOLO("runs/detect/train/weights/best.pt")
r = m.val(data="VisDrone.yaml", split="test", imgsz=960, save_json=True)
# requires ultralytics >= 8.4.133 and faster-coco-eval >= 1.6.7 (auto-installed)
# -> results_dict now contains, among others:
#    metrics/mAP_small(B), metrics/mAP_medium(B), metrics/mAP_large(B)
```

Pin `ultralytics>=8.4.133` in the repo's requirements and record the exact version in the paper — on any earlier version this silently returns nothing and you would have to hand-roll a pycocotools script.

> **CORRECTION (2026-09-12, *Freeze the experimental invariants*, verified against the ultralytics 8.4.149 and faster-coco-eval 1.8.0 source).** The recipe above is wrong in three ways. Use `study/scripts/studyeval.py`, not `model.val()`, for every reported number:
>
> 1. **`r.results_dict` does NOT contain `metrics/mAP_small|medium|large(B)`.** `model.val()` returns `validator.metrics`, whose `results_dict` is built from the native keys only; the area splits exist only in the stats dict the validator returns, which `model.val()` discards (`engine/model.py`).
> 2. **§5.4's "`save_json=True` gives you both" is half true.** For a custom dataset, `coco_evaluate` overwrites `mAP50`/`mAP50-95` with the faster-coco-eval values only when `is_coco or is_lvis` (`detect/val.py` L609). On VisDrone, `metrics/mAP50-95(B)` stays on the trapezoidal integrator and the COCO-style `AP_all` is only *logged*, never returned.
> 3. **The area splits are computed at `maxDets=100`, not 500.** faster-coco-eval defaults to `maxDets=[1, 10, 100]` and computes every AP stat at `maxDets[-1]` (`core/cocoeval.py` L684, L504-510), whatever Ultralytics' `max_det` is. 107/548 val images (19.5%) and 127/1,610 test-dev images (7.9%) hold more than 100 objects.
>
> Also: training-time (per-epoch) validation never computes area splits at all. `engine/validator.py` returns from the training branch before `eval_json` runs.

---

## Q6 — Epochs and augmentation

### 6.1 Epochs: modal value 100–300

| Work | Epochs | Init | Source |
|---|---|---|---|
| Ultralytics VisDrone docs example | **100** | COCO (`yolo26n.pt`) | [docs](https://docs.ultralytics.com/datasets/detect/visdrone/) |
| MASF-YOLO (YOLOv11-s) | 100 | not stated | [arXiv 2504.18136](https://arxiv.org/pdf/2504.18136) |
| YOLO26-vs-YOLOv8 benchmark | 100 | Ultralytics defaults | [arXiv 2605.24831](https://arxiv.org/html/2605.24831) |
| DroneScan-YOLO | 100 | not stated | [arXiv 2604.13278](https://arxiv.org/html/2604.13278v1) |
| YOLO11s-UAV | 200 | not stated | [PMC12942582](https://pmc.ncbi.nlm.nih.gov/articles/PMC12942582/) |
| RTUAV-YOLO | 300 | not stated | [PMC12608591](https://pmc.ncbi.nlm.nih.gov/articles/PMC12608591/) |
| MFP-YOLO (Sensors 2023) | 300 | **explicitly from scratch** | [PMC10346989](https://pmc.ncbi.nlm.nih.gov/articles/PMC10346989/) |
| LW-YOLOv8s (Sci Rep 2025) | 300 | **explicitly no pretrained weights** | [s41598-025-00341-7](https://www.nature.com/articles/s41598-025-00341-7) |
| SL-YOLO | **600** | not stated | [arXiv 2411.11477v3](https://arxiv.org/html/2411.11477v3) |
| TPH-YOLOv5 | 65 (2 warm-up) | COCO | [arXiv 2108.11539](https://arxiv.org/pdf/2108.11539) |
| Ultralytics staff, VisDrone thread | 600, `patience=100` | `yolo11s.pt` = COCO | [thread 1614](https://community.ultralytics.com/t/standard-epochs-and-imgsz-for-training-yolo11-yolov12-on-visdrone-dataset/1614) |

**Important split in the literature:** architecture papers often train **from scratch on purpose** (MFP-YOLO and LW-YOLOv8s state this explicitly, "for experimental fairness") to isolate the architectural delta. That is why some published absolutes look low and are not comparable to a COCO-fine-tuned baseline. **For a deployment study, COCO-pretrained is correct** and is what Ultralytics and TPH-YOLOv5 do.

**Epoch → mAP evidence on VisDrone (YOLOv8s, val, 640):** ~100 ep → 38.7/23.3; ~300 ep → 39.1/23.4; **600 ep → 43.0/26.0**. So the 100→600 jump is worth roughly **+4 mAP50 / +2.7 mAP50-95** — real, but ~1/2 of what 640→960 buys, at 6× the GPU cost. **On an 80–150 GPU-hour budget, spend the hours on resolution before epochs.**

### 6.2 Optimizer / lr / warmup / batch, as reported

| Work | Optimizer | lr0 | Schedule | Momentum / WD | Warmup | Batch | GPU |
|---|---|---|---|---|---|---|---|
| MASF-YOLO | SGD | 0.01 | cosine | 0.937 / — | — | 12 | RTX 4090D 24G |
| RTUAV-YOLO | SGD | 0.01 → 1e-4 | cosine | 0.937 / 5e-4 | — | 16 | — |
| SOD-YOLO | SGD | 0.005 | cosine | 0.937 / 5e-4 | 3 ep linear | 8 | RTX 4090 |
| DroneScan-YOLO | **AdamW** | 1e-3 | — | — / 5e-4 | 5 ep | **4** (@1280) | RTX 4090 Laptop 16G |
| TPH-YOLOv5 | **Adam** | 3e-4 | cosine → 0.12·lr0 | — | 2 ep | **2** (@1536) | — |
| YOLO11s-UAV | auto | — | — | — | — | 15 | RTX 3090 |
| **Ultralytics defaults** | `auto` | 0.01 (lrf 0.01) | linear | 0.937 / 0.0005 | 3.0 (warmup_momentum 0.8) | 16 | — |

Ultralytics defaults ([usage/cfg](https://docs.ultralytics.com/usage/cfg/)): `epochs=100, imgsz=640, batch=16, optimizer='auto', lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, box=7.5, cls=0.5, dfl=1.5, patience=100, cos_lr=False, amp=True, rect=False, multi_scale=0.0`.

**Nobody deviates much from SGD/0.01/cosine at 640.** The AdamW/1e-3 choices appear at high resolution where batch is forced to 2–4 and SGD becomes noisy — relevant if you run 1280 on the 4050.

### 6.3 Augmentation

**Ultralytics defaults** ([usage/cfg](https://docs.ultralytics.com/usage/cfg/)): `hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0, translate=0.1, scale=0.5, shear=0, perspective=0, flipud=0, fliplr=0.5, mosaic=1.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, erasing=0.4, close_mosaic=10`.

**Ultralytics' own small/dense-object guidance** ([yolo-data-augmentation](https://docs.ultralytics.com/guides/yolo-data-augmentation/)):
- **Mosaic: keep it on.** *"Highly effective for improving small object detection and context understanding."* This **contradicts the folk claim that mosaic hurts small objects** — I found no VisDrone-specific evidence that it is harmful.
- `close_mosaic` disables mosaic **and** `copy_paste`/`mixup`/`cutmix` for the last N epochs (default 10).
- CutMix suggested for dense/occluded scenes; mixup for dense/overlapping; copy_paste for rare classes (VisDrone's `awning-tricycle`, `bicycle`, `tricycle` are exactly this).
- `fliplr=0.5` fine (drone yaw is arbitrary); `flipud` unnecessary.

**Empirical VisDrone results — and they disagree with the generic advice:**
- [xuanandsix](https://github.com/xuanandsix/VisDrone-yolov8) (YOLOv8s-p2 @960): **`mixup=0.4` helped** — test mAP50 0.417 → 0.431 (+1.4). **`copy_paste` FAILED and tuning `scale` FAILED**, both explicitly listed as approaches that did not work on VisDrone.
- [DroneScan-YOLO](https://arxiv.org/html/2604.13278v1) by contrast used `mosaic=1.0, copy_paste=0.3, scale ±0.9` at 1280.
- **Sources disagree on `copy_paste`.** Treat it as one ablation cell, not a default.
- `multi_scale=True`: essentially a wash in xuanandsix's runs (0.496/0.396 default vs 0.492/0.399 multi-scale), though Ultralytics staff recommend it.
- [Select-Mosaic (arXiv 2406.05412)](https://arxiv.org/pdf/2406.05412) argues standard mosaic is *not optimized* for dense small-object scenes and beats it on VisDrone — an argument for tuning mosaic, not disabling it.

**Also relevant:** `val_iou=0.6` (NMS IoU at validation, vs default 0.7) was worth **+0.9 val mAP50 / +0.6 test mAP50** in xuanandsix's best config. On ~70-objects-per-image scenes, NMS settings are not free. Fix `iou` and `max_det` across *all* compression variants or you will confound NMS tuning with compression damage. Note `max_det` defaults to 300 while VisDrone images can hold 500+ objects — **raise `max_det` to at least 500** and hold it constant, or you will cap recall on the densest images and mis-attribute the loss to compression.

---

## Recommended baseline recipe (synthesised)

```bash
# Headline config — best mAP-per-GPU-hour on 6 GB
yolo detect train data=VisDrone.yaml model=yolo11s.pt \
     imgsz=960 epochs=200 batch=-1 patience=50 \
     optimizer=SGD lr0=0.01 cos_lr=True \
     mosaic=1.0 close_mosaic=10 mixup=0.4 \
     cache=False amp=True

# Evaluation — ALWAYS with save_json for the area splits
yolo detect val model=runs/detect/train/weights/best.pt \
     data=VisDrone.yaml split=test imgsz=960 \
     save_json=True iou=0.7 max_det=500
```

- Also run `split=val` for the development-loop numbers; report both columns.
- Pin `ultralytics>=8.4.133`.
- Run a 20–30 epoch sanity job first to settle batch size at your chosen imgsz.
- Hold `iou`, `max_det`, `imgsz` and `conf` **identical** across every compression variant.

**Expected baseline** (YOLO11s / YOLO26s, COCO-pretrained):

| Config | val mAP50 | val mAP50-95 | test-dev mAP50 | test-dev mAP50-95 |
|---|---|---|---|---|
| 640, 100–300 ep | 38–43 | 23–26 | 31–35 | 18–21 |
| 960, 200 ep | 48–52 | 30–33 | 39–42 | 23–26 |
| 1280, 200 ep | 52–56 | 33–36 | — | — |

---

## Open questions / what I could not establish

- **RESOLVED, no longer open:** whether category-0/category-11 rows always carry `score = 0`. Verified by downloading all three splits and counting every annotation row — they do, with zero exceptions (§3.2). The Ultralytics converter is safe.
- **No quantified A/B of identical predictions scored by Ultralytics vs `evalDET.m`.** The five protocol deltas are derivable from both codebases (§3.3), and the *direction* is solid (Ultralytics biased low), but nobody has measured the gap. Searched Ultralytics issues #17536, #18031, #4047, #6876, #12758, yolov5 #2340/#10631, discussion #1594, plus web/arXiv. **Running this A/B once on your own baseline is cheap and is itself a publishable methodological note.**
- **No official acknowledgement of the `evalClass` image-frequency weighting** in `calcAccuracy.m`. That is a code reading only.
- **No official Python port of the DET toolkit** in the [VisDrone org](https://github.com/VisDrone) — only the MATLAB toolkit. Third-party ports exist (e.g. [dronefreak/VisDrone-dataset-python-toolkit](https://github.com/dronefreak/VisDrone-dataset-python-toolkit)) but none was verified against `dropObjectsInIgr.m`'s 50%-area rule. Diff before trusting one.
- **No official Ultralytics VRAM-vs-imgsz table for 6 GB cards.** The batch-size figures in §1.4 are extrapolated from measurements on 16–24 GB cards and are flagged as such.
- **No single paper ablates 640 vs 960 vs 1280 for one s-scale model in one table with a fixed recipe.** The closest are MFP-YOLO (640 vs 1024, but from scratch), xuanandsix (640 vs 960, YOLOv8s, COCO-pretrained), and the forum's 1280→640 anecdote. **Running that ablation on YOLO11s + YOLO26s is a genuine, currently-unpublished contribution** — cheap to do and worth a table.
- **No published APs/APm/APl numbers for a YOLOv8s/YOLO11s/YOLO26s baseline on VisDrone.** The EdgeYOLO table (§5.5) is YOLOX-X/EdgeYOLO. Searched for s-scale YOLO area splits on VisDrone and found none. Your baseline's APs/APm/APl row would itself be a novel reference point.
- **No published measurement of how quantization/pruning damage distributes across object size on VisDrone.** That is exactly this study's claimed contribution — good news for novelty, but it means there is no prior estimate of effect size to plan around. Related work is ticket 08.
- MDPI *Appl. Sci.* 16(7):3559 returned HTTP 403 on both article and PDF; its reported absolutes are irreconcilable with all other sources and are excluded.
- CF-YOLO (Sci Rep s41598-025-99634-0 / PMC12078580) was behind an IDP redirect / reCAPTCHA and was not extracted.
- The Ultralytics Platform run `visdrone-search-b1-640px-baseline` returned HTTP 403 (login required).

---

## Source index

**Primary — official docs / repos / source code**
- [VisDrone-Dataset (official)](https://github.com/VisDrone/VisDrone-Dataset)
- [VisDrone2018-DET-toolkit (official evaluator)](https://github.com/VisDrone/VisDrone2018-DET-toolkit) — and its sources: [`evalDET.m`](https://raw.githubusercontent.com/VisDrone/VisDrone2018-DET-toolkit/master/evalDET.m) · [`dropObjectsInIgr.m`](https://raw.githubusercontent.com/VisDrone/VisDrone2018-DET-toolkit/master/utils/dropObjectsInIgr.m) · [`saveAnnoRes.m`](https://raw.githubusercontent.com/VisDrone/VisDrone2018-DET-toolkit/master/utils/saveAnnoRes.m) · [`calcAccuracy.m`](https://raw.githubusercontent.com/VisDrone/VisDrone2018-DET-toolkit/master/utils/calcAccuracy.m) · [`evalRes.m`](https://raw.githubusercontent.com/VisDrone/VisDrone2018-DET-toolkit/master/utils/evalRes.m) · [`VOCap.m`](https://raw.githubusercontent.com/VisDrone/VisDrone2018-DET-toolkit/master/utils/VOCap.m)
- VisDrone challenge site: [test guidelines](https://aiskyeye.com/evaluate/test-guidelines-2022/) · [FAQ](https://aiskyeye.com/faq/) · [evaluate](https://aiskyeye.com/evaluate/object-detection-2022/) · [leaderboard](https://aiskyeye.com/leaderboard/)
- [Ultralytics `utils/metrics.py` (AP integration)](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/utils/metrics.py) · [`cfg/default.yaml`](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/default.yaml) · [legacy yolov5 `VisDrone.yaml`](https://raw.githubusercontent.com/ultralytics/yolov5/master/data/VisDrone.yaml)
- [Ultralytics `VisDrone.yaml` (raw source)](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/VisDrone.yaml)
- [Ultralytics `models/yolo/detect/val.py` (raw source)](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/models/yolo/detect/val.py)
- [Ultralytics `utils/autobatch.py` (raw source)](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/utils/autobatch.py)
- [Ultralytics VisDrone docs](https://docs.ultralytics.com/datasets/detect/visdrone/) · [val mode](https://docs.ultralytics.com/modes/val/) · [usage/cfg](https://docs.ultralytics.com/usage/cfg/) · [data augmentation](https://docs.ultralytics.com/guides/yolo-data-augmentation/) · [training tips](https://docs.ultralytics.com/guides/model-training-tips/) · [SAHI guide](https://docs.ultralytics.com/guides/sahi-tiled-inference/) · [COCO-JSON training](https://docs.ultralytics.com/guides/coco-json-training/)
- [YOLO11 model card (COCO table)](https://docs.ultralytics.com/models/yolo11/) · [YOLO26 model card (COCO table)](https://docs.ultralytics.com/models/yolo26/)
- [ultralytics v8.4.133 release notes (PR #25981)](https://newreleases.io/project/github/ultralytics/ultralytics/release/v8.4.133)
- [Ultralytics community thread 1614 (VisDrone epochs/imgsz)](https://community.ultralytics.com/t/standard-epochs-and-imgsz-for-training-yolo11-yolov12-on-visdrone-dataset/1614)
- Ultralytics issues [#6043](https://github.com/ultralytics/ultralytics/issues/6043) · [#13363](https://github.com/ultralytics/ultralytics/issues/13363) · [#17488](https://github.com/ultralytics/ultralytics/issues/17488) · [#18849](https://github.com/ultralytics/ultralytics/issues/18849) · [#19917](https://github.com/ultralytics/ultralytics/issues/19917)

**Papers with numbers**
- [arXiv 2605.24831 — YOLO26 vs YOLOv8 benchmark (VisDrone test, all scales)](https://arxiv.org/html/2605.24831)
- [arXiv 2302.07483 — EdgeYOLO (VisDrone val APs/APm/APl)](https://ar5iv.labs.arxiv.org/html/2302.07483)
- [arXiv 2202.06934 — SAHI (VisDrone val, area-split table)](https://arxiv.org/abs/2202.06934)
- [arXiv 2203.04799 — Keleş et al., sliced inference with YOLO on VisDrone test-dev](https://arxiv.org/abs/2203.04799)
- [arXiv 2604.19233 — ASAHI (SAHI throughput cost)](https://arxiv.org/html/2604.19233v1)
- [arXiv 2604.13278 — DroneScan-YOLO (1280px, YOLOv8s baseline)](https://arxiv.org/html/2604.13278v1)
- [arXiv 2411.11477v3 — SL-YOLO (YOLOv8s 600-epoch baseline)](https://arxiv.org/html/2411.11477v3)
- [arXiv 2606.09081 — VisDrone object-size statistics](https://arxiv.org/html/2606.09081)
- [arXiv 2505.20612 — Roboflow100-VL (Ultralytics vs pycocotools mAP)](https://arxiv.org/html/2505.20612v2)
- [arXiv 2108.11539 — TPH-YOLOv5 (1536px, batch 2)](https://arxiv.org/pdf/2108.11539)
- [arXiv 2504.18136 — MASF-YOLO](https://arxiv.org/pdf/2504.18136)
- [arXiv 2406.05412 — Select-Mosaic](https://arxiv.org/pdf/2406.05412)
- [PMC12608591 — RTUAV-YOLO (YOLO11s / YOLOv8s val baselines)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12608591/)
- [PMC12942582 — YOLO11s-UAV (YOLO11s val + test baselines)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12942582/)
- [PMC10346989 — MFP-YOLO, Sensors 2023 (640 vs 1024 ablation)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10346989/)
- [Sci Rep s41598-025-00341-7 — LW-YOLOv8s](https://www.nature.com/articles/s41598-025-00341-7)

**Reproducible community runs**
- [xuanandsix/VisDrone-yolov8 — imgsz + augmentation sweep with val AND test numbers](https://github.com/xuanandsix/VisDrone-yolov8)
- [dronefreak/visdrone-yolov11s — HF model card, per-class AP](https://huggingface.co/dronefreak/visdrone-yolov11s)
