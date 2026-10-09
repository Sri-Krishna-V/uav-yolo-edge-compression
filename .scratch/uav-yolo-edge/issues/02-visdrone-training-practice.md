# VisDrone training practice

Type: research
Status: resolved
Blocked by: none

## Question

What does a defensible VisDrone2019-DET training recipe look like for an s-scale YOLO, and what baseline mAP should we expect?

Establish, with sources:

1. **Input resolution.** What `imgsz` do published VisDrone YOLO results use (640 / 960 / 1024 / 1280)? VisDrone objects are tiny and dense, so resolution is likely the single largest mAP lever. Find the reported mAP-vs-imgsz relationship if anyone has published it.
2. **Realistic baselines.** What mAP@0.5 and mAP@0.5:0.95 do YOLOv8s / YOLO11s-class models actually reach on the VisDrone val and test-dev splits? Needed to sanity-check our own baseline. NOTE: the proposal deck currently cites ~40.1 / ~47.8 mAP50-95 from the Ultralytics Pi 5 guide, but those are COCO numbers, not VisDrone. Find the real VisDrone figures.
3. **Annotation handling.** VisDrone ships an `ignored regions` category (class 0) and an `others` category (class 11) alongside the 10 evaluated classes. How do published works and the Ultralytics `VisDrone.yaml` converter handle them? Getting this wrong silently corrupts mAP.
4. **Split discipline.** `val` (548) vs `test-dev` (1,610) vs `test-challenge` - which split do papers report on, and are test-dev labels public?
5. **Small-object evaluation.** How is mAP broken out by object area (AP-small / AP-medium / AP-large) on VisDrone, and does the standard COCO-style eval give us that for free? The paper central claim depends on measuring small-object degradation specifically.
6. **Epochs and augmentation** typical for VisDrone fine-tuning from COCO weights.

Write findings to `.scratch/uav-yolo-edge/findings/02-visdrone.md` and summarise under an `## Answer` heading here.

## Answer

Full findings: `findings/02-visdrone.md`.

**2. The deck is wrong, and by a lot.** `40.1 / 47.8` are verbatim the YOLO26n / YOLO26s **COCO val** mAP50-95 (e2e) figures from the Ultralytics model card. Real VisDrone @640, s-scale, COCO-pretrained: **val 38-43 mAP50 / 23-26 mAP50-95**, **test-dev 31-35 / 18-21**. Specifically YOLO26s test-dev 32.5/18.2, YOLOv8s 31.5/17.9 (arXiv 2605.24831); YOLO11s val 39.4/23.6 (RTUAV-YOLO). Budget ~40% of the COCO number. Also: YOLO26s beats YOLOv8s by only +1.0 mAP50 and measured *slower* on both CPU and GPU.

**5. Area splits: YES, free — with a version floor.** The official MATLAB toolkit emits AP/AP50/AP75/AR{1,10,100,500} and **never** area splits; every published VisDrone APs came from a private COCO conversion. But Ultralytics **>= 8.4.133** (PR #25981) writes `metrics/mAP_small|medium|large(B)` for any custom detection dataset when `save_json=True`, via faster-coco-eval, with areas in **native image pixels** — so directly comparable to published APs. Pin the version. Baseline spread is wide (EdgeYOLO val: APs 16.3 vs APl 53.1), so the asymmetry is measurable.

**1. Resolution is the biggest lever** — bigger than model scale. YOLOv8s val 640->960: 40.4->49.6 mAP50 (+9.2); 640->1024 is +8 to +12 across 8 models. `yolov8s@960` beats `yolov8l@640`. On 6GB: ~batch 8-16@640, ~4@1024, ~2-4@1280.

**3. Converter is safe.** It filters on `score`, not category — but verified over all three splits, `score==0` is exactly `category in {0,11}`, so 10 clean classes result. The risk is at *eval* time: no ignored-region masking, `max_det=300` vs official 500, different class averaging. Set `max_det=500`.

**4. Splits.** test-dev labels public; test-challenge withheld, server dormant since 2024. Authors recommend test-dev. val->test-dev gap ~6 mAP50. Headline test-dev, tune on val.

**6. Epochs/aug.** 100-300, COCO-pretrained. 600 epochs buys +4 mAP50 at 6x cost — spend GPU-hours on resolution first. SGD/0.01/cosine; keep mosaic on; `mixup=0.4` helped, `copy_paste` disputed.
