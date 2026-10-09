# YOLO26 reality check

Type: research
Status: resolved
Blocked by: none

## Question

Is YOLO26 actually usable as a study target, and does it survive the compression pipeline this study depends on?

Establish, with sources:

1. **Availability** - is YOLO26 released in the `ultralytics` package, at what version, and are `yolo26s.pt` / `yolo26m.pt` COCO-pretrained weights publicly downloadable? If it is preview or partial, say exactly what exists.
2. **Licence** - confirm the licence on the weights and the package (AGPL-3.0 vs enterprise). This constrains what the paper can release as an open artifact.
3. **Architecture** - YOLO26 is reported as end-to-end and NMS-free (DFL removed, ProgLoss/STAL). Confirm what actually changed versus YOLO11, and identify the detection head structure.
4. **Export coverage** - which of ONNX, NCNN, TFLite/LiteRT, TensorRT are supported for YOLO26s, at what opset, and are there known failures.
5. **Compression compatibility - the load-bearing part.** Does the NMS-free head break (a) standard PTQ INT8 calibration, (b) structured channel pruning, (c) output-level knowledge distillation, which normally assumes NMS-based class/box logits? Flag any known workaround or published precedent.
6. **Published VisDrone or UAV results** for YOLO26 at any scale, if any exist.

Report findings with citations to primary sources (Ultralytics docs, GitHub issues, papers). Where the answer is unknown or not yet published, say so plainly rather than inferring.

Write findings to `.scratch/uav-yolo-edge/findings/01-yolo26.md` and summarise under an `## Answer` heading here.


## Answer

**Usable, yes. But its headline feature does not survive the pipeline.**

Full findings: `findings/01-yolo26.md`.

1. **Released** in `ultralytics` **v8.4.0, 2026-01-14**. `yolo26n/s/m/l/x.pt` COCO-pretrained, public, auto-downloading. Not preview.
2. **AGPL-3.0** on both package and weights (Enterprise alternative). Release the artifact AGPL-3.0; don't make legal claims about weights-as-derivative-work.
3. **Architecture**: same `Detect` class and near-identical YOLO11 backbone/neck. The entire diff is `reg_max: 1` (DFL -> `nn.Identity`), `end2end: True` (duplicates the head into a one-to-one branch), and three C3k2/SPPF arg changes. ProgLoss/STAL/MuSGD are training-only. **YOLO26s is bigger than YOLO11s**: 10.0M/22.8 GFLOPs vs 9.5M/21.7.
4. **Export**: ONNX (opset 18; ORT static INT8 rejects opset>=21), TensorRT, LiteRT, NCNN all work. Two plan-breaking gaps: **NCNN has no INT8 export path in Ultralytics**, and **LiteRT has no FP16**. Issue #23282 (YOLO26 TFLite on Android GPU delegate: GATHER_ND/CAST-INT64) closed *not planned*.
5. **Compression — the load-bearing answer. Nothing is broken, because the NMS-free head is automatically switched off.** `exporter.py` disables the end2end branch for `ncnn`/`rknn`/`edgetpu`/`qnn`/etc. at any precision ("formats without top-k support"), for LiteRT `quantize in {8,"w8a16"}` ("static activation quantization collapses the end2end class-index output"), and for TensorRT <8.5.0 and TRT 10.3 INT8 on JetPack 6. **So on Pi 5, the M35, and Jetson INT8, NMS-free is simply unavailable** — you get a YOLO11-shaped one-to-many head (which is also 0.6-0.8 mAP *more* accurate). PTQ is therefore no harder than YOLO11, and arguably easier (no DFL softmax). Distillation is solved: first-party KD shipped v8.4.77 is **feature-based on the three neck layers**, head-agnostic, same-family, worth ~+0.6 mAP. Pruning is the only genuine unknown — **zero published YOLO26 pruning precedent**; de-risk with a DepGraph trace on a fused `end2end=False` model in week 1. Note `reg_max=1` makes **LD/DFL-distribution distillation inapplicable** to YOLO26.
6. **VisDrone**: arXiv:2605.24831v2 has a full table — YOLO26s .325 mAP50 / .182 mAP50-95 @640, vs YOLOv8s .315/.179. YOLO26n *loses* to YOLOv8n. No YOLO11-controlled VisDrone comparison exists, and no YOLO26 VisDrone result under compression on any edge device. **That gap is the paper.**
