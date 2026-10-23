# UAV YOLO edge-compression study

Glossary for the UAV YOLO edge-compression study. Terms only - no implementation detail, no decisions. Decisions live on the map, [issue #1](https://github.com/Sri-Krishna-V/uav-yolo-edge-compression/issues/1).

## Model

A **model** is one architecture at one scale, e.g. `YOLO11s`. This study has two **study models** (YOLO11s, YOLO26s) and two **teachers** (YOLO11m, YOLO26m). Teachers exist only to distill from and are never deployed or benchmarked.

## Configuration

A **configuration** is one point in the compression pipeline applied to a study model. Five are defined:

- `baseline` - fine-tuned on VisDrone, no compression step; it runs at its runtime's default precision on that platform
- `quantized` - post-training quantization (or QAT) to INT8, with calibration
- `pruned` - structured channel pruning plus recovery fine-tuning
- `distilled` - trained with a teacher
- `combined` - pruned, recovered with a teacher (distillation does the recovery), then quantized to INT8

"Compressed" is the cover term for the four non-baseline configurations.

The **pruning ratio** of a pruned configuration is the fraction of channels removed from each prunable layer. It is not the fraction of parameters removed: the two differ, and by a different amount for each study model.
_Avoid_: "30% smaller" for a 30% pruning ratio.

## Platform vs runtime

A **platform** is a physical device: Raspberry Pi 5, Raspberry Pi 4B, Jetson, Galaxy M34, laptop CPU. A **runtime** is the inference engine and model format used on it: TensorRT, NCNN, TFLite/LiteRT, ONNX Runtime, PyTorch. One platform may support several runtimes, and the runtime is part of the experimental condition, never an implementation detail - runtime choice alone moves Pi 5 throughput by ~2.8x.

Do not write "device" when you mean platform, or "format" when you mean runtime.

The **precision** is the arithmetic a runtime computes in on a platform: FP32, FP16 or INT8. A baseline's precision is its runtime's default on that platform, so it is FP16 where that default is FP16 (NCNN on the Pi 5, TensorRT on the Jetson). Precision is part of a cell's experimental condition and is always stated with the cell.
_Avoid_: calling an FP16 baseline "quantized"; FP16 is a runtime default here, never a configuration.

## Head

A **one-to-many head** predicts many overlapping boxes per object and needs NMS to keep one. Every cell runs it unless stated otherwise. YOLO26 also has an **NMS-free head** (its one-to-one head), which predicts one box per object and needs no NMS.
_Avoid_: "end2end" or "e2e" in prose.

## Latency and confidence thresholds

**Latency** is the end-to-end time for one image at batch 1 on one platform: pre-processing, the runtime call and post-processing, measured at the deployment threshold. Reading and decoding the image file is not part of it. Every compression budget and equal-latency comparison is in latency.

The **evaluation threshold** is the low confidence threshold accuracy is measured at, so that every low-score box counts. The **deployment threshold** is the confidence threshold a deployed detector uses, and the one latency is measured at. NMS cost depends on the threshold, so a post-processing time is always stated with its threshold.

## Rail sum

The **rail sum** is the Pi 5's on-board power as its PMIC reports it: the sum of voltage × current over its output rails. It is not wall power, which also includes what bypasses those rails.
_Avoid_: "power draw" or "wall power" for a rail sum.

## Cell

A **cell** is one measured combination: model x configuration x platform (x runtime, where more than one is benchmarked). The **matrix** is the set of cells the study commits to running. Cells are the unit of scope cutting.

## Input resolution

The **input resolution** is the square image size a model is trained, evaluated, and exported at. Like the runtime, it is part of a cell's experimental condition.

The **matrix resolution** is the one input resolution at which every configuration is run. Cells at any other input resolution exist only for equal-latency comparisons.

The **resolution study** is the set of baselines trained at input resolutions above the matrix resolution. These baselines, and the compressed configurations built from them, are cells: the equal-latency comparisons are made from them.

The **reference baseline** is the baseline at the matrix resolution. Every compressed configuration at the matrix resolution is measured against it. Where resolution matters, "the baseline" means the reference baseline.

## Compression budget and equal-latency comparison

The **compression budget** is the latency that a compressed configuration saves against the baseline at the same input resolution, on one platform. It is always measured latency, never FLOPs or parameters.

An **equal-latency comparison** puts two cells on one platform side by side at the same measured latency: typically a compressed configuration at a larger input resolution against a baseline at a smaller one. It asks whether a compression budget is better spent on input resolution.
_Avoid_: iso-latency, matched-latency.

## Small-object degradation

The hypothesised effect that compression that costs only a few points of aggregate mAP costs disproportionately more **AP-small**, because VisDrone is dominated by tiny aerial objects that COCO-calibrated compression recipes do not protect. Always stated in terms of COCO-style area splits (AP-small / AP-medium / AP-large), never as aggregate mAP alone.

## Study models vs literature anchor

The study evaluates s-scale models only. Any n-scale run exists purely as a **literature anchor** - a reference point so readers can align these numbers with published COCO and VisDrone results. An anchor is not a study cell and is not part of any claim.

## Dataset splits

VisDrone2019-DET: `train` (6,471), `val` (548), `test-dev` (1,610). The split that results are reported on is fixed once and applies to every cell. VisDrone also ships an `ignored regions` category and an `others` category that sit outside the 10 evaluated classes.

## Destination vs execution

The **map** decides _how_ the study runs; **execution** is running it: training runs, exports, and benchmarks. Baseline training executes alongside the map (since 2026-10-09, once the recipe was frozen); exports and benchmarks each wait for the decision ticket that governs them.
