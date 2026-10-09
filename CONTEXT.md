# CONTEXT

Glossary for the UAV YOLO edge-compression study. Terms only - no implementation detail, no decisions. Decisions live on the map at `.scratch/uav-yolo-edge/map.md`.

## Model

A **model** is one architecture at one scale, e.g. `YOLO11s`. This study has two **study models** (YOLO11s, YOLO26s) and two **teachers** (YOLO11m, YOLO26m). Teachers exist only to distill from and are never deployed or benchmarked.

## Configuration

A **configuration** is one point in the compression pipeline applied to a study model. Five are defined:

- `baseline` - FP32, fine-tuned on VisDrone, no compression
- `quantized` - PTQ (or QAT) to FP16 or INT8
- `pruned` - structured channel pruning plus recovery fine-tuning
- `distilled` - trained with a teacher
- `combined` - pruned and quantized, then distilled

"Compressed" is the cover term for the four non-baseline configurations.

## Platform vs runtime

A **platform** is a physical device: Raspberry Pi 5, Jetson, Galaxy M35, laptop CPU. A **runtime** is the inference engine and model format used on it: TensorRT, NCNN, TFLite/LiteRT, ONNX Runtime, PyTorch. One platform may support several runtimes, and the runtime is part of the experimental condition, never an implementation detail - runtime choice alone moves Pi 5 throughput by ~2.8x.

Do not write "device" when you mean platform, or "format" when you mean runtime.

## Cell

A **cell** is one measured combination: model x configuration x platform (x runtime, where more than one is benchmarked). The **matrix** is the set of cells the study commits to running. Cells are the unit of scope cutting.

## Input resolution

The **input resolution** is the square image size a model is trained, evaluated, and exported at. Like the runtime, it is part of a cell's experimental condition.

The **matrix resolution** is the one input resolution that every compressed configuration uses.

The **resolution study** is a set of baselines trained at several input resolutions, to show what accuracy and latency each resolution buys. Its extra baselines are not cells of the compression matrix.

The **reference baseline** is the baseline at the matrix resolution. Every compressed configuration is measured against it. Where resolution matters, "the baseline" means the reference baseline.

## Small-object degradation

The hypothesised effect at the centre of the paper: compression that costs only a few points of aggregate mAP costs disproportionately more **AP-small**, because VisDrone is dominated by tiny aerial objects that COCO-calibrated compression recipes do not protect. Always stated in terms of COCO-style area splits (AP-small / AP-medium / AP-large), never as aggregate mAP alone.

## Study models vs literature anchor

The study evaluates s-scale models only. Any n-scale run exists purely as a **literature anchor** - a reference point so readers can align these numbers with published COCO and VisDrone results. An anchor is not a study cell and is not part of any claim.

## Dataset splits

VisDrone2019-DET: `train` (6,471), `val` (548), `test-dev` (1,610). The split that results are reported on is fixed once and applies to every cell. VisDrone also ships an `ignored regions` category and an `others` category that sit outside the 10 evaluated classes.

## Destination vs execution

This repo is currently running a **wayfinder map**: the work in progress is deciding *how* to run the study, not running it. Training runs, exports, and benchmarks are **execution**, and begin only once the map is clear.
