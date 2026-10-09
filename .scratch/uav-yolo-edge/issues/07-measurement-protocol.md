# Measurement protocol

Type: grilling
Status: open
Blocked by: none

## Question

Define exactly what gets measured and how, so that every cell of the matrix is comparable across four platforms, four runtimes, and five configurations.

This is where benchmark papers get shredded in review. Decisions to reach:

1. **What latency includes.** Pure model forward pass, or the full pipeline with letterbox preprocessing, NMS, and postprocessing? These differ by 2-3x on a CPU target, and YOLO26 being NMS-free makes the comparison unfair by construction unless the boundary is fixed and stated.
2. **What FPS means.** 1/latency single-stream, or sustained throughput with batching and pipelining? Single-stream is the honest choice for UAV, but say so.
3. **Warmup.** How many discarded iterations before timing starts, since runtimes JIT and cache on first calls.
4. **Thermal soak.** The Pi 5 and the phone both throttle. Does each configuration run to thermal steady state before timing, is there a cooldown between configurations, and is ambient temperature recorded? Without this, benchmark ORDER silently determines results.
5. **Repeats and variance.** How many runs per cell, and is the reported figure mean, median, or p95? What dispersion is reported? A benchmark paper with no error bars gets desk-rejected.
6. **Accuracy evaluation location.** Is mAP computed once on a workstation using the exported model, or re-measured on-device? On-device mAP over 1,610 images at ~3 FPS is about 9 minutes per configuration - affordable, and much more defensible, since it proves the deployed artifact is the one measured.
7. **Memory and utilisation.** Which counters, sampled how often, and peak vs mean.
8. **Energy.** How joules-per-frame is derived, at what sampling rate, and whether idle baseline is subtracted.

9. **FLOPs and parameters are not results.** TensorRT issue #3884 records 20% structured pruning of YOLOv8s producing ZERO FPS gain on a Jetson Nano, and NVIDIA ModelOpt reports 34% FLOPs reduction buying only 16% latency. Decide that every compression claim is reported in MEASURED DEVICE LATENCY, with FLOPs and parameter counts as secondary descriptive columns only. This is a protocol rule because it must bind every cell uniformly rather than be argued per result.
10. **Schema carries runtime and precision as first-class columns** - see Precision and export matrix. A row labelled only `quantized` is not interpretable when NCNN offers no INT8 and LiteRT offers no FP16.

Output is a written protocol that the benchmark harness implements literally, plus the table schema every result row conforms to.
