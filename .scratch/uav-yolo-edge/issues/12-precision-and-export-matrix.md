# Precision and export matrix

Type: grilling
Status: open
Blocked by: none

## Question

What precisely does the `quantized` configuration MEAN on each platform, given that the available precisions are not the same across runtimes?

The YOLO26 reality check destroyed the assumption that `quantized` is one uniform treatment applied four times. Constraints now known:

- **NCNN has no INT8 export path in Ultralytics** - FP32 and FP16 only. INT8 on NCNN requires a manual `ncnn2int8` step with its own calibration, which is extra work nobody has budgeted.
- **LiteRT has no FP16 path** for this purpose.
- The **NMS-free head is auto-disabled** for NCNN at any precision, for LiteRT INT8, for TensorRT <8.5, and for TensorRT 10.3 on JetPack 6 with INT8.
- **QAT is first-party** now (`quantize=8` at train time) but emits Q/DQ nodes only for `onnx` and `engine`; other formats reject a QAT checkpoint outright. So QAT is Jetson-and-laptop-only.

Decisions to reach:

1. **Fill in the real grid**: for each platform x runtime, which precisions are actually reachable, and with how much manual work.
2. **Is the manual `ncnn2int8` path worth it?** NCNN is the fastest Pi 5 runtime (~2.8x over PyTorch) and the Pi is the primary platform, so "our fastest runtime has no INT8" is a conspicuous hole. Weigh the manual calibration work against dropping INT8-on-NCNN and reporting FP16 there.
3. **Decide what gets compared to what.** If the Pi runs NCNN-FP16 and the Jetson runs TensorRT-INT8, a cross-platform "quantized" column is comparing different things. Either normalise, or report per-platform and say so loudly.
4. **QAT scope** - in or out, given it only reaches two of four platforms and would therefore be another non-uniform column.
5. **Is `quantized` one column or several?** It may be more honest to split it into explicit named conditions per platform than to pretend one label covers them all.

This ticket is upstream of the harness design: the table schema in *Measurement protocol* has to carry precision and runtime as first-class columns, not footnotes.
