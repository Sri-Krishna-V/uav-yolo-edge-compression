# Findings: can the department's Coral USB Accelerator and Coral Dev Board be used in the study, and how?

Ticket: none yet (to be filed).
Researched: 2026-10-28. Paper due 2026-10-31.

**Question.** The department has a Coral USB Accelerator and a Coral Dev Board. Can either one add Edge TPU cells to
the study (YOLO11s and YOLO26s on VisDrone, Ultralytics 8.4.149, INT8 post-training quantization calibrated on the
548 val images, matrix resolution 640, main platform a Raspberry Pi 5 at 2000 MHz)? What would it take, what would
the numbers mean, and does it fit before results freeze on 31 October?

**Verdict in one line:** the USB Accelerator on the Pi 5 is usable in principle, but not cheaply. The pinned exporter
builds Edge TPU models only on x86_64 Linux (Kaggle), and at the pin it needs two hand fixes: Google's compiler apt
repository is gone (403), and Kaggle's Python 3.13 makes the exporter pick an onnx2tf release that leaves 668 of 688
ops on the CPU. Nobody has published an s-size YOLO at 640 on the Edge TPU. Both s models are larger than the ~7 MiB
on-chip parameter cache. Ultralytics' own Pi 5 numbers, extrapolated, put a YOLOv8s-class model at 640 near or above
NCNN's measured 171 ms. The TPU's power is invisible to our energy method. A full Pi 5 column costs about 11–20 dev
hours plus Pi 5 time that #37 also needs. A Kaggle-only compile note (op mapping and memory per model, no Pi time) or
future work fits the three days. The Dev Board (Debian 10, Python 3.7, Cortex-A53) cannot run the study's bench stack
and adds nothing the USB Accelerator does not.

Legend: ✅ = verified against the primary source itself (official doc, source code at the tag, release page, PyPI
metadata, wheel contents, or paper full text). ⚠️ = verified only from an abstract, metadata or a secondary page, or
partly unverifiable, or my own inference/extrapolation. ❌ = claim did not hold up.

---

## 0. Key facts at a glance

| # | Fact | Status | Source |
|---|---|---|---|
| 1 | At 8.4.149, Edge TPU export runs PyTorch → ONNX → **onnx2tf** → `*_full_integer_quant.tflite` → `edgetpu_compiler`. It is **not** the `litert` (litert_torch) path. x86_64 Linux only. | ✅ | [exporter.py L606, L820-827, L972-977](https://github.com/ultralytics/ultralytics/blob/v8.4.149/ultralytics/engine/exporter.py#L820-L827) |
| 2 | The exporter installs the compiler from `packages.cloud.google.com/apt coral-edgetpu-stable`. That repository now returns **403** (checked today). Fixed on `main` on 2026-09-30, after the pin (2026-09-12). | ✅ | [tensorflow.py L253-269 at tag](https://github.com/ultralytics/ultralytics/blob/v8.4.149/ultralytics/utils/export/tensorflow.py#L253-L269), [PR #26456](https://github.com/ultralytics/ultralytics/pull/26456) |
| 3 | On Python ≥3.13 the pin selects `onnx2tf>=2.3.0,<2.3.16`. With 2.3.15, YOLO26n at 384 put **20 of 688** ops on the TPU. With onnx2tf 1.28.x it put **652 of 658**. **Our Kaggle machines record Python 3.13.15.** | ✅ | [tensorflow.py L105, L115](https://github.com/ultralytics/ultralytics/blob/v8.4.149/ultralytics/utils/export/tensorflow.py#L105), [#26847](https://github.com/ultralytics/ultralytics/issues/26847), [PR #26849](https://github.com/ultralytics/ultralytics/pull/26849), `study/verification/smoke_results-kaggle-*.json` |
| 4 | "roughly 8 MB of SRAM" caches parameters. Compiler logs show 7.0–7.5 MiB actually available. YOLO11s (9.4 M params) and YOLO26s (9.5 M) will stream several MiB off-chip. | ✅ doc, ⚠️ estimate | [compiler doc](https://coral.ai/docs/edgetpu/compiler/), §3 |
| 5 | No published Edge TPU number exists for any YOLOv8/11/26 **s** model at **640**. Ultralytics' table stops at 512; papers use 320 or 224. | ✅ | §6 |
| 6 | The Pi 5's PMIC rail sum leaves out USB current, so the Accelerator's power never appears in our J/frame. | ✅ | [vcgencmd.1](https://github.com/raspberrypi/utils/blob/master/vcgencmd/vcgencmd.1), INVARIANTS.md §4.6 |
| 7 | `tflite-runtime` (last 2.14.0) is built against NumPy 1.x and cannot share the bench venv (numpy 2.4.6, opencv 5.0.0.93 needs numpy ≥2). `ai-edge-litert` 2.3.0 is a NumPy 2 build and keeps `load_delegate`. | ✅ | §4 |
| 8 | Dev Board: newest OS is Mendel 5.3 (Nov 2021), Debian 10 (LTS ended 30 June 2024), Python 3.7, 4× Cortex-A53 at 1.5 GHz. | ✅ | §5 |

---

## 1. Edge TPU export at the pinned Ultralytics 8.4.149

All line numbers are from the tag `v8.4.149` (published 2026-09-12,
[release](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.149)).

| Claim | Status | What the source says |
|---|---|---|
| The path is TF SavedModel via onnx2tf, then the Edge TPU compiler | ✅ | `is_tf_format = fmt in {"saved_model", "pb", "edgetpu"}` (L606). For `edgetpu`, `export_saved_model()` runs, then `self.export_edgetpu(tflite_model=Path(f) / f"{self.file.stem}_full_integer_quant.tflite")` (L972-977). `export_saved_model` exports ONNX, then calls `onnx2saved_model` (L1421-1463). [exporter.py](https://github.com/ultralytics/ultralytics/blob/v8.4.149/ultralytics/engine/exporter.py#L972-L977), [tensorflow.py](https://github.com/ultralytics/ultralytics/blob/v8.4.149/ultralytics/utils/export/tensorflow.py#L68-L201) |
| It is not the study's LiteRT path | ✅ | `format="litert"` uses `torch2litert` (litert_torch) and asserts "LiteRT export only supported on Linux x86 and macOS" (L1260-1280). The Edge TPU path never calls it. So the LiteRT notes in `01-yolo26.md` (§5a) do not transfer as-is. |
| x86_64 Linux only | ✅ | `if not LINUX or ARM64: raise SystemError("Edge TPU export only supported on non-aarch64 Linux. ...")` (L821-823). The compiler doc: "x86-64 system architecture" and "no longer available for ARM64 systems (such as the Coral Dev Board), beginning with version 2.1". [compiler doc](https://coral.ai/docs/edgetpu/compiler/). The laptop (Windows) is out; Kaggle (x86_64 Linux) qualifies. |
| INT8 is forced; FP32 is refused | ✅ | `if fmt in {..., "edgetpu", ...} and self.args.quantize not in {8, "w8a16"}: LOGGER.warning(f"{fmt} export requires INT8 quantization, enabling it.")` (L626-628); `FP32_UNSUPPORTED_FORMATS` includes `edgetpu` (L438). |
| Batch is forced to 1 | ✅ | "Edge TPU export requires batch size 1, setting batch=1." (L824-827). |
| `imgsz` is not capped | ✅ | Only `check_imgsz(self.args.imgsz, stride=model.stride, min_dim=2)` (L720). No Edge-TPU-specific limit in the exporter. |
| Accepted arguments | ✅ | `["data", "fraction", "quantize", "opset"]` (L200-207). `studyexport.py` passes `batch=1, device="cpu"`; both equal the exporter defaults, so `validate_args` (L448-483) accepts them. |
| YOLO26 head | ✅ | `model.end2end = self.args.nms is False` (L683); then for `edgetpu` the one-to-one branch is disabled with "This export format does not support end2end models" (L685-688). With the study's `nms=None` both models export the one-to-many head, as INVARIANTS.md §5 requires. |
| Box decode is rewritten for TF formats | ✅ | `tf_wrapper` (L873-875) swaps in `_tf_decode_boxes`, which multiplies by `norm = self.strides / (self.stride[0] * grid_size)`: boxes come out as fractions of the input size, in the same 0–1 range as the class scores. The Edge TPU backend multiplies them back by `w`, `h`. [tensorflow.py L38-49](https://github.com/ultralytics/ultralytics/blob/v8.4.149/ultralytics/utils/export/tensorflow.py#L38-L49), [backends/tensorflow.py L137-147](https://github.com/ultralytics/ultralytics/blob/v8.4.149/ultralytics/nn/backends/tensorflow.py#L137-L147) |
| Graph edits for the Edge TPU | ✅ | C2f keeps `chunk` for TF formats: "EdgeTPU does not support FlexSplitV while split provides cleaner ONNX graph" (L911-913). `disable_group_convolution=self.args.format == "edgetpu"` (L1454). |
| Calibration | ✅ | The 548 VisDrone val images through `get_int8_calibration_dataloader` (letterboxed at `imgsz`), concatenated into one float array (`torch.cat(images, 0).float()`, L1436-1440), handed to onnx2tf with mean 0 and std 255 (tensorflow.py L146). onnx2tf defaults: `quant_type='per-channel'` weights, `int8` input and output ([onnx2tf README @1.28.8](https://github.com/PINTO0309/onnx2tf/blob/1.28.8/README.md)). That array is 548×3×640²×4 B ≈ 2.7 GB at 640, ≈ 6.1 GB at 960, ≈ 10.8 GB at 1280, plus an interpolated copy (my arithmetic from the code). |
| Toolchain pins at the tag | ✅ | `tensorflow>2.19.0` on Python ≥3.13, else `>=2.0.0,<=2.19.0` (L99); `onnx2tf>=2.3.0,<2.3.16` on ≥3.13, else `>=1.26.3,<1.29.0` (L105, L115); plus `tf_keras`, `sng4onnx`, `onnx_graphsurgeon`, `ai-edge-litert>=1.2.0`, `onnxslim`, protobuf. [tensorflow.py L96-125](https://github.com/ultralytics/ultralytics/blob/v8.4.149/ultralytics/utils/export/tensorflow.py#L96-L125) |
| How the compiler is installed at the tag | ✅ | If `edgetpu_compiler --version` fails, it adds `deb ... https://packages.cloud.google.com/apt coral-edgetpu-stable main` and installs `edgetpu-compiler` with apt (L253-270). |
| ❌ That install works today | ❌ | The repository answers 403: I got 403 for `.../dists/coral-edgetpu-stable/Release` and the package index on 2026-10-28. PR #26456: "Google's Coral apt repo (`packages.cloud.google.com/apt coral-edgetpu-stable`) now returns **403** for every client and path". Merged 2026-09-30, after the pin; `main` now downloads `edgetpu-compiler_16.0_amd64.tar.gz` from `ultralytics/assets` ("Google's Coral apt repo is gone", [main tensorflow.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/tensorflow.py)). **At the pin: put `edgetpu_compiler` on `PATH` first.** The archived [google-coral/edgetpu `compiler/x86_64`](https://github.com/google-coral/edgetpu/tree/master/compiler/x86_64) still holds the binary (19.6 MB) and its launcher script. The study also sets `YOLO_AUTOINSTALL=false` (`studyenv.py` L29). |
| Compiler version | ✅ | Every log found reports "Edge TPU Compiler version 16.0.384591198". The compiler doc's version table tops out at 16.0, which needs runtime 14 by default. [compiler doc, Table 3](https://coral.ai/docs/edgetpu/compiler/) |
| Compiler flags | ✅ | `--out_dir <dir> --show_operations --search_delegate --delegate_search_step 30 --timeout_sec 180` (L276-287). No `--min_runtime_version`, no `--num_segments`, no `--intermediate_tensors`. `--search_delegate` "automatically re-attempts to compile but will stop compilation before reaching the operation that failed" ([compiler doc](https://coral.ai/docs/edgetpu/compiler/)). The compiler writes a `.log` "with the same name and location as the compiled model". |
| Known issue 1: onnx2tf 2.x breaks mapping | ✅ | [#26847](https://github.com/ultralytics/ultralytics/issues/26847) (2026-10-23, Ultralytics 8.4.184, Python 3.13, TF 2.21.0, onnx2tf 2.3.15, YOLO26n at 384): "Number of operations that will run on Edge TPU: 20 / ... on CPU: 668". [PR #26849](https://github.com/ultralytics/ultralytics/pull/26849) (merged 2026-10-23): onnx2tf 2.x "inserts NHWC to NCHW `TRANSPOSE` ops around residual `Add` outputs. The Edge TPU compiler rejects these early in the graph". With onnx2tf 1.28.8 on Python 3.13 and TF 2.21.0: 652 TPU / 6 CPU. "Newer onnx2tf releases do not fix it." |
| What issue 1 means for us | ✅ / ⚠️ | ✅ Kaggle runs Python 3.13.15 (`study/verification/smoke_results-kaggle-krishna.json`, `-vidhyarth.json`), so the pinned exporter asks for onnx2tf 2.3.x. ⚠️ Workaround from reading the code, not tested: install `onnx2tf==1.28.8` and the TF stack by hand. With `YOLO_AUTOINSTALL=false`, `check_requirements` returns `False` without installing ([checks.py L604-625](https://github.com/ultralytics/ultralytics/blob/v8.4.149/ultralytics/utils/checks.py#L604-L625)), and `onnx2saved_model` ignores that return value, so the hand-installed 1.28.8 is used. The other route is a Python 3.12 venv on Kaggle, where the pin itself selects onnx2tf 1.x. |
| Known issue 2: the NMS-free head on the TPU | ✅ | [#23723](https://github.com/ultralytics/ultralytics/issues/23723) (2026-02): YOLO26n compile log full of `TOPK_V2`, `GATHER_ND`, `CAST`, `FLOOR_MOD` "not supported". That is the end-to-end head, which the pin disables for `edgetpu` anyway. |
| Ultralytics' own caveat | ✅ | "The Edge TPU compiler maps the operations it supports onto the accelerator and leaves the rest on the CPU, so task support does not mean every operation runs on the TPU." [docs/en/integrations/edge-tpu.md @v8.4.149](https://github.com/ultralytics/ultralytics/blob/v8.4.149/docs/en/integrations/edge-tpu.md) |
| No CI test | ✅ | `tests/test_exports.py` at the tag has no `edgetpu` test (grep: 0 hits). [test_exports.py](https://github.com/ultralytics/ultralytics/blob/v8.4.149/tests/test_exports.py) |

## 2. Which ops map to the TPU, and which fall to the CPU

**The rule** ([models-intro](https://coral.ai/docs/edgetpu/models-intro/), ✅):

- Models must be "fully 8-bit quantized". "Tensors are either 1-, 2-, or 3-dimensional. If a tensor has more than 3
  dimensions, then only the 3 innermost dimensions may have a size greater than 1."
- "At the first point in the model graph where an unsupported operation occurs, the compiler partitions the graph into
  two parts." And: "the Edge TPU compiler cannot partition the model more than once, so as soon as an unsupported
  operation occurs, that operation and everything after it executes on the CPU, even if supported operations occur
  later."
- "if even a small fraction of your model executes on the CPU, it can potentially slow the inference speed by an order
  of magnitude".

**The ops in a YOLO11/YOLO26 INT8 TFLite graph, against Coral's Table 1** (✅ for the doc text; the YOLO op list is
from the compile logs in the next table):

| TFLite op | Where in YOLO | Coral's limitation |
|---|---|---|
| CONV_2D, DEPTHWISE_CONV_2D | everywhere | same dilation in x and y |
| LOGISTIC + MUL | SiLU (x·sigmoid x), class sigmoid | none listed |
| ADD, SUB, MUL | residuals, box decode | none listed |
| CONCATENATION | C2f/C3k2, neck, head output | "No fused activation function"; constant inputs only as zero padding |
| SPLIT, STRIDED_SLICE | C3k2 chunk, slicing | Split: not in batch dim. StridedSlice: "all strides are equal to 1" |
| MAX_POOL_2D | SPPF | no fused activation |
| RESIZE_NEAREST_NEIGHBOR | neck upsample | "Depending on input/output size, this operation might not be mapped" |
| RESHAPE, TRANSPOSE | head, attention | Reshape: "Certain reshapes might not be mapped for large tensor sizes". Transpose: runtime ≥14 |
| SOFTMAX | C2PSA attention; YOLO11's DFL (16 bins) | "Supports only 1-D input tensor with a max of 16,000 elements" |
| BATCH_MATMUL | C2PSA attention | **not in Table 1**, yet mapped in the YOLO26n log below. Table 1 is not complete for compiler 16. |
| TOPK_V2, GATHER(_ND), CAST, FLOOR_MOD | NMS-free head only | not in Table 1; "Operation not supported" in #23723 |

**Observed compiles** (✅ copied from the logs; each is someone else's run):

| Source | Model, imgsz | Toolchain | Ops on TPU / total | On-chip used / off-chip |
|---|---|---|---|---|
| [#26847](https://github.com/ultralytics/ultralytics/issues/26847) reporter | YOLO26n, 384 | UL 8.4.184, Py 3.13, TF 2.21.0, onnx2tf 2.3.15 | **20 / 688** | 30.5 KiB / 0 |
| [#26847](https://github.com/ultralytics/ultralytics/issues/26847) maintainer (Y-T-G), in the comments | YOLO26n, 384 | Py 3.12, onnx2tf 1.28.x | **652 / 658** | 3.21 MiB / 69.19 KiB (3.92 MiB left) |
| [PR #26849](https://github.com/ultralytics/ultralytics/pull/26849) | YOLO26n, 384 | Py 3.13, TF 2.21.0, onnx2tf 1.28.8 | 652 / 658 | not given |
| [#4754](https://github.com/ultralytics/ultralytics/issues/4754) (2023) | YOLOv8n, 320 | Ultralytics 8.0.x | 233 / 257 (SOFTMAX and the head tail on CPU) | 3.06 MiB / 0 |
| [#4754](https://github.com/ultralytics/ultralytics/issues/4754) (2023) | YOLOv8n, 224 | same | log cut off; output model 6.24 MiB vs 3.31 MiB at 320, so more of the graph mapped | 5.81 MiB / 18.19 KiB |
| [#4754](https://github.com/ultralytics/ultralytics/issues/4754) (2023) | YOLOv5s, 320 | YOLOv5 repo | 262 / 262 | 6.85 MiB / 24.75 KiB |

- In the good YOLO26n compile, the 6 CPU ops are 2 CONCATENATION, 1 RESHAPE, 1 TRANSPOSE, 1 LOGISTIC and 1 QUANTIZE:
  the tail of the head. Backbone, neck, C2PSA attention (BATCH_MATMUL ×4, SOFTMAX ×2) and the head convs map. ✅
- **YOLO11: no recent primary compile log found.** ⚠️ The maintainer's question in #26847, "Does YOLO11n map
  correctly?", went unanswered. [#19336](https://github.com/ultralytics/ultralytics/issues/19336) (Feb 2025, YOLO11s at
  224, older Ultralytics) reports many ops as "More than one subgraph is not supported", without a full log. My
  expectation, not evidence: YOLO11's DFL is a SOFTMAX over 16 bins on a 4-D tensor, outside the 1-D rule. It sits after
  all the convs, so the cut would leave the DFL and the box decode on the CPU, as in the 2023 YOLOv8n log (233/257).
  YOLO26 has no DFL (`reg_max=1`), which is one reason it maps better.
- **Resolution matters, and nobody has logged 640 for an s model.** ⚠️ #4754 mapped more of YOLOv8n at 224 than at 320.
  The two papers that ran YOLOv8 on the TPU chose 320 "to fit the constraints of TPU-based deployment" and 224 "to
  satisfy the Edge TPU memory constraint" (§6). No compile log at 640 for any s-size YOLO was found.

## 3. The ~8 MB on-chip parameter cache

| Claim | Status | Source and text |
|---|---|---|
| Cache size | ✅ | "The Edge TPU has roughly 8 MB of SRAM that can cache the model's parameter data. However, a small amount of the RAM is first reserved for the model's inference executable, so the parameter data uses whatever space remains after that." [compiler doc](https://coral.ai/docs/edgetpu/compiler/) |
| What is really available | ✅ | Doc example: "On-chip memory available for caching model parameters: 6.91MiB". In the logs of §2, used + remaining = 7.13 MiB (YOLO26n), 7.01 MiB (YOLOv5s), 7.53 MiB (YOLOv8n at 320). |
| Granularity | ✅ | "Parameter data is allocated to cache one layer at a time—either all parameter data from a given layer fits into cache and is written there, or that layer's data is too big to fit and all data for that layer must be fetched from external memory." |
| What happens past the cache | ✅ / ⚠️ | ✅ The compiler reports "Off-chip memory used for streaming uncached model parameters", and "saving the parameter data on the Edge TPU RAM enables faster inferencing speed compared to fetching the parameter data from external memory". **The docs give no figure for the cost.** ⚠️ The USB Accelerator datasheet lists no memory on the device ([datasheet v1.4](https://coral.ai/static/files/Coral-USB-Accelerator-datasheet.pdf)), so "external memory" there must be host memory over USB. That is my inference; the docs do not say it. |
| Fixes the docs offer | ✅ | Co-compilation and `--num_segments` pipelining across several Edge TPUs. With one Accelerator on the Pi, neither applies. |
| Input-size limit | ✅ / ⚠️ | ✅ No maximum input size is stated. The limits are per-op: tensor rank, "Certain reshapes might not be mapped for large tensor sizes", the resize ops "Depending on input/output size", Softmax ≤16,000 elements. ✅ Coral's own model page: "EfficientDet-Lite3x is not compatible with Edge TPUs over USB; it can be used only with PCIe-based devices" (640×640, 20.6 MB) ([object-detection models](https://coral.ai/models/object-detection/)). ⚠️ The docs say nothing about activation memory. |

**Estimate for our models** (⚠️ an estimate; the compile log gives the real values):

| Model | Params (Ultralytics table, 80-class COCO) | INT8 weights at 1 B/param | Parameter data at the compiler's observed 0.95–1.37 MiB per M params | Expected off-chip |
|---|---|---|---|---|
| YOLO26n (measured, for scale) | 2.4 M | 2.3 MiB | measured 3.28 MiB | measured 69 KiB |
| YOLO11s | 9.4 M ([yolo11.md](https://github.com/ultralytics/ultralytics/blob/v8.4.149/docs/en/models/yolo11.md)) | ≈ 9.0 MiB | ≈ 9–13 MiB | ≈ 2–6 MiB |
| YOLO26s | 9.5 M ([yolo-det-perf.md](https://github.com/ultralytics/ultralytics/blob/v8.4.149/docs/macros/yolo-det-perf.md)) | ≈ 9.1 MiB | ≈ 9–13 MiB | ≈ 2–6 MiB |

The ratios are YOLOv5s (6.87 MiB / 7.2 M params) and YOLO26n (3.28 MiB / 2.4 M). Our 10-class heads are slightly
smaller than the 80-class counts. Either way, both s models exceed the cache. Parameter size does not change with
`imgsz`; activation sizes and op mapping do.

**One more INT8 effect specific to this path** (⚠️ derived from the source, not measured): the graph's single output
is boxes (as 0–1 fractions, §1) concatenated with class scores. TFLite quantizes activations per tensor, as int8 in
[−128, 127] ([LiteRT quantization spec](https://developers.google.com/edge/litert/models/quantization_spec)). onnx2tf's
default output dtype is int8. So the output step is about 1/255 of the input size: about 2.5 px at 640 (about 5 px in
native VisDrone pixels at a 1360-px-wide image). Scores also get a step of about 0.004, above the 0.001 evaluation
threshold. The ONNX Runtime, OpenVINO and TensorRT INT8 cells keep the decode in float (studyexport.py docstring;
INVARIANTS.md §8), so this hazard is new to the Edge TPU cell, and it lands on AP-small.

## 4. Running it on the Raspberry Pi 5 (Bookworm, Python 3.11)

| Item | Finding | Status | Source |
|---|---|---|---|
| Google's runtime | `google-coral/libedgetpu` is **archived**; last release "Grouper", 2021-07-26. `pycoral` is archived (last 2.0.0, 2021-07-26) and supports "Python 3.6 through 3.9". | ✅ | [libedgetpu releases](https://github.com/google-coral/libedgetpu/releases), [pycoral](https://github.com/google-coral/pycoral), [USB get-started](https://coral.ai/docs/accelerator/get-started/) |
| Community runtime | `feranick/libedgetpu` **16.0TF2.19.1-1** (2025-09-17) has `libedgetpu1-std_16.0tf2.19.1-1.bookworm_arm64.deb` and `libedgetpu1-max_...bookworm_arm64.deb`. Release note: "Built against TensorFlow v2.19.1 stable." "`ai-edge-litert` (v2.0.2) - recommended. Alternatively: `tflite_runtime` from v2.11.1 to v2.19.1 (soon to be deprecated)". No release in the 13 months since. | ✅ | [release 16.0TF2.19.1-1](https://github.com/feranick/libedgetpu/releases/tag/16.0TF2.19.1-1), [README](https://github.com/feranick/libedgetpu) |
| Ultralytics' advice | Use feranick's debs: "the original Coral runtime builds no longer work with current TensorFlow Lite runtime versions, and the project saw no updates between 2021 and 2025." | ✅ | [coral-edge-tpu-on-raspberry-pi.md @v8.4.149](https://github.com/ultralytics/ultralytics/blob/v8.4.149/docs/en/guides/coral-edge-tpu-on-raspberry-pi.md) |
| `tflite-runtime` | Last release **2.14.0** (2023-10-03). cp38–cp311 `manylinux_2_34_aarch64` (Bookworm's glibc is 2.36, so it installs). Declares `numpy >=1.23.2`. | ✅ | [PyPI](https://pypi.org/project/tflite-runtime/2.14.0/), [Debian glibc](https://sources.debian.org/src/glibc/) |
| `tflite-runtime` and NumPy 2 | Its `_pywrap_tensorflow_interpreter_wrapper.so` references only `numpy.core._multiarray_umath`: a NumPy 1.x build (wheel contents inspected). NumPy: "When you build wheels for your package using a NumPy 1.xx version at build time, those **will not work** with NumPy 2.0." The bench's `opencv-python-headless==5.0.0.93` requires `numpy>=2`. **So `tflite-runtime` cannot share the bench venv.** It would need numpy 1.x and an older OpenCV, which changes the preprocessing build. | ✅ | [NumPy, depending on numpy](https://numpy.org/doc/stable/dev/depending_on_numpy.html), [opencv-python-headless 5.0.0.93](https://pypi.org/project/opencv-python-headless/5.0.0.93/) |
| `ai-edge-litert` | Latest **2.3.0** (2026-10-06), cp310–cp314 `manylinux_2_27_aarch64` and `win_amd64`. Its `libpywrap_litert_common.so` references `numpy._core._multiarray_umath`: a NumPy 2 build, which NumPy says "**will work** with NumPy 1.xx", so it runs with numpy 2.4.6. `interpreter.py` keeps `load_delegate(library, options=None)`: "This is typically used to leverage EdgeTPU". | ✅ | [PyPI 2.3.0](https://pypi.org/project/ai-edge-litert/2.3.0/), wheel contents |
| `ai-edge-litert` 2.0.2 (feranick's tested version) | cp311 aarch64 wheel exists (70 MB). Its NumPy ABI was **not** inspected (the download stalled). Whether feranick's TF-2.19.1 build loads under 2.3.0 is **untested** by anyone I found. | ⚠️ | [PyPI 2.0.2](https://pypi.org/project/ai-edge-litert/2.0.2/) |
| Ultralytics' runtime | `TensorFlowBackend` tries `tflite_runtime.interpreter`, then full `tensorflow`. It never tries `ai_edge_litert` (L64-72). It quantizes the input as `(im / scale + zero_point).astype(dtype)`, a truncation, not a rounding (L118). `studybench.py` imports its own runtimes, so this matters only for parity. | ✅ | [backends/tensorflow.py](https://github.com/ultralytics/ultralytics/blob/v8.4.149/ultralytics/nn/backends/tensorflow.py#L64-L147) |
| USB 3 | Accelerator: "USB 3.0 (USB 3.1 Gen 1) port and cable (SuperSpeed, 5 Gbps)"; "for the best performance, use a USB 3.0 port". It works on USB 2, slower. Pi 5: "2 × USB 3.0 ports, supporting simultaneous 5Gbps operation". | ✅ | [datasheet](https://coral.ai/static/files/Coral-USB-Accelerator-datasheet.pdf), [Pi 5 product page](https://www.raspberrypi.com/products/raspberry-pi-5/) |
| Power draw | "at least 500 mA at 5 V ... when the device runs at maximum speed, it can draw a peak current of 900 mA." | ✅ | [datasheet §5](https://coral.ai/static/files/Coral-USB-Accelerator-datasheet.pdf) |
| Pi 5 USB budget | "Raspberry Pi 5 provides 1.6 A of power to downstream USB peripherals when connected to a power supply capable of 5 A at +5 V (25 W). When connected to any other compatible power supply, Raspberry Pi 5 restricts downstream USB devices to 600 mA". "The power budget is shared between the USB ports and the fan header." With the official 27 W supply, 0.9 A peak fits. On a 3 A supply, max mode's peak would not. | ✅ | [power-supplies.adoc](https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/raspberry-pi/power-supplies.adoc), [usb-bus-on-raspberry-pi.adoc](https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/raspberry-pi/usb-bus-on-raspberry-pi.adoc) |
| Our energy method cannot see it | `pmic_read_adc`: "The readings do not include USB current or anything else powered directly from the 5V supply". A J/frame for a TPU cell would leave out the TPU. | ✅ | [vcgencmd.1](https://github.com/raspberrypi/utils/blob/master/vcgencmd/vcgencmd.1), INVARIANTS.md §4.6 |
| std vs max | `libedgetpu1-max` "runs at the maximum frequency (500 MHz)", `libedgetpu1-std` "at the reduced frequency (250 MHz)". Only one can be installed. Datasheet: "The maximum frequency is twice the reduced setting". | ✅ | [Coral software page](https://coral.ai/software/), [datasheet §4.1](https://coral.ai/static/files/Coral-USB-Accelerator-datasheet.pdf) |
| Thermal warning | "the metal on the USB Accelerator can become very hot to the touch. This might cause burn injuries." Maximum ambient: **35 °C reduced, 25 °C maximum**. The Pi 5 pilot recorded a **26 °C** room, so max mode would run outside Google's recommended range. | ✅ | [get-started](https://coral.ai/docs/accelerator/get-started/), [datasheet Table 3](https://coral.ai/static/files/Coral-USB-Accelerator-datasheet.pdf), `study/verification/pi5_pilot_record.md` |
| TPU temperature readout | The docs have a temperature page for PCIe modules only. I found no readout for the USB Accelerator, so a TPU-side throttle would go unrecorded. | ⚠️ | [docs index](https://coral.ai/docs/) |
| First inference | "The first inference on Edge TPU is slow because it includes loading the model into Edge TPU memory." The bench's 20 warm-up images cover it. | ✅ | [get-started](https://coral.ai/docs/accelerator/get-started/) |

## 5. The Coral Dev Board

| Item | Finding | Status | Source |
|---|---|---|---|
| Hardware | NXP i.MX 8M, quad Cortex-A53 at 1.5 GHz plus Cortex-M4F; Edge TPU over PCIe; 1 GB or 4 GB LPDDR4; 8 or 16 GB eMMC; 5 V 2–3 A over USB-C; heat sink and fan. | ✅ | [Dev Board datasheet v1.7 (Dec 2022)](https://coral.ai/static/files/Coral-Dev-Board-datasheet.pdf) |
| Newest OS | Mendel **5.3 "Eagle"**, Nov 2021 (`enterprise-eagle-20211117215217`). Mendel 4.0 was the "First release based on Debian 10 (buster)" with "Support for Python 3.7". Debian 10's LTS ended "30th June, 2024". | ✅ | [Coral software page](https://coral.ai/software/), [Mendel release notes](https://coral.googlesource.com/docs/+/refs/heads/master/Releases.md), [Debian buster](https://www.debian.org/releases/buster/) |
| Flashing | A microSD "flashcard" image written with balenaEtcher, boot switches set to SD, then back to eMMC. The host may be "Linux (recommended), Mac, or Windows 10"; Windows users run Git Bash. Mendel runs from eMMC, "not on the microSD card". MDT installs with pip. So the laptop can flash it. | ✅ | [Dev Board get-started](https://coral.ai/docs/dev-board/get-started/) |
| Still maintained? | No explicit end-of-life notice found. ⚠️ But: coral.ai now redirects to `gweb-coral-full.uc.r.appspot.com`; Google's Coral pages are now about Coral NPU and link the old docs as "First-gen Coral Edge TPU"; `libedgetpu`, `pycoral`, `libcoral`, `edgetpu` and `tflite` are archived on GitHub; the `coral-edgetpu-stable` apt repository answers 403. I did not check whether Mendel's own apt sources still answer. | ✅ / ⚠️ | [developers.google.com/coral intro](https://developers.google.com/coral/guides/intro-platform), [announcements](https://developers.google.com/coral/news/announcements), GitHub repo pages |
| Which board? | Google and Synaptics launched a "Next-Generation Coral Dev Board" in March 2026. It is a different product built on Coral NPU, not the Edge TPU. Check the department's unit. | ✅ (announcement title) | [announcements](https://developers.google.com/coral/news/announcements) |
| Fits the bench? | No, as it stands. numpy 2.4.6 needs Python ≥3.11 ([PyPI](https://pypi.org/project/numpy/2.4.6/)) and opencv 5.0.0.93 needs numpy ≥2; Mendel has Python 3.7. It would need a separate, older bench stack, so preprocessing would not be the same build as on the other platforms. | ✅ | PyPI metadata |
| Faster than the Pi 5 + USB? | Unlikely. In Coral's own table, Dev Board + TPU is slower than a desktop + USB 3 Accelerator on most models (MobileNet v2 SSD 14 vs 7.2 ms; Inception v4 102 vs 85 ms). The CPU tail would run on A53 cores. Its one advantage is PCIe instead of USB for the TPU. | ✅ (table), ⚠️ (inference) | [Coral benchmarks](https://coral.ai/docs/edgetpu/benchmarks/) |

## 6. Published numbers

**Ultralytics' own table** ([coral-edge-tpu-on-raspberry-pi.md @v8.4.149](https://github.com/ultralytics/ultralytics/blob/v8.4.149/docs/en/guides/coral-edge-tpu-on-raspberry-pi.md), ✅).
Raspberry Pi OS Bookworm 64-bit and a USB Coral; "inference time only (pre-/postprocessing excluded)"; "These
benchmarks were recorded with YOLOv8 models." **Only 320 and 512.** The Pi clock and the Ultralytics version are not
stated.

| Pi | imgsz | Model | std (ms) | max (ms) |
|---|---|---|---|---|
| Pi 5 8GB | 320 | YOLOv8n | 22.2 | 16.7 |
| Pi 5 8GB | 320 | YOLOv8s | 40.1 | 32.2 |
| Pi 5 8GB | 512 | YOLOv8n | 53.5 | 41.6 |
| Pi 5 8GB | 512 | YOLOv8s | 132.0 | 103.3 |
| Pi 4B 2GB | 512 | YOLOv8s | 149.6 | 125.3 |

"The high-frequency mode is 28.4% faster than the standard mode" (their average).

**Extrapolation to 640** (⚠️ my arithmetic, not a measurement): YOLOv8s went from 40.1 to 132.0 ms for 2.56× the
pixels, an exponent of about 1.27. 640 has 1.5625× the pixels of 512, so about 132 × 1.76 ≈ **230 ms std** and
103 × 1.76 ≈ **180 ms max**, before pre- and post-processing. Our Pi 5's measured `e2e_deploy` medians for YOLO11s at
640 are NCNN FP16 171 ms, ONNX Runtime INT8 272 ms and OpenVINO FP32 294 ms (PLATFORMS.md). YOLO11s has fewer FLOPs
than YOLOv8s (21.6 vs 28.6 B, [yolov8.md @v8.4.149](https://github.com/ultralytics/ultralytics/blob/v8.4.149/docs/en/models/yolov8.md)),
and our Pi runs at 2000 MHz, not the stock 2400. So the TPU at 640 may land near NCNN,
not far below it.

**Coral's own detection models** (desktop Xeon + USB 3 Accelerator, "time to perform one inference";
[object-detection models](https://coral.ai/models/object-detection/), ✅):

| Model | Input | Latency | COCO mAP | Size |
|---|---|---|---|---|
| SSD MobileNet V2 | 300 | 7.3 ms | 25.6 % | 6.6 MB |
| SSDLite MobileDet | 320 | 9.1 ms | 32.9 % | 5.1 MB |
| EfficientDet-Lite0 | 320 | 37.4 ms | 30.4 % | 5.7 MB |
| EfficientDet-Lite3 | 512 | 107.6 ms | 39.4 % | 14.4 MB |
| SSD/FPN MobileNet V1 | 640 | 229.4 ms | 31.1 % | 37.7 MB |
| EfficientDet-Lite3x | 640 | 197.0 ms (PCIe only) | 43.9 % | 20.6 MB |

Above about 7 MB, even Google's own models leave single-digit milliseconds behind.

**Papers:**

| Paper | Status | What it reports |
|---|---|---|
| Suchý & Turčaník, "Review of large YOLOv8 and RT-DETR energy efficiency on edge devices for real-time detection", *Scientific Reports* 2026 (doi:10.1038/s41598-026-46453-6; full text via [Europe PMC PMC13039985](https://europepmc.org/article/PMC/PMC13039985)) | ✅ peer-reviewed, full text read | YOLOv8**l** on a Pi 5 + Coral USB at **224×224**: 140 ms inference, 6.89 FPS end-to-end, 5.23 W, mAP50-95 **0.324**. The same model at 640: PyTorch 0.530, TFLite FP32 0.527, TFLite INT8 (CPU) 0.526. "This reduction is linked to mandatory INT8 quantization and the required decrease in input resolution to 224 × 224 in order to satisfy the Edge TPU memory constraint." Their own TFLite INT8 row at 640 lost almost nothing, so most of the drop goes with 224 and the TPU path, not with INT8 itself. |
| Alqahtani et al., "A Comprehensive Evaluation of Deep Learning Object Detection Models on Heterogeneous Edge Devices", [arXiv:2409.16808v3](https://arxiv.org/html/2409.16808v3) (2026-07-14) | ⚠️ preprint, no venue stated | YOLOv8n/s/m on Pis with Coral USB Accelerators at **320×320** "in order to fit the constraints of TPU-based deployment" (CPU runs at 640). COCO mAP for the TPU group: n 16, s 25, m 32, against 30, 39, 44 for the Pi CPU group. Pi 5 + TPU YOLOv8 latency 21–97 ms vs 316–1,348 ms CPU-only. No std/max clock and no runtime versions are given. |
| Minott, Siddiqui & Haddad, SoutheastCon 2025, doi:10.1109/southeastcon56624.2025.10971592 | ⚠️ abstract only ([listing](https://scholars.georgiasouthern.edu/en/publications/benchmarking-edge-ai-platforms-performance-analysis-of-nvidia-jet/)) | **YOLOv5**, not YOLOv8: Pi 5 + Coral 21.5 FPS, about 8.3 W, hottest board without active cooling. No input size in the abstract. (Corrects `novelty-check.md`, which does not name the model.) |
| Bestari & Zulfa, [arXiv:2610.34304](https://arxiv.org/html/2610.34304) (2026-10-25) | ⚠️ preprint, weak | YOLO11n at 320 on a Pi 5 + Edge TPU, 30–40 FPS, 70–90 ms camera-to-display. No op counts, no clock mode, no runtime versions, no pre- vs post-quantization mAP. |

**Summary:** every published YOLO-on-Edge-TPU number is at 224, 320 or 512. None is for YOLO11s or YOLO26s, none
is at 640, and none is on VisDrone or reports AP-small.

## 7. Fit to our study, and the time cost

**What a cell could be.** Only these are possible:

- Runtime: Edge TPU, INT8 only (FP32/FP16 are refused, §1). The one-to-many head plus our NMS, as everywhere (§5 of
  INVARIANTS). No NMS-free head.
- Platform: Pi 5 + USB Accelerator on a USB 3 port, `libedgetpu1-std` by default (max is outside the recommended
  ambient at 26 °C, §4).
- Latency and accuracy, yes. **Energy, no**: the rail sum cannot see the Accelerator (§4).

| Candidate cell | Sense for the study | Risk |
|---|---|---|
| YOLO11s and YOLO26s, Edge TPU INT8 at **640**, from the 640-trained checkpoints | The matrix resolution; joins Result A/B as one more runtime | Op mapping at 640 for s models is unknown; 2–6 MiB off-chip; extrapolated latency is near NCNN's (§6) |
| The same at 960 / 1280 | The equal-latency cells | Likely worse mapping, more activations, calibration arrays of 6–11 GB; no published evidence at all |
| The same models exported at 320 or 480 | Off-matrix; a 640-trained model at 320 is a different experiment for small objects | Only a feasibility point, never a matrix cell |
| Control: the same `*_full_integer_quant.tflite` on the Pi 5 CPU (ai-edge-litert, no delegate) | Separates "the TPU" from "this INT8 model" | Needs a CPU `litert` runtime as well |
| Accuracy reference: `*_float32.tflite` at the same imgsz | INVARIANTS §4.4 asks for a same-format FP32 reference | Can run on the laptop (ai-edge-litert has `win_amd64` wheels) |

**What an `edgetpu` runtime in `studybench.py` would look like** (from the bench's runtime classes and Ultralytics'
backend):

- `find_model`: `*_edgetpu.tflite`.
- `__init__`: `from ai_edge_litert.interpreter import Interpreter, load_delegate`;
  `Interpreter(model_path, experimental_delegates=[load_delegate("libedgetpu.so.1")], num_threads=threads)` (threads
  for the CPU tail).
- `input_shape`: report NCHW, because the model is NHWC `(1, H, W, 3)` and `main()` checks `input_shape[-2:]`.
- `__call__`: NCHW→NHWC, quantize with the input's `(scale, zero_point)` (decide: truncate like Ultralytics or round),
  `invoke`, dequantize the output, multiply rows 0 and 2 by W and rows 1 and 3 by H. Then the existing
  `postprocess_raw` takes the `(1, 4+nc, anchors)` float output unchanged.
- `precision`: INT8, from the int8 input/output tensors and the delegate's custom op in `_get_ops_details()` (present
  in both ai-edge-litert 2.3.0 and tflite-runtime 2.14.0). `runtime_info` should record: TPU vs CPU op counts (from the
  runtime op list and the compiler `.log`), on-chip/off-chip MiB (compiler log), `libedgetpu1-std` or `-max` (dpkg),
  the libedgetpu and ai-edge-litert versions, and the USB link speed (5000 Mbps expected).
- Decide where quantize/dequantize time goes. Ultralytics counts it in inference; for the bench it is `net` or `pre`/`post`.

**What would have to be built, with honest hours** (⚠️ estimates; they assume nothing breaks badly):

| Piece | Where | Dev hours | Machine time |
|---|---|---|---|
| Kaggle export env: `edgetpu_compiler` on `PATH` (archived google-coral binary or the Ultralytics asset tarball); onnx2tf 1.28.8 plus TF/tf_keras/sng4onnx/onnx_graphsurgeon/protobuf by hand on Python 3.13, or a Python 3.12 venv. Check that the compile log maps the head as in #26847. | Kaggle | 2–4 | — |
| `studyexport.py`: an `edgetpu-int8` variant (`format="edgetpu", quantize=8, data=...`); add `tensorflow`, `onnx2tf` and the compiler version to the toolchain record; keep the compile `.log`, op counts and memory lines in the manifest; also keep `*_float32.tflite` and `*_full_integer_quant.tflite` | repo | 1–2 | — |
| Exports: 2 models × 640 (+ optionally 320) | Kaggle | 0.5 | 1–3 h unattended. In #26847 the SavedModel step alone took 606 s for YOLO26n at 384 with 320 images on a desktop CPU. |
| `EdgeTPU` runtime plus a CPU `LiteRT` runtime in `studybench.py`, tests on the laptop through the CPU path | repo | 3–5 | — |
| `studysession.py` (`RUNTIMES`, `PI_PINS`, setup checks: Accelerator present at 5000M, std/max package) and a `pi/wizard.sh` stage for the deb, the udev replug and the venv | repo | 2–3 | — |
| Pi 5 install: feranick `bookworm_arm64` std deb; ai-edge-litert in a copy of `~/sb` (the session runs the bench with `sys.executable`, and the drift reference cell needs onnxruntime in the same venv); first-light run | Pi 5 | 1–2 | — |
| Bench: 2 cells at 640 at about 200 ms ≈ 6–7 min each, plus the session's reference runs and gates; CPU control cells take longer | Pi 5 | 0.5 | 0.75–2 h |
| Accuracy reference (`*_float32.tflite`, 1,610 images) on the laptop; scoring | laptop | 0.5–1 | under 1 h |
| INVARIANTS §1.3 toolchain row, §8 precision row (with the mapped-op count), PLATFORMS section, paper text | repo | 1–2 | — |
| **Total** | | **≈ 11–20 h** | **2–4 h Kaggle, 1–2.5 h Pi 5** |

**The options, plainly:**

- **(a) Edge TPU cells in the paper by 31 Oct.** Everything in the table. About 11–20 dev hours in three days, and
  1–2.5 Pi 5 hours taken from #37, the main matrix, which is not exported yet. What it buys: one more runtime column
  on the Pi 5 at 640 (latency and accuracy, no energy). Risks: poor mapping at 640 for YOLO11 (DFL), off-chip
  streaming, the onnx2tf and compiler workarounds, and an untested libedgetpu + ai-edge-litert pairing. The result could
  be "the TPU is no faster than NCNN at 640", which is still a finding, but the cost is the same.
- **(b) A small feasibility note.** Kaggle only, no Pi time: compile YOLO11s and YOLO26s at 640 (and 320) with the
  pinned exporter plus the two fixes. Report the compiler's TPU/CPU op counts and on-chip/off-chip memory per model and
  size, and cite §6. About 4–8 dev hours and 1–3 h of unattended Kaggle time. Optional: one 200-image latency run on the
  Pi 5 for one model, if the Pi has a free hour after #37 (+2–4 dev hours for the bench class and install). Labelled as
  outside the matrix.
- **(c) Future work.** A paragraph with the constraints (x86-only compiler, retired apt repo, ~7 MiB cache, first-cut
  partitioning, INT8-only output, USB power outside the rail sum) and the published numbers. Under 1 hour of writing.

In all three, the Dev Board stays out (§5).

## Open questions for the dev

1. **Which option, (a), (b) or (c)?** The deciding constraint is Pi 5 time against #37.
2. If (a) or (b): **which resolutions?** 640 only (the matrix), or also 320 as a mapping and latency anchor? 960 and
   1280 look out of reach.
3. **Kaggle environment:** hand-pin onnx2tf 1.28.8 on Python 3.13 (the PR's tested combination), or build a Python 3.12
   venv where the pin selects onnx2tf 1.x itself? Either way the record gains `tensorflow`, `onnx2tf` and the compiler
   version.
4. **Runtime package on the Pi:** ai-edge-litert **2.0.2** (feranick's recommendation; NumPy ABI not checked here) or
   **2.3.0** (NumPy 2 build verified; pairing with feranick's build untested)? In a copy of `~/sb` or in `~/sb` itself?
5. **Clock:** std only (within Google's ambient spec at 26 °C), or also max (outside it)?
6. **Timing boundary:** does input quantization and output dequantization count as `net` (as in Ultralytics) or as
   `pre`/`post`? Truncate (Ultralytics) or round the input?
7. **Precision rule wording:** should a TPU cell whose compile leaves more than a few ops on the CPU still count as an
   "Edge TPU" cell, or be labelled "Edge TPU + CPU, k of n ops on the TPU"?
8. **Energy:** report no energy for TPU cells, or borrow a USB power meter for the Accelerator's port?
9. **Which Dev Board does the department have:** the 2019 NXP i.MX 8M board or the 2026 Synaptics "next-generation"
   board? It does not change the answer for this paper, but it matters for future work.

## Sources

Ultralytics, at tag v8.4.149 unless noted:
- `ultralytics/engine/exporter.py`: <https://github.com/ultralytics/ultralytics/blob/v8.4.149/ultralytics/engine/exporter.py>
- `ultralytics/utils/export/tensorflow.py`: <https://github.com/ultralytics/ultralytics/blob/v8.4.149/ultralytics/utils/export/tensorflow.py>
- `ultralytics/nn/backends/tensorflow.py`: <https://github.com/ultralytics/ultralytics/blob/v8.4.149/ultralytics/nn/backends/tensorflow.py>
- `ultralytics/utils/checks.py`: <https://github.com/ultralytics/ultralytics/blob/v8.4.149/ultralytics/utils/checks.py>
- `tests/test_exports.py`: <https://github.com/ultralytics/ultralytics/blob/v8.4.149/tests/test_exports.py>
- Docs: [coral-edge-tpu-on-raspberry-pi.md](https://github.com/ultralytics/ultralytics/blob/v8.4.149/docs/en/guides/coral-edge-tpu-on-raspberry-pi.md), [integrations/edge-tpu.md](https://github.com/ultralytics/ultralytics/blob/v8.4.149/docs/en/integrations/edge-tpu.md), [models/yolo11.md](https://github.com/ultralytics/ultralytics/blob/v8.4.149/docs/en/models/yolo11.md), [macros/yolo-det-perf.md](https://github.com/ultralytics/ultralytics/blob/v8.4.149/docs/macros/yolo-det-perf.md)
- `main` (after the pin): [tensorflow.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/tensorflow.py); [PR #26456](https://github.com/ultralytics/ultralytics/pull/26456) (compiler without apt, merged 2026-09-30); [PR #26849](https://github.com/ultralytics/ultralytics/pull/26849) (onnx2tf 1.x on all Pythons, merged 2026-10-23)
- Issues: [#26847](https://github.com/ultralytics/ultralytics/issues/26847), [#23723](https://github.com/ultralytics/ultralytics/issues/23723), [#4754](https://github.com/ultralytics/ultralytics/issues/4754), [#19336](https://github.com/ultralytics/ultralytics/issues/19336), [#22400](https://github.com/ultralytics/ultralytics/issues/22400)
- Release: <https://github.com/ultralytics/ultralytics/releases/tag/v8.4.149>

Coral / Google (coral.ai now redirects to `gweb-coral-full.uc.r.appspot.com`; fetched 2026-10-28):
- Models intro and supported ops: <https://coral.ai/docs/edgetpu/models-intro/>
- Compiler: <https://coral.ai/docs/edgetpu/compiler/>
- Benchmarks: <https://coral.ai/docs/edgetpu/benchmarks/>
- Object-detection models: <https://coral.ai/models/object-detection/>
- USB Accelerator get-started: <https://coral.ai/docs/accelerator/get-started/>; datasheet v1.4: <https://coral.ai/static/files/Coral-USB-Accelerator-datasheet.pdf>
- Dev Board get-started: <https://coral.ai/docs/dev-board/get-started/>; datasheet v1.7: <https://coral.ai/static/files/Coral-Dev-Board-datasheet.pdf>
- Software and packages: <https://coral.ai/software/>; FAQ: <https://coral.ai/docs/edgetpu/faq/>
- Mendel release notes: <https://coral.googlesource.com/docs/+/refs/heads/master/Releases.md>
- New Coral platform: <https://developers.google.com/coral/guides/intro-platform>, <https://developers.google.com/coral/news/announcements>
- Apt repository status: `https://packages.cloud.google.com/apt/dists/coral-edgetpu-stable/Release` → HTTP 403 (checked 2026-10-28)
- Archived repos: [google-coral/libedgetpu](https://github.com/google-coral/libedgetpu), [google-coral/pycoral](https://github.com/google-coral/pycoral), [google-coral/edgetpu](https://github.com/google-coral/edgetpu) (compiler binary in `compiler/x86_64`), [google-coral/libcoral](https://github.com/google-coral/libcoral), [google-coral/tflite](https://github.com/google-coral/tflite)
- LiteRT quantization spec: <https://developers.google.com/edge/litert/models/quantization_spec>

Runtime packages:
- feranick/libedgetpu: <https://github.com/feranick/libedgetpu/releases/tag/16.0TF2.19.1-1>, <https://github.com/feranick/libedgetpu>
- tflite-runtime 2.14.0: <https://pypi.org/project/tflite-runtime/2.14.0/> (wheel `cp311-manylinux_2_34_aarch64` inspected)
- ai-edge-litert 2.3.0: <https://pypi.org/project/ai-edge-litert/2.3.0/> (wheel `cp311-manylinux_2_27_aarch64` inspected); 2.0.2: <https://pypi.org/project/ai-edge-litert/2.0.2/> (metadata only)
- onnx2tf 1.28.8 README: <https://github.com/PINTO0309/onnx2tf/blob/1.28.8/README.md>
- NumPy ABI: <https://numpy.org/doc/stable/dev/depending_on_numpy.html>; numpy 2.4.6: <https://pypi.org/project/numpy/2.4.6/>; opencv-python-headless 5.0.0.93: <https://pypi.org/project/opencv-python-headless/5.0.0.93/>
- Debian bookworm glibc 2.36: <https://sources.debian.org/src/glibc/>; Debian 10: <https://www.debian.org/releases/buster/>

Raspberry Pi:
- Power supplies: <https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/raspberry-pi/power-supplies.adoc>
- USB: <https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/raspberry-pi/usb-bus-on-raspberry-pi.adoc>
- Pi 5 product page: <https://www.raspberrypi.com/products/raspberry-pi-5/> (the product brief PDF was too large to fetch)
- vcgencmd man page: <https://github.com/raspberrypi/utils/blob/master/vcgencmd/vcgencmd.1>

Papers:
- Suchý I., Turčaník M., *Scientific Reports* 2026, doi:10.1038/s41598-026-46453-6; full text: <https://europepmc.org/article/PMC/PMC13039985> (nature.com required a login redirect; read through Europe PMC instead)
- Alqahtani D.K. et al., arXiv:2409.16808v3: <https://arxiv.org/html/2409.16808v3>
- Minott D., Siddiqui S., Haddad R.J., SoutheastCon 2025, doi:10.1109/southeastcon56624.2025.10971592 (abstract): <https://scholars.georgiasouthern.edu/en/publications/benchmarking-edge-ai-platforms-performance-analysis-of-nvidia-jet/>
- Bestari M.N.B., Zulfa M.I., arXiv:2610.34304: <https://arxiv.org/html/2610.34304>

Repo files read: `study/INVARIANTS.md`, `study/PLATFORMS.md`, `study/scripts/studybench.py`, `study/scripts/studyexport.py`,
`study/scripts/studysession.py`, `study/verification/smoke_results-kaggle-*.json`, `study/verification/pi5_facts.txt`,
`study/verification/pi5_pilot_record.md`, `.scratch/uav-yolo-edge/findings/01-yolo26.md`, `.scratch/uav-yolo-edge/findings/novelty-check.md`.

Could not reach or did not read: the Pi 5 product brief PDF (over 10 MB; the product page was used instead); the
`ai-edge-litert` 2.0.2 wheel's contents (download stalled); the full text of Minott et al. (abstract only); whether
Mendel's own apt sources still answer.
