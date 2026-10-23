"""On-platform benchmark of one cell: every test-dev image through one exported artifact, accuracy and latency in one pass.

  python3 studybench.py --artifact exports/yolo11s-640-s0/640/onnxruntime-fp32 --runtime onnxruntime --imgsz 640 \
      --images VisDrone/images/test --out bench/pi5/yolo11s-640-s0/640/onnxruntime-fp32 --platform-tag pi5

Runs ON the platform. Needs only the standard library, numpy, cv2 and the one runtime module (imported lazily); runs on
Python 3.11 (Raspberry Pi OS Bookworm), 3.14 (Termux), 3.10 (Jetson, JetPack 6) and the laptop. Never import
ultralytics or studyenv here, and torch only inside the TensorRT runner (see there).

Warm-up first (the first --warmup images, untimed), then every image is timed, batch 1, single stream, in three stages.
Reading and decoding the file is not timed.
  pre   what DetectionValidator feeds a static export (rect=False, batch 1): long side resized to imgsz with ceil()
        rounding (data/base.py load_image, INTER_LINEAR), LetterBox pad of 114 (scaleup=False, so it only pads),
        BGR->RGB, CHW, float32 / 255. Not the predictor's LetterBox, which rounds the short side and differs by a row
        on 1,084 of the 1,610 test-dev images.
  net   one runtime call. For TensorRT that is the whole GPU round trip: copy into pinned memory, host-to-device copy,
        execute, device-to-host copy, stream sync.
  post  box decode + NMS exactly as DetectionValidator calls utils/nms.py non_max_suppression (conf 0.001, iou 0.7,
        max_det 500, multi_label, class-offset NMS, max_nms 30000), or the end2end branch for an NMS-free head, then
        scale_boxes with the validator's ratio_pad. Converting to COCO json is not timed.
        It runs twice per image, on the same model output, each pass timed on its own (#22): at the evaluation
        threshold, 0.001 (t_post_ms; its detections are saved), and at the deployment threshold, 0.25 (t_post_deploy_ms
        and n_dets_deploy; timed only). The 0.25 pass goes first on odd image indices, so a warm cache favours neither.
Latency is e2e_deploy, pre + net + the 0.25 pass, in meta.json's timing_ms beside post_deploy; t_total_ms is pre + net
+ post at 0.001, which NMS over up to 30,000 candidates makes look slower than a deployed detector.
The detections are written exactly as DetectionValidator.pred_to_json writes them (bbox rounded to 3 decimals, score
to 5, category = class + 1). Score them on the laptop with `python studyscore.py <out>`.

--threads is part of the cell, not only a speed knob: ONNX Runtime's INT8 output changes with intra_op_num_threads
(measured on the laptop: 1, 2 and 4 threads agree, 8 and 16 each differ, by up to 49 px on a box; FP32 does not move).
Ultralytics leaves ONNX Runtime at its default, one thread per physical core.

On a Jetson (/etc/nv_tegra_release exists) the run also records the GPU temperature and clock per image, and the
board's power mode, jetson_clocks state, clocks, temperatures and over-current counters at the start, every 50
images and the end, with jetson_probe.py's readers (that file must sit beside this one; only a Jetson imports it).

Crash-visible: every image is appended to predictions.jsonl and timings.csv and flushed as it finishes (fsync every
50 images), and meta.json says "running" until the run completes ("interrupted", with the error, if Python saw it
die). A run that died resumes with --resume: it re-warms, skips the images already recorded, and logs each session.
"""

import argparse
import collections
import csv
import functools
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# cfg/invariants.yaml `shared:` plus the non_max_suppression defaults the validator leaves alone. Copied, not read:
# yaml and studyenv are not on the platform. studyscore.py refuses a run whose meta disagrees with the invariants.
CONF, IOU, MAX_DET = 0.001, 0.7, 500
MAX_NMS, MAX_WH = 30000, 7680
DEPLOY_CONF = 0.25  # the deployment threshold (INVARIANTS.md section 4.5): latency only, its detections are not saved
PAD_VALUE = 114
IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
FSYNC_EVERY = THROTTLE_EVERY = 50
COLUMNS = ["index", "file", "t_pre_ms", "t_net_ms", "t_post_ms", "n_cand", "n_dets", "temp_c", "cpu_mhz", "throttled", "img_crc32", "session",
           "gpu_temp_c", "gpu_mhz",  # these two: Jetson only
           "t_post_deploy_ms", "n_dets_deploy"]  # the post-processing pass at the deployment threshold


# ---------------------------------------------------------------- pre-processing


def preprocess(im, imgsz):
    """BGR uint8 HWC image -> (1, 3, imgsz, imgsz) float32 input and the validator's ratio_pad."""
    h0, w0 = im.shape[:2]
    r = imgsz / max(h0, w0)
    if r != 1:  # load_image: ceil, not LetterBox's round
        im = cv2.resize(im, (min(math.ceil(w0 * r), imgsz), min(math.ceil(h0 * r), imgsz)), interpolation=cv2.INTER_LINEAR)
    h, w = im.shape[:2]
    # LetterBox(scaleup=False) on an image whose long side is already imgsz: ratio 1, pad only
    dw, dh = (imgsz - w) / 2, (imgsz - h) / 2
    top, bottom, left, right = round(dh - 0.1), round(dh + 0.1), round(dw - 0.1), round(dw + 0.1)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    x = np.ascontiguousarray(im.transpose(2, 0, 1)[::-1]).astype(np.float32)[None]
    x /= np.float32(255)  # a division, as the validator does; * (1/255) rounds differently
    return x, ((h / h0, w / w0), (left, top))


# ---------------------------------------------------------------- post-processing


def nms_greedy(boxes, scores, cls, iou, max_det):
    """torchvision.ops.nms on class-offset boxes, stopped once max_det boxes are kept (the validator keeps only those).

    Boxes of different classes never overlap after the offset, so each kept box is compared only with its own class.
    The next box kept is the highest-scoring survivor of any class, ties broken by input order (torchvision's sort is
    stable). Arithmetic is float32 in torchvision's order, so near-threshold decisions match it.
    """
    order = np.argsort(-scores, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    x1, y1, x2, y2 = (np.ascontiguousarray(boxes[:, k]) for k in range(4))
    areas = (x2 - x1) * (y2 - y1)
    cls_sorted = cls[order]
    queues = [order[cls_sorted == c] for c in np.unique(cls_sorted)]
    zero, thr = np.float32(0), np.float32(iou)
    keep = []
    with np.errstate(divide="ignore", invalid="ignore"):  # zero-area pairs give nan, which never suppresses
        while queues and len(keep) < max_det:
            q = min(range(len(queues)), key=lambda k: rank[queues[k][0]])
            i, rest = queues[q][0], queues[q][1:]
            keep.append(i)
            if rest.size:
                w = np.maximum(zero, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
                h = np.maximum(zero, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
                inter = w * h
                rest = rest[~(inter / (areas[i] + areas[rest] - inter) > thr)]
            if rest.size:
                queues[q] = rest
            else:
                del queues[q]
    return np.array(keep, dtype=np.int64)


def postprocess_raw(pred, conf):
    """Raw one-to-many head (4 + nc, anchors), xywh in input pixels -> boxes xyxy, scores, classes, candidate count."""
    conf = np.float32(conf)
    pred = pred[:, pred[4:].max(0) > conf].T  # candidates: best class above conf
    xy, wh = pred[:, :2], pred[:, 2:4] / 2
    box = np.concatenate((xy - wh, xy + wh), 1)
    i, j = np.nonzero(pred[:, 4:] > conf)  # multi_label: one row per (box, class) above conf, row-major like torch.where
    box, score, cls = box[i], pred[i, 4 + j], j
    n_cand = int(score.size)
    if n_cand > MAX_NMS:  # torch's argsort here is unstable, so tied scores may order differently (640: max 6,514 on test-dev)
        top = np.argsort(-score, kind="stable")[:MAX_NMS]
        box, score, cls = box[top], score[top], cls[top]
    if not score.size:
        return box, score, cls, n_cand
    keep = nms_greedy(box + (cls.astype(np.float32) * np.float32(MAX_WH))[:, None], score, cls, IOU, MAX_DET)
    return box[keep], score[keep], cls[keep], n_cand


def postprocess_e2e(pred, conf):
    """NMS-free one-to-one head (max_det, 6): x1 y1 x2 y2 score class, already top-k sorted -> the validator's end2end branch."""
    pred = pred[pred[:, 4] > np.float32(conf)]
    n_cand = int(pred.shape[0])
    pred = pred[:MAX_DET]
    return pred[:, :4], pred[:, 4], pred[:, 5].astype(np.int64), n_cand


def postprocess(pred, head, conf, ratio_pad, shape0):
    """One post-processing pass at one confidence threshold: decode and NMS (or the NMS-free head's branch), then scale_boxes."""
    boxes, scores, cls, n_cand = (postprocess_e2e if head == "end2end" else postprocess_raw)(pred, conf)
    return scale_boxes(boxes, ratio_pad, shape0), scores, cls, n_cand


def two_passes(pred, head, ratio_pad, shape0, deploy_first):
    """Both post-processing passes on one model output, each timed on its own, in the order given:
    -> (result at the evaluation threshold, its ms, result at the deployment threshold, its ms).

    Only the evaluation pass's detections are saved. The deployment pass is what a deployed detector runs, and its time
    is the one latency uses. Neither pass writes to pred, so the order cannot change a detection, only which pass finds
    the caches warm."""
    passes = {"eval": CONF, "deploy": DEPLOY_CONF}
    res, ms = {}, {}
    for name in ("deploy", "eval") if deploy_first else ("eval", "deploy"):
        t0 = time.perf_counter_ns()
        res[name] = postprocess(pred, head, passes[name], ratio_pad, shape0)
        ms[name] = (time.perf_counter_ns() - t0) / 1e6
    return res["eval"], ms["eval"], res["deploy"], ms["deploy"]


def scale_boxes(boxes, ratio_pad, shape0):
    """ops.scale_boxes with the validator's ratio_pad (separate x and y gains), then clip to the original image."""
    (gain_y, gain_x), (pad_x, pad_y) = ratio_pad
    b = boxes.copy()
    b[:, [0, 2]] -= np.float32(pad_x)
    b[:, [1, 3]] -= np.float32(pad_y)
    b[:, [0, 2]] /= np.float32(gain_x)
    b[:, [1, 3]] /= np.float32(gain_y)
    h0, w0 = shape0
    b[:, [0, 2]] = b[:, [0, 2]].clip(np.float32(0), np.float32(w0))
    b[:, [1, 3]] = b[:, [1, 3]].clip(np.float32(0), np.float32(h0))
    return b


def to_coco(boxes, scores, cls):
    """pred_to_json: xyxy -> top-left xywh in float32, rounded as the validator rounds it. Rows are [x, y, w, h, score, category]."""
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    x = (boxes[:, 0] + boxes[:, 2]) / np.float32(2) - w / np.float32(2)
    y = (boxes[:, 1] + boxes[:, 3]) / np.float32(2) - h / np.float32(2)
    return [[round(a, 3), round(b, 3), round(c, 3), round(d, 3), round(s, 5), k + 1]
            for a, b, c, d, s, k in zip(x.tolist(), y.tolist(), w.tolist(), h.tolist(), scores.tolist(), cls.tolist())]


# ---------------------------------------------------------------- runtimes


def find_model(artifact, runtime):
    p = Path(artifact)
    pattern = {"onnxruntime": "*.onnx", "openvino": "*.xml", "ncnn": "*.param", "tensorrt": "*.engine"}[runtime]
    if p.is_file():
        return p
    found = sorted(p.rglob(pattern))
    if len(found) != 1:
        sys.exit(f"expected exactly one {pattern} under {p}, found {[str(f) for f in found]}")
    return found[0]


ONNX_INT8_OPS = {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear", "QLinearConv", "QLinearMatMul",
                 "ConvInteger", "MatMulInteger"}


def _protobuf_fields(buf, pos, end):
    """(field number, value) of each field in buf[pos:end]; a length-delimited value is its (start, end) span."""
    def varint():
        nonlocal pos
        n = shift = 0
        while True:
            b = buf[pos]
            pos += 1
            n |= (b & 0x7F) << shift
            shift += 7
            if b < 0x80:
                return n

    while pos < end:
        key = varint()
        wire = key & 7
        if wire == 0:
            yield key >> 3, varint()
        elif wire == 2:
            n = varint()
            yield key >> 3, (pos, pos + n)
            pos += n
        elif wire in (1, 5):
            pos += 8 if wire == 1 else 4
        else:
            raise ValueError(f"protobuf wire type {wire} at byte {pos}")


def onnx_precision(model):
    """INT8 when the graph quantizes (ONNX Runtime's CPU provider then runs those convolutions on integers), else FP32:
    that provider never lowers float arithmetic. Reads the operator of every top-level node (ModelProto.graph = field 7,
    GraphProto.node = 1, NodeProto.op_type = 4) with no onnx package, which the Pis and the phone do not have."""
    buf = memoryview(Path(model).read_bytes())
    ops = set()
    for field, graph in _protobuf_fields(buf, 0, len(buf)):
        if field == 7:
            for field, node in _protobuf_fields(buf, *graph):
                if field == 1:
                    ops.update(bytes(buf[slice(*v)]).decode() for k, v in _protobuf_fields(buf, *node) if k == 4)
    return "INT8" if ops & ONNX_INT8_OPS else "FP32"


class OnnxRuntime:
    def __init__(self, model, threads):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name, self.input_shape = inp.name, inp.shape
        self.version = ort.__version__
        self.info = {"providers": self.session.get_providers(), "intra_op_num_threads": threads, "inter_op_num_threads": 1,
                     "input_type": inp.type, "outputs": [(o.name, o.shape) for o in self.session.get_outputs()],
                     "precision_from": "the graph's operators: quantize/dequantize or integer operators mean INT8"}
        self.precision = _try(lambda: onnx_precision(model))

    def __call__(self, x):
        return self.session.run(None, {self.input_name: x})[0]


OPENVINO_COMPUTE = {"Convolution", "GroupConvolution", "Deconvolution", "MatMul", "FullyConnected"}
OPENVINO_TYPES = {"u8": "INT8", "i8": "INT8", "f16": "FP16", "bf16": "BF16", "f32": "FP32"}


def openvino_precision(layers):
    """[(layerType, runtimePrecision), ...] of a compiled model's runtime layers -> FP32, BF16, FP16, INT8, or None.

    INT8 when any convolution or matmul runs on integers (the quantized model's Detect head stays float), else the
    lowest float type a compute layer runs in: OpenVINO lowers an FP32 model to bf16 on a CPU with AVX-512 BF16 and to
    f16 on ARM unless told not to."""
    found = {OPENVINO_TYPES.get(p, p) for t, p in layers if t in OPENVINO_COMPUTE}
    return next((v for v in ("INT8", "FP16", "BF16", "FP32") if v in found), None)


class OpenVINO:
    def __init__(self, model, threads):
        import openvino as ov

        core = ov.Core()
        net = core.read_model(model=str(model), weights=str(model.with_suffix(".bin")))
        # What Ultralytics' OpenVINOBackend compiles with, on CPU explicitly (it picks AUTO when a GPU is visible), with
        # f32 accuracy mode forced on every platform: the ARM CPU plugin otherwise runs an FP32 model in f16, and an x86
        # CPU with AVX-512 BF16 (the laptop's Zen 4) in bf16. Ultralytics forces it on Linux ARM64 only. A quantized
        # model keeps its INT8 layers under these hints; only its float layers stay f32.
        config = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": threads,
                  "EXECUTION_MODE_HINT": ov.properties.hint.ExecutionMode.ACCURACY, "INFERENCE_PRECISION_HINT": ov.Type.f32}
        compiled = core.compile_model(net, "CPU", config)
        self.request = compiled.create_infer_request()
        self.input_shape = [d.get_length() if d.is_static else None for d in net.input().get_partial_shape()]
        self.version = ov.get_version()
        props = ("PERFORMANCE_HINT", "INFERENCE_NUM_THREADS", "INFERENCE_PRECISION_HINT", "EXECUTION_MODE_HINT",
                 "NUM_STREAMS", "ENABLE_HYPER_THREADING", "ENABLE_CPU_PINNING", "SCHEDULING_CORE_TYPE", "EXECUTION_DEVICES")
        self.info = {"device": "CPU", "requested": {k: str(v) for k, v in config.items()}, "effective": {k: _try(lambda k=k: str(compiled.get_property(k))) for k in props}}
        layers = _try(lambda: [(op.get_rt_info()["layerType"].astype(str), op.get_rt_info()["runtimePrecision"].astype(str))
                               for op in compiled.get_runtime_model().get_ops()], [])
        self.precision = openvino_precision(layers)
        self.info["compute_layers"] = {f"{t} {p}": n for (t, p), n in sorted(collections.Counter(x for x in layers if x[0] in OPENVINO_COMPUTE).items())}
        self.info["precision_from"] = "the runtime precision of the compiled model's convolution and matmul layers"
        if self.precision in ("BF16", "FP16"):
            sys.exit(f"OpenVINO {self.version} compiled {model} in {self.precision} despite INFERENCE_PRECISION_HINT=f32: "
                     f"not a cell (INVARIANTS.md §8). Layers: {self.info['compute_layers']}")

    def __call__(self, x):
        self.request.infer({0: x})
        return self.request.get_output_tensor(0).data.copy()


def ncnn_precision(opt, cpu):
    """ncnn's options and CPU probes -> FP16 or FP32 (None when the probe failed): what its convolutions compute in.

    On the CPU, ncnn computes a layer in FP16 only with fp16 storage and fp16 arithmetic on (its defaults) and ARMv8.2
    FP16 arithmetic in the CPU (cpu_support_arm_asimdhp, a 64-bit OS only), e.g. the Pi 5's Cortex-A76 (net.cpp
    convert_layout, layer/arm/convolution_arm.cpp, tag 20260526). The Pi 4B's Cortex-A72 and x86 compute in FP32; they
    store at most a few layers' weights in fp16."""
    asimdhp = cpu.get("cpu_support_arm_asimdhp")
    if asimdhp is None:
        return None
    return "FP16" if opt.get("use_fp16_storage") and opt.get("use_fp16_arithmetic") and asimdhp else "FP32"


class NCNN:
    def __init__(self, model, threads):
        import ncnn

        self.ncnn, self.net = ncnn, ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = threads  # everything else stays at ncnn's defaults, fp16 storage/arithmetic included
        self.net.load_param(str(model))
        self.net.load_model(str(model.with_suffix(".bin")))
        self.input_name, self.output_name = self.net.input_names()[0], sorted(self.net.output_names())[0]
        self.input_shape = None
        self.version = ncnn.__version__
        opt = self.net.opt
        probes = [f for f in dir(ncnn) if f.startswith("cpu_support_") or (f.startswith("get_") and f.endswith("_count"))]
        self.info = {"opt": {k: getattr(opt, k) for k in dir(opt) if k.startswith(("use_", "num_threads", "lightmode")) and isinstance(getattr(opt, k), (bool, int))},
                     "cpu": {f: _try(getattr(ncnn, f)) for f in probes},
                     "precision_from": "ncnn's fp16 storage and arithmetic options and cpu_support_arm_asimdhp (ARMv8.2 FP16)"}
        self.precision = ncnn_precision(self.info["opt"], self.info["cpu"])

    def __call__(self, x):
        with self.net.create_extractor() as ex:
            ex.input(self.input_name, self.ncnn.Mat(x[0]))
            return np.array(ex.extract(self.output_name)[1])[None]

    def close(self):
        """The pip wheel creates a Vulkan instance even for CPU inference; left to interpreter shutdown it segfaults (Windows)."""
        self.net.clear()
        del self.net
        if hasattr(self.ncnn, "destroy_gpu_instance"):
            self.ncnn.destroy_gpu_instance()


def engine_header(path):
    """The metadata header Ultralytics writes ahead of an engine (4-byte little-endian length, then JSON): (offset, dict)."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(4), "little")
        if 0 < n <= f.seek(0, 2) - 4:
            f.seek(4)
            try:
                return 4 + n, json.loads(f.read(n))
            except ValueError:  # engine bytes, not a header
                pass
    return 0, {}


TRT_TYPES = (("INT8", re.compile(r"\bInt8\b|\bINT8\b")), ("FP16", re.compile(r"FP16|\bHalf\b")), ("FP32", re.compile(r"FP32|\bFloat\b")))
TRT_QUANTIZE = {8: "INT8", 16: "FP16", None: "FP32", 32: "FP32"}


def tensorrt_precision(header, layers):
    """(precision, where it came from) of an engine, from its inspector's "Layers" and its Ultralytics header.

    Ultralytics builds INT8 engines with DETAILED profiling, so their layers list their input/output formats and weight
    types: INT8 when any layer runs on Int8, else FP16 when any runs in half, else FP32. It builds FP16 engines without,
    so their layers are names only and the header's args.quantize (8, 16, or unset for FP32) is all there is."""
    built = TRT_QUANTIZE.get((header.get("args") or {}).get("quantize", "missing"))
    detail = [x for x in layers or [] if isinstance(x, dict)]
    if not detail:
        return built, "the engine header's args.quantize (the engine lists layer names only)" if built else "unknown: no layer detail and no header"
    tokens = [s for x in detail for s in [*(t.get("Format/Datatype", "") for t in x.get("Inputs", []) + x.get("Outputs", [])),
                                          *((x.get(k) or {}).get("Type", "") for k in ("Weights", "Bias"))]]
    found = next((name for name, pattern in TRT_TYPES if any(pattern.search(s) for s in tokens)), None)
    note = "" if found == built or built is None else f"; the header says {built}: they disagree"
    return found, f"the engine's layer formats ({len(detail)} layers){note}"


class TensorRT:
    """A static batch-1 engine built by studyexport.py on this platform, through TensorRT's named-tensor API (8.5 and later,
    so JetPack 6.0's 8.6 and 6.1/6.2's 10.3 alike). One non-default stream; net time covers the whole GPU round trip."""

    def __init__(self, model, threads):
        # Device and pinned host buffers and the stream come from torch, imported here and nowhere else in this file.
        # JetPack has no Python CUDA binding (pycuda and cuda-python are extra installs), torch with CUDA is on the board
        # anyway for the export, and it is what Ultralytics' TensorRTBackend uses. The Pi and the phone never get here.
        import torch
        import tensorrt as trt

        self.torch, offset, header = torch, *engine_header(model)
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(model, "rb") as f:
            f.seek(offset)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            sys.exit(f"TensorRT {trt.__version__} cannot load {model}: an engine runs only on the GPU and TensorRT version that "
                     "built it. Build it on this platform with studyexport.py")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outputs = sorted(n for n in names if n not in inputs)  # Ultralytics' order: output0 first
        if len(inputs) != 1:
            sys.exit(f"{model}: expected one input tensor, found {inputs}")
        self.buffers, io = {}, []
        for n in names:
            shape = tuple(self.engine.get_tensor_shape(n))
            if -1 in shape:
                sys.exit(f"{model}: {n} has a dynamic shape {shape}; studyexport.py builds static batch-1 engines")
            dtype = trt.nptype(self.engine.get_tensor_dtype(n))
            tdtype = torch.from_numpy(np.empty(0, dtype)).dtype
            host = torch.empty(shape, dtype=tdtype, pin_memory=True)
            device = torch.empty(shape, dtype=tdtype, device="cuda")
            self.context.set_tensor_address(n, device.data_ptr())
            self.buffers[n] = (host, device)
            io.append([n, "input" if n in inputs else "output", list(shape), np.dtype(dtype).name])
        self.inputs, self.outputs = inputs, outputs
        self.host_in = self.buffers[inputs[0]][0].numpy()
        self.host_out = self.buffers[outputs[0]][0].numpy()
        self.input_shape = [int(d) for d in self.engine.get_tensor_shape(inputs[0])]
        self.version = trt.__version__
        mem = getattr(self.engine, "device_memory_size_v2", None)
        self.info = {"device": torch.cuda.get_device_name(0), "compute_capability": list(torch.cuda.get_device_capability(0)),
                     "torch": torch.__version__, "torch_cuda": torch.version.cuda, "io": io,
                     "device_memory_bytes": mem if mem is not None else getattr(self.engine, "device_memory_size", None),
                     "engine_header": {k: header.get(k) for k in ("version", "date", "imgsz", "batch", "end2end", "args")},
                     "buffers": "torch pinned host + device tensors, one non-default CUDA stream",
                     "net": "np.copyto into pinned input, H2D copy, execute_async_v3, D2H copy, stream sync",
                     "threads": f"TensorRT does not use --threads; it sets OpenCV's threads only ({threads})"}
        layers = _try(lambda: json.loads(self.engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))["Layers"])
        self.precision, self.info["precision_from"] = tensorrt_precision(header, layers)

    def __call__(self, x):
        """The output is the pinned host buffer itself, valid until the next call (post-processing reads it at once)."""
        np.copyto(self.host_in, x, casting="same_kind")
        with self.torch.cuda.stream(self.stream):
            for n in self.inputs:
                host, device = self.buffers[n]
                device.copy_(host, non_blocking=True)
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 failed")
            for n in self.outputs:
                host, device = self.buffers[n]
                host.copy_(device, non_blocking=True)
        self.stream.synchronize()
        return self.host_out

    def close(self):
        """Free in dependency order while CUDA is still up; left to interpreter shutdown, TensorRT may crash on exit."""
        self.stream.synchronize()
        del self.context, self.engine, self.runtime, self.buffers


RUNTIMES = {"onnxruntime": OnnxRuntime, "openvino": OpenVINO, "ncnn": NCNN, "tensorrt": TensorRT}


# -------------------------------------------------------------- platform state (fails soft: Android hides most of it)


def _try(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _read(path):
    return _try(lambda: Path(path).read_text(encoding="utf-8", errors="replace").strip())


def cpu_temp_c():
    v = _read("/sys/class/thermal/thermal_zone0/temp")
    return round(int(v) / 1000, 1) if v and v.lstrip("-").isdigit() else None


def cpu_mhz():
    v = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    return round(int(v) / 1000) if v and v.isdigit() else None


def throttled():
    """vcgencmd get_throttled as an int (Raspberry Pi only), else None."""
    if not shutil.which("vcgencmd"):
        return None
    out = _try(lambda: subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5).stdout)
    return _try(lambda: int(out.strip().split("=")[1], 16))


@functools.lru_cache(maxsize=None)
def _jetson():
    """jetson_probe (beside this file) on a Jetson, else None. Imported lazily, so the Pi and the phone never need it."""
    if not os.path.exists("/etc/nv_tegra_release"):
        return None
    import jetson_probe

    return jetson_probe


@functools.lru_cache(maxsize=None)
def _gpu_nodes():
    """The Jetson's GPU thermal-zone and devfreq clock files, found once; (None, None) elsewhere."""
    jp = _jetson()
    if jp is None:
        return None, None
    zone = next((z / "temp" for z in sorted(Path("/sys/devices/virtual/thermal").glob("thermal_zone*"))
                 if "gpu" in (_read(z / "type") or "").lower()), None)
    dev = jp.gpu_devfreq()
    return zone, (dev / "cur_freq" if dev else None)


def gpu_state():
    """(GPU temperature in C, GPU clock in MHz) for the per-image columns; (None, None) off a Jetson."""
    t, f = (_read(p) if p else None for p in _gpu_nodes())
    return (round(int(t) / 1000, 1) if t and t.lstrip("-").isdigit() else None, round(int(f) / 1e6) if f and f.isdigit() else None)


def jetson_summary(samples, rows, start_state):
    """samples: [[image index, jetson_probe.sample()], ...] taken at the start, every 50 images and the end of each session.

    The throttle signals a user can read without root: an over-current counter that rose between the first and the last
    sample, and images whose GPU clock was below the maximum it was pinned to (jetson_clocks) at the start."""
    s = [x for _, x in samples if x]
    if not s:
        return None
    oc = {k: s[-1]["oc_events"].get(k, v) - v for k, v in s[0]["oc_events"].items()}
    pinned = ((start_state or {}).get("gpu_clocks") or {}).get("max")
    below = sum(1 for r in rows if r.get("gpu_mhz") and pinned and float(r["gpu_mhz"]) < pinned)
    return {"temp_c_max": {z: max(x["temp_c"][z] for x in s if z in x["temp_c"]) for z in s[0]["temp_c"]},
            "oc_events_during_run": oc, "over_current_throttled": any(v > 0 for v in oc.values()) if oc else None,
            "gpu_max_mhz_at_start": pinned, "images_gpu_below_max": below if pinned else None, "samples": samples}


def platform_info():
    cpuinfo = _read("/proc/cpuinfo") or ""
    fields = {}
    for line in cpuinfo.splitlines():
        k, _, v = line.partition(":")
        fields.setdefault(k.strip(), v.strip())
    cpu = fields.get("model name") or fields.get("Hardware") or fields.get("Model")
    if not cpu and platform.system() == "Windows":
        import winreg

        cpu = _try(lambda: winreg.QueryValueEx(winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"), "ProcessorNameString")[0].strip())
    getprop = shutil.which("getprop")
    android = {k: _try(lambda k=k: subprocess.run([getprop, k], capture_output=True, text=True, timeout=5).stdout.strip()) for k in
               ("ro.product.manufacturer", "ro.product.model", "ro.soc.model", "ro.build.version.release")} if getprop else None
    return {
        "uname": platform.uname()._asdict(),
        "cpu_model": cpu or platform.processor() or None,
        "cpu_part": fields.get("CPU part"),
        "device_model": (_read("/proc/device-tree/model") or "").rstrip("\x00") or None,
        "android": android,
        "cpu_count": os.cpu_count(),
        "cpu_affinity": _try(lambda: len(os.sched_getaffinity(0))),
        "cpu_max_mhz": [_try(lambda c=c: round(int(_read(f"/sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq")) / 1000))
                        for c in range(os.cpu_count() or 0)] if os.path.isdir("/sys/devices/system/cpu") else None,
        "governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "mem_total_kb": _try(lambda: int(next(l for l in _read("/proc/meminfo").splitlines() if l.startswith("MemTotal")).split()[1])),
        "jetson": _jetson().board_state() if _jetson() else None,
    }


def max_rss_mb():
    import resource  # not on Windows

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1 << 20 if sys.platform == "darwin" else 1 << 10), 1)


# ---------------------------------------------------------------- bookkeeping


def sha256_files(artifact):
    p = Path(artifact)
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file() and f.name != "manifest.json")
    root = p.parent if p.is_file() else p
    digests = {}
    for f in files:
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        digests[f.relative_to(root).as_posix()] = h.hexdigest()
    return digests


def find_manifest(artifact):
    p = Path(artifact).resolve()
    for d in [p if p.is_dir() else p.parent, *list(p.parents)[:2]]:
        if (d / "manifest.json").is_file():
            return json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    return None


def imread(path):
    """utils/patches.py imread: np.fromfile + cv2.imdecode, IMREAD_COLOR (BGR)."""
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_progress(out):
    """Images already recorded by a previous session: the ones complete in both predictions.jsonl and timings.csv."""
    preds, rows = {}, {}
    if not (out / "predictions.jsonl").exists() or not (out / "timings.csv").exists():
        return set()
    with open(out / "predictions.jsonl", "rb") as f:
        for line in f:
            try:
                rec = json.loads(line)
                preds[rec["file"]] = line
            except ValueError:  # a line cut off by the crash
                break
    with open(out / "timings.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("session") not in (None, ""):
                rows[row["file"]] = row
    done = [k for k in preds if k in rows]
    with open(out / "predictions.jsonl", "wb") as f:  # rewrite both to the common, complete set
        f.writelines(preds[k] for k in done)
    with open(out / "timings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(rows[k] for k in done)
    return set(done)


def stage_stats(out):
    with open(out / "timings.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = {k: np.array([float(r[k]) for r in rows]) for k in ("t_pre_ms", "t_net_ms", "t_post_ms", "t_post_deploy_ms")}
    cols["t_total_ms"] = cols["t_pre_ms"] + cols["t_net_ms"] + cols["t_post_ms"]  # at the evaluation threshold
    cols["post_deploy"] = cols.pop("t_post_deploy_ms")
    cols["e2e_deploy"] = cols["t_pre_ms"] + cols["t_net_ms"] + cols["post_deploy"]  # the latency
    stats = {k: {"median": round(float(np.median(v)), 3), "p95": round(float(np.percentile(v, 95)), 3),
                 "mean": round(float(v.mean()), 3), "min": round(float(v.min()), 3), "max": round(float(v.max()), 3)} for k, v in cols.items()}

    def spread(col):
        v = [float(r[col]) for r in rows if r.get(col)]
        return {"min": min(v), "max": max(v), "median": float(np.median(v))} if v else None

    n_cand = [int(r["n_cand"]) for r in rows]
    return rows, stats, {
        "temp_c": spread("temp_c"),
        "cpu_mhz": spread("cpu_mhz"),
        "gpu_temp_c": spread("gpu_temp_c"),
        "gpu_mhz": spread("gpu_mhz"),
        "n_cand_max": max(n_cand) if n_cand else None,
        "images_over_max_nms": sum(n > MAX_NMS for n in n_cand),
    }


def finalize_predictions(out):
    """predictions.jsonl -> predictions.json, the list DetectionValidator.pred_to_json builds, streamed to keep memory flat."""
    n = 0
    with open(out / "predictions.jsonl", encoding="utf-8") as src, open(out / "predictions.json.tmp", "w", encoding="utf-8") as dst:
        dst.write("[")
        for line in src:
            rec = json.loads(line)
            stem = Path(rec["file"]).stem
            image_id = int(stem) if stem.isnumeric() else stem
            for x, y, w, h, s, c in rec["dets"]:
                dst.write(("," if n else "") + json.dumps({"image_id": image_id, "file_name": rec["file"], "category_id": c, "bbox": [x, y, w, h], "score": s}))
                n += 1
        dst.write("]")
    os.replace(out / "predictions.json.tmp", out / "predictions.json")
    return n


def throttle_summary(samples):
    """samples: [[image index, vcgencmd get_throttled value or None], ...] taken at start, every 50 images and end."""
    vals = [v for _, v in samples if v is not None]
    if not vals:
        return None
    seen = 0
    for v in vals:
        seen |= v
    # bits 0-3 are the state now, 16-19 "has occurred since boot"
    flags = {"under_voltage": 0, "freq_capped": 1, "throttled": 2, "soft_temp_limit": 3}
    return {"start": hex(vals[0]), "end": hex(vals[-1]), "or_of_samples": hex(seen),
            "during_run": {k: bool(seen & (1 << b)) for k, b in flags.items()},
            "since_boot": {k: bool(seen & (1 << (b + 16))) for k, b in flags.items()},
            "samples": [[i, hex(v)] for i, v in samples if v is not None]}


# ---------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--artifact", required=True, help="a cell directory written by studyexport.py, or the model file/directory")
    p.add_argument("--runtime", required=True, choices=sorted(RUNTIMES))
    p.add_argument("--imgsz", type=int, required=True)
    p.add_argument("--images", required=True, help="the VisDrone test-dev images directory")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=4, help="runtime and OpenCV threads (recorded)")
    p.add_argument("--warmup", type=int, default=20, help="images run untimed first")
    p.add_argument("--limit", type=int, default=None, help="time only the first N images (a quick check, not a cell)")
    p.add_argument("--platform-tag", default=None, help="e.g. pi5, pi4b, m34, jetson, laptop")
    p.add_argument("--resume", action="store_true", help="continue a run that died")
    p.add_argument("--overwrite", action="store_true", help="discard an existing run in --out")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    all_files = sorted(f for f in os.listdir(a.images) if f.lower().endswith(IMG_SUFFIXES))  # the validator's order
    files = all_files[: a.limit] if a.limit else all_files
    model_file = find_model(a.artifact, a.runtime)
    digests = sha256_files(a.artifact)
    manifest = find_manifest(a.artifact)
    if manifest and manifest.get("files") and any(manifest["files"].get(k) not in (None, v) for k, v in digests.items()):
        sys.exit(f"{a.artifact} does not match its manifest.json: re-copy the artifact")
    if manifest and (manifest.get("imgsz") != a.imgsz or manifest.get("runtime") != a.runtime):
        sys.exit(f"--runtime {a.runtime} --imgsz {a.imgsz}, but the manifest says {manifest.get('runtime')} at {manifest.get('imgsz')}")
    config = {"runtime": a.runtime, "imgsz": a.imgsz, "threads": a.threads, "artifact_sha256": digests,
              "images": len(files), "images_sha1": hashlib.sha1("\n".join(files).encode()).hexdigest(),
              "deploy_conf": DEPLOY_CONF}  # also keeps a run started before the deployment pass from resuming into this one

    meta_path = out / "meta.json"
    old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    if old and a.overwrite:
        for f in ("meta.json", "predictions.json", "predictions.jsonl", "timings.csv"):
            (out / f).unlink(missing_ok=True)
        old = None
    if old and old.get("status") == "complete":
        sys.exit(f"{out} holds a complete run; pass --overwrite to replace it")
    if old and not a.resume:
        sys.exit(f"{out} holds an unfinished run; pass --resume to continue it or --overwrite to restart")
    if old and old["config"] != config:
        sys.exit(f"{out} was started with a different configuration; --overwrite to restart:\n{old['config']}\nvs\n{config}")
    if not old:  # leftovers without a meta.json belong to no run
        for f in ("predictions.json", "predictions.jsonl", "timings.csv"):
            (out / f).unlink(missing_ok=True)
    done = load_progress(out) if old else set()

    cv2.setNumThreads(a.threads)
    t_load = time.perf_counter()
    rt = RUNTIMES[a.runtime](model_file, a.threads)
    load_s = time.perf_counter() - t_load
    if rt.input_shape and rt.input_shape[-2:] != [a.imgsz, a.imgsz]:
        sys.exit(f"--imgsz {a.imgsz} but the model input is {rt.input_shape}")

    meta = old or {
        "tool": "studybench.py",
        "status": "running",
        "platform_tag": a.platform_tag,
        "config": config,
        "artifact": str(a.artifact),
        "model_file": str(model_file),
        "manifest": manifest,
        "imgsz": a.imgsz,
        "threads": a.threads,
        "warmup": a.warmup,
        "limit": a.limit,
        "images_dir": str(a.images),
        "complete_testdev": a.limit is None,
        "preprocess": "validator rect=False batch 1: ceil long-side resize INTER_LINEAR, LetterBox pad 114 scaleup=False, BGR->RGB, CHW, /255",
        "postprocess": {"conf": CONF, "iou": IOU, "max_det": MAX_DET, "max_nms": MAX_NMS, "max_wh": MAX_WH, "multi_label": True, "agnostic": False},
        "deploy_conf": DEPLOY_CONF,
        "post_passes": f"two per image on the same model output, each timed: at conf {CONF} (t_post_ms, saved for accuracy) and at "
                       f"{DEPLOY_CONF} (t_post_deploy_ms, timed only); the {DEPLOY_CONF} pass runs first on odd image indices",
        "timed": "pre + net + post per image; image read/decode and COCO conversion excluded",
        "latency": f"e2e_deploy = pre + net + post at {DEPLOY_CONF}; t_total_ms is pre + net + post at {CONF}",
        "sessions": [],
    }
    meta["status"] = "running"
    meta["platform"] = platform_info()
    meta["versions"] = {"python": platform.python_version(), "numpy": np.__version__, "cv2": cv2.__version__, a.runtime: rt.version}
    if old and old.get("precision") != rt.precision:
        sys.exit(f"{out} was started computing in {old.get('precision') or 'an unrecorded precision'}, and this session "
                 f"computes in {rt.precision}: one run, one precision. --overwrite to restart")
    meta["runtime_info"] = rt.info
    meta["precision"] = rt.precision  # what the runtime computes in here, from its own evidence (runtime_info.precision_from)
    meta["cv2_threads"] = cv2.getNumThreads()
    session = {"index": len(meta["sessions"]), "start": now(), "first_image": len(done), "model_load_s": round(load_s, 3), "warmup": a.warmup}
    meta["sessions"].append(session)
    throttle_samples = meta.setdefault("throttle_samples", [])
    throttle_samples.append([len(done), throttled()])
    board = meta["platform"]["jetson"]
    if board:
        meta.setdefault("jetson_samples", []).append([len(done), _jetson().sample()])
        meta.setdefault("jetson_state_at_start", board)
        if board["jetson_clocks"] is not True or (board["highest_mode"] and board["nvpmodel_mode"] != board["highest_mode"]):
            print(f"WARNING: the study runs a Jetson in its highest nvpmodel mode {board['highest_mode']} with jetson_clocks on; "
                  f"this board is in {board['nvpmodel_mode']}, jetson_clocks {board['jetson_clocks']}. meta.json records it.")
    write_json(meta_path, meta)

    head = None

    def infer(im, i):
        """Image i through pre, net and both post-processing passes; the deployment pass goes first on odd i."""
        nonlocal head
        t0 = time.perf_counter_ns()
        x, ratio_pad = preprocess(im, a.imgsz)
        t1 = time.perf_counter_ns()
        y = rt(x)[0]
        t2 = time.perf_counter_ns()
        if head is None:
            head = "end2end" if y.shape[-1] == 6 else "raw"
        saved, t_post, deploy, t_post_deploy = two_passes(y, head, ratio_pad, im.shape[:2], deploy_first=i % 2 == 1)
        return saved, len(deploy[1]), ((t1 - t0) / 1e6, (t2 - t1) / 1e6, t_post, t_post_deploy)

    for i, f in enumerate(all_files[: a.warmup]):
        infer(imread(Path(a.images) / f), i)
    session["warmup_done"] = now()

    todo = [(i, f) for i, f in enumerate(files) if f not in done]
    try:
        run_images(a, out, todo, infer, session, throttle_samples, meta)
    except BaseException as e:  # Ctrl-C, a dropped SSH session, a runtime error: leave the reason where the next session looks
        session["end"], meta["status"], meta["error"] = now(), "interrupted", f"{type(e).__name__}: {e}"
        write_json(meta_path, meta)
        raise
    finally:
        getattr(rt, "close", lambda: None)()

    throttle_samples.append([len(files), throttled()])
    if "jetson_samples" in meta:
        meta["jetson_samples"].append([len(files), _jetson().sample()])
    rows, stats, extra = stage_stats(out)
    if len(rows) != len(files):
        sys.exit(f"timings.csv has {len(rows)} rows for {len(files)} images: run again with --resume")
    session["end"] = now()
    meta.pop("throttle_samples")
    meta.pop("error", None)
    meta.update(status="complete", head=head, start=meta["sessions"][0]["start"], end=session["end"], images_timed=len(rows),
                detections=finalize_predictions(out), timing_ms=stats, device_state=extra, throttle=throttle_summary(throttle_samples),
                jetson=jetson_summary(meta.pop("jetson_samples", []), rows, meta.get("jetson_state_at_start")),
                max_rss_mb=_try(max_rss_mb), resumed=len(meta["sessions"]) > 1)
    write_json(meta_path, meta)
    (out / "predictions.jsonl").unlink()
    t = stats
    print(f"done: {len(rows)} images, {a.runtime} {rt.precision}, head {head}; median pre {t['t_pre_ms']['median']} net {t['t_net_ms']['median']} ms; "
          f"latency (e2e_deploy, conf {DEPLOY_CONF}) {t['e2e_deploy']['median']} ms (p95 {t['e2e_deploy']['p95']}), post {t['post_deploy']['median']} ms; "
          f"at conf {CONF}: total {t['t_total_ms']['median']}, post {t['t_post_ms']['median']} ms")


def run_images(a, out, todo, infer, session, throttle_samples, meta):
    """The timed pass: one line in predictions.jsonl and one row in timings.csv per image, flushed as it finishes."""
    new_csv = not (out / "timings.csv").exists()
    with open(out / "predictions.jsonl", "a", encoding="utf-8") as pj, open(out / "timings.csv", "a", newline="", encoding="utf-8") as tc:
        writer = csv.DictWriter(tc, COLUMNS)
        if new_csv:
            writer.writeheader()
        for k, (i, f) in enumerate(todo, 1):
            im = imread(Path(a.images) / f)
            if im is None:
                sys.exit(f"cannot read {f}")
            (boxes, scores, cls, n_cand), n_dets_deploy, (t_pre, t_net, t_post, t_post_deploy) = infer(im, i)
            pj.write(json.dumps({"file": f, "dets": to_coco(boxes, scores, cls)}, separators=(",", ":")) + "\n")
            thr = throttled() if k % THROTTLE_EVERY == 0 else None
            if thr is not None:
                throttle_samples.append([i, thr])
            if k % THROTTLE_EVERY == 0 and "jetson_samples" in meta:
                meta["jetson_samples"].append([i, _jetson().sample()])
            gpu_temp, gpu_mhz = gpu_state()
            writer.writerow({"index": i, "file": f, "t_pre_ms": f"{t_pre:.3f}", "t_net_ms": f"{t_net:.3f}", "t_post_ms": f"{t_post:.3f}",
                             "n_cand": n_cand, "n_dets": len(scores), "temp_c": cpu_temp_c(), "cpu_mhz": cpu_mhz(),
                             "throttled": hex(thr) if thr is not None else None, "img_crc32": f"{zlib.crc32(im):08x}", "session": session["index"],
                             "gpu_temp_c": gpu_temp, "gpu_mhz": gpu_mhz, "t_post_deploy_ms": f"{t_post_deploy:.3f}", "n_dets_deploy": n_dets_deploy})
            pj.flush()
            tc.flush()
            if k % FSYNC_EVERY == 0:
                os.fsync(pj.fileno())
                os.fsync(tc.fileno())
                session["last_image"], session["last_update"] = i, now()
                write_json(out / "meta.json", meta)
            if k % 100 == 0 or k == len(todo):
                print(f"[{k}/{len(todo)}] {f} pre {t_pre:.1f} net {t_net:.1f} post {t_post:.1f} (at {DEPLOY_CONF}: {t_post_deploy:.1f}) ms, "
                      f"{len(scores)} dets ({n_dets_deploy} at {DEPLOY_CONF})", flush=True)


if __name__ == "__main__":
    main()
