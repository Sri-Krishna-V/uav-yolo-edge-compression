"""Check the benchmark's precision verdicts: what each runtime really computes in, from the evidence it gives. Runs on any
machine with numpy and cv2 (the study venv):

  python study/scripts/tests/test_precision.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import studybench as sb  # noqa: E402

FAILED = []


def check(label, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        FAILED.append(label)
        if detail:
            print(f"     {detail}")


# ---------------------------------------------------------------- 1: OpenVINO, from the compiled model's runtime layers
# Layer counts as the laptop's OpenVINO 2026.4.1 CPU plugin compiled yolo11s-640 (Ryzen 7 8845HS, 22 October 2026).
LAPTOP_DEFAULT_FP32 = [("Convolution", "bf16")] * 88 + [("MatMul", "bf16")] * 2 + [("Concat", "bf16"), ("Parameter", "f32")]
FORCED_FP32 = [("Convolution", "f32")] * 88 + [("MatMul", "f32")] * 2 + [("Concat", "f32"), ("Parameter", "f32")]
FORCED_INT8 = [("Convolution", "u8")] * 86 + [("Convolution", "f32")] * 2 + [("MatMul", "f32")] * 2 + [("FakeQuantize", "f32")]
ARM_DEFAULT_FP32 = [("Convolution", "f16")] * 88 + [("MatMul", "f16")] * 2

v = sb.openvino_precision(FORCED_FP32)
check("1 OpenVINO: f32 compute layers are FP32", v == "FP32", v)
v = sb.openvino_precision(LAPTOP_DEFAULT_FP32)
check("1 OpenVINO: an FP32 model computed in bf16 is BF16, not FP32", v == "BF16", v)
v = sb.openvino_precision(FORCED_INT8)
check("1 OpenVINO: u8 convolutions are INT8, though the head stays f32", v == "INT8", v)
v = sb.openvino_precision(ARM_DEFAULT_FP32)
check("1 OpenVINO: f16 compute layers are FP16", v == "FP16", v)
v = sb.openvino_precision([("Concat", "bf16"), ("Parameter", "f32")])
check("1 OpenVINO: no compute layer gives no verdict", v is None, v)

# ---------------------------------------------------------------- 2: ONNX Runtime, from the graph's operators
def varint(n):
    out = bytearray()
    while True:
        out.append((n & 0x7F) | (0x80 if n > 0x7F else 0))
        n >>= 7
        if not n:
            return bytes(out)


def field(num, payload):
    """A length-delimited protobuf field; an int payload is a varint field instead."""
    if isinstance(payload, int):
        return varint(num << 3) + varint(payload)
    return varint(num << 3 | 2) + varint(len(payload)) + payload


def onnx_model(*ops):
    """A minimal ModelProto: ir_version, a graph of nodes (op_type is NodeProto field 4) with an initializer, an opset."""
    nodes = b"".join(field(1, field(1, b"x") + field(2, b"y") + field(3, f"n{i}".encode()) + field(4, op.encode())) for i, op in enumerate(ops))
    graph = nodes + field(2, b"main") + field(5, field(1, 1) + field(8, b"w") + field(9, bytes(300)))
    return field(1, 10) + field(2, b"pytorch") + field(7, graph) + field(8, field(2, 17))


def write(blob):
    path = Path(tempfile.mkdtemp()) / "model.onnx"
    path.write_bytes(blob)
    return path


def onnx_verdict(*ops):
    return sb.onnx_precision(write(onnx_model(*ops)))


v = onnx_verdict("Conv", "Sigmoid", "Mul", "Concat")
check("2 ONNX: a float graph is FP32", v == "FP32", v)
v = onnx_verdict("QuantizeLinear", "DequantizeLinear", "Conv", "Sigmoid")
check("2 ONNX: a QDQ graph (quantize_static) is INT8", v == "INT8", v)
v = onnx_verdict("QLinearConv", "Sigmoid")
check("2 ONNX: an operator-oriented quantized graph is INT8", v == "INT8", v)
v = onnx_verdict("Conv", "MatMulInteger")
check("2 ONNX: an integer matmul is INT8", v == "INT8", v)
v = sb.onnx_precision(write(field(7, field(1, field(3, b"QuantizeLinear_7") + field(4, b"Conv")))))
check("2 ONNX: an operator name inside a node's name does not count", v == "FP32", v)
EXPORTS = Path(__file__).resolve().parents[2] / "exports" / "yolo11s-640-s0" / "640"
if (EXPORTS / "onnxruntime-int8").is_dir():
    v = sb.onnx_precision(EXPORTS / "onnxruntime-fp32" / "model.onnx"), sb.onnx_precision(EXPORTS / "onnxruntime-int8" / "model_int8.onnx")
    check("2 ONNX: the laptop's real yolo11s exports are FP32 and INT8", v == ("FP32", "INT8"), v)
else:
    print("skip 2 (no yolo11s exports on this machine)")

# ---------------------------------------------------------------- 3: TensorRT, from the engine, else its Ultralytics header
# Ultralytics builds only INT8 engines with DETAILED profiling (utils/export/engine.py onnx2engine), so only they list
# each layer's formats; an FP16 engine lists layer names alone, and its header's args.quantize says what was built.
def trt_layer(fmt, weights=None):
    layer = {"Name": "conv", "LayerType": "CaskConvolution", "Inputs": [{"Name": "x", "Format/Datatype": fmt}],
             "Outputs": [{"Name": "y", "Format/Datatype": fmt}]}
    return {**layer, "Weights": {"Type": weights, "Count": 9}} if weights else layer


INT8_LAYERS = [trt_layer("Row major linear FP32 format"), trt_layer("Thirty-two wide channel vectorized row major Int8 format.", "Int8"),
               trt_layer("Row major linear FP32 format", "Float")]
NAMES_ONLY = ["/model.0/conv/Conv + PWN(/model.0/act/Sigmoid)", "/model.1/conv/Conv"]

v = sb.tensorrt_precision({"args": {"quantize": 16}}, NAMES_ONLY)
check("3 TensorRT: an FP16 engine without layer detail is FP16, from its header", v[0] == "FP16" and "header" in v[1], v)
v = sb.tensorrt_precision({"args": {"quantize": 8}}, INT8_LAYERS)
check("3 TensorRT: Int8 layer formats are INT8, from the engine", v[0] == "INT8" and "engine" in v[1], v)
v = sb.tensorrt_precision({"args": {"quantize": 8}}, [trt_layer("Row major linear FP32 format", "Float")] * 3)
check("3 TensorRT: an INT8 build whose layers all stayed FP32 is FP32, and says the header disagrees",
      v[0] == "FP32" and "disagree" in v[1], v)
v = sb.tensorrt_precision({"args": {"quantize": 16}}, [trt_layer("Channel major FP16 format where channel % 8 == 0", "Half")])
check("3 TensorRT: Half layer formats are FP16", v[0] == "FP16", v)
v = sb.tensorrt_precision({"args": {"quantize": None}}, NAMES_ONLY)
check("3 TensorRT: no quantize in the header is FP32", v[0] == "FP32", v)
v = sb.tensorrt_precision({}, NAMES_ONLY)
check("3 TensorRT: neither layer detail nor a header gives no verdict", v[0] is None, v)

# ---------------------------------------------------------------- 4: NCNN, from its options and its CPU probes
# The five probes the ncnn 1.0.20260526 wheel exposes. They return raw hwcap bitmasks, not booleans (the laptop's
# cpu_support_x86_avx2() is 32). HWCAP_ASIMDHP is bit 10 on aarch64 Linux.
DEFAULT_OPT = {"use_fp16_packed": True, "use_fp16_storage": True, "use_fp16_arithmetic": True, "use_bf16_storage": False,
               "use_int8_inference": True, "use_vulkan_compute": False, "num_threads": 4}
A76 = {"cpu_support_arm_neon": 1, "cpu_support_arm_vfpv4": 1, "cpu_support_arm_asimdhp": 1024, "cpu_support_x86_avx": 0, "cpu_support_x86_avx2": 0}
A72 = {**A76, "cpu_support_arm_asimdhp": 0}
ZEN4 = {"cpu_support_arm_neon": 0, "cpu_support_arm_vfpv4": 0, "cpu_support_arm_asimdhp": 0, "cpu_support_x86_avx": 1, "cpu_support_x86_avx2": 32}

v = sb.ncnn_precision(DEFAULT_OPT, A76)
check("4 NCNN: defaults on a Cortex-A76 (Pi 5) compute in FP16", v == "FP16", v)
v = sb.ncnn_precision(DEFAULT_OPT, A72)
check("4 NCNN: defaults on a Cortex-A72 (Pi 4B, no ARMv8.2 FP16) compute in FP32", v == "FP32", v)
v = sb.ncnn_precision(DEFAULT_OPT, ZEN4)
check("4 NCNN: defaults on x86 compute in FP32", v == "FP32", v)
v = sb.ncnn_precision({**DEFAULT_OPT, "use_fp16_arithmetic": False}, A76)
check("4 NCNN: FP16 storage without FP16 arithmetic computes in FP32", v == "FP32", v)
v = sb.ncnn_precision({**DEFAULT_OPT, "use_fp16_storage": False}, A76)
check("4 NCNN: FP16 arithmetic without FP16 storage computes in FP32 (ncnn turns it off)", v == "FP32", v)
v = sb.ncnn_precision(DEFAULT_OPT, {"cpu_support_arm_asimdhp": None})
check("4 NCNN: a probe that failed gives no verdict", v is None, v)

if FAILED:
    sys.exit(f"{len(FAILED)} failed: {FAILED}")
print("all passed")
