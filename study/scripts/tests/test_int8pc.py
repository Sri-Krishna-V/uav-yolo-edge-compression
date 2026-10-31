"""Check the per-channel INT8 build (#41) on a small model quantized by Ultralytics' own recipe: it stores what ONNX
Runtime's quantize_static(per_channel=True) stores, and refuses an FP32 model the stock INT8 export was not made from.
Needs the study venv (numpy, onnx, onnxruntime, ultralytics); about 30 s, most of it three studyexport.py runs:

  python study/scripts/tests/test_int8pc.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import studyenv  # noqa: E402, F401  must precede ultralytics

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402
from onnxruntime.quantization import CalibrationDataReader, quantize_static  # noqa: E402
from ultralytics.utils.export.onnx import onnx_int8_quantize  # noqa: E402

import studyqdq  # noqa: E402

FAILED = []


def check(label, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        FAILED.append(label)
        if detail:
            print(f"     {detail}")


# ---------------------------------------------------------------- a small detector-like model, and its INT8 exports
# Conv + SiLU (Sigmoid, Mul) as Ultralytics exports it, a depthwise conv, and a 1x1 output conv. The first conv's output
# channels span an 80x weight range, as YOLO26s's 32-channel box convs do (4-10x there), so one per-tensor scale
# rounds the small channels' weights mostly away.
SPREAD = np.array([1, 0.1, 0.5, 2, 0.05, 1, 0.3, 4], np.float32)


def fp32_model(path, seed):
    rng = np.random.default_rng(seed)
    w = lambda *s: rng.normal(0, 0.3, s).astype(np.float32)  # noqa: E731
    inits = [numpy_helper.from_array(w(8, 3, 3, 3) * SPREAD[:, None, None, None], "model.0.conv.weight"),
             numpy_helper.from_array(w(8), "model.0.conv.bias"),
             numpy_helper.from_array(w(8, 1, 3, 3), "model.1.conv.weight"),
             numpy_helper.from_array(w(8), "model.1.conv.bias"),
             numpy_helper.from_array(w(4, 8, 1, 1), "model.2.weight"),
             numpy_helper.from_array(w(4), "model.2.bias")]
    nodes = [helper.make_node("Conv", ["images", "model.0.conv.weight", "model.0.conv.bias"], ["c0"], "/model.0/conv/Conv", pads=[1] * 4),
             helper.make_node("Sigmoid", ["c0"], ["s0"], "/model.0/act/Sigmoid"),
             helper.make_node("Mul", ["c0", "s0"], ["a0"], "/model.0/act/Mul"),
             helper.make_node("Conv", ["a0", "model.1.conv.weight", "model.1.conv.bias"], ["c1"], "/model.1/conv/Conv", pads=[1] * 4, group=8),
             helper.make_node("Sigmoid", ["c1"], ["s1"], "/model.1/act/Sigmoid"),
             helper.make_node("Mul", ["c1", "s1"], ["a1"], "/model.1/act/Mul"),
             helper.make_node("Conv", ["a1", "model.2.weight", "model.2.bias"], ["output0"], "/model.2/Conv")]
    graph = helper.make_graph(nodes, "main", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 16, 16])],
                              [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4, 16, 16])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    model.metadata_props.add(key="imgsz", value="[16, 16]")
    onnx.save(model, str(path))
    return path


CALIBRATION = [np.random.default_rng(100 + i).random((1, 3, 16, 16), np.float32) for i in range(8)]


class Reader(CalibrationDataReader):
    def __init__(self):
        self.it = iter(CALIBRATION)

    def get_next(self):
        x = next(self.it, None)
        return None if x is None else {"images": x}


def stock_int8(fp32, path):
    """The stock export: Ultralytics' onnx_int8_quantize, i.e. quantize_static's defaults over Conv, Gemm and MatMul."""
    onnx_int8_quantize(str(fp32), str(path), CALIBRATION, lambda x: x, "images", 1)
    return path


def ort_per_channel(fp32, path):
    """ONNX Runtime's own per-channel quantization with the same calibration and the same exclusions: the oracle."""
    exclude = [n.name for n in onnx.load(str(fp32)).graph.node if n.op_type not in {"Conv", "Gemm", "MatMul"}]
    quantize_static(str(fp32), str(path), Reader(), nodes_to_exclude=exclude, per_channel=True)
    return path


def run(path, x):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"images": x})[0]


def initializers(path):
    return {i.name: numpy_helper.to_array(i) for i in onnx.load(str(path)).graph.initializer}


def nodes(path):
    return [(n.op_type, n.name, list(n.input), list(n.output), {a.name: helper.get_attribute_value(a) for a in n.attribute})
            for n in onnx.load(str(path)).graph.node]


TMP = Path(tempfile.mkdtemp())
FP32 = fp32_model(TMP / "model.onnx", seed=0)
STOCK = stock_int8(FP32, TMP / "model_int8.onnx")
ORACLE = ort_per_channel(FP32, TMP / "oracle.onnx")

# ---------------------------------------------------------------- 1: the build stores what quantize_static(per_channel=True) stores
OUT = TMP / "model_int8pc.onnx"
stats = studyqdq.per_channel(STOCK, FP32, OUT)
check("1 per-channel: every conv's weights and bias are re-quantized", stats == {"convs": 3, "biases": 3}, stats)

got, want = initializers(OUT), initializers(ORACLE)
check("1 per-channel: the same initializers as ONNX Runtime's per-channel export", set(got) == set(want), sorted(set(got) ^ set(want)))


def same(a, b):
    """Integers exactly; a float scale to float32 rounding (the bias scale's product is rounded once here, twice there)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return np.allclose(a, b, rtol=2e-7, atol=0) if a.dtype == np.float32 else np.array_equal(a, b)


diff = sorted(k for k in set(got) & set(want) if not same(got[k], want[k]))
check("1 per-channel: each initializer's values, shape and type are ONNX Runtime's", not diff, diff)
check("1 per-channel: the same nodes, wiring and axis attributes as ONNX Runtime's", nodes(OUT) == nodes(ORACLE))

stock_nodes = [(t, n, i, o) for t, n, i, o, _ in nodes(STOCK)]
check("1 per-channel: the stock export's nodes and wiring are kept", [(t, n, i, o) for t, n, i, o, _ in nodes(OUT)] == stock_nodes)
stock = initializers(STOCK)
activations = [k for k in stock if not k.startswith("model.")]  # the activation scales and zero points, from calibration
check("1 per-channel: the stock activation scales and zero points are kept",
      activations and all(np.array_equal(got[k], stock[k]) for k in activations), activations)
meta = lambda p: {m.key: m.value for m in onnx.load(str(p)).metadata_props}  # noqa: E731
check("1 per-channel: the stock export's metadata is kept", meta(OUT) == meta(STOCK) and "imgsz" in meta(OUT), meta(OUT))

x = np.random.default_rng(7).random((1, 3, 16, 16), np.float32)
ref = run(FP32, x)
err_stock, err_pc = np.abs(run(STOCK, x) - ref).mean(), np.abs(run(OUT, x) - ref).mean()
check("1 per-channel: its output is closer to the FP32 model's than the stock INT8's", err_pc < err_stock,
      f"mean abs error {err_pc:.5f} against stock {err_stock:.5f}")

# ---------------------------------------------------------------- 2: it refuses an FP32 export of another checkpoint
OTHER = fp32_model(TMP / "other.onnx", seed=1)  # the same graph and names, other weights
try:
    studyqdq.per_channel(STOCK, OTHER, TMP / "wrong.onnx")
    check("2 source: an FP32 export of another checkpoint is refused", False, "it built a model")
except ValueError as e:
    check("2 source: an FP32 export of another checkpoint is refused", "same checkpoint" in str(e), e)
check("2 source: and nothing is written", not (TMP / "wrong.onnx").exists())

# ---------------------------------------------------------------- 3: studyexport.py writes the variant as a cell directory
# A run's FP32 and stock INT8 cells as studyexport.py writes them, then the CLI (on Windows, in its memory-capped child).
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402

import studybench as sb  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "studyexport.py"
WEIGHTS = TMP / "toy-32-s0" / "best.pt"
WEIGHTS.parent.mkdir()
WEIGHTS.write_bytes(b"not a real checkpoint")
SHA = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
CALIB = {"data": "VisDrone.yaml", "split": "val", "fraction": 1.0, "imgsz": 32, "images": 548, "images_sha256": "c" * 64}


def cell(root, variant, model, artifact, source_sha=SHA):
    d = root / "toy-32-s0" / "32" / variant
    d.mkdir(parents=True)
    shutil.copy2(model, d / artifact)
    tagged = onnx.load(str(d / artifact))  # Ultralytics' metadata, as the exporter writes it
    for k, v in {"stride": "32", "imgsz": "[32, 32]", "end2end": "False", "names": "{0: 'a'}"}.items():
        tagged.metadata_props.add(key=k, value=v)
    onnx.save(tagged, str(d / artifact))
    int8 = "int8" in variant
    manifest = {"run": "toy-32-s0", "variant": variant, "runtime": "onnxruntime", "precision": "int8" if int8 else "fp32",
                "head": "raw", "imgsz": 32, "batch": 1, "artifact": artifact,
                "files": {artifact: hashlib.sha256((d / artifact).read_bytes()).hexdigest()}, "source_weights": str(WEIGHTS),
                "source_sha256": source_sha, "export_args": {"imgsz": 32, "batch": 1, "device": "cpu", "format": "onnx",
                                                             **({"quantize": 8, "data": "VisDrone.yaml"} if int8 else {})},
                "calibration": CALIB if int8 else None}
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


def export(root, *extra):
    r = subprocess.run([sys.executable, str(SCRIPT), "--weights", str(WEIGHTS), "--imgsz", "32", "--out", str(root),
                        "--only", "onnxruntime-int8pc", "--mem-cap-gb", "2", "--timeout-min", "5", *extra],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


ROOT = TMP / "exports"
fp32_dir = cell(ROOT, "onnxruntime-fp32", FP32, "model.onnx")
int8_dir = cell(ROOT, "onnxruntime-int8", STOCK, "model_int8.onnx")
code, log = export(ROOT)
pc = ROOT / "toy-32-s0" / "32" / "onnxruntime-int8pc"
check("3 cell: the CLI builds onnxruntime-int8pc from the run's two exports", code == 0 and (pc / "manifest.json").is_file(), log[-1500:])
if (pc / "manifest.json").is_file():
    m = json.loads((pc / "manifest.json").read_text(encoding="utf-8"))
    check("3 cell: the directory holds its manifest and one ONNX model, as studybench.py needs",
          sorted(f.name for f in pc.iterdir()) == ["manifest.json", "model_int8pc.onnx"] and sb.find_model(pc, "onnxruntime").name == "model_int8pc.onnx")
    check("3 cell: the manifest's file hashes are the files'", m["files"] == sb.sha256_files(pc), (m["files"], sb.sha256_files(pc)))
    check("3 cell: studybench.py reads it as INT8", sb.onnx_precision(pc / "model_int8pc.onnx") == "INT8")
    want = {"run": "toy-32-s0", "variant": "onnxruntime-int8pc", "runtime": "onnxruntime", "precision": "int8", "head": "raw",
            "imgsz": 32, "batch": 1, "artifact": "model_int8pc.onnx", "source_sha256": SHA, "calibration": CALIB}
    check("3 cell: the manifest names the cell, its checkpoint and the stock calibration it keeps",
          {k: m.get(k) for k in want} == want, {k: m.get(k) for k in want})
    sources = (m.get("per_channel") or {}).get("from") or {}
    check("3 cell: the manifest names both exports it was built from, by their files' hashes",
          sources == {"onnxruntime-fp32": json.loads((fp32_dir / "manifest.json").read_text())["files"],
                      "onnxruntime-int8": json.loads((int8_dir / "manifest.json").read_text())["files"]}, sources)
    studyqdq.per_channel(int8_dir / "model_int8.onnx", fp32_dir / "model.onnx", TMP / "direct.onnx")
    check("3 cell: its model is the per-channel build of those two exports, byte for byte",
          (pc / "model_int8pc.onnx").read_bytes() == (TMP / "direct.onnx").read_bytes())

# ---------------------------------------------------------------- 4: it refuses to build from a missing or a foreign export
ROOT = TMP / "no-int8"
cell(ROOT, "onnxruntime-fp32", FP32, "model.onnx")
code, log = export(ROOT)
check("4 refuse: with no stock INT8 export it stops, and says to export onnxruntime-int8 first",
      code != 0 and "export onnxruntime-int8 first" in log, log[-800:])
check("4 refuse: and writes no cell", not (ROOT / "toy-32-s0" / "32" / "onnxruntime-int8pc").exists())

ROOT = TMP / "foreign"
cell(ROOT, "onnxruntime-fp32", FP32, "model.onnx")
cell(ROOT, "onnxruntime-int8", STOCK, "model_int8.onnx", source_sha="f" * 64)
code, log = export(ROOT)
check("4 refuse: a stock INT8 export of another checkpoint stops it", code != 0 and "another checkpoint" in log, log[-800:])
check("4 refuse: and writes no cell", not (ROOT / "toy-32-s0" / "32" / "onnxruntime-int8pc").exists())

if FAILED:
    sys.exit(f"{len(FAILED)} failed: {FAILED}")
print("all passed")
