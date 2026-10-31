"""Export one trained checkpoint to every artifact a cell needs, each with a manifest. Laptop side; TensorRT on the Jetson.

  python studyexport.py --weights C:/Users/srikr/uav-yolo-checkpoints/yolo11s-640-s0/best.pt --imgsz 640
  python studyexport.py --weights ... --imgsz 640 --only onnxruntime-fp32 onnxruntime-int8
  python studyexport.py --weights yolo26s.pt --imgsz 640 --e2e      # adds the NMS-free head variants (YOLO26 only)

Layout: exports/<run>/<imgsz>/<runtime>-<precision>[-e2e]/ holding the Ultralytics artifact and manifest.json
(git-ignored). studybench.py takes that directory as --artifact and checks the files against the manifest.

Export arguments are the issue #9 quantization arm's (prototype quant_arm.py), so its numbers reproduce: imgsz,
batch=1 (static), device=cpu, and for INT8 quantize=8 with data=VisDrone.yaml, i.e. Ultralytics' default static
calibration on all 548 val images at the export imgsz. ONNX INT8 is ONNX Runtime quantize_static over Conv/Gemm/MatMul
only; OpenVINO INT8 is NNCF with the Detect head and every Sigmoid ignored. NCNN has no INT8 in Ultralytics, and
runs at ncnn's own default precision on the device. Each artifact is exported in its own staging directory because
ONNX INT8 export deletes the FP32 .onnx beside it.

onnxruntime-int8pc is the ONNX Runtime INT8 the matrix adopted on 31 October (#41): not an Ultralytics export, built by studyqdq.py
from the run's onnxruntime-fp32 and onnxruntime-int8 exports at the same imgsz, which it needs beside it (on the same
checkpoint, matching their manifests). It keeps the stock INT8 export's graph and its calibrated activation scales and
stores each Conv's weights per output channel, so it runs no calibration of its own; its manifest names both sources
by their files' hashes.
onnxruntime-int8pc-e2e is built the same way from the two -e2e exports. A cell with no stock INT8 export (YOLO26s at
1280) has none:

  python studyexport.py --weights .../yolo26s-960-s0/best.pt --imgsz 800 --only onnxruntime-int8pc

Every artifact keeps the default nms=None: the raw one-to-many head, decoded and NMSed by studybench.py, so YOLO11 and
YOLO26 share one post-processing (INVARIANTS.md section 5). The -e2e variants pass nms=False, which keeps YOLO26's
NMS-free one-to-one head for ONNX and OpenVINO at any precision (exporter.py L682-713 drops it for NCNN, RKNN,
ExecuTorch, Paddle, IMX, EdgeTPU, QNN, LiteRT INT8 and old TensorRT), and max_det=500 so its baked top-k matches the
invariant. The head actually exported is read back from the artifact's metadata, never assumed.

TensorRT engines are built ON the Jetson, because an engine runs only on the GPU and TensorRT version that built it:

  python3 studyexport.py --weights .../yolo11s-640-s0/best.pt --imgsz 640 --only tensorrt-fp16 tensorrt-int8

They are never in the default set; name them with --only. format=engine, device=0, static batch 1, workspace left to
TensorRT unless --workspace is given. The exporter writes an FP32 ONNX (opset 18 at torch >= 2.4, onnxslim), then
builds from it: tensorrt-fp16 with the FP16 builder flag (quantize=16, what the old half=True means), tensorrt-int8
with the INT8 flag alone (quantize=8, the old int8=True; layers TensorRT cannot run in INT8 fall back to FP32, not
FP16) and an IInt8Calibrator (MINMAX) over all 548 val images, batch 1, at the export imgsz, with the head's Sigmoid
kept in FP32 (utils/export/engine.py onnx2engine). The calibration cache and the ONNX stay beside the engine. The
manifest adds what the engine is tied to: TensorRT, GPU, CUDA, L4T and power mode.

On Windows every INT8 variant is exported in its own child process under studycap's memory cap and timeout (#20),
because calibration is the heaviest step on the laptop: 4 GB and 60 minutes unless --mem-cap-gb or --timeout-min
says otherwise (INT8 at 960 and 1280 needs 4-5 GB). A per-channel build runs the same way, though it calibrates
nothing. A variant that goes over either is stopped and reported failed; the others go on. Elsewhere, and for FP32,
nothing changes.

INT8 needs datasets/VisDrone/images/val and labels/val. Without them Ultralytics would run VisDrone.yaml's download
script (about 2 GB), so this script stops first. studyenv imports only the standard library, yaml and ultralytics'
version, so it runs on the Jetson unchanged.
"""

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import studycap
import studyenv  # noqa: F401  must precede ultralytics

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.backends.base import BaseBackend  # noqa: E402

VARIANTS = {
    "onnxruntime-fp32": dict(format="onnx"),
    "onnxruntime-int8": dict(format="onnx", quantize=8, data=studyenv.DATA),
    "openvino-fp32": dict(format="openvino"),
    "openvino-int8": dict(format="openvino", quantize=8, data=studyenv.DATA),
    "ncnn-fp32": dict(format="ncnn"),
}
BOARD_VARIANTS = {  # built on the Jetson, only when named in --only
    "tensorrt-fp16": dict(format="engine", device=0, quantize=16),
    "tensorrt-int8": dict(format="engine", device=0, quantize=8, data=studyenv.DATA),
}
DERIVED = {  # built by studyqdq.py from two exports of the same run and imgsz, not by Ultralytics: (FP32, stock INT8)
    "onnxruntime-int8pc": ("onnxruntime-fp32", "onnxruntime-int8"),
}
E2E_FORMATS = ("onnx", "openvino")
TOOLCHAIN = ("ultralytics", "torch", "onnx", "onnxruntime", "onnxslim", "openvino", "nncf", "ncnn", "pnnx")
VAL_IMAGES = 548


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def variants(e2e):
    """Name -> export arguments, or {"from": (FP32 source, INT8 source)} for a DERIVED variant, which comes after both."""
    out = dict(VARIANTS)
    if e2e:
        out.update({f"{k}-e2e": {**v, "nms": False, "max_det": studyenv.EVAL_ARGS["max_det"]} for k, v in VARIANTS.items() if v["format"] in E2E_FORMATS})
    out.update({k: {"from": v} for k, v in DERIVED.items()})
    if e2e:
        out.update({f"{k}-e2e": {"from": tuple(f"{s}-e2e" for s in v)} for k, v in DERIVED.items()})
    return out


def run_name(weights):
    """yolo11s-640-s0 for .../yolo11s-640-s0/best.pt or .../yolo11s-640-s0/weights/best.pt, else the file stem."""
    w = Path(weights)
    if w.stem in ("best", "last"):
        return w.parent.parent.name if w.parent.name == "weights" else w.parent.name
    return w.stem


def toolchain():
    def v(p):
        try:
            return version(p)
        except PackageNotFoundError:
            return None

    return {p: v(p) for p in TOOLCHAIN}


def val_images():
    """The val images INT8 calibrates on, with one hash over their names and contents, or exit when they are missing."""
    images, labels = studyenv.VISDRONE_DIR / "images" / "val", studyenv.VISDRONE_DIR / "labels" / "val"
    files = sorted(images.glob("*.jpg")) if images.is_dir() else []
    n_labels = len(list(labels.glob("*.txt"))) if labels.is_dir() else 0
    if len(files) != VAL_IMAGES or n_labels != VAL_IMAGES:
        raise SystemExit(f"INT8 calibration needs the {VAL_IMAGES} VisDrone val images in {images} and their labels in "
                         f"{labels} (found {len(files)} and {n_labels}). Copy them first (PLATFORMS.md); otherwise "
                         "Ultralytics would start VisDrone's 2 GB download.")
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.name} {sha256(f)}\n".encode())
    return {"images": len(files), "images_sha256": h.hexdigest()}


def engine_record():
    """What a TensorRT engine is tied to. Read after the export: Ultralytics imports tensorrt only after its ONNX step."""
    import tensorrt as trt
    import torch

    import jetson_probe  # standard library only; every board value is None off a Jetson

    return {"tensorrt": trt.__version__, "gpu": torch.cuda.get_device_name(0), "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), **jetson_probe.board_state()}


def export_one(weights, imgsz, name, kwargs, dest, source_sha, calibration=None):
    """Export into a staging directory next to dest, write the manifest, then rename it into place."""
    stage = dest.with_name(f".{dest.name}.staging")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    pt = stage / "model.pt"  # Ultralytics writes the artifact beside the weights, named after them
    shutil.copy2(weights, pt)
    args = {"imgsz": imgsz, "batch": 1, "device": "cpu", **kwargs}  # TensorRT variants override the device
    t = time.time()
    artifact = Path(YOLO(pt).export(**args))
    seconds = round(time.time() - t, 1)
    pt.unlink()
    for d in list(stage.rglob("__pycache__")):  # pnnx leaves its model_pnnx.py bytecode behind
        shutil.rmtree(d)

    meta = BaseBackend.read_metadata(artifact)
    end2end = str(meta.get("end2end", False)) == "True"
    if end2end != (kwargs.get("nms") is False):
        raise RuntimeError(f"{name}: asked for {'end2end' if kwargs.get('nms') is False else 'raw'} head, the exporter wrote end2end={end2end}")
    files = sorted(f for f in stage.rglob("*") if f.is_file())
    runtime, precision = name.split("-")[:2]
    manifest = {
        "run": run_name(weights),
        "variant": name,
        "runtime": runtime,
        "precision": precision,
        "head": "end2end" if end2end else "raw",
        "imgsz": imgsz,
        "batch": 1,
        "artifact": artifact.relative_to(stage).as_posix(),
        "files": {f.relative_to(stage).as_posix(): sha256(f) for f in files},
        "source_weights": str(Path(weights).resolve()),
        "source_sha256": source_sha,
        "export_args": {k: str(v) if isinstance(v, Path) else v for k, v in args.items()},
        "calibration": {"data": kwargs["data"], "split": "val", "fraction": 1.0, "imgsz": imgsz, **(calibration or {})} if kwargs.get("quantize") == 8 else None,
        "artifact_metadata": {k: meta.get(k) for k in ("version", "date", "stride", "task", "head", "batch", "imgsz", "end2end", "args", "names")},
        "toolchain": toolchain(),
        "engine": engine_record() if kwargs["format"] == "engine" else None,
        "export_seconds": seconds,
        "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "git_commit": studyenv.git_commit(),
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    shutil.rmtree(dest, ignore_errors=True)
    stage.rename(dest)
    return manifest


def derive_one(weights, imgsz, name, sources, dest, source_sha):
    """Build a per-channel INT8 variant (#41) from the run's FP32 export and its stock INT8 export, the directories beside
    dest named in sources, in a staging directory like export_one. No calibration: the stock export's activation scales."""
    import onnxruntime as ort

    import studyqdq  # numpy and onnx, imported only here

    made = {}
    for s in sources:
        d = dest.with_name(s)
        if not (d / "manifest.json").exists():
            raise RuntimeError(f"it is built from the {s} export, which {d} does not hold: export {s} first")
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        if m["source_sha256"] != source_sha:
            raise RuntimeError(f"{d} was exported from another checkpoint ({m['source_weights']}): export {s} again")
        if any(sha256(d / f) != h for f, h in m["files"].items()):
            raise RuntimeError(f"{d} does not match its manifest.json: export {s} again")
        made[s] = m
    fp32, int8 = (dest.with_name(s) / made[s]["artifact"] for s in sources)
    stage = dest.with_name(f".{dest.name}.staging")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    artifact = stage / "model_int8pc.onnx"
    t = time.time()
    stats = studyqdq.per_channel(int8, fp32, artifact)
    ort.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])  # it loads
    seconds = round(time.time() - t, 1)

    meta = BaseBackend.read_metadata(artifact)
    end2end = str(meta.get("end2end", False)) == "True"
    if end2end != name.endswith("-e2e"):
        raise RuntimeError(f"{name}: built from {sources[1]}, whose head is end2end={end2end}")
    stock = made[sources[1]]
    manifest = {
        "run": run_name(weights),
        "variant": name,
        "runtime": "onnxruntime",
        "precision": "int8",
        "head": "end2end" if end2end else "raw",
        "imgsz": imgsz,
        "batch": 1,
        "artifact": artifact.name,
        "files": {artifact.name: sha256(artifact)},
        "source_weights": str(Path(weights).resolve()),
        "source_sha256": source_sha,
        "export_args": stock["export_args"],  # the stock export's, whose graph and activation scales this keeps
        "calibration": stock["calibration"],
        "per_channel": {"method": "studyqdq.py (#41): each Conv's weights per output channel (symmetric INT8, absmax/127) and its "
                                  "int32 bias at input scale x weight scale, from the FP32 export; every node and activation scale "
                                  "is the stock INT8 export's",
                        "from": {s: made[s]["files"] for s in sources}, **stats},
        "artifact_metadata": {k: meta.get(k) for k in ("version", "date", "stride", "task", "head", "batch", "imgsz", "end2end", "args", "names")},
        "toolchain": toolchain(),
        "engine": None,
        "export_seconds": seconds,
        "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "git_commit": studyenv.git_commit(),
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    shutil.rmtree(dest, ignore_errors=True)
    stage.rename(dest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--weights", required=True, help="a trained best.pt")
    p.add_argument("--imgsz", type=int, required=True)
    p.add_argument("--name", default=None, help="run name for the folder (default: from the weights path)")
    p.add_argument("--out", default=str(studyenv.ROOT / "exports"))
    p.add_argument("--e2e", action="store_true", help="also export the NMS-free one-to-one head (YOLO26) for ONNX Runtime and OpenVINO")
    p.add_argument("--only", nargs="+", default=None, help=f"subset of variants, from {[*variants(True), *BOARD_VARIANTS]}")
    p.add_argument("--force", action="store_true", help="re-export variants that already have a manifest")
    p.add_argument("--workspace", type=float, default=None, help="TensorRT builder workspace in GiB (default: TensorRT decides); "
                   "lower it if an engine build runs out of memory. Recorded in the manifest")
    studycap.add_arguments(p)
    a = p.parse_args()
    studycap.use(a)
    if a.imgsz % 32:
        raise SystemExit(f"--imgsz {a.imgsz} is not a multiple of the stride 32: Ultralytics would round it and the folder would lie")

    todo = variants(a.e2e or any(v.endswith("-e2e") for v in a.only or []))
    if a.only:
        todo.update(BOARD_VARIANTS)
        unknown = set(a.only) - set(todo)
        if unknown:
            raise SystemExit(f"unknown variants {sorted(unknown)}; choose from {list(todo)}")
        todo = {k: v for k, v in todo.items() if k in a.only}
    if a.workspace is not None:
        todo = {k: {**v, "workspace": a.workspace} if v.get("format") == "engine" else v for k, v in todo.items()}
    calibration = val_images() if any(v.get("quantize") == 8 for v in todo.values()) else None
    if any(k.endswith("-e2e") for k in todo) and getattr(YOLO(a.weights).model.model[-1], "one2one_cv2", None) is None:
        raise SystemExit(f"{a.weights} has no one-to-one head: --e2e is for YOLO26")
    root = Path(a.out) / (a.name or run_name(a.weights)) / str(a.imgsz)
    source_sha = sha256(a.weights)
    failed = []
    for name, kwargs in todo.items():
        dest = root / name
        if (dest / "manifest.json").exists() and not a.force:
            print(f"[{name}] exists, skipped: {dest}")
            continue
        if (kwargs.get("quantize") == 8 or "from" in kwargs) and studycap.applies():  # calibration, or a per-channel build: this variant alone, in a capped process
            argv = [str(Path(__file__).resolve()), *sys.argv[1:], "--only", name, "--force"]  # the last --only wins
            try:
                if studycap.run(argv, f"the {name} export"):
                    failed.append(name)  # the child said why
            except studycap.Stopped as e:
                failed.append(name)
                print(f"[{name}] FAILED: {e}")
            continue
        try:
            if "from" in kwargs:
                m = derive_one(a.weights, a.imgsz, name, kwargs["from"], dest, source_sha)
            else:
                m = export_one(a.weights, a.imgsz, name, kwargs, dest, source_sha, calibration)
            print(f"[{name}] {m['head']} head, {m['export_seconds']}s -> {dest / m['artifact']}")
        except Exception as e:  # keep exporting the others; report at the end
            failed.append(name)
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
