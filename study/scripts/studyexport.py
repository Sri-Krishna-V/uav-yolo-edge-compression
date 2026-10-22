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

INT8 needs datasets/VisDrone/images/val and labels/val. Without them Ultralytics would run VisDrone.yaml's download
script (about 2 GB), so this script stops first. studyenv imports only the standard library, yaml and ultralytics'
version, so it runs on the Jetson unchanged.
"""

import argparse
import hashlib
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

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
    out = dict(VARIANTS)
    if e2e:
        out.update({f"{k}-e2e": {**v, "nms": False, "max_det": studyenv.EVAL_ARGS["max_det"]} for k, v in VARIANTS.items() if v["format"] in E2E_FORMATS})
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
    a = p.parse_args()
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
        todo = {k: {**v, "workspace": a.workspace} if v["format"] == "engine" else v for k, v in todo.items()}
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
        try:
            m = export_one(a.weights, a.imgsz, name, kwargs, dest, source_sha, calibration)
            print(f"[{name}] {m['head']} head, {m['export_seconds']}s -> {dest / m['artifact']}")
        except Exception as e:  # keep exporting the others; report at the end
            failed.append(name)
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
