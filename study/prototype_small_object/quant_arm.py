"""PROTOTYPE - throwaway, lives only on branch prototype/small-object-effect. Map ticket: Confirm the small-object effect.

Quantization arm: export A1 (yolo11s-640-s0) as FP32 and naive INT8 for two toolchains, evaluate all four on
val and test-dev through studyeval, and dump predictions + GT for the paired bootstrap (bootstrap.py).

  .venv/Scripts/python prototype_small_object/quant_arm.py

Exports land in runs/prototype-small-object/ (git-ignored); records in prototype_small_object/results/.
"""

import json
import shutil
import sys
import time
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import studyenv  # noqa: E402  must precede ultralytics
import studyeval  # noqa: E402

from ultralytics import YOLO  # noqa: E402

A1 = Path(r"C:\Users\srikr\uav-yolo-checkpoints\yolo11s-640-s0\best.pt")
WORK = studyenv.ROOT / "runs" / "prototype-small-object"
OUT = Path(__file__).resolve().parent / "results"
IMGSZ = 640

# (name, export kwargs). Every artifact gets its own directory: ONNX INT8 export deletes the FP32 .onnx beside it.
ARTIFACTS = [
    ("onnx-fp32", dict(format="onnx")),
    ("onnx-int8", dict(format="onnx", quantize=8, data=studyenv.DATA)),
    ("openvino-fp32", dict(format="openvino")),
    ("openvino-int8", dict(format="openvino", quantize=8, data=studyenv.DATA)),
]

# Capture the in-memory COCO GT and predictions studyeval feeds faster-coco-eval, for the bootstrap.
CAPTURE = {}
_orig = studyeval.StudyValidator.coco_evaluate


def _capturing(self, stats, pred_json, anno_json, *args, **kwargs):
    CAPTURE["pred"], CAPTURE["anno"] = pred_json, anno_json
    return _orig(self, stats, pred_json, anno_json, *args, **kwargs)


studyeval.StudyValidator.coco_evaluate = _capturing


def export(name, kwargs):
    d = WORK / name
    d.mkdir(parents=True, exist_ok=True)
    pt = d / "best.pt"
    if not pt.exists():
        shutil.copy2(A1, pt)
    existing = [p for p in d.iterdir() if p.name != "best.pt"]
    if existing:
        return str(existing[0])
    t = time.time()
    f = YOLO(pt).export(imgsz=IMGSZ, batch=1, device="cpu", **kwargs)
    print(f"[{name}] exported in {time.time() - t:.0f}s -> {f}")
    return f


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    toolchain = {p: version(p) for p in ("onnx", "onnxruntime", "onnxslim", "openvino", "nncf", "ultralytics")}
    for name, kwargs in ARTIFACTS:
        artifact = export(name, kwargs)
        for split in ("val", "test"):
            rec_file = OUT / f"{name}-{split}.json"
            if rec_file.exists():
                continue
            t = time.time()
            result = studyeval.evaluate(artifact, split=split, imgsz=IMGSZ, device="cpu")
            result.update(artifact=name, export_kwargs=kwargs, split=split, eval_seconds=round(time.time() - t), toolchain=toolchain)
            rec_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
            (WORK / f"{name}-{split}.coco.json").write_text(json.dumps({"anno": CAPTURE["anno"], "pred": CAPTURE["pred"]}), encoding="utf-8")
            s = result["stats"]
            print(f"[{name} {split}] AP_all {s['coco/AP_all']:.4f} small {s['coco/AP_small']:.4f} medium {s['coco/AP_medium']:.4f} large {s['coco/AP_large']:.4f} ({result['eval_seconds']}s)")


if __name__ == "__main__":
    main()
