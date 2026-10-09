"""Smoke run for the experimental invariants. NOT a baseline.

Proves, on this machine and in this environment, that:
  1. the Ultralytics conversion of VisDrone matches the raw-zip fingerprint, row for row;
  2. training receives the shared + train invariants;
  3. studyeval.evaluate() on split=val and split=test (test-dev) emits the area splits AND the full
     coco/* family at maxDets=500, with every eval invariant reaching the validator;
  4. the headline evaluation scores the one-to-many head for both families.

1 epoch on 10% of train per study model: the mAP values are meaningless, only the plumbing matters.
Run fingerprint_visdrone.py first - conversion deletes the raw annotations it needs.
"""

import json
import math
import platform
import subprocess
import sys
from collections import Counter
from datetime import date
from importlib.metadata import version
from pathlib import Path

import studyenv  # noqa: F401  must precede ultralytics
import studyeval

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402

ROOT = studyenv.ROOT
OUT = ROOT / "verification" / "smoke_results.json"
SMOKE_MODELS = ["yolo11s.pt", "yolo26s.pt"]
SPLIT_TO_FINGERPRINT = {"train": "train", "val": "val", "test": "test-dev"}
REQUIRED_STATS = [*studyenv.AREA_KEYS, "coco/AP_all", "coco/AP_50", "coco/AP_75", "coco/AP_small", "coco/AR_500"]


def verify_conversion(failures):
    fp_file = ROOT / "verification" / "visdrone_fingerprint.json"
    if not fp_file.exists():
        sys.exit("Run fingerprint_visdrone.py first: conversion deletes the raw annotations.")
    fp = json.loads(fp_file.read_text(encoding="utf-8"))
    if not fp["checks_passed"]:
        sys.exit("Raw-zip fingerprint failed its checks; refusing to convert a suspect download.")

    data = check_det_dataset(studyenv.DATA)  # first call converts, reusing the already-downloaded zips
    base = Path(data["path"])
    report = {}
    for split, fp_split in SPLIT_TO_FINGERPRINT.items():
        raw = fp["splits"][fp_split]
        images = len(list((base / "images" / split).glob("*.jpg")))
        label_files = list((base / "labels" / split).glob("*.txt"))
        classes = Counter()
        max_objects = 0
        for f in label_files:
            lines = [line for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
            max_objects = max(max_objects, len(lines))
            classes.update(int(line.split()[0]) for line in lines)
        per_class = [classes[k] for k in range(10)]
        report[split] = {"images": images, "label_files": len(label_files), "label_rows": sum(classes.values()), "per_class": per_class, "max_objects_per_image": max_objects}
        if images != raw["images"] or len(label_files) != raw["annotation_files"]:
            failures.append(f"{split}: {images} images / {len(label_files)} label files, raw zip had {raw['images']}")
        if per_class != raw["kept_per_class_1_to_10"]:
            failures.append(f"{split}: converted per-class rows {per_class} != raw kept rows {raw['kept_per_class_1_to_10']}")
        if set(classes) - set(range(10)):
            failures.append(f"{split}: out-of-range class ids {sorted(set(classes) - set(range(10)))}")
    return report


def check_eval(where, result, failures):
    stats = result["stats"]
    missing = [k for k in REQUIRED_STATS if k not in stats]
    if missing:
        failures.append(f"{where}: missing {missing}")
    non_finite = [k for k in REQUIRED_STATS if k in stats and not math.isfinite(stats[k])]
    if non_finite:
        failures.append(f"{where}: non-finite {non_finite}")
    if result["coco_max_dets"] != studyenv.COCO_MAX_DETS:
        failures.append(f"{where}: faster-coco-eval ran at maxDets={result['coco_max_dets']}")
    for k, v in studyenv.EVAL_ARGS.items():
        if result["validator_args"].get(k) != v:
            failures.append(f"{where}: validator {k}={result['validator_args'].get(k)!r}, invariant is {v!r}")
    return result


def smoke_model(weights, failures):
    stem = Path(weights).stem
    runs = str(ROOT / "runs" / "smoke")
    out = {}

    model = YOLO(weights)
    model.train(data=studyenv.DATA, epochs=1, fraction=0.1, imgsz=640, batch=8, workers=4, project=runs, name=f"{stem}-train", exist_ok=True, plots=False, **studyenv.TRAIN_ARGS)
    trainer_args = vars(model.trainer.args)
    out["train_args"] = {k: trainer_args.get(k) for k in studyenv.TRAIN_ARGS}
    for k, v in studyenv.TRAIN_ARGS.items():
        if trainer_args.get(k) != v:
            failures.append(f"{stem} training: {k}={trainer_args.get(k)!r}, invariant is {v!r}")

    best = model.trainer.best
    for split in ("val", "test"):
        result = studyeval.evaluate(best, split, 640, project=runs, name=f"{stem}-eval-{split}", exist_ok=True, plots=False, workers=4)
        out[f"eval_{split}"] = check_eval(f"{stem} eval split={split}", result, failures)
        if result["end2end_head_scored"]:
            failures.append(f"{stem} eval split={split}: scored the one-to-one head; headline must be one-to-many + NMS")

    if stem.startswith("yolo26"):
        # Informational only: YOLO26's NMS-free head, deliberately outside the invariants.
        probe = studyeval.evaluate(best, "val", 640, nms=False, project=runs, name=f"{stem}-probe-e2e-val", exist_ok=True, plots=False, workers=4)
        out["probe_nms_free_head_val"] = probe
        if not probe["end2end_head_scored"]:
            failures.append(f"{stem} probe: nms=False did not select the one-to-one head")
    return out


def environment():
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return {
        "recorded_on": date.today().isoformat(),
        "host": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_driver": driver or None,
        **{pkg: version(pkg) for pkg in ("ultralytics", "faster-coco-eval", "torchvision", "numpy", "opencv-python")},
    }


def main():
    failures = []
    result = {"environment": environment(), "invariants": studyenv.INVARIANTS}
    result["conversion"] = verify_conversion(failures)
    result["models"] = {w: smoke_model(w, failures) for w in SMOKE_MODELS}
    result["checks_passed"] = not failures
    result["failures"] = failures
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"checks_passed": result["checks_passed"], "failures": failures}, indent=2))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
