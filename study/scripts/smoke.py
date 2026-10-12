"""Smoke run for the experimental invariants. NOT a baseline.

Proves, on this machine and in this environment, that:
  1. the Ultralytics conversion of VisDrone matches the raw-zip fingerprint, row for row;
  2. training receives the shared + train invariants;
  3. studyeval.evaluate() on split=val and split=test (test-dev) emits the area splits AND the full
     coco/* family at maxDets=500, with every eval invariant reaching the validator;
  4. the headline evaluation scores the one-to-many head for both families.

1 epoch on 10% of train per study model: the mAP values are meaningless, only the plumbing matters.
Run fingerprint_visdrone.py first - conversion deletes the raw annotations it needs.

  python smoke.py                    # the laptop: verification/smoke_results.json
  python smoke.py --tag kaggle-srikr # any other machine: verification/smoke_results-kaggle-srikr.json
"""

import argparse
import json
import math
import platform
import subprocess
import sys
from datetime import date
from importlib.metadata import version
from pathlib import Path

import studyenv  # noqa: F401  must precede ultralytics
import studyeval
from fingerprint_visdrone import verify_conversion

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

ROOT = studyenv.ROOT
SMOKE_MODELS = ["yolo11s.pt", "yolo26s.pt"]
REQUIRED_STATS = [*studyenv.AREA_KEYS, "coco/AP_all", "coco/AP_50", "coco/AP_75", "coco/AP_small", "coco/AR_500"]


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
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=None, help="machine tag for the record file, e.g. kaggle-srikr. Omit only on the laptop")
    a = p.parse_args()
    out = ROOT / "verification" / (f"smoke_results-{a.tag}.json" if a.tag else "smoke_results.json")

    failures = []
    result = {"environment": environment(), "invariants": studyenv.INVARIANTS, "git_commit": studyenv.git_commit()}
    result["conversion"] = verify_conversion(failures)
    result["models"] = {w: smoke_model(w, failures) for w in SMOKE_MODELS}
    result["checks_passed"] = not failures
    result["failures"] = failures
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"checks_passed": result["checks_passed"], "failures": failures, "written": str(out)}, indent=2))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
