"""Launch one study training run under the frozen invariants plus the baseline recipe.

  python studytrain.py --model yolo11s.pt --imgsz 640 --seed 0 --device 0    --batch-per-gpu 16
  python studytrain.py --model yolo26s.pt --imgsz 1280 --seed 0 --device 0,1 --batch-per-gpu 4   # one run, two GPUs
  python studytrain.py --model yolo11s.pt --imgsz 640 --seed 0 --device 0    --batch-per-gpu 16 --resume

The run name is derived from (model, imgsz, seed), so --resume finds the same directory. `batch` is
per-GPU-batch x GPU count, because Ultralytics splits `batch` across GPUs and does not sync BatchNorm:
holding the PER-GPU batch constant is what keeps one-GPU and two-GPU runs comparable.

On success it writes verification/runs/<name>.json: environment, every resolved argument, both
studyeval results (val and test-dev), and wall-clock. That file is the run's record for the paper.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import studyenv  # noqa: F401  must precede ultralytics
import studyeval

import yaml  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

RECIPE_FILE = studyenv.ROOT / "cfg" / "train_baseline.yaml"
RUNS_DIR = studyenv.ROOT / "runs" / "baseline"
RECORD_DIR = studyenv.ROOT / "verification" / "runs"


def recipe():
    """Flatten the grouped recipe file into Ultralytics train arguments."""
    groups = yaml.safe_load(RECIPE_FILE.read_text(encoding="utf-8"))
    flat = {}
    for group in groups.values():
        flat.update(group)
    return flat


def environment():
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return {
        "recorded_on": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "nvidia_driver": driver or None,
        "cpu_count": __import__("os").cpu_count(),
        **{pkg: version(pkg) for pkg in ("ultralytics", "faster-coco-eval", "torchvision", "numpy")},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="yolo11s.pt | yolo26s.pt (or a teacher: yolo11m.pt | yolo26m.pt)")
    p.add_argument("--imgsz", type=int, required=True)
    p.add_argument("--seed", type=int, default=None, help="default: the invariant seed (0). 1 and 2 are the labelled noise runs")
    p.add_argument("--device", default="0", help="'0' for one GPU, '0,1' for one run across two GPUs")
    p.add_argument("--batch-per-gpu", type=int, required=True)
    p.add_argument("--workers", type=int, default=None, help="default: the recipe value. Recorded, because it changes the augmentation RNG stream")
    p.add_argument("--name", default=None, help="default: <model stem>-<imgsz>-s<seed>")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-eval", action="store_true", help="skip the studyeval pass (use when the run will be evaluated later)")
    a = p.parse_args()

    n_gpus = len([d for d in str(a.device).split(",") if d != ""])
    args = {**studyenv.TRAIN_ARGS, **recipe()}
    if a.seed is not None:
        args["seed"] = a.seed
    if a.workers is not None:
        args["workers"] = a.workers
    seed = args["seed"]
    name = a.name or f"{Path(a.model).stem}-{a.imgsz}-s{seed}"
    run_dir = RUNS_DIR / name
    last = run_dir / "weights" / "last.pt"

    if a.resume and not last.exists():
        sys.exit(f"--resume given but {last} does not exist. Start the run without --resume.")
    if not a.resume and last.exists():
        sys.exit(f"{run_dir} already holds a run. Pass --resume to continue it, or --name for a new one.")

    weights = str(last) if a.resume else a.model
    model = YOLO(weights)
    started = time.time()
    model.train(
        data=studyenv.DATA,
        imgsz=a.imgsz,
        batch=a.batch_per_gpu * max(n_gpus, 1),
        device=a.device,
        project=str(RUNS_DIR),
        name=name,
        exist_ok=True,
        resume=a.resume,
        **args,
    )
    elapsed_h = round((time.time() - started) / 3600, 3)

    record = {
        "run": name,
        "model": a.model,
        "imgsz": a.imgsz,
        "seed": seed,
        "device": a.device,
        "gpus_used": n_gpus,
        "batch_per_gpu": a.batch_per_gpu,
        "batch_total": a.batch_per_gpu * max(n_gpus, 1),
        "resumed": a.resume,
        "train_hours_this_session": elapsed_h,
        "environment": environment(),
        "train_args": {k: getattr(model.trainer.args, k, None) for k in args},
        "best_weights": str(model.trainer.best),
    }

    if not a.no_eval:
        best = model.trainer.best
        for split in ("val", "test"):  # "test" is test-dev, the headline split
            record[f"eval_{split}"] = studyeval.evaluate(best, split, a.imgsz, device="0", project=str(RUNS_DIR), name=f"{name}-eval-{split}", exist_ok=True, plots=False)

    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    out = RECORD_DIR / f"{name}.json"
    out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    headline = record.get("eval_test", {}).get("stats", {})
    print(json.dumps({
        "run": name,
        "record": str(out),
        "train_hours_this_session": elapsed_h,
        "test_dev_AP_50": headline.get("coco/AP_50"),
        "test_dev_AP_all": headline.get("coco/AP_all"),
        "test_dev_AP_small": headline.get("coco/AP_small"),
    }, indent=2))


if __name__ == "__main__":
    main()
