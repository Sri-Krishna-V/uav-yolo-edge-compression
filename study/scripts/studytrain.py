"""Launch one study training run under the frozen invariants plus the baseline recipe.

  python studytrain.py --model yolo11s.pt --imgsz 640 --seed 0 --device 0 --batch-per-gpu 16
  python studytrain.py --model yolo11s.pt --imgsz 640 --seed 0 --device 1 --batch-per-gpu 16 --resume
  python studytrain.py --model yolo11s.pt --imgsz 640 --seed 0 --device 0 --batch-per-gpu 16 --eval-only
  python studytrain.py --model yolo26s.pt --imgsz 1280 --seed 0 --device 0,1 --batch-per-gpu 4   # one run, two GPUs (DDP)
  python studytrain.py --model PRUNED.pt --pruned --epochs 30 --imgsz 640 --seed 0 --device 0 --batch-per-gpu 24 --name yolo11s-640-s0-prune30
  python studytrain.py --model yolo11s.pt --distill TEACHER.pt --imgsz 640 --seed 0 --device 0 --batch-per-gpu 24 --name yolo11s-640-s0-distill
  python studytrain.py --model PRUNED.pt --pruned --distill TEACHER.pt --epochs 30 ... --name yolo11s-640-s0-prune30-distill

On Kaggle, do not call this directly: kaggle/session.py picks fresh / --resume / --eval-only per run and
sets --stop-by. The study runs one run per GPU, never DDP (kaggle/README.md says why).

The run name is derived from (model, imgsz, seed), so --resume finds the same directory. `batch` is
per-GPU-batch x GPU count, because Ultralytics splits `batch` across GPUs and does not sync BatchNorm:
holding the PER-GPU batch constant is what keeps one-GPU and two-GPU runs comparable.

Two guards, both single-GPU only (Ultralytics does not pass callbacks into its DDP subprocess):
  --stop-by T   right after an epoch is saved, exit with code 75 (PAUSED) if the next epoch would not finish
                before unix time T. Exiting there, not via the trainer's own stop, matters: a trainer that stops
                runs final_eval, which strips last.pt to epoch -1 and makes the run impossible to resume.
  batch check   Ultralytics silently halves `batch` on an out-of-memory error in the first epoch. That would
                break the constant-batch rule, so the run is renamed *-INVALID-batch and exits with code 76.

On success it writes verification/runs/<name>.json: environment, code commit, every resolved argument,
both studyeval results (val and test-dev), and per-session wall-clock. That file is the run's record for
the paper. Every session also appends one line to runs/baseline/<name>/sessions.jsonl.

--pruned trains a studyprune.py checkpoint (a pruning-recovery run). A stock train() would rebuild the model
from its yaml and keep only the weights whose shapes still match, i.e. silently train an unpruned model.

--distill TEACHER.pt trains with Ultralytics' own distillation (distill_model, default `dis`): the trainer wraps the
student in a DistillationModel with the frozen teacher. Checkpoints drop the teacher, so --resume needs --distill
again to rebuild it; the final best.pt holds only the student, so it evaluates and exports like any other run.
With --pruned, the student is the pruned module (a combined run).

--fallback-from N: this run trains at a smaller batch than the N it was planned at, because it ran out of GPU
memory at N in its first epoch (kaggle/session.py restarts it). The record says so (batch_fallback_from).
"""

import argparse
import json
import os
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
from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402

RECIPE_FILE = studyenv.ROOT / "cfg" / "train_baseline.yaml"
RUNS_DIR = studyenv.ROOT / "runs" / "baseline"
RECORD_DIR = studyenv.ROOT / "verification" / "runs"

EXIT_PAUSED = 75  # stopped before the session limit; resume next session
EXIT_BATCH_CHANGED = 76  # Ultralytics halved the batch on OOM; the run is invalid


class PrunedTrainer(DetectionTrainer):
    """Trains the checkpoint's own pruned module, fresh or resumed, instead of rebuilding it from yaml."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        if not isinstance(weights, torch.nn.Module):
            raise RuntimeError("--pruned needs a pruned checkpoint (scripts/studyprune.py) as --model")
        return self.set_model_names_for_load(weights)


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
        "cpu_count": os.cpu_count(),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "git_commit": studyenv.git_commit(),
        **{pkg: version(pkg) for pkg in ("ultralytics", "faster-coco-eval", "torchvision", "numpy")},
    }


def checkpoint_epoch(path):
    """Epochs completed in a checkpoint: -1 once Ultralytics has stripped it at the end of training."""
    return torch.load(path, map_location="cpu", weights_only=False).get("epoch", -1)


def log_session(run_dir, entry):
    with open(run_dir / "sessions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_sessions(run_dir):
    f = run_dir / "sessions.jsonl"
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()] if f.exists() else []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="yolo11s.pt | yolo26s.pt (or a teacher: yolo11m.pt | yolo26m.pt)")
    p.add_argument("--imgsz", type=int, required=True)
    p.add_argument("--seed", type=int, default=None, help="default: the invariant seed (0). 1 and 2 are the labelled noise runs")
    p.add_argument("--device", default="0", help="'0' or '1' for one GPU, '0,1' for one run across two GPUs")
    p.add_argument("--batch-per-gpu", type=int, required=True)
    p.add_argument("--workers", type=int, default=None, help="default: the recipe value. Recorded, because it changes the augmentation RNG stream")
    p.add_argument("--name", default=None, help="default: <model stem>-<imgsz>-s<seed>")
    p.add_argument("--pruned", action="store_true", help="--model is a studyprune.py checkpoint: train its pruned module as is")
    p.add_argument("--epochs", type=int, default=None, help="default: the recipe value (100). Recovery runs are shorter")
    p.add_argument("--distill", default=None, metavar="TEACHER.pt", help="train with this teacher (Ultralytics distill_model); give it on --resume too")
    p.add_argument("--fallback-from", type=int, default=None, metavar="N", help="the per-GPU batch this run was planned at, before it fell back to --batch-per-gpu")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="continue an unfinished run from its last.pt")
    mode.add_argument("--eval-only", action="store_true", help="training finished but no record: evaluate best.pt and write the record")
    p.add_argument("--no-eval", action="store_true", help="skip the studyeval pass (use when the run will be evaluated later)")
    p.add_argument("--stop-by", type=float, default=None, help="unix time: pause (exit 75) after the last epoch that finishes before it")
    a = p.parse_args()

    devices = [d for d in str(a.device).split(",") if d != ""]
    n_gpus = len(devices)
    if a.stop_by and n_gpus > 1:
        sys.exit("--stop-by needs a single-GPU run: Ultralytics does not pass callbacks into its DDP subprocess.")
    args = {**studyenv.TRAIN_ARGS, **recipe()}
    if a.seed is not None:
        args["seed"] = a.seed
    if a.workers is not None:
        args["workers"] = a.workers
    if a.epochs is not None:
        args["epochs"] = a.epochs
    pruning = torch.load(a.model, map_location="cpu", weights_only=False).get("pruning") if a.pruned else None
    if a.pruned and not pruning:
        sys.exit(f"--pruned given but {a.model} has no 'pruning' record: make it with scripts/studyprune.py.")
    seed = args["seed"]
    planned_batch = a.batch_per_gpu * max(n_gpus, 1)
    name = a.name or f"{Path(a.model).stem}-{a.imgsz}-s{seed}"
    run_dir = RUNS_DIR / name
    last = run_dir / "weights" / "last.pt"
    best = run_dir / "weights" / "best.pt"
    done_epochs = checkpoint_epoch(last) if last.exists() else None
    finished = done_epochs == -1

    if a.resume and not last.exists():
        sys.exit(f"--resume given but {last} does not exist. Start the run without --resume.")
    if a.resume and finished:
        sys.exit(f"{name} has finished training (last.pt is final). Use --eval-only to write its record.")
    if a.eval_only and not (finished and best.exists()):
        sys.exit(f"--eval-only needs a finished run, but {last} is {'missing' if done_epochs is None else f'at epoch {done_epochs + 1}'}.")
    if not (a.resume or a.eval_only) and last.exists():
        sys.exit(f"{run_dir} already holds a run. Pass --resume to continue it, or --name for a new one.")
    if a.distill and a.eval_only:
        sys.exit("--eval-only reads the teacher from the run's args.yaml: drop --distill.")
    if a.resume and (run_dir / "args.yaml").exists():  # resume takes batch and distill_model from this command line
        started_with = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8"))
        if started_with.get("batch") != planned_batch:
            sys.exit(f"{name} started at batch {started_with.get('batch')}; resuming it at {planned_batch} would change its batch mid-run.")
        if bool(started_with.get("distill_model")) != bool(a.distill):
            sys.exit(f"{name} started {'with' if a.distill is None else 'without'} a teacher: resume it the same way "
                     f"({'add' if a.distill is None else 'drop'} --distill).")

    started = time.time()
    session = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "device": a.device, "host": platform.node(),
               "git_commit": studyenv.git_commit(), "from_epoch": 0 if done_epochs is None else done_epochs + 1}

    if a.eval_only:
        model = YOLO(str(best))
        resolved = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8"))
        best_weights = best
    else:
        model = YOLO(str(last) if a.resume else a.model)
        clock = {"t": time.time()}

        def check_batch(trainer):
            if trainer.batch_size != planned_batch:
                invalid = run_dir.with_name(f"{name}-INVALID-batch-{int(time.time())}")
                print(f"\nSTUDY: Ultralytics changed batch {planned_batch} -> {trainer.batch_size} after running out of GPU memory. "
                      f"This run breaks the constant-batch rule; moved to {invalid.name}. Do NOT lower the batch on your own: report it.", flush=True)
                trainer.train_loader = trainer.test_loader = None
                run_dir.rename(invalid)
                os._exit(EXIT_BATCH_CHANGED)

        def pause_if_out_of_time(trainer):
            now = time.time()
            took, clock["t"] = now - clock["t"], now
            if a.stop_by is None or trainer.epoch + 1 >= trainer.epochs:
                return  # the final epoch always completes, so final_eval can finish the run
            if now + 1.15 * took > a.stop_by:
                log_session(run_dir, {**session, "ended": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                      "hours": round((now - started) / 3600, 3), "to_epoch": trainer.epoch + 1, "outcome": "paused"})
                print(f"\nSTUDY: paused after epoch {trainer.epoch + 1}/{trainer.epochs}: the next epoch (~{took / 60:.0f} min) "
                      f"would run past this session's limit. last.pt is saved; resume in the next session.", flush=True)
                os._exit(EXIT_PAUSED)

        model.add_callback("on_train_epoch_end", check_batch)
        model.add_callback("on_model_save", pause_if_out_of_time)
        model.train(
            data=studyenv.DATA,
            imgsz=a.imgsz,
            batch=planned_batch,
            device=a.device,
            project=str(RUNS_DIR),
            name=name,
            exist_ok=True,
            resume=a.resume,
            **({"trainer": PrunedTrainer} if a.pruned else {}),  # with --distill too: the pruned module is the student
            **({"distill_model": a.distill} if a.distill else {}),
            **({"save_dir": str(run_dir)} if a.resume else {}),  # resume into this checkout even if the old path differs
            **args,
        )
        resolved = vars(model.trainer.args)
        best_weights = model.trainer.best
        log_session(run_dir, {**session, "ended": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "hours": round((time.time() - started) / 3600, 3), "to_epoch": model.trainer.epochs, "outcome": "finished"})

    teacher = resolved.get("distill_model")
    train_args = {k: resolved.get(k) for k in [*args, *(("distill_model", "dis") if teacher else ())]}
    sessions = read_sessions(run_dir)
    record = {
        "run": name,
        "model": a.model,
        "imgsz": a.imgsz,
        "seed": seed,
        "device": a.device,
        "gpus_used": n_gpus,
        "batch_per_gpu": a.batch_per_gpu,
        "batch_total": planned_batch,
        "resumed": a.resume or len(sessions) > 1,
        "eval_only_session": a.eval_only,
        "train_hours_this_session": round((time.time() - started) / 3600, 3),
        "train_hours_total": round(sum(s.get("hours", 0) for s in sessions), 3),
        "sessions": sessions,
        "environment": environment(),
        "train_args": train_args,
        "best_weights": str(best_weights),
        **({"pruning": pruning} if pruning else {}),
        **({"distillation": {"teacher": str(teacher), "teacher_run": Path(teacher).parent.parent.name or None, "dis": resolved.get("dis")}} if teacher else {}),
        **({"batch_fallback_from": a.fallback_from} if a.fallback_from else {}),
    }

    if not a.no_eval:
        eval_device = devices[0] if devices else "0"  # the run's own GPU: the other one may be training another run
        for split in ("val", "test"):  # "test" is test-dev, the headline split
            record[f"eval_{split}"] = studyeval.evaluate(str(best_weights), split, a.imgsz, device=eval_device, project=str(RUNS_DIR), name=f"{name}-eval-{split}", exist_ok=True, plots=False)

    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    out = RECORD_DIR / f"{name}.json"
    out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    headline = record.get("eval_test", {}).get("stats", {})
    print(json.dumps({
        "run": name,
        "record": str(out),
        "train_hours_total": record["train_hours_total"],
        "test_dev_AP_50": headline.get("coco/AP_50"),
        "test_dev_AP_all": headline.get("coco/AP_all"),
        "test_dev_AP_small": headline.get("coco/AP_small"),
    }, indent=2))


if __name__ == "__main__":
    main()
