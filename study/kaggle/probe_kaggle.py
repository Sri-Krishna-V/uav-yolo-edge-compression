"""Measure, on a Kaggle T4 x2 session, the numbers the run schedule needs. Run once per machine class.

  python probe_kaggle.py                 # full probe, writes verification/kaggle_probe.json
  python probe_kaggle.py --quick         # 640 only

Reports, per (model, imgsz): the largest per-GPU batch that fits, its peak memory and images/s, and the
estimated minutes per training epoch. Then at the matrix resolution it compares the two ways of using
both GPUs: ONE run across both (DDP) against TWO concurrent single-GPU runs.

Nothing here is a study run: each attempt trains for ~70 iterations and saves nothing.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import studyenv  # noqa: E402  must precede ultralytics
import studytrain  # noqa: E402  for the shared recipe

WARM, MEASURE = 10, 60
TRAIN_IMAGES = 6471
CANDIDATES = {640: [32, 24, 16, 12, 8], 960: [16, 12, 8, 6, 4], 1280: [8, 6, 4, 3, 2]}
MODELS = ["yolo11s.pt", "yolo26s.pt"]
OUT = studyenv.ROOT / "verification" / "kaggle_probe.json"


class ProbeDone(Exception):
    pass


def attempt(model_name, imgsz, batch_per_gpu, device):
    import torch
    from ultralytics import YOLO

    n_gpus = len(str(device).split(","))
    stamps = []

    def on_batch_end(trainer):
        stamps.append(time.perf_counter())
        if len(stamps) >= WARM + MEASURE + 1:
            raise ProbeDone

    args = {**studyenv.TRAIN_ARGS, **studytrain.recipe()}
    args.update(val=False, plots=False, save=False, epochs=1)
    model = YOLO(model_name)
    model.add_callback("on_train_batch_end", on_batch_end)
    torch.cuda.reset_peak_memory_stats()
    rec = {"model": model_name, "imgsz": imgsz, "batch_per_gpu": batch_per_gpu, "device": device, "gpus": n_gpus}
    try:
        model.train(data=studyenv.DATA, imgsz=imgsz, batch=batch_per_gpu * n_gpus, device=device,
                    project="/tmp/probe_runs", name=f"{Path(model_name).stem}-{imgsz}-b{batch_per_gpu}-g{n_gpus}",
                    exist_ok=True, **args)
        rec["status"] = "epoch_ended_early"
    except ProbeDone:
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    actual = getattr(getattr(model, "trainer", None), "batch_size", None)
    rec["oom_halved"] = actual is not None and actual != batch_per_gpu * n_gpus
    rec["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
    if len(stamps) > WARM + 1 and not rec["oom_halved"]:
        dts = [b - a for a, b in zip(stamps[WARM:], stamps[WARM + 1:])]
        s_it = statistics.median(dts)
        rec["images_per_s"] = round(batch_per_gpu * n_gpus / s_it, 1)
        rec["est_train_epoch_min"] = round(TRAIN_IMAGES / rec["images_per_s"] / 60, 2)
        rec["est_100_epoch_hours"] = round(rec["est_train_epoch_min"] * 100 / 60, 2)
    rec["fits"] = rec["status"] == "ok" and not rec["oom_halved"]
    print("PROBE_RESULT " + json.dumps(rec), flush=True)


def run_child(model_name, imgsz, batch, device, wait=True):
    cmd = [sys.executable, __file__, "--child", model_name, str(imgsz), str(batch), str(device)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    if not wait:
        return proc
    return parse(proc, model_name, imgsz, batch, device)


def parse(proc, model_name, imgsz, batch, device):
    out, err = proc.communicate()
    lines = [ln for ln in out.splitlines() if ln.startswith("PROBE_RESULT ")]
    if lines:
        return json.loads(lines[-1][len("PROBE_RESULT "):])
    return {"model": model_name, "imgsz": imgsz, "batch_per_gpu": batch, "device": device,
            "status": "crashed", "fits": False, "stderr_tail": err[-600:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="640 only")
    a = ap.parse_args()
    sizes = [640] if a.quick else [640, 960, 1280]
    result = {"cpu_count": os.cpu_count(), "attempts": [], "largest_fitting": {}, "pairing": {}}

    for model_name in MODELS:
        for imgsz in sizes:
            for batch in CANDIDATES[imgsz]:
                rec = run_child(model_name, imgsz, batch, "0")
                result["attempts"].append(rec)
                print(json.dumps(rec), flush=True)
                if rec.get("fits"):
                    result["largest_fitting"][f"{Path(model_name).stem}@{imgsz}"] = rec
                    break

    # How to use both GPUs at the matrix resolution: one run on two GPUs, against two concurrent runs.
    chosen = result["largest_fitting"].get("yolo11s@640")
    if chosen:
        b = chosen["batch_per_gpu"]
        result["pairing"]["single_gpu"] = chosen
        result["pairing"]["ddp_two_gpus"] = run_child("yolo11s.pt", 640, b, "0,1")
        procs = [run_child("yolo11s.pt", 640, b, "0", wait=False), run_child("yolo26s.pt", 640, b, "1", wait=False)]
        paired = [parse(procs[0], "yolo11s.pt", 640, b, "0"), parse(procs[1], "yolo26s.pt", 640, b, "1")]
        result["pairing"]["two_concurrent_runs"] = paired
        thr = [p.get("images_per_s") for p in paired]
        if all(thr):
            result["pairing"]["verdict"] = {
                "concurrent_total_images_per_s": round(sum(thr), 1),
                "ddp_total_images_per_s": result["pairing"]["ddp_two_gpus"].get("images_per_s"),
                "note": "higher total wins: it is more finished runs per hour of Kaggle quota",
            }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"cpu_count": result["cpu_count"], "largest_fitting": {k: v["batch_per_gpu"] for k, v in result["largest_fitting"].items()}, "written": str(OUT)}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        attempt(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    else:
        main()
