"""Gather every scored platform cell into one table, results/cells.csv, one row per cell (#44).

  python studyresults.py                       # reads bench/, writes results/cells.csv
  python studyresults.py --bench B --out F.csv

Every table and figure of the paper reads cells.csv, never the session folders. Re-run it when a session lands: a new
session directory that SESSIONS names becomes rows with no code change.

Which runs are cells: every row of summary.csv, but the reference runs (drift-start, drift-end), in the Pi 5 matrix
sessions (bench/pi5/pi5-matrix-*) and the Galaxy M34's full-cell and matrix sessions (bench/m34/m34-fullcell*,
bench/m34/m34-matrix-*). Not cells: the speed tests (200 images), the Pi 5 pilot and full-cell checks (pi5-fullcell*),
the Pi 5 speed test and the energy check. A session with no summary.csv yet (still running) is skipped and reported.
The attempt that counts is the one summary.csv names in run_dir, as in the session records. A cell with no attempt that
counts, or with no score.json there, is still a row, with empty accuracy and a note: no cell is skipped silently.

Light on the laptop: standard library only, no torch or ultralytics. It reads summary.csv, session.json, and each
counted attempt's score.json, meta.json and timings.csv, never the detections (predictions.json).

The columns, a contract for #46-#48: their names do not change.
  platform          pi5 or m34, from session.json
  session           the session directory, e.g. pi5-matrix-1-yolo11s
  cell              summary.csv's name, <run>/<imgsz>/<variant>
  run               the checkpoint's run name, e.g. yolo26s-960-s0-prune30
  checkpoint        its RUNS.md code, e.g. P4
  model             YOLO11s or YOLO26s
  trained_imgsz     the input resolution the checkpoint was trained at
  imgsz             the input resolution the cell ran at
  untrained_imgsz   1 when imgsz is not trained_imgsz (800 from the 960 models, 1120 from the 1280 ones), else 0
  configuration     GLOSSARY.md's, from the run's suffix and the precision: no suffix is baseline, or quantized at INT8;
                    -prune30 pruned; -distill distilled; -prune30-distill at INT8 combined. Empty for any other mix
  runtime           onnxruntime, openvino or ncnn
  variant           the export's folder name, e.g. onnxruntime-int8pc-e2e. Never read the precision from it
  precision         what the runtime really computed in (score.json, else meta.json): NCNN's fp32 export is FP16 on
                    the Pi 5
  head              one-to-many, or NMS-free (score.json's head: raw or end2end)
  int8_recipe       stock (onnxruntime-int8*), per-channel (onnxruntime-int8pc*, the matrix's ONNX Runtime INT8
                    recipe, INVARIANTS.md section 8), or empty
  ap, ap50, ap75, ap_small, ap_medium, ap_large, ar500
                    score.json's coco/AP_all, AP_50, AP_75, AP_small, AP_medium, AP_large and AR_500: COCO at maxDets
                    500, from the detections at the evaluation threshold (0.001)
  latency_ms        the median e2e_deploy, pre + net + post at the deployment threshold (0.25): the cell's latency
  latency_p95_ms    its 95th percentile
  pre_ms, net_ms    the medians of pre-processing and of the runtime call
  post_deploy_ms    the median post-processing at the deployment threshold, 0.25
  post_eval_ms      the median post-processing at the evaluation threshold, 0.001 (the scoring pass)
                    The latency columns are meta.json's timing_ms, the bench's own medians.
  images            the images the attempt timed, from timings.csv (1,610 for a full cell)
  throttled         the Pi 5's throttle flag as summary.csv recorded it (throttled or none)
  throttle_reads    the get_throttled reads (every 50th image: 32 in a full cell)
  soft_limit_reads  of those, the reads with bit 3, the soft temperature limit, set
  below_2000_mhz    the images whose clock read below the 2000 MHz lock
  clock_min_mhz, clock_median_mhz, peak_temp_c
                    the clock's min and median and the peak CPU temperature over the attempt
                    The throttle columns are studyclock.py's (#43), by INVARIANTS.md section 4.7: the paper reports
                    throttling by the clock. They are empty for the phone, which has no clock lock and no throttle signal.
  attempts          the cell's attempts in its session; attempt is the one that counts (empty if none does)
  run_dir           that attempt's directory, relative to the study folder
  drift             the session's drift verdict (ok, re-run or unknown); outcome is summary.csv's (ok, throttled, failed)
  python_version, numpy_version, opencv_version, runtime_version
                    the platform's software, from meta.json: the phone's sessions of 25 October (OpenCV 4.14.0,
                    ONNX Runtime 1.31.0) and 1 November (5.0.0, 1.31.1) differ by these
  start_utc         when the attempt started
  note              anything a table must footnote, joined by "; ": no attempt counts or no score.json; an earlier
                    attempt interrupted; a flagged drift check; and SESSION_NOTES, the decisions recorded for a session
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import studyclock

STUDY = Path(__file__).resolve().parents[1]
SESSIONS = ["pi5/pi5-matrix-*", "m34/m34-fullcell*", "m34/m34-matrix-*"]
SESSION_NOTES = {  # decided on #44, #37 and #27, 1 November
    "pi5-matrix-6-1120": "latency about 6 % high against the other sessions' (measured by session 7)",
    "pi5-matrix-7-int8pc": "drift flagged across a restart; kept (#37)",
    "m34-matrix-1": "drift +4.9 %, flagged; kept, latency provisional (#27, #34)",
    "m34-matrix-2": "drift +3.4 %, flagged; kept, latency provisional (#27, #34)",
    "m34-matrix-3": "drift +5.9 %, flagged; kept, latency provisional (#27, #34)",
}
DRIFT_DECIDED = {"pi5-matrix-7-int8pc", "m34-matrix-1", "m34-matrix-2", "m34-matrix-3"}  # their drift notes are in SESSION_NOTES
ACCURACY = {"ap": "coco/AP_all", "ap50": "coco/AP_50", "ap75": "coco/AP_75", "ap_small": "coco/AP_small", "ap_medium": "coco/AP_medium",
            "ap_large": "coco/AP_large", "ar500": "coco/AR_500"}
LATENCY = {"latency_ms": ("e2e_deploy", "median"), "latency_p95_ms": ("e2e_deploy", "p95"), "pre_ms": ("t_pre_ms", "median"),
           "net_ms": ("t_net_ms", "median"), "post_deploy_ms": ("post_deploy", "median"), "post_eval_ms": ("t_post_ms", "median")}
THROTTLE_COLUMNS = ["throttled", "throttle_reads", "soft_limit_reads", "below_2000_mhz", "clock_min_mhz", "clock_median_mhz", "peak_temp_c"]
COLUMNS = (["platform", "session", "cell", "run", "checkpoint", "model", "trained_imgsz", "imgsz", "untrained_imgsz", "configuration", "runtime",
            "variant", "precision", "head", "int8_recipe"] + list(ACCURACY) + list(LATENCY) + ["images"] + THROTTLE_COLUMNS
           + ["attempts", "attempt", "run_dir", "drift", "outcome", "python_version", "numpy_version", "opencv_version", "runtime_version",
              "start_utc", "note"])
RUN = re.compile(r"^(yolo11s|yolo26s)-(\d+)-s0(-prune30)?(-distill)?$")


def checkpoints():
    """RUNS.md's checkpoint codes by run name: yolo11s-640-s0 -> A1."""
    rows = re.findall(r"^\| ([A-Z]\d) \|.*?\| `([^`]+)` \|", (STUDY / "RUNS.md").read_text(encoding="utf-8"), re.M)
    return {run: code for code, run in rows}


def configuration(run, precision):
    """GLOSSARY.md's configuration of a cell: the run's suffix, and INT8 for the quantized and combined ones."""
    _, _, pruned, distilled = RUN.match(run).groups()
    int8 = precision == "INT8"
    if pruned and distilled:
        return "combined" if int8 else ""
    if pruned or distilled:
        return "" if int8 else ("pruned" if pruned else "distilled")
    return "quantized" if int8 else "baseline"


def _int(x):
    return None if x is None else int(x)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def cell_row(sdir, platform, s, codes):
    """One summary.csv row of a session -> its cells.csv row, from the attempt run_dir names."""
    adir = sdir / s["run_dir"] if s["run_dir"] else None
    score, meta = (_json(adir / "score.json") or {}, _json(adir / "meta.json") or {}) if adir else ({}, {})
    stats, timing, versions = score.get("stats", {}), meta.get("timing_ms", {}), meta.get("versions", {})
    run, imgsz, variant = s["cell"].split("/")
    model, trained = RUN.match(run).group(1, 2)
    precision = score.get("precision") or meta.get("precision") or s["precision"]
    head = score.get("head") or meta.get("head") or ("end2end" if variant.endswith("-e2e") else "raw")
    row = {"platform": platform, "session": sdir.name, "cell": s["cell"], "run": run, "checkpoint": codes.get(run, ""),
           "model": model.replace("yolo", "YOLO"), "trained_imgsz": int(trained), "imgsz": int(imgsz),
           "untrained_imgsz": int(int(imgsz) != int(trained)), "configuration": configuration(run, precision), "runtime": s["runtime"],
           "variant": variant, "precision": precision, "head": "NMS-free" if head == "end2end" else "one-to-many",
           "int8_recipe": "per-channel" if variant.startswith("onnxruntime-int8pc") else "stock" if variant.startswith("onnxruntime-int8") else ""}
    row.update({col: stats.get(key) for col, key in ACCURACY.items()})
    row.update({col: (timing.get(key) or {}).get(stat) for col, (key, stat) in LATENCY.items()})
    row.update(dict.fromkeys(["images"] + THROTTLE_COLUMNS))
    if adir and (adir / "timings.csv").is_file():
        c = studyclock.attempt_clock(adir)
        row["images"] = c["images"]
        if platform == "pi5":
            row.update(throttled=s["throttle"], throttle_reads=c["samples_read"], soft_limit_reads=c["samples_soft_limit"],
                       below_2000_mhz=c["below_lock"], clock_min_mhz=_int(c["clock_min_mhz"]), clock_median_mhz=_int(c["clock_median_mhz"]),
                       peak_temp_c=c["peak_temp_c"])
    row.update(attempts=int(s["attempts"]), attempt=int(s["run_dir"].rsplit("-", 1)[1]) if adir else None,
               run_dir=adir.relative_to(sdir.parents[2]).as_posix() if adir else "", drift=s["drift"], outcome=s["outcome"],
               python_version=versions.get("python"), numpy_version=versions.get("numpy"), opencv_version=versions.get("cv2"),
               runtime_version=versions.get(s["runtime"]), start_utc=meta.get("start"))
    notes = []
    if not adir:
        notes.append(f"{s['outcome']}: no attempt counts")
    elif not score:
        notes.append("not scored: no score.json in run_dir")
    if "interrupted" in s["attempt_log"]:
        notes.append("an earlier attempt was interrupted; run_dir's attempt counts")
    if s["drift"] not in ("ok", "") and sdir.name not in DRIFT_DECIDED:
        notes.append(f"drift check: {s['drift']}")
    if sdir.name in SESSION_NOTES:
        notes.append(SESSION_NOTES[sdir.name])
    row["note"] = "; ".join(notes)
    return row


def sessions(bench):
    """The session directories SESSIONS names under bench/, in order."""
    return [d for pattern in SESSIONS for d in sorted(Path(bench).glob(pattern)) if d.is_dir()]


def build_rows(bench=STUDY / "bench"):
    """Every cell of every session that has its summary.csv, in session order, then summary.csv's order."""
    out, codes = [], checkpoints()
    for sdir in sessions(bench):
        if not (sdir / "summary.csv").is_file():
            continue
        platform = json.loads((sdir / "session.json").read_text(encoding="utf-8"))["plan"]["platform"]
        with open(sdir / "summary.csv", newline="", encoding="utf-8") as f:
            out += [cell_row(sdir, platform, s, codes) for s in csv.DictReader(f) if not s["cell"].startswith("drift-")]
    return out


def write_cells(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows({k: ("" if r[k] is None else r[k]) for k in COLUMNS} for r in rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", type=Path, default=STUDY / "bench")
    ap.add_argument("--out", type=Path, default=STUDY / "results" / "cells.csv")
    args = ap.parse_args(argv)
    rows = build_rows(args.bench)
    write_cells(rows, args.out)
    for sdir in sessions(args.bench):
        n = sum(1 for r in rows if r["session"] == sdir.name)
        print(f"{sdir.name}: " + (f"{n} cells" if (sdir / "summary.csv").is_file() else "skipped, no summary.csv yet (still running?)"))
    unscored = sum(1 for r in rows if r["ap"] is None)
    print(f"{len(rows)} cells -> {args.out}" + (f"; {unscored} with no accuracy (see their note)" if unscored else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
