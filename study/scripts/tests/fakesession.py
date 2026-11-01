"""A small fake platform session on disk, laid out as studysession.py writes one, for the tests of the scripts that
read sessions (studyclock.py, studyresults.py). Nothing here runs a benchmark: it writes the files only.

  make_session(root, "pi5-matrix-9-fake", cells, platform="pi5")   # -> the session directory

Each cell is a dict: name (`<run>/<imgsz>/<variant>`), runtime, imgsz, and attempts, a list of attempt dicts:
  pass_    "main" or "rerun"
  result   "complete" or "interrupted"
  start    seconds after the session's start; the gate (gate_s, default 180) ends there
  seconds  how long the attempt ran (complete attempts only)
  rows     timings.csv rows, each a dict of its columns (missing columns are left empty)
  nul_tail True to end timings.csv with a NUL-filled line, as a restart leaves it
  score    the score.json to write (None: no score.json, as for a run not scored yet)
  meta     extra keys for meta.json
counted is the attempt summary.csv names in run_dir (default: the last one).
"""
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

COLUMNS = ["index", "file", "t_pre_ms", "t_net_ms", "t_post_ms", "n_cand", "n_dets", "temp_c", "cpu_mhz", "throttled", "img_crc32", "session",
           "gpu_temp_c", "gpu_mhz", "t_post_deploy_ms", "n_dets_deploy"]
SUMMARY = ["cell", "runtime", "precision", "imgsz", "images", "attempts", "outcome", "latency_ms", "latency_source", "throttle", "drift", "run_dir",
           "attempt_log"]
T0 = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)


def iso(seconds):
    return (T0 + timedelta(seconds=seconds)).isoformat(timespec="seconds")


def write_timings(path, rows, nul_tail=False):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"index": i, "file": f"{i:07d}.jpg", "session": 0, **r})
    if nul_tail:
        with open(path, "ab") as f:
            f.write(b"\x00" * 64)


def make_session(root, name, cells, platform="pi5", precision="FP32", drift="ok"):
    out = Path(root) / name
    units, summary = {}, []
    for c in cells:
        attempts = []
        for n, a in enumerate(c["attempts"], 1):
            rel = f"runs/{c['name']}/attempt-{n}"
            d = out / rel
            d.mkdir(parents=True, exist_ok=True)
            write_timings(d / "timings.csv", a.get("rows", []), a.get("nul_tail", False))
            rec = {"n": n, "pass": a.get("pass_", "main"), "dir": rel,
                   "gate": {"waited_s": float(a.get("gate_s", 180)), "opened": "3 min passed", "cpu_temp_c": 55.0, "battery_temp_c": None},
                   "start": iso(a["start"]), "result": a.get("result", "complete")}
            if rec["result"] == "complete":
                rec.update(end=iso(a["start"] + a["seconds"]), throttle=a.get("throttle", "none"), precision=a.get("precision", precision))
                (d / "meta.json").write_text(json.dumps({"status": "complete", "precision": a.get("precision", precision), **a.get("meta", {})}),
                                             encoding="utf-8")
            else:
                rec["error"] = "the session stopped during this run"
            if a.get("score") is not None:
                (d / "score.json").write_text(json.dumps(a["score"]), encoding="utf-8")
            attempts.append(rec)
        units[c["name"]] = {"attempts": attempts}
        k = c.get("counted", len(attempts))
        a = c["attempts"][k - 1]
        summary.append({"cell": c["name"], "runtime": c["runtime"], "precision": a.get("precision", precision), "imgsz": c["imgsz"],
                        "images": c.get("images", "all"), "attempts": len(attempts), "outcome": c.get("outcome", "ok"),
                        "latency_ms": c.get("latency_ms", ""), "latency_source": "e2e_deploy", "throttle": a.get("throttle", "none"),
                        "drift": c.get("drift", drift), "run_dir": f"runs/{c['name']}/attempt-{k}", "attempt_log": "a1 complete"})
    first = min(a["start"] - a.get("gate_s", 180) for c in cells for a in c["attempts"])
    session = {"tool": "studysession.py", "plan": {"platform": platform, "note": "a fake session"}, "units": units,
               "invocations": [{"start": iso(first)}], "status": "complete"}
    (out / "session.json").write_text(json.dumps(session), encoding="utf-8")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY)
        w.writeheader()
        w.writerows(summary)
    return out
