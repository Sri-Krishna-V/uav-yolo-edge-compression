"""Report a Pi 5 session's clock and throttle per attempt, and where its time went (#43). Reads the session directory only.

  python studyclock.py bench/pi5/pi5-matrix-1-yolo11s [more session directories...]

The session records' clock tables (verification/pi5_matrix_*_record.md) and the paper's throttle columns come from here
(INVARIANTS.md section 4.7, #40, #42). For every attempt, from its timings.csv:
  throttle     the get_throttled reads with bit 3 (the soft temperature limit) set, of those read. The bench reads it at
               every 50th image only (32 reads in a full cell, 4 in a 200-image reference run), so none means that no
               read caught the limit, not that the cell ran at 2000 MHz throughout
  clock        the images below the 2000 MHz lock, and the clock's min and median: the clock is read at every image
  peak CPU     the highest CPU temperature read
  latency      the median e2e_deploy, pre + net + post at the deployment threshold (0.25), as the bench computes it
A timings.csv that a restart cut short ends in a NUL-filled line: that line is skipped, and the attempt counts the
images it timed. The attempt that counts is the one summary.csv names in run_dir; the table marks it.
From session.json, the session's time, split by pass (the first attempts, then the throttle re-runs) into the cells'
attempts, their start gates and the reference runs, plus any interrupted attempts, which have no end. The reference
runs' gates count with the first pass. Each complete attempt's overhead is its time beyond its images x its median
latency: the model load, the 20 warm-up images, the second post-processing pass at 0.001 and the untimed part of each
image's loop. The phone has no clock lock and no throttle signal, so for a session on any platform but the Pi 5 the
table leaves the throttle, clock and temperature columns empty, and only the latency and the time split mean anything.
Checked by tests/test_studyclock.py: every row of the clock tables of sessions 1-6's records, and the time tables of
sessions 2, 4, 5 and 7, come out of this script unchanged. The records round latency half up (393.15 -> 393.2); so does
this. Standard library only; attempt_clock() and session_attempts() are what studyresults.py reuses.
"""

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

LOCK_MHZ = 2000
SOFT_LIMIT = 0x8  # get_throttled bit 3: the soft temperature limit is active
CLOCK_KEYS = ["images", "samples_read", "samples_soft_limit", "below_lock", "clock_min_mhz", "clock_median_mhz", "peak_temp_c", "e2e_deploy_ms"]


def _num(v):
    return float(v) if v not in (None, "") else None


def read_timings(path):
    """timings.csv's rows, without the NUL-filled tail a restart leaves."""
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(line for line in f if "\0" not in line) if r.get("file")]


def attempt_clock(attempt_dir):
    """One attempt's images, throttle reads, clock, peak temperature and median e2e_deploy, from its timings.csv."""
    rows = read_timings(Path(attempt_dir) / "timings.csv")
    reads = [int(r["throttled"], 16) for r in rows if r.get("throttled")]
    mhz = [m for m in (_num(r.get("cpu_mhz")) for r in rows) if m is not None]
    temp = [t for t in (_num(r.get("temp_c")) for r in rows) if t is not None]
    e2e = [float(r["t_pre_ms"]) + float(r["t_net_ms"]) + float(r["t_post_deploy_ms"]) for r in rows if r.get("t_post_deploy_ms")]
    return {"images": len(rows), "samples_read": len(reads), "samples_soft_limit": sum(1 for v in reads if v & SOFT_LIMIT),
            "below_lock": sum(1 for m in mhz if m < LOCK_MHZ),
            "clock_min_mhz": min(mhz) if mhz else None, "clock_median_mhz": statistics.median(mhz) if mhz else None,
            "peak_temp_c": max(temp) if temp else None, "e2e_deploy_ms": round(statistics.median(e2e), 3) if e2e else None}


def _t(iso):
    return datetime.fromisoformat(iso) if iso else None


def counted_dirs(session_dir):
    """summary.csv's run_dir for every row: the attempt that counts, for each cell and reference run."""
    with open(Path(session_dir) / "summary.csv", newline="", encoding="utf-8") as f:
        return {r["cell"]: r["run_dir"] for r in csv.DictReader(f)}


def session_attempts(session_dir):
    """Every attempt of the session, in the order it ran: session.json's record of it, its clock (attempt_clock), whether
    it is the attempt that counts, and its time and overhead in seconds (None for an interrupted attempt)."""
    session_dir = Path(session_dir)
    state = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    counted, platform = counted_dirs(session_dir), state["plan"]["platform"]
    out = []
    for unit, u in state["units"].items():
        for a in u["attempts"]:
            start, end = _t(a.get("start")), _t(a.get("end"))
            rec = {"platform": platform, "unit": unit, "reference": unit.startswith("drift-"), "n": a["n"], "pass": a["pass"], "result": a["result"],
                   "dir": a["dir"], "counted": counted.get(unit) == a["dir"], "gate_s": (a.get("gate") or {}).get("waited_s") or 0.0,
                   "start": start, "end": end, "seconds": (end - start).total_seconds() if start and end else None}
            timings = session_dir / a["dir"] / "timings.csv"
            rec.update(attempt_clock(session_dir / a["dir"]) if timings.exists() else dict.fromkeys(CLOCK_KEYS))
            rec["overhead_s"] = (rec["seconds"] - rec["images"] * rec["e2e_deploy_ms"] / 1000
                                 if rec["seconds"] is not None and rec["e2e_deploy_ms"] is not None else None)
            out.append(rec)
    return sorted(out, key=lambda r: r["start"])


def time_split(attempts):
    """session_attempts()'s attempts -> each pass's cells, gates, reference runs and interrupted attempts, in seconds, and
    the whole session's span. An interrupted attempt's time runs to the next gate: the restart is part of it."""
    out = {p: dict.fromkeys(["cells", "cells_s", "gates", "gates_s", "reference_s", "interrupted", "interrupted_s"], 0) for p in ("main", "rerun")}
    for i, a in enumerate(attempts):
        p = out[a["pass"]]
        p["gates"] += 1
        p["gates_s"] += a["gate_s"]
        if a["result"] != "complete":
            p["interrupted"] += 1
            nxt = attempts[i + 1] if i + 1 < len(attempts) else None
            if nxt:
                p["interrupted_s"] += (nxt["start"] - a["start"]).total_seconds() - nxt["gate_s"]
        elif a["reference"]:
            p["reference_s"] += a["seconds"]
        else:
            p["cells"] += 1
            p["cells_s"] += a["seconds"]
    for p in out.values():
        p["total_s"] = p["cells_s"] + p["gates_s"] + p["reference_s"] + p["interrupted_s"]
    first = min(a["start"].timestamp() - a["gate_s"] for a in attempts)
    out["session_s"] = max(a["end"].timestamp() for a in attempts if a["end"]) - first
    return out


def _r1(x):
    """One decimal, half up, as the records round (393.15 -> 393.2)."""
    return str(Decimal(repr(x)).quantize(Decimal("0.1"), ROUND_HALF_UP))


def _h(s):
    return f"{s / 3600:.2f} h ({s / 60:.0f} min)"


def report(session_dir):
    """The session's clock table and time split, as Markdown."""
    attempts = session_attempts(session_dir)
    lines = [f"## {Path(session_dir).as_posix()}", "",
             f"Clock and throttle per attempt, from timings.csv (the lock is {LOCK_MHZ} MHz; get_throttled is read at every 50th image). "
             "**Bold** is the attempt summary.csv names in run_dir. Overhead is the attempt's time beyond its images x its median latency.", "",
             "| Cell | Attempt | Pass | Median latency, ms | Throttle during the run | Images below 2000 MHz | Clock min / median, MHz | Peak CPU | Overhead, min |",
             "|---|---|---|---|---|---|---|---|---|"]
    for a in attempts:
        name = f"**{a['unit']}**" if a["counted"] else a["unit"]
        if a["result"] != "complete":
            name += f" ({a['result']})"
        if a["images"] is None:
            lines.append(f"| {name} | {a['n']} | {a['pass']} | no timings.csv | | | | | |")
            continue
        if a["platform"] != "pi5":  # no clock lock and no throttle signal: only the latency and the time mean anything
            a = {**a, "samples_read": 0, "clock_min_mhz": None, "peak_temp_c": None}
        thr = ("no reads" if not a["samples_read"] else
               f"{'soft limit' if a['samples_soft_limit'] else 'none'} ({a['samples_soft_limit']} of {a['samples_read']})")
        clock = "-" if a["clock_min_mhz"] is None else f"{a['clock_min_mhz']:.0f} / {a['clock_median_mhz']:.0f}"
        peak = "-" if a["peak_temp_c"] is None else f"{a['peak_temp_c']:.1f} °C"
        ms = "-" if a["e2e_deploy_ms"] is None else _r1(a["e2e_deploy_ms"])
        over = "-" if a["overhead_s"] is None else f"{a['overhead_s'] / 60:.2f}"
        below = "-" if a["clock_min_mhz"] is None else f"{a['below_lock']:,} of {a['images']:,}"
        lines.append(f"| {name} | {a['n']} | {a['pass']} | {ms} | {thr} | {below} | {clock} | {peak} | {over} |")
    t = time_split(attempts)
    lines += ["", f"The session's time, {_h(t['session_s'])} from the first gate to the last attempt's end. The reference runs' gates "
              "count with the first attempts; an interrupted attempt's time runs to the next gate.", "",
              "| Pass | Time | Cells | Gates | Reference runs | Interrupted attempts |", "|---|---|---|---|---|---|"]
    for p, label in (("main", "first attempts"), ("rerun", "throttle re-runs")):
        x = t[p]
        lines.append(f"| {label} | {_h(x['total_s'])} | {x['cells']}: {_h(x['cells_s'])} | {x['gates']}: {x['gates_s'] / 60:.0f} min | "
                     f"{x['reference_s'] / 60:.1f} min | {x['interrupted']}: {x['interrupted_s'] / 60:.1f} min |")
    over = [a["overhead_s"] / 60 for a in attempts if a["overhead_s"] is not None and not a["reference"]]
    if over:
        lines += ["", f"The cells' attempts ran {min(over):.2f}-{max(over):.2f} min each beyond images x median latency, "
                  f"{sum(over):.1f} min in all."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sessions", nargs="+", type=Path, help="session directories, e.g. bench/pi5/pi5-matrix-1-yolo11s")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # the degree sign, on a Windows console
    for s in args.sessions:
        print(report(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
