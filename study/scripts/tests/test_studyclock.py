"""Check studyclock.py, the clock and time tables of a Pi 5 session (#43): on a small fake session, and against the clock
tables of the session records (verification/pi5_matrix_{1..6}_record.md), which earlier one-off scripts computed from
the same files. Runs on any machine with the repo's bench files:

  python study/scripts/tests/test_studyclock.py
"""
import contextlib
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fakesession as fake  # noqa: E402
import studyclock as sc  # noqa: E402

FAILED = []


def check(label, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        FAILED.append(label)
        if detail:
            print(f"     {detail}")


def row(pre, net, post, deploy, temp, mhz, throttled=""):
    return {"t_pre_ms": pre, "t_net_ms": net, "t_post_ms": post, "t_post_deploy_ms": deploy, "temp_c": temp, "cpu_mhz": mhz, "throttled": throttled}


# e2e_deploy per image: 11.5, 14, 13.5, 21.5, 12.5 -> median 13.5. Clock: 1500, 1911, 1966, 2000, 2000 -> median 1966.
FIVE = [row(1, 10, 5, 0.5, 70.0, 2000), row(1, 12, 5, 1.0, 75.5, 1911, "0xe0008"), row(2, 11, 5, 0.5, 80.1, 1500),
        row(1, 20, 5, 0.5, 72.0, 2000, "0xe0000"), row(1, 10, 5, 1.5, 71.0, 1966)]


def test_attempt_clock():
    tmp = Path(tempfile.mkdtemp())
    s = fake.make_session(tmp, "s", [{"name": "m-640-s0/640/onnxruntime-fp32", "runtime": "onnxruntime", "imgsz": 640,
                                      "attempts": [{"start": 180, "seconds": 600, "rows": FIVE, "nul_tail": True}]}])
    c = sc.attempt_clock(s / "runs/m-640-s0/640/onnxruntime-fp32/attempt-1")
    want = {"images": 5, "samples_read": 2, "samples_soft_limit": 1, "below_lock": 3, "clock_min_mhz": 1500, "clock_median_mhz": 1966,
            "peak_temp_c": 80.1, "e2e_deploy_ms": 13.5}
    for k, v in want.items():
        check(f"attempt clock: {k} = {v}", c[k] == v, f"got {c[k]!r}")


def test_time_split():
    """A first pass of two reference runs, cell A and cell B (B interrupted once, by a restart), then A's re-run."""
    tmp = Path(tempfile.mkdtemp())
    ref = {"runtime": "onnxruntime", "imgsz": 640}
    s = fake.make_session(tmp, "s", [
        {"name": "drift-start", **ref, "attempts": [{"start": 180, "seconds": 100, "rows": FIVE}]},
        {"name": "a-640-s0/640/onnxruntime-fp32", **ref, "attempts": [{"start": 460, "seconds": 600, "rows": FIVE, "throttle": "throttled"},
                                                                     {"start": 2280, "seconds": 650, "rows": FIVE, "pass_": "rerun"}]},
        {"name": "b-640-s0/640/onnxruntime-fp32", **ref, "attempts": [{"start": 1240, "result": "interrupted", "rows": FIVE[:2], "nul_tail": True},
                                                                     {"start": 1600, "seconds": 500, "rows": FIVE}]},
        {"name": "drift-end", **ref, "attempts": [{"start": 3110, "seconds": 100, "rows": FIVE}]}])
    attempts = sc.session_attempts(s)
    b = [a for a in attempts if a["unit"].startswith("b-")]
    check("the interrupted attempt has no time and no overhead, and counts its 2 images",
          b[0]["seconds"] is None and b[0]["overhead_s"] is None and b[0]["images"] == 2, f"{b[0]}")
    check("a complete attempt's overhead is its time beyond images x median latency (500 - 5 x 13.5 ms)",
          abs(b[1]["overhead_s"] - 499.9325) < 1e-9, f"{b[1]['overhead_s']}")
    check("summary.csv's run_dir marks the attempt that counts", [a["counted"] for a in attempts if a["unit"].startswith("a-")] == [False, True])
    t = sc.time_split(attempts)
    want = {"main": {"cells": 2, "cells_s": 1100, "gates": 5, "gates_s": 900, "reference_s": 200, "interrupted": 1, "interrupted_s": 180,
                     "total_s": 2380},
            "rerun": {"cells": 1, "cells_s": 650, "gates": 1, "gates_s": 180, "reference_s": 0, "interrupted": 0, "interrupted_s": 0, "total_s": 830}}
    for p, w in want.items():
        check(f"time split, {p} pass: {w}", t[p] == w, f"got {t[p]}")
    check("the whole session runs from the first gate to the last end (3210 s)", t["session_s"] == 3210, f"got {t['session_s']}")


STUDY = Path(__file__).resolve().parents[2]
RECORDS = {k: (f"verification/pi5_matrix_{k}_record.md", f"bench/pi5/{d}") for k, d in
           [(1, "pi5-matrix-1-yolo11s"), (2, "pi5-matrix-2-yolo26s"), (3, "pi5-matrix-3-yolo26s-compressed"),
            (4, "pi5-matrix-4-1280-baselines"), (5, "pi5-matrix-5-800"), (6, "pi5-matrix-6-1120")]}


def record_clock_table(path):
    """The record's clock table, one dict per row: what the record's one-off scripts printed, by hand-written rules."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("| Cell | Attempt | Median latency"))
    out = []
    for ln in lines[start + 2:]:
        if not ln.startswith("|"):
            break
        cell, n, ms, thr, below, clock, peak = [c.strip() for c in ln.strip("|").split("|")]
        soft, of = 0, None
        if "(" in thr:
            inner = thr[thr.index("(") + 1:thr.index(")")].split()
            soft, of = (0, int(inner[1])) if inner[0] == "of" else (int(inner[0]), int(inner[2]))
        lo, hi = clock.split("/")
        out.append({"label": cell.strip("*"), "bold": cell.startswith("**"), "n": int(n), "ms": float(ms), "soft": soft, "of": of,
                    "below": int(below.split()[0].replace(",", "")), "min": int(lo), "median": int(hi), "peak": float(peak.split()[0])})
    return out


def test_records():
    """Every row of sessions 1-6's clock tables, found by its attempt and median latency (to 0.05: the records round a tie
    either way, 393.15 up and 232.95 down),
    gives the record's numbers, and the attempts the records bold are the ones that count."""
    for k, (rec, sdir) in RECORDS.items():
        table, got = record_clock_table(STUDY / rec), sc.session_attempts(STUDY / sdir)
        bold = any(r["bold"] for r in table)
        bad = []
        for r in table:
            hit = [a for a in got if a["n"] == r["n"] and a["e2e_deploy_ms"] is not None and abs(a["e2e_deploy_ms"] - r["ms"]) < 0.0501]
            if len(hit) != 1:
                bad.append(f"{r['label']} a{r['n']}: {len(hit)} attempts at {r['ms']} ms")
                continue
            a = hit[0]
            mine = {"soft": a["samples_soft_limit"], "below": a["below_lock"], "min": a["clock_min_mhz"], "median": a["clock_median_mhz"],
                    "peak": a["peak_temp_c"]}
            if r["of"] is not None:
                mine["of"] = a["samples_read"]
            if bold and not a["reference"]:  # the records bold the cells' attempts only
                mine["bold"] = a["counted"]
            diff = {key: (r[key], v) for key, v in mine.items() if r[key] != v}
            if diff:
                bad.append(f"{r['label']} a{r['n']}: record vs script {diff}")
        check(f"session {k}: the {len(table)} rows of the record's clock table", not bad, "; ".join(bad))
        attempts = [a for a in got if a["result"] == "complete"]
        check(f"session {k}: every complete attempt is a row of the record ({len(attempts)})", len(attempts) == len(table),
              f"{len(attempts)} attempts, {len(table)} rows")


# The records' time tables (sessions 2, 4 and 5; session 7's): pass -> (total h, cells h, gates, gates min), as rounded there.
TIME_TABLES = {2: {"main": (3.1, 2.4, 13, 39), "rerun": (1.8, 1.5, 6, 18)}, 4: {"main": (4.1, 3.6, 8, 24), "rerun": (3.0, 2.8, 4, 12)},
               5: {"main": (2.3, 1.8, 10, 27), "rerun": (0.7, 0.6, 2, 6)}}


def test_record_time_tables():
    for k, passes in TIME_TABLES.items():
        t = sc.time_split(sc.session_attempts(STUDY / RECORDS[k][1]))
        for p, (total_h, cells_h, gates, gates_min) in passes.items():
            mine = (round(t[p]["total_s"] / 3600, 1), round(t[p]["cells_s"] / 3600, 1), t[p]["gates"], round(t[p]["gates_s"] / 60))
            check(f"session {k}, {p} pass: the record's time table {total_h} h, cells {cells_h} h, {gates} gates {gates_min} min",
                  mine == (total_h, cells_h, gates, gates_min), f"got {mine}")
    t7 = sc.time_split(sc.session_attempts(STUDY / "bench/pi5/pi5-matrix-7-int8pc"))["main"]
    check("session 7: 12 cells 3.05 h, 15 gates 45 min, the interrupted attempt and the restart 8 min",
          (round(t7["cells_s"] / 3600, 2), t7["gates"], round(t7["gates_s"] / 60), round(t7["interrupted_s"] / 60)) == (3.05, 15, 45, 8), f"{t7}")


def test_report():
    """What the script prints for a session: the clock table in the records' terms, the attempt that counts in bold."""
    tmp = Path(tempfile.mkdtemp())
    s = fake.make_session(tmp, "s", [{"name": "m-640-s0/640/onnxruntime-fp32", "runtime": "onnxruntime", "imgsz": 640,
                                      "attempts": [{"start": 180, "seconds": 60, "rows": FIVE}]}])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = sc.main([str(s)])
    text = out.getvalue()
    line = "| **m-640-s0/640/onnxruntime-fp32** | 1 | main | 13.5 | soft limit (1 of 2) | 3 of 5 | 1500 / 1966 | 80.1 °C | 1.00 |"
    check("the report prints the attempt's row, in the records' terms", line in text, text)
    check("the report prints the time split", "| first attempts | " in text and "| throttle re-runs | " in text, text)
    check("exit code 0", code == 0)


if __name__ == "__main__":
    test_attempt_clock()
    test_time_split()
    test_report()
    test_record_time_tables()
    test_records()
    print(f"\n{len(FAILED)} failed" if FAILED else "\nall passed")
    sys.exit(1 if FAILED else 0)
