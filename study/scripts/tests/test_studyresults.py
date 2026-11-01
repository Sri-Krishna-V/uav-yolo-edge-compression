"""Check studyresults.py, which gathers every scored cell into results/cells.csv (#44): on a small fake bench, and against
the results tables of the session records (verification/pi5_matrix_{1..7}_record.md and the two Galaxy M34 full-cell
records). Runs on any machine with the repo's bench files:

  python study/scripts/tests/test_studyresults.py
"""
import builtins
import csv
import io
import sys
import tempfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fakesession as fake  # noqa: E402
import studyresults as sr  # noqa: E402

FAILED = []


def check(label, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        FAILED.append(label)
        if detail:
            print(f"     {detail}")


def score(ap, ap_small, precision="FP32", head="raw"):
    stats = {"coco/AP_all": ap, "coco/AP_50": ap + 0.1, "coco/AP_75": ap - 0.01, "coco/AP_small": ap_small, "coco/AP_medium": ap + 0.05,
             "coco/AP_large": ap + 0.2, "coco/AR_500": ap + 0.15}
    return {"stats": stats, "runtime": "onnxruntime", "precision": precision, "head": head, "platform_tag": "pi5"}


def meta(e2e, precision="FP32", cv2="5.0.0", ort="1.31.0"):
    timing = {"t_pre_ms": {"median": 3.0}, "t_net_ms": {"median": e2e - 4.0}, "t_post_ms": {"median": 15.0},
              "post_deploy": {"median": 1.0}, "e2e_deploy": {"median": e2e, "p95": e2e + 10}}
    return {"precision": precision, "timing_ms": timing, "start": "2026-10-28T12:03:00+00:00",
            "versions": {"python": "3.11.2", "numpy": "2.4.6", "cv2": cv2, "onnxruntime": ort}}


def rows(n, mhz=2000):
    return [{"t_pre_ms": 3, "t_net_ms": 100, "t_post_ms": 15, "t_post_deploy_ms": 1, "temp_c": 70.0, "cpu_mhz": mhz} for _ in range(n)]


def fake_bench():
    """A bench with one Pi 5 matrix session: A1 FP32 (throttled, re-run; its two attempts score differently) and C1 INT8,
    plus a pilot full-cell check and a phone speed test that are not cells."""
    root = Path(tempfile.mkdtemp()) / "bench"
    a1 = {"name": "yolo11s-640-s0/640/onnxruntime-fp32", "runtime": "onnxruntime", "imgsz": 640, "attempts": [
        {"start": 180, "seconds": 700, "rows": rows(5, 1856), "throttle": "throttled", "score": score(0.10, 0.05), "meta": meta(404.4)},
        {"start": 1100, "seconds": 700, "rows": rows(5), "throttle": "throttled", "pass_": "rerun", "score": score(0.1875, 0.0904),
         "meta": meta(389.8)}]}
    c1 = {"name": "yolo11s-640-s0-prune30-distill/640/onnxruntime-int8", "runtime": "onnxruntime", "imgsz": 640, "attempts": [
        {"start": 900, "seconds": 400, "rows": rows(5), "precision": "INT8", "score": score(0.1420, 0.0607, "INT8"), "meta": meta(192.8, "INT8")}]}
    ref = {"name": "drift-start", "runtime": "onnxruntime", "imgsz": 640, "images": 200, "attempts": [{"start": 0, "seconds": 90, "rows": rows(2)}]}
    fake.make_session(root / "pi5", "pi5-matrix-9-fake", [ref, a1, c1])
    def one(name, precision, head="raw", scored=True, **cell):
        a = {"start": 0, "seconds": 300, "rows": rows(3), "precision": precision, "meta": meta(300.0, precision),
             "score": score(0.2, 0.1, precision, head) if scored else None}
        return {"name": name, "runtime": name.split("/")[2].split("-")[0], "imgsz": int(name.split("/")[1]), "attempts": [a], **cell}
    interrupted = one("yolo26s-960-s0/960/onnxruntime-int8", "INT8")
    interrupted["attempts"].insert(0, {"start": 0, "result": "interrupted", "rows": rows(2), "nul_tail": True})
    fake.make_session(root / "pi5", "pi5-matrix-8-fake", [one("yolo26s-640-s0/640/ncnn-fp32", "FP16"),
                                                          one("yolo26s-960-s0/800/onnxruntime-int8pc", "INT8"),
                                                          one("yolo26s-640-s0/640/openvino-fp32-e2e", "FP32", "end2end"),
                                                          one("yolo26s-640-s0/640/onnxruntime-fp32", "FP32", scored=False), interrupted,
                                                          one("yolo26s-640-s0-distill/640/openvino-fp32", "FP32", outcome="failed", counted=0)],
                      drift="re-run")
    fake.make_session(root / "pi5", "pi5-fullcell", [dict(a1, attempts=a1["attempts"][:1], counted=1)])
    phone = {"name": "yolo11s-640-s0/640/onnxruntime-fp32", "runtime": "onnxruntime", "imgsz": 640, "attempts": [
        {"start": 0, "seconds": 1100, "rows": rows(4, 1920), "throttle": "unknown", "score": score(0.1875, 0.0904), "meta": meta(640.0, ort="1.31.1")}]}
    fake.make_session(root / "m34", "m34-matrix-1", [phone], platform="m34")
    fake.make_session(root / "m34", "m34-speedtest", [dict(a1, attempts=a1["attempts"][:1], counted=1, images=200)], platform="m34")
    return root


def test_attempt_rule():
    got = [r for r in sr.build_rows(fake_bench()) if r["session"] == "pi5-matrix-9-fake"]
    check("only the matrix session's two cells are rows (no reference run, pilot or speed test)",
          sorted(r["cell"] for r in got) == ["yolo11s-640-s0-prune30-distill/640/onnxruntime-int8", "yolo11s-640-s0/640/onnxruntime-fp32"],
          f"{[(r['session'], r['cell']) for r in got]}")
    a1 = next(r for r in got if r["cell"].endswith("onnxruntime-fp32"))
    check("the re-run cell takes run_dir's attempt (2): its AP, AP-small and latency", (a1["attempt"], a1["ap"], a1["ap_small"], a1["latency_ms"])
          == (2, 0.1875, 0.0904, 389.8), f"{a1}")


def test_configuration():
    """GLOSSARY.md's five configurations, from the run's suffix and the precision the runtime really computed in."""
    cases = [("yolo11s-640-s0", "FP32", "baseline"), ("yolo11s-640-s0", "FP16", "baseline"), ("yolo26s-960-s0", "INT8", "quantized"),
             ("yolo11s-960-s0-prune30", "FP32", "pruned"), ("yolo26s-640-s0-distill", "FP16", "distilled"),
             ("yolo26s-640-s0-prune30-distill", "INT8", "combined")]
    for run, precision, want in cases:
        check(f"configuration: {run} at {precision} is {want}", sr.configuration(run, precision) == want, f"got {sr.configuration(run, precision)}")


def test_identity():
    got = {r["cell"]: r for r in sr.build_rows(fake_bench()) if r["session"] == "pi5-matrix-8-fake"}
    ncnn, pc, nf = (got[k] for k in ["yolo26s-640-s0/640/ncnn-fp32", "yolo26s-960-s0/800/onnxruntime-int8pc", "yolo26s-640-s0/640/openvino-fp32-e2e"])
    want = {"platform": "pi5", "run": "yolo26s-640-s0", "checkpoint": "A2", "model": "YOLO26s", "trained_imgsz": 640, "imgsz": 640,
            "untrained_imgsz": 0, "runtime": "ncnn", "variant": "ncnn-fp32", "precision": "FP16", "configuration": "baseline", "head": "one-to-many",
            "int8_recipe": ""}
    check("an NCNN cell: its precision is score.json's FP16, not the folder's fp32", {k: ncnn[k] for k in want} == want, f"{ncnn}")
    want = {"checkpoint": "B2", "trained_imgsz": 960, "imgsz": 800, "untrained_imgsz": 1, "precision": "INT8", "configuration": "quantized",
            "int8_recipe": "per-channel", "head": "one-to-many"}
    check("a per-channel INT8 cell at 800 from the 960 model: an untrained input resolution", {k: pc[k] for k in want} == want, f"{pc}")
    check("an end2end cell has the NMS-free head", (nf["head"], nf["variant"]) == ("NMS-free", "openvino-fp32-e2e"), f"{nf}")
    stock = next(r for r in sr.build_rows(fake_bench()) if r["variant"] == "onnxruntime-int8")
    check("a stock INT8 cell's recipe is stock", stock["int8_recipe"] == "stock", f"{stock}")



def test_measures():
    """Accuracy from score.json, latency medians from meta.json and the clock from timings.csv, all of run_dir's attempt;
    the phone's throttle columns empty, and every cell's software versions from meta.json."""
    got = sr.build_rows(fake_bench())
    a1 = next(r for r in got if r["session"] == "pi5-matrix-9-fake" and r["variant"] == "onnxruntime-fp32")
    want = {"ap": 0.1875, "ap50": 0.2875, "ap75": 0.1775, "ap_small": 0.0904, "ap_medium": 0.2375, "ap_large": 0.3875, "ar500": 0.3375,
            "latency_ms": 389.8, "latency_p95_ms": 399.8, "pre_ms": 3.0, "net_ms": 385.8, "post_deploy_ms": 1.0, "post_eval_ms": 15.0}
    bad = {k: (v, a1[k]) for k, v in want.items() if a1[k] is None or abs(a1[k] - v) > 1e-9}
    check("A1's accuracy and latency columns, from its attempt 2", not bad, f"want, got: {bad}")
    want = {"throttled": "throttled", "throttle_reads": 0, "soft_limit_reads": 0, "images": 5, "below_2000_mhz": 0, "clock_min_mhz": 2000,
            "clock_median_mhz": 2000, "peak_temp_c": 70.0}
    check("A1's throttle columns: the flag as recorded, and attempt 2's clock (attempt 1 ran at 1856 MHz)",
          {k: a1[k] for k in want} == want, f"{ {k: a1[k] for k in want} }")
    want = {"python_version": "3.11.2", "numpy_version": "2.4.6", "opencv_version": "5.0.0", "runtime_version": "1.31.0"}
    check("A1's software versions", {k: a1[k] for k in want} == want, f"{ {k: a1[k] for k in want} }")
    m34 = next(r for r in got if r["platform"] == "m34")
    empty = [k for k in sr.THROTTLE_COLUMNS if m34[k] not in (None, "")]
    check("the phone's throttle columns are empty: it has no clock lock and no throttle signal", not empty, f"{ {k: m34[k] for k in empty} }")
    check("the phone cell carries its own ONNX Runtime version (1.31.1)", (m34["session"], m34["runtime_version"]) == ("m34-matrix-1", "1.31.1"))



def test_notes():
    """No cell is skipped silently: an unscored or failed cell is a row with empty accuracy and a note saying why."""
    got = {r["cell"]: r for r in sr.build_rows(fake_bench()) if r["session"] == "pi5-matrix-8-fake"}
    unscored, failed = got["yolo26s-640-s0/640/onnxruntime-fp32"], got["yolo26s-640-s0-distill/640/openvino-fp32"]
    check("an unscored cell: a row, no AP, its latency, and a note", unscored["ap"] is None and unscored["latency_ms"] == 300.0
          and "not scored" in unscored["note"], f"{unscored}")
    check("a failed cell: a row with no attempt, no AP, no latency, and a note", failed["attempt"] is None and failed["ap"] is None
          and failed["latency_ms"] is None and "failed" in failed["note"], f"{failed}")
    i = got["yolo26s-960-s0/960/onnxruntime-int8"]
    check("a cell whose first attempt was interrupted: attempt 2 counts, with a note", i["attempt"] == 2 and "interrupted" in i["note"], f"{i}")
    check("a flagged drift check, with no decision recorded for the session, is in every row's note",
          all("drift check: re-run" in r["note"] for r in got.values()), f"{[r['note'] for r in got.values()]}")



def test_contract():
    """cells.csv has COLUMNS in order, empty where a value is missing, and the docstring documents every column."""
    out = Path(tempfile.mkdtemp()) / "results" / "cells.csv"
    sr.write_cells(sr.build_rows(fake_bench()), out)
    with open(out, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    check("cells.csv's header is COLUMNS", r.fieldnames == sr.COLUMNS, f"{r.fieldnames}")
    failed = next(x for x in rows if x["outcome"] == "failed")
    check("a missing value is an empty field, not None", failed["ap"] == "" and failed["attempt"] == "", f"{failed}")
    words = set(sr.__doc__.replace(",", " ").split())
    check("the docstring documents every column", all(c in words for c in sr.COLUMNS), f"{[c for c in sr.COLUMNS if c not in words]}")


STUDY = Path(__file__).resolve().parents[2]
PI5 = {1: "pi5-matrix-1-yolo11s", 2: "pi5-matrix-2-yolo26s", 3: "pi5-matrix-3-yolo26s-compressed", 4: "pi5-matrix-4-1280-baselines",
       5: "pi5-matrix-5-800", 6: "pi5-matrix-6-1120", 7: "pi5-matrix-7-int8pc"}
# The two Galaxy M34 full cells, from m34_fullcell_record.md and m34_fullcell_int8_record.md: AP, AP50, AP-small, latency.
M34 = {"m34-fullcell": (0.1875, 0.3265, 0.0904, 636.6), "m34-fullcell-int8": (0.1682, 0.3056, 0.0769, 304.8)}


def _r(x, places):
    return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP))


def record_results(path):
    """The record's results table: one dict per row, the columns by their headers."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("| Cell | Precision recorded |"))
    head = [c.strip() for c in lines[start].strip("|").split("|")]
    out = []
    for ln in lines[start + 2:]:
        if not ln.startswith("|"):
            break
        out.append(dict(zip(head, [c.strip() for c in ln.strip("|").split("|")])))
    return out


def test_records():
    """Every cell of sessions 1-7 and the two M34 full cells equal the session records' results tables."""
    cells = sr.build_rows()
    for k, name in PI5.items():
        mine = [r for r in cells if r["session"] == name]
        table, bad = record_results(STUDY / f"verification/pi5_matrix_{k}_record.md"), []
        for t in table:
            ms = float(t["Median latency, ms"].split()[0])
            hit = [r for r in mine if r["latency_ms"] is not None and abs(r["latency_ms"] - ms) < 0.0501]
            if len(hit) != 1:
                bad.append(f"{t['Cell']}: {len(hit)} cells at {ms} ms")
                continue
            r = hit[0]
            # the records round a latency tie either way (session 2's 393.15 up, session 3's 232.95 down): the match is within 0.05
            got = {"Precision recorded": r["precision"], "AP": _r(r["ap"], 4), "AP50": _r(r["ap50"], 4), "AP-small": _r(r["ap_small"], 4),
                   "AR@500": _r(r["ar500"], 4), "Median latency, ms": ms}
            want = {c: (t[c] if c == "Precision recorded" else float(t[c].split()[0])) for c in got}
            if got != want:
                bad.append(f"{t['Cell']}: record {want} vs cells.csv {got}")
        check(f"session {k}: {len(table)} cells, each equal to its row of the record (AP, AP50, AP-small, AR@500, latency, precision)",
              not bad and len(mine) == len(table), f"{len(mine)} cells; " + "; ".join(bad))
    for name, want in M34.items():
        r = [x for x in cells if x["session"] == name]
        got = tuple(_r(v, p) for v, p in zip([r[0]["ap"], r[0]["ap50"], r[0]["ap_small"], r[0]["latency_ms"]], [4, 4, 4, 1])) if len(r) == 1 else r
        check(f"{name}: one cell, equal to its record {want}", got == want, f"got {got}")
    s7 = [r for r in cells if r["session"] == PI5[7]]
    check("session 7: 12 per-channel INT8 rows, picked up by SESSIONS with no code change",
          len(s7) == 12 and all(r["int8_recipe"] == "per-channel" and r["precision"] == "INT8" for r in s7))
    check("session 7's rows note the drift decision, and none says 'drift check'",
          all("kept (#37)" in r["note"] and "drift check" not in r["note"] for r in s7), f"{[r['note'] for r in s7]}")
    s6 = [r for r in cells if r["session"] == PI5[6]]
    check("session 6's rows note their latency, about 6 % high", s6 and all("about 6 % high" in r["note"] for r in s6))
    pi5 = [r for r in cells if r["platform"] == "pi5"]
    check(f"the Pi 5 matrix: 74 cells ({len(pi5)}), every one scored", len(pi5) == 74 and all(r["ap"] is not None for r in pi5))


def test_light():
    """No torch or ultralytics, and no detections read: build_rows over the whole bench opens no predictions file."""
    opened, real = [], builtins.open

    def spy(file, *a, **kw):
        opened.append(str(file))
        return real(file, *a, **kw)

    builtins.open, io.open = spy, spy
    try:
        n = len(sr.build_rows())
    finally:
        builtins.open, io.open = real, real
    check(f"build_rows over the bench ({n} cells, {len(opened)} files opened) reads no predictions file",
          n > 0 and opened and not [f for f in opened if "predictions" in Path(f).name], f"{[f for f in opened if 'predictions' in f][:3]}")
    check("neither torch nor ultralytics is imported", "torch" not in sys.modules and "ultralytics" not in sys.modules)


if __name__ == "__main__":
    test_attempt_rule()
    test_configuration()
    test_identity()
    test_measures()
    test_notes()
    test_contract()
    test_records()
    test_light()
    print(f"\n{len(FAILED)} failed" if FAILED else "\nall passed")
    sys.exit(1 if FAILED else 0)
