"""Check studypairs.py (#46, result A: the equal-latency comparisons) at its three seams: the pair rule on literal cells,
the real results/cells.csv against the session records and #46's check list, and the figure's data.

  python study/scripts/tests/test_studypairs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import studypairs as sp  # noqa: E402

FAILED = []


def check(label, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        FAILED.append(label)
        if detail:
            print(f"     {detail}")


def cell(checkpoint, imgsz, configuration, latency, ap_small, model="YOLO11s", runtime="onnxruntime", trained=None, recipe="",
         head="one-to-many", session="pi5-matrix-1", platform="pi5", ap=0.2):
    """One cells.csv row as load_cells gives it, with only what the pair rule reads, plus enough to tell the cells apart."""
    precision = "INT8" if configuration in ("quantized", "combined") else "FP32"
    trained = imgsz if trained is None else trained
    return {"platform": platform, "session": session, "checkpoint": checkpoint, "model": model, "trained_imgsz": trained, "imgsz": imgsz,
            "untrained_imgsz": int(trained != imgsz), "configuration": configuration, "runtime": runtime, "precision": precision,
            "variant": f"{runtime}-{'int8pc' if recipe == 'per-channel' else 'int8' if precision == 'INT8' else 'fp32'}", "head": head,
            "int8_recipe": recipe, "ap": ap, "ap_small": ap_small, "ap_medium": ap + 0.05, "ap_large": ap + 0.2, "latency_ms": latency,
            "throttled": "none", "below_2000_mhz": 0, "clock_median_mhz": 2000, "start_utc": "2026-10-28T12:00:00+00:00",
            "cell": f"{checkpoint}/{imgsz}/{runtime}-{precision}{recipe}{head}", "note": ""}


def names(pairs):
    return sorted((p["compressed"]["checkpoint"], p["compressed"]["imgsz"], p["compressed"]["configuration"], p["baseline"]["checkpoint"],
                   p["baseline"]["imgsz"]) for p in pairs)


# ---------------------------------------------------------------- 1: the pair rule, on literal cells

def test_pair_rule():
    """A compressed configuration at a larger input resolution against a baseline at a smaller one: same platform, model, runtime
    and head, median latencies 10 % apart or less, divided by the smaller of the two. Distilled cells are never the compressed side."""
    cells = [cell("A1", 640, "baseline", 100.0, 0.09),
             cell("B1", 800, "baseline", 150.0, 0.12, trained=960),
             cell("B1", 800, "quantized", 110.0, 0.10, trained=960, recipe="stock"),  # 10.0 % above A1: a pair (10 % or less)
             cell("B1", 960, "quantized", 165.0, 0.13, recipe="stock"),  # 10.0 % above B1 at 800: a pair
             cell("P2", 960, "pruned", 140.0, 0.11),  # 150 / 140: 7.1 % apart, a pair
             cell("P9", 960, "pruned", 136.0, 0.11),  # 150 / 136: 10.3 % over the smaller (9.3 % over the larger): no pair
             cell("D9", 960, "distilled", 152.0, 0.13),  # distilled: never the compressed side
             cell("A1", 640, "quantized", 100.0, 0.08, recipe="stock"),  # the same input resolution as A1's baseline: no pair
             cell("A1", 640, "baseline", 105.0, 0.09, runtime="openvino"),  # another runtime: no pair with B1 INT8 at 800
             cell("A2", 640, "baseline", 105.0, 0.10, model="YOLO26s"),  # another model: no pair with B1 INT8 at 800
             cell("A2", 640, "baseline", 105.0, 0.10, model="YOLO26s", head="NMS-free"),
             cell("B2", 800, "quantized", 112.0, 0.10, model="YOLO26s", trained=960, recipe="stock"),  # pairs A2 one-to-many only
             cell("A2", 640, "baseline", 109.0, 0.10, model="YOLO26s"),
             cell("A1", 640, "baseline", 104.0, 0.09, platform="m34", session="m34-matrix-1")]  # another platform: no pair with B1 INT8 at 800
    want = [("B1", 800, "quantized", "A1", 640), ("B1", 960, "quantized", "B1", 800), ("B2", 800, "quantized", "A2", 640),
            ("B2", 800, "quantized", "A2", 640), ("P2", 960, "pruned", "B1", 800)]
    got = sp.equal_latency_pairs(cells)
    check("1 the pairs: compressed at the larger input resolution, same platform, model, runtime and head, within 10 % of the smaller",
          names(got) == sorted(want), f"{names(got)}")
    p = next(p for p in got if p["compressed"]["checkpoint"] == "P2")
    check("1 apart is the larger latency over the smaller (150 / 140), the ratio compressed over baseline (140 / 150)",
          (round(p["apart_pct"], 4), round(p["latency_ratio"], 4)) == (7.1429, 0.9333), f"{p['apart_pct']}, {p['latency_ratio']}")
    check("1 differences are compressed minus baseline: AP-small 0.11 - 0.12, AP, AP-medium, AP-large",
          (round(p["ap_small_diff"], 6), round(p["ap_diff"], 6), round(p["ap_medium_diff"], 6), round(p["ap_large_diff"], 6))
          == (-0.01, 0.0, 0.0, 0.0), f"{p}")


def test_int8_headline():
    """The per-channel cells are the matrix's ONNX Runtime INT8 cells (INVARIANTS.md section 8): once one is scored, its pair is
    the headline and the stock cell's pair stays as its own row; until then the stock cell's pair is the headline, with a note."""
    base = [cell("A1", 640, "baseline", 100.0, 0.09), cell("P1", 640, "pruned", 80.0, 0.07), cell("P2", 960, "pruned", 105.0, 0.10)]
    stock = cell("B1", 800, "quantized", 104.0, 0.100, trained=960, recipe="stock", session="pi5-matrix-5")
    pc = cell("B1", 800, "quantized", 103.5, 0.105, trained=960, recipe="per-channel", session="pi5-matrix-7")
    before = {p["compressed"]["checkpoint"] + p["compressed"]["int8_recipe"]: p for p in sp.equal_latency_pairs(base + [stock])}
    check("2 before the per-channel cell: the stock INT8 pair is the headline, with a note saying so",
          before["B1stock"]["headline"] and "no per-channel cell" in before["B1stock"]["note"], f"{before['B1stock']}")
    after = {p["compressed"]["checkpoint"] + p["compressed"]["int8_recipe"]: p for p in sp.equal_latency_pairs(base + [stock, pc])}
    check("2 after it: the per-channel pair is the headline, and the stock pair is still a row, not the headline, with a note",
          sorted(after) == ["B1per-channel", "B1stock", "P2"] and after["B1per-channel"]["headline"] and not after["B1stock"]["headline"]
          and "per-channel cell is the headline" in after["B1stock"]["note"], f"{ {k: (p['headline'], p['note']) for k, p in after.items()} }")
    check("2 a pair with no INT8 cell is the headline, with no INT8 note", after["P2"]["headline"] and "INT8" not in after["P2"]["note"],
          f"{after['P2']}")


# ---------------------------------------------------------------- 3: the real cells.csv against the records and #46's check list

STUDY = Path(__file__).resolve().parents[2]


def record_pairs():
    """pi5_matrix_7_record.md, 'What it changes for the equal-latency pairs': compressed cell, its ms, baseline, its ms, whether the
    baseline was flagged throttled (t), apart (%), and the AP-small difference, per-channel and stock."""
    lines = (STUDY / "verification" / "pi5_matrix_7_record.md").read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("| Compressed cell | ms | Baseline |"))
    out = []
    for ln in lines[start + 2:]:
        if not ln.startswith("|"):
            break
        c, cms, b, bms, apart, diff = [x.strip().strip("*").replace("−", "-") for x in ln.strip("|").split("|")]
        pc, stock = diff.rstrip(")").split(" (")
        out.append((c, float(cms), b, float(bms.split()[0]), bms.endswith(" t"), float(apart.rstrip(" %")), float(pc), float(stock)))
    return out


# #46's check list: the ONNX Runtime pairs from sessions 1-6 (stock INT8), compressed cell and ms against baseline and ms.
CHECK_LIST = [("B1 INT8 at 800", 416.7, "A1 FP32 at 640", 389.8), ("B1 INT8 at 960", 614.0, "B1 FP32 at 800", 629.9),
              ("P2 FP32 at 960", 574.6, "B1 FP32 at 800", 629.9), ("B3 INT8 at 1120", 883.5, "B1 FP32 at 960", 909.7),
              ("B2 INT8 at 960", 644.4, "B2 FP32 at 800", 611.9), ("P4 FP32 at 960", 571.7, "B2 FP32 at 800", 611.9),
              ("B4 INT8 at 1120", 943.1, "B2 FP32 at 960", 926.5)]


def ms(c):
    return round(c["latency_ms"], 1)


def test_real_pairs():
    cells, _ = sp.usable(sp.load_cells(STUDY / "results" / "cells.csv"))
    pi5 = [c for c in cells if c["platform"] == "pi5"]
    before = [c for c in pi5 if c["session"] != "pi5-matrix-7-int8pc"]
    ort = sorted((sp.label(p["compressed"]), ms(p["compressed"]), sp.label(p["baseline"]), ms(p["baseline"]), p["headline"])
                 for p in sp.equal_latency_pairs(before) if p["compressed"]["runtime"] == "onnxruntime")
    check("3 sessions 1-6 (no session 7): the ONNX Runtime pairs are #46's check list, each the headline",
          ort == sorted(x + (True,) for x in CHECK_LIST), f"{ort}")
    pairs = sp.equal_latency_pairs(pi5)
    find = lambda c, b, recipe: [p for p in pairs if (sp.label(p["compressed"]), sp.label(p["baseline"]), p["compressed"]["int8_recipe"])  # noqa: E731
                                 == (c, b, recipe)]
    for c, cms, b, bms, t, apart, pc, stock in record_pairs():
        hp, st = find(c, b, "per-channel"), find(c, b, "stock")
        got = (len(hp), len(st)) == (1, 1) and (ms(hp[0]["compressed"]), ms(hp[0]["baseline"]), hp[0]["baseline"]["throttled"] == "throttled",
                                                 round(hp[0]["apart_pct"], 1), round(hp[0]["ap_small_diff"], 4), round(st[0]["ap_small_diff"], 4),
                                                 hp[0]["headline"], st[0]["headline"])
        check(f"3 session 7 record: {c} ({cms}) against {b} ({bms}{' t' if t else ''}), {apart} % apart, AP-small {pc:+.4f} per-channel "
              f"({stock:+.4f} stock); the per-channel pair is the headline", got == (cms, bms, t, apart, pc, stock, True, False), f"got {got}")
    ort = [p for p in pairs if p["compressed"]["runtime"] == "onnxruntime"]
    check("3 with session 7: 7 ONNX Runtime headline pairs (5 per-channel INT8, P2, P4) and the 5 stock pairs as their own rows",
          (sum(p["headline"] for p in ort), sum(not p["headline"] for p in ort)) == (7, 5)
          and all(p["compressed"]["int8_recipe"] == "stock" for p in ort if not p["headline"]), f"{[(sp.label(p['compressed']), p['headline']) for p in ort]}")
    check("3 B2 INT8 at 800 (430.3 ms) is 12.3 % above A2 FP32 at 640 (383.1): no pair",
          not find("B2 INT8 at 800", "A2 FP32 at 640", "per-channel") and round(100 * sp.apart(
              next(c for c in pi5 if sp.label(c) == "B2 INT8 at 800" and c["int8_recipe"] == "per-channel"),
              next(c for c in pi5 if sp.label(c) == "A2 FP32 at 640" and c["runtime"] == "onnxruntime" and c["head"] == "one-to-many")), 1) == 12.3)
    spans = {(sp.label(p["compressed"]), p["compressed"]["int8_recipe"]): p["spans_session_6"] for p in ort if p["compressed"]["imgsz"] == 1120}
    check("3 spans session 6: the stock INT8 cells at 1120 (session 6) against session 1 and 2's baselines; not the per-channel ones",
          spans == {("B3 INT8 at 1120", "stock"): True, ("B4 INT8 at 1120", "stock"): True, ("B3 INT8 at 1120", "per-channel"): False,
                    ("B4 INT8 at 1120", "per-channel"): False}, f"{spans}")
    every = sp.all_pairs(pi5)
    b3 = [p for p in every if {(sp.label(p["a"]), p["a"]["runtime"]), (sp.label(p["b"]), p["b"]["runtime"])}
          == {("B3 INT8 at 1280", "onnxruntime"), ("B3 FP32 at 1120", "onnxruntime")} and p["a"]["int8_recipe"] == "per-channel"]
    check("3 all pairs: B3 INT8 at 1280 / B3 FP32 at 1120 (session 6) is a row, 14.1 % apart as measured, about 7-8 % without session 6's "
          "offset, not within 10 %, not a paper pair, and its note says so",
          len(b3) == 1 and round(b3[0]["apart_pct"], 1) == 14.1 and 7 <= b3[0]["apart_pct_without_offset"] <= 8 and not b3[0]["within_limit"]
          and not b3[0]["paper_pair"] and "session 6" in b3[0]["note"], f"{b3}")
    inside = [p for p in every if p["a"]["session"] == p["b"]["session"] == "pi5-matrix-6-1120"]
    check("3 all pairs: a pair inside session 6 does not span it", inside and not any(p["spans_session_6"] for p in inside))
    cross = [p for p in every if p["a"]["runtime"] != p["b"]["runtime"] and p["within_limit"]]
    paper = sum(p["paper_pair"] for p in every)
    check(f"3 all pairs: every pair within 10 % across runtimes too ({len(cross)} cross-runtime), and each paper pair once ({paper})",
          cross and paper == len(pairs) and all(p["within_limit"] or p["spans_session_6"] for p in every))


# ---------------------------------------------------------------- 4: the figure's data

def test_figure_data():
    """AP-small against median latency: one panel per runtime, both study models in every panel, a line through each model's
    baselines from 640 to 1280, the headline cells as points (no stock INT8 cell once its per-channel cell exists, no NMS-free
    head), and each headline pair as a connector. The phone gets a figure once it has three or more cells."""
    cells, _ = sp.usable(sp.load_cells(STUDY / "results" / "cells.csv"))
    pairs = sp.equal_latency_pairs(cells)
    panels = sp.figure_data(cells, pairs, "pi5")
    check("4 the Pi 5: one panel per runtime, ONNX Runtime, OpenVINO, NCNN", [p["runtime"] for p in panels] == ["onnxruntime", "openvino", "ncnn"],
          f"{[p['runtime'] for p in panels]}")
    lines = {(p["runtime"], m): [c["imgsz"] for c in s["baselines"]] for p in panels for m, s in p["models"].items()}
    check("4 both study models in every panel, each with its baselines at 640, 800, 960, 1120 and 1280 in that order",
          len(lines) == 6 and all(v == [640, 800, 960, 1120, 1280] for v in lines.values()), f"{lines}")
    points = [c for p in panels for s in p["models"].values() for c in s["points"]]
    check("4 the points: no stock INT8 cell (each has its per-channel cell) and no NMS-free head",
          points and not [c for c in points if c["int8_recipe"] == "stock" or c["head"] == "NMS-free"],
          f"{[sp.label(c) for c in points if c['int8_recipe'] == 'stock' or c['head'] == 'NMS-free']}")
    configs = {c["configuration"] for c in panels[0]["models"]["YOLO26s"]["points"]}
    check("4 the ONNX Runtime panel has all five configurations", configs == {"baseline", "quantized", "pruned", "distilled", "combined"}, f"{configs}")
    conn = {p["runtime"]: sorted((sp.label(x["compressed"]), sp.label(x["baseline"])) for x in p["connectors"]) for p in panels}
    want = {"onnxruntime": sorted([("B1 INT8 at 800", "A1 FP32 at 640"), ("B1 INT8 at 960", "B1 FP32 at 800"), ("P2 FP32 at 960", "B1 FP32 at 800"),
                                   ("B3 INT8 at 1120", "B1 FP32 at 960"), ("B2 INT8 at 960", "B2 FP32 at 800"), ("P4 FP32 at 960", "B2 FP32 at 800"),
                                   ("B4 INT8 at 1120", "B2 FP32 at 960")]),
            "openvino": [("P2 FP32 at 960", "B1 FP32 at 800"), ("P4 FP32 at 960", "B2 FP32 at 800")],
            "ncnn": [("P2 FP16 at 960", "B1 FP16 at 800"), ("P4 FP16 at 960", "B2 FP16 at 800")]}
    check("4 a connector for each headline pair, in its runtime's panel", conn == want, f"{conn}")
    check("4 the phone's matrix sessions 1 and 2 give it a figure", sp.figure_platforms(cells) == ["pi5", "m34"], f"{sp.figure_platforms(cells)}")
    phone = [cell("A1", 640, "baseline", 640.0, 0.09, platform="m34", session="m34-fullcell"),
             cell("A1", 640, "quantized", 300.0, 0.08, platform="m34", session="m34-fullcell-int8", recipe="stock")]
    again = dict(phone[0], session="m34-matrix-1", start_utc="2026-11-01T06:00:00+00:00")
    check("4 the same phone cell run again is one cell: still 2, no figure", sp.figure_platforms(sp.usable(phone + [again])[0]) == [])
    three = phone + [cell("A2", 640, "baseline", 650.0, 0.10, model="YOLO26s", platform="m34", session="m34-matrix-1")]
    check("4 three phone cells: the phone gets the figure", sp.figure_platforms(sp.usable(three)[0]) == ["m34"])


if __name__ == "__main__":
    test_pair_rule()
    test_int8_headline()
    test_real_pairs()
    test_figure_data()
    print(f"\n{len(FAILED)} failed" if FAILED else "\nall passed")
    sys.exit(1 if FAILED else 0)
