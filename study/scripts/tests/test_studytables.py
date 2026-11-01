"""Check studytables.py (#47: the accuracy and latency tables, results C and D, the small-object section) at its three seams:
the rules on literal cells, the real results/cells.csv against the session records, and the figure's data.

  python study/scripts/tests/test_studytables.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import studytables as st  # noqa: E402

FAILED = []


def check(label, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        FAILED.append(label)
        if detail:
            print(f"     {detail}")


def cell(checkpoint, imgsz, configuration, latency=100.0, ap=0.2, ap_small=0.1, model="YOLO11s", runtime="onnxruntime", trained=None,
         recipe="", head="one-to-many", session="pi5-matrix-1", platform="pi5", below=0, clock=2000, throttled="none"):
    """One cells.csv row as studypairs.load_cells gives it, with what the tables read."""
    precision = "INT8" if configuration in ("quantized", "combined") else "FP16" if runtime == "ncnn" else "FP32"
    trained = imgsz if trained is None else trained
    return {"platform": platform, "session": session, "checkpoint": checkpoint, "model": model, "trained_imgsz": trained, "imgsz": imgsz,
            "untrained_imgsz": int(trained != imgsz), "configuration": configuration, "runtime": runtime, "precision": precision, "head": head,
            "int8_recipe": recipe, "ap": ap, "ap50": ap + 0.15, "ap_small": ap_small, "ap_medium": ap + 0.1, "ap_large": ap + 0.2,
            "latency_ms": latency, "pre_ms": 3.0, "net_ms": latency - 4.0, "post_deploy_ms": 1.0, "post_eval_ms": 15.0, "throttled": throttled,
            "below_2000_mhz": None if platform != "pi5" else below, "clock_median_mhz": None if platform != "pi5" else clock,
            "start_utc": "2026-10-28T12:00:00+00:00", "cell": f"{checkpoint}/{imgsz}/{runtime}-{precision}{recipe}{head}", "note": ""}


def picked(rows):
    return [(r["cell"]["model"], r["cell"]["configuration"], r["cell"]["checkpoint"], r["cell"]["runtime"], r["cell"]["precision"],
             r["cell"]["int8_recipe"], r["headline"]) for r in rows]


# ---------------------------------------------------------------- 1: the rules, on literal cells

def test_accuracy_rows():
    """Configuration x study model at the matrix resolution, 640, on the Pi 5. Baseline, pruned and distilled from the ONNX Runtime
    FP32 cells; quantized and combined from ONNX Runtime INT8, per-channel the headline (INVARIANTS.md section 8), stock beside it."""
    cells = [cell("A1", 640, "baseline", ap=0.1875),
             cell("A1", 640, "baseline", runtime="openvino"), cell("A1", 640, "baseline", runtime="ncnn"),  # not ONNX Runtime
             cell("A1", 640, "quantized", recipe="stock", ap=0.1682), cell("A1", 640, "quantized", recipe="per-channel", ap=0.1696),
             cell("P1", 640, "pruned"), cell("P1", 640, "pruned", runtime="ncnn"), cell("D1", 640, "distilled"),
             cell("C1", 640, "combined", recipe="stock"),  # no per-channel C1: the stock cell is the headline
             cell("B1", 960, "baseline"), cell("B1", 960, "quantized", recipe="per-channel"),  # not at 640
             cell("A2", 640, "baseline", model="YOLO26s"), cell("A2", 640, "baseline", model="YOLO26s", head="NMS-free"),  # result B's head
             cell("A1", 640, "baseline", platform="m34", session="m34-fullcell")]  # not the Pi 5
    rows = st.accuracy_rows(cells)
    want = [("YOLO11s", "baseline", "A1", "onnxruntime", "FP32", "", True),
            ("YOLO11s", "quantized", "A1", "onnxruntime", "INT8", "per-channel", True),
            ("YOLO11s", "quantized", "A1", "onnxruntime", "INT8", "stock", False),
            ("YOLO11s", "pruned", "P1", "onnxruntime", "FP32", "", True),
            ("YOLO11s", "distilled", "D1", "onnxruntime", "FP32", "", True),
            ("YOLO11s", "combined", "C1", "onnxruntime", "INT8", "stock", True),
            ("YOLO26s", "baseline", "A2", "onnxruntime", "FP32", "", True)]
    check("1 accuracy: ONNX Runtime FP32 for baseline, pruned and distilled, INT8 per-channel then stock, one-to-many head, 640, the Pi 5",
          picked(rows) == want, f"{picked(rows)}")
    notes = {r["cell"]["checkpoint"] + r["cell"]["int8_recipe"]: r["note"] for r in rows}
    check("1 accuracy: a stock row whose per-channel cell exists says so; one without says none is scored yet",
          "per-channel cell is the headline" in notes["A1stock"] and "no per-channel cell" in notes["C1stock"], f"{notes}")


def test_int8_rows():
    """Result D: for each checkpoint and input resolution, INT8's speed-up over the same runtime's FP32 (FP32 latency over INT8's)
    and the AP and AP-small it changes (INT8 minus FP32, and that over FP32), stock and per-channel side by side, on every platform."""
    cells = [cell("A1", 640, "baseline", latency=400.0, ap=0.2, ap_small=0.1),
             cell("A1", 640, "baseline", runtime="openvino", latency=300.0),  # another runtime: not the FP32 side
             cell("A1", 640, "quantized", latency=250.0, ap=0.18, ap_small=0.085, recipe="stock"),
             cell("A1", 640, "quantized", latency=200.0, ap=0.19, ap_small=0.09, recipe="per-channel", session="pi5-matrix-7-int8pc"),
             cell("B3", 1120, "baseline", latency=1300.0, trained=1280, session="pi5-matrix-6-1120"),
             cell("B3", 1120, "quantized", latency=880.0, trained=1280, recipe="stock", session="pi5-matrix-6-1120"),
             cell("B3", 1120, "quantized", latency=832.0, trained=1280, recipe="per-channel", session="pi5-matrix-7-int8pc"),
             cell("C1", 640, "combined", latency=193.0, recipe="stock"),  # no FP32 export of the combined checkpoint
             cell("A1", 640, "baseline", latency=636.56, platform="m34", session="m34-fullcell"),
             cell("A1", 640, "quantized", latency=304.779, recipe="stock", platform="m34", session="m34-fullcell-int8")]
    rows = st.int8_rows(cells)
    got = [(r["platform"], r["checkpoint"], r["imgsz"]) for r in rows]
    check("1 result D: one row per platform, checkpoint, input resolution and head with an INT8 cell, the Pi 5 first",
          got == [("pi5", "A1", 640), ("pi5", "C1", 640), ("pi5", "B3", 1120), ("m34", "A1", 640)], f"{got}")
    a1 = rows[0]
    check("1 result D: speed-up is FP32 over INT8 (400 / 200 per-channel, 400 / 250 stock), on the same runtime's FP32 cell",
          (a1["per_channel_speedup"], a1["stock_speedup"]) == (2.0, 1.6) and a1["fp32"]["runtime"] == "onnxruntime",
          f"{a1['per_channel_speedup']}, {a1['stock_speedup']}")
    got = tuple(round(a1[k], 6) for k in ("per_channel_ap_diff", "per_channel_ap_diff_pct", "per_channel_ap_small_diff",
                                          "per_channel_ap_small_diff_pct", "stock_ap_diff_pct", "stock_ap_small_diff_pct"))
    check("1 result D: the change is INT8 minus FP32, and that over FP32 in %: AP -0.01 (-5 %), AP-small -0.01 (-10 %); stock -10 %, -15 %",
          got == (-0.01, -5.0, -0.01, -10.0, -10.0, -15.0), f"{got}")
    b3 = rows[2]
    check("1 result D: a speed-up across session 6 is flagged (per-channel, session 7, against session 6's FP32); within it, not",
          b3["per_channel_spans_session_6"] and not b3["stock_spans_session_6"] and round(b3["stock_speedup"], 4) == 1.4773, f"{b3}")
    c1 = rows[1]
    check("1 result D: a checkpoint with no FP32 cell keeps its row, with no speed-up and a note",
          c1["fp32"] is None and c1["stock_speedup"] is None and "no FP32" in c1["note"], f"{c1}")
    check("1 result D: the phone, A1 at 640: 636.56 / 304.779 = 2.09x (m34_fullcell_int8_record.md)",
          round(rows[3]["stock_speedup"], 2) == 2.09 and rows[3]["per_channel"] is None, f"{rows[3]}")
    got = [(r["checkpoint"], r["imgsz"], r["speedup_recipe"], r["speedup"] and round(r["speedup"], 4)) for r in rows]
    check("1 result D: the headline speed-up is the per-channel one, but the stock one from within session 6 where the per-channel "
          "one spans it (#47's comment of 1 November), and the stock one where there is no per-channel cell",
          got == [("A1", 640, "per-channel", 2.0), ("C1", 640, None, None), ("B3", 1120, "stock", 1.4773), ("A1", 640, "stock", 2.0886)], f"{got}")


def test_model_pairs():
    """Result C: YOLO11s against YOLO26s, the same configuration on the same platform, runtime, precision, INT8 recipe, input resolution
    and head. YOLO26s's latency over YOLO11s's, and the AP and AP-small differences, YOLO26s minus YOLO11s. FP32 and INT8 apart."""
    cells = [cell("A1", 640, "baseline", latency=389.752, ap=0.1875, ap_small=0.0904, below=381, clock=2000, throttled="throttled"),
             cell("A2", 640, "baseline", latency=383.063, ap=0.1943, ap_small=0.0995, model="YOLO26s", below=1373, clock=1911, throttled="throttled"),
             cell("A1", 640, "baseline", runtime="ncnn", latency=171.51), cell("A2", 640, "baseline", runtime="ncnn", latency=173.755, model="YOLO26s"),
             cell("A1", 640, "quantized", latency=272.297, recipe="per-channel"), cell("A2", 640, "quantized", latency=277.445, recipe="per-channel", model="YOLO26s"),
             cell("A1", 640, "quantized", latency=272.486, recipe="stock"), cell("A2", 640, "quantized", latency=275.892, recipe="stock", model="YOLO26s"),
             cell("A2", 640, "baseline", model="YOLO26s", head="NMS-free"),  # YOLO11s has no NMS-free head: no pair
             cell("B3", 1280, "quantized", recipe="stock"),  # B4 has no INT8 cell at 1280: no pair
             cell("P1", 640, "pruned", latency=239.66), cell("P3", 640, "pruned", latency=232.95, model="YOLO26s")]
    pairs = st.model_pairs(cells)
    got = [(p["runtime"], p["precision"], p["int8_recipe"], p["configuration"], p["imgsz"], p["yolo11s"]["checkpoint"], p["yolo26s"]["checkpoint"])
           for p in pairs]
    want = [("onnxruntime", "FP32", "", "baseline", 640, "A1", "A2"), ("ncnn", "FP16", "", "baseline", 640, "A1", "A2"),
            ("onnxruntime", "INT8", "per-channel", "quantized", 640, "A1", "A2"), ("onnxruntime", "INT8", "stock", "quantized", 640, "A1", "A2"),
            ("onnxruntime", "FP32", "", "pruned", 640, "P1", "P3")]
    check("1 result C: one pair per configuration, runtime, precision, INT8 recipe and input resolution that both models have", got == want, f"{got}")
    ort, ncnn = pairs[0], pairs[1]
    check("1 result C: latency is YOLO26s over YOLO11s: NCNN 173.755 / 171.51, +1.3 % (pi5_matrix_2_record.md); ONNX Runtime FP32 -1.7 %",
          (round(ncnn["latency_pct"], 1), round(ort["latency_pct"], 1)) == (1.3, -1.7), f"{ncnn['latency_pct']}, {ort['latency_pct']}")
    check("1 result C: the differences are YOLO26s minus YOLO11s: AP +0.0068, AP-small +0.0091",
          (round(ort["ap_diff"], 4), round(ort["ap_small_diff"], 4)) == (0.0068, 0.0091), f"{ort}")
    check("1 result C: clean when neither cell ran an image below 2000 MHz: NCNN yes, the throttled ONNX Runtime FP32 pair no",
          (ncnn["clean"], ort["clean"]) == (True, False))
    check("1 result C: the per-channel INT8 pair is the headline, the stock pair is not", (pairs[2]["headline"], pairs[3]["headline"]) == (True, False))


def test_small_object_rows():
    """The supporting section: quantized, pruned and combined against the reference baseline (the baseline at 640) on the Pi 5. The
    relative drop per bucket is (reference - compressed) / reference, as #9 pre-registered it; the ratio is the AP-small drop over the
    AP drop, banded by #9's bar: below 1.3x absent, 1.3x to 2x weak, 2x or more (#9's 'confirmed' also needs a CI that excludes 0)."""
    a1 = cell("A1", 640, "baseline", ap=0.1875, ap_small=0.0904)
    cells = [a1, cell("A1", 640, "quantized", ap=0.1696, ap_small=0.0781, recipe="per-channel"),
             cell("A1", 640, "quantized", ap=0.1682, ap_small=0.0769, recipe="stock"),
             cell("P1", 640, "pruned", ap=0.1556, ap_small=0.0724),  # the laptop's pruning arm: 17.0 % and 19.9 %, 1.17x
             cell("D1", 640, "distilled", ap=0.1950, ap_small=0.0963),  # not a compression the section is about
             cell("C1", 640, "combined", ap=0.1, ap_small=0.03, recipe="per-channel")]  # 46.7 % and 66.8 %, 1.43x
    rows = st.small_object_rows(cells)
    got = [(r["cell"]["checkpoint"], r["cell"]["configuration"], r["cell"]["int8_recipe"], r["reference"]["checkpoint"], r["headline"]) for r in rows]
    check("1 small objects: quantized (per-channel, stock), pruned and combined, each against the reference baseline; no distilled row",
          got == [("A1", "quantized", "per-channel", "A1", True), ("A1", "quantized", "stock", "A1", False), ("P1", "pruned", "", "A1", True),
                  ("C1", "combined", "per-channel", "A1", True)], f"{got}")
    pc = rows[0]
    got = (round(pc["ap_drop_pct"], 2), round(pc["ap_small_drop_pct"], 2), round(pc["small_over_all"], 2), pc["band"])
    check("1 small objects: A1 per-channel drops 9.55 % AP and 13.61 % AP-small (0.0179 / 0.1875, 0.0123 / 0.0904): 1.43x, weak",
          got == (9.55, 13.61, 1.43, "weak"), f"{got}")
    check("1 small objects: 1.17x is absent", (round(rows[2]["small_over_all"], 2), rows[2]["band"]) == (1.17, "absent"), f"{rows[2]}")
    check("1 small objects: the medium and large drops are there too", all(k in pc for k in ("ap_medium_drop_pct", "ap_large_drop_pct")))


def test_runtime_rows():
    """Result D: the three Pi 5 runtimes side by side for each checkpoint and input resolution (one-to-many head): NCNN FP16, OpenVINO
    FP32, ONNX Runtime FP32 and INT8 (per-channel and stock), each with its speed-up over ONNX Runtime FP32 (that latency over its own)."""
    cells = [cell("A1", 640, "baseline", runtime="ncnn", latency=171.51), cell("A1", 640, "baseline", runtime="openvino", latency=294.008),
             cell("A1", 640, "baseline", latency=389.752), cell("A1", 640, "quantized", latency=272.297, recipe="per-channel"),
             cell("A1", 640, "quantized", latency=272.486, recipe="stock"),
             cell("C1", 640, "combined", latency=193.204, recipe="per-channel"),  # no FP32 cell: no speed-up
             cell("A2", 640, "baseline", model="YOLO26s", head="NMS-free"),  # result B's head: not here
             cell("A1", 640, "baseline", latency=636.56, platform="m34", session="m34-fullcell")]  # not the Pi 5
    rows = st.runtime_rows(cells)
    check("1 runtimes: one row per checkpoint and input resolution on the Pi 5, the one-to-many head, configuration from its float cell",
          [(r["checkpoint"], r["imgsz"], r["configuration"]) for r in rows] == [("A1", 640, "baseline"), ("C1", 640, "combined")], f"{rows}")
    a1 = rows[0]
    got = {k: round(a1[f"{k}_speedup"], 2) for k in st.RUNTIME_COLUMNS}
    check("1 runtimes: speed-up over ONNX Runtime FP32: NCNN 2.27x, OpenVINO 1.33x, ONNX Runtime FP32 1, INT8 1.43x either recipe",
          got == {"ncnn_fp16": 2.27, "openvino_fp32": 1.33, "onnxruntime_fp32": 1.0, "onnxruntime_int8_per_channel": 1.43, "onnxruntime_int8_stock": 1.43},
          f"{got}")
    c1 = rows[1]
    check("1 runtimes: a checkpoint with no ONNX Runtime FP32 cell keeps its cells, with no speed-up",
          c1["onnxruntime_int8_per_channel"]["latency_ms"] == 193.204 and c1["onnxruntime_int8_per_channel_speedup"] is None
          and c1["ncnn_fp16"] is None, f"{c1}")


def test_latency_rows():
    """The latency table: every cell, platform x runtime x configuration x input resolution, the Pi 5 first. Session 6's cells carry
    the footnote that their latencies are about 6 % high against the other sessions'."""
    s6 = dict(cell("B3", 1120, "baseline", trained=1280, session="pi5-matrix-6-1120"),
              note="latency about 6 % high against the other sessions' (measured by session 7)")  # as studyresults.py writes it
    cells = [cell("A1", 640, "baseline", platform="m34", session="m34-fullcell"), s6, cell("A1", 640, "baseline", runtime="ncnn"),
             cell("A1", 640, "baseline"), cell("A1", 640, "quantized", recipe="stock"), cell("A1", 640, "quantized", recipe="per-channel"),
             cell("A2", 640, "baseline", model="YOLO26s", head="NMS-free"), cell("A2", 640, "baseline", model="YOLO26s")]
    rows = st.latency_rows(cells)
    got = [(r["cell"]["platform"], r["cell"]["runtime"], r["cell"]["model"], r["cell"]["checkpoint"], r["cell"]["imgsz"], r["cell"]["int8_recipe"],
            r["cell"]["head"]) for r in rows]
    want = [("pi5", "onnxruntime", "YOLO11s", "A1", 640, "", "one-to-many"), ("pi5", "onnxruntime", "YOLO11s", "B3", 1120, "", "one-to-many"),
            ("pi5", "onnxruntime", "YOLO11s", "A1", 640, "per-channel", "one-to-many"), ("pi5", "onnxruntime", "YOLO11s", "A1", 640, "stock", "one-to-many"),
            ("pi5", "onnxruntime", "YOLO26s", "A2", 640, "", "one-to-many"), ("pi5", "onnxruntime", "YOLO26s", "A2", 640, "", "NMS-free"),
            ("pi5", "ncnn", "YOLO11s", "A1", 640, "", "one-to-many"), ("m34", "onnxruntime", "YOLO11s", "A1", 640, "", "one-to-many")]
    check("1 latency: every cell, by platform (the Pi 5 first), runtime, model, configuration, INT8 recipe, input resolution, head",
          got == want, f"{got}")
    check("1 latency: session 6's cell, and only it, carries the footnote",
          [r["session_6"] for r in rows] == [False, True] + [False] * 6 and "6 %" in rows[1]["note"], f"{[(r['session_6'], r['note']) for r in rows]}")


# ---------------------------------------------------------------- 2: the real cells.csv against the session records

STUDY = Path(__file__).resolve().parents[2]
SESSION_7 = "pi5-matrix-7-int8pc"


def real(without_session_7=False):
    cells, _ = st.sp.usable(st.sp.load_cells(STUDY / "results" / "cells.csv"))
    return [c for c in cells if not (without_session_7 and c["session"] == SESSION_7)]


def record_table(name, header, nth=0):
    """The rows of the nth table in verification/<name> whose header line starts with header: each a list of its cells, bold marks
    taken out and the records' minus sign made ASCII."""
    lines = (STUDY / "verification" / name).read_text(encoding="utf-8").splitlines()
    start = [i for i, ln in enumerate(lines) if ln.startswith(header)][nth]
    rows = []
    for ln in lines[start + 2:]:
        if not ln.startswith("|"):
            break
        rows.append([x.strip().replace("**", "").replace("−", "-") for x in ln.strip().strip("|").split("|")])
    return rows


def r4(x):
    return None if x is None else round(x, 4)


def r1(x):
    return None if x is None else round(x, 1)


def test_real_accuracy():
    """Every AP and AP-small of the accuracy table against the records: session 7's per-channel table (A1, A2 and their INT8 cells),
    session 3's compressed table (P, D and the stock C cells), session 7's results table (the per-channel C cells)."""
    rows = st.accuracy_rows(real())
    acc = {(r["cell"]["checkpoint"], r["cell"]["configuration"], r["cell"]["int8_recipe"]): r["cell"] for r in rows}
    check("2 accuracy: 14 rows, 7 per model: baseline, quantized (per-channel, stock), pruned, distilled, combined (per-channel, stock)",
          len(rows) == 14 and sum(r["headline"] for r in rows) == 10, f"{[(k, r['headline']) for k, r in zip(acc, rows)]}")
    for name, imgsz, fp32, stock, pc, _, small, _ in record_table("pi5_matrix_7_record.md", "| Checkpoint | imgsz | AP, FP32 |"):
        ck = name.split(",")[0]
        if imgsz != "640" or "NMS-free" in name:
            continue
        got = (r4(acc[(ck, "baseline", "")]["ap"]), r4(acc[(ck, "quantized", "stock")]["ap"]), r4(acc[(ck, "quantized", "per-channel")]["ap"]),
               " -> ".join(f"{acc[k]['ap_small']:.4f}" for k in ((ck, "baseline", ""), (ck, "quantized", "stock"), (ck, "quantized", "per-channel"))))
        want = (float(fp32), float(stock.split()[0]), float(pc.split()[0]), small.replace("→", "->"))
        check(f"2 accuracy, session 7's per-channel table, {ck}: AP FP32 / stock / per-channel {want[:3]}, AP-small {want[3]}", got == want, f"{got}")
    names = {"Pruned 640 (P1 / P3)": ("P1", "P3", "pruned", ""), "Distilled 640 (D1 / D2)": ("D1", "D2", "distilled", ""),
             "Pruned and distilled, INT8 (C1 / C2)": ("C1", "C2", "combined", "stock")}
    for conf, ap, small, *_ in record_table("pi5_matrix_3_record.md", "| Configuration | AP, YOLO11s / YOLO26s | AP-small |"):
        if conf not in names:  # Pruned 960: not at the matrix resolution
            continue
        a, b, configuration, recipe = names[conf]
        got = [r4(acc[(x, configuration, recipe)][k]) for k in ("ap", "ap_small") for x in (a, b)]
        want = [float(v) for v in ap.split(" / ") + small.split(" / ")]
        check(f"2 accuracy, session 3's table, {conf}: AP {ap}, AP-small {small}", got == want, f"{got}")
    s7 = {r[0].split()[0]: r for r in record_table("pi5_matrix_7_record.md", "| Cell | Precision recorded | AP |")}
    got = [(r4(acc[(x, "combined", "per-channel")]["ap"]), r4(acc[(x, "combined", "per-channel")]["ap_small"])) for x in ("C1", "C2")]
    want = [(float(s7[x][2]), float(s7[x][4])) for x in ("C1", "C2")]
    check(f"2 accuracy, session 7's results table: C1 and C2 per-channel AP and AP-small {want}", got == want, f"{got}")
    check("2 accuracy: A1's AP50 is 0.3265 (pi5_matrix_1_record.md)", r4(acc[("A1", "baseline", "")]["ap50"]) == 0.3265)


def test_real_latency():
    """Five spot checks of the latency table against the records."""
    rows = st.latency_rows(real())
    find = lambda platform, ck, imgsz, runtime, precision, head="one-to-many": next(  # noqa: E731
        r for r in rows if (r["cell"]["platform"], r["cell"]["checkpoint"], r["cell"]["imgsz"], r["cell"]["runtime"], r["cell"]["precision"],
                            r["cell"]["head"]) == (platform, ck, imgsz, runtime, precision, head))
    spot = lambda r: (r1(r["cell"]["latency_ms"]), r["cell"]["below_2000_mhz"], r["cell"]["clock_median_mhz"], r["cell"]["throttled"], r["session_6"])  # noqa: E731
    check("2 latency, pi5_matrix_1_record.md: A1 640 ONNX Runtime FP32 389.8 ms, 381 images below 2000 MHz, median 2000, throttled",
          spot(find("pi5", "A1", 640, "onnxruntime", "FP32")) == (389.8, 381, 2000, "throttled", False), f"{spot(find('pi5', 'A1', 640, 'onnxruntime', 'FP32'))}")
    a2 = find("pi5", "A2", 640, "onnxruntime", "FP32")["cell"]
    check("2 latency, pi5_matrix_2_record.md (result B table): A2 640 ONNX Runtime FP32 383.1 ms, network 378.4, post at 0.25 1.11, "
          "1,373 below 2000 MHz, median 1911", (r1(a2["latency_ms"]), r1(a2["net_ms"]), round(a2["post_deploy_ms"], 2), a2["below_2000_mhz"],
                                                a2["clock_median_mhz"]) == (383.1, 378.4, 1.11, 1373, 1911), f"{a2}")
    check("2 latency, pi5_matrix_6_record.md: B3 1120 ONNX Runtime FP32 1315.4 ms, 513 below 2000 MHz, session 6's footnote",
          spot(find("pi5", "B3", 1120, "onnxruntime", "FP32")) == (1315.4, 513, 2000, "throttled", True) and "6 %" in find("pi5", "B3", 1120, "onnxruntime", "FP32")["note"])
    check("2 latency, pi5_matrix_1_record.md: A1 640 NCNN FP16 171.5 ms, none below 2000 MHz",
          spot(find("pi5", "A1", 640, "ncnn", "FP16")) == (171.5, 0, 2000, "none", False))
    m34 = find("m34", "A1", 640, "onnxruntime", "INT8")["cell"]
    check("2 latency, m34_fullcell_int8_record.md: the phone's A1 640 INT8 304.8 ms, no clock reading",
          (r1(m34["latency_ms"]), m34["below_2000_mhz"]) == (304.8, None), f"{m34}")


RECORD_RUNTIMES = {"ONNX Runtime FP32": ("onnxruntime", "FP32", ""), "OpenVINO FP32": ("openvino", "FP32", ""), "NCNN FP16": ("ncnn", "FP16", ""),
                   "ONNX Runtime INT8": ("onnxruntime", "INT8", "stock")}


def test_real_model_pairs():
    """Result C against the records' YOLO11s-against-YOLO26s tables: each pair's two APs and its latency gap, every runtime at 640, 800,
    960, 1120 and 1280 (sessions 2, 5, 6 and 4), and the per-channel INT8 pairs (session 7)."""
    pairs = [p for p in st.model_pairs(real()) if p["platform"] == "pi5" and p["configuration"] in ("baseline", "quantized") and p["head"] == "one-to-many"]
    by = {(p["runtime"], p["precision"], p["int8_recipe"], p["imgsz"]): p for p in pairs}
    want = []
    for rt, ap640, _, pct640, ap960, _, pct960 in record_table("pi5_matrix_2_record.md", "| Runtime | AP, A1 / A2 |"):
        want += [(RECORD_RUNTIMES[rt] + (640,), ap640, pct640), (RECORD_RUNTIMES[rt] + (960,), ap960, pct960)]
    for name, imgsz in (("pi5_matrix_5_record.md", 800), ("pi5_matrix_6_record.md", 1120), ("pi5_matrix_4_record.md", 1280)):
        want += [(RECORD_RUNTIMES[rt] + (imgsz,), ap, pct) for rt, ap, _, _, pct in record_table(name, "| Runtime | AP, B")]
    want += [(("onnxruntime", "INT8", "per-channel", int(imgsz)), ap, pct)
             for imgsz, ap, _, _, pct, _ in record_table("pi5_matrix_7_record.md", "| imgsz | AP, YOLO11s / YOLO26s |")]
    for k, ap, pct in want:
        p = by.get(k)
        got = p and (f"{p['yolo11s']['ap']:.4f} / {p['yolo26s']['ap']:.4f}", f"{p['latency_pct']:+.1f} %")
        check(f"2 result C, {k}: AP {ap}, YOLO26s's latency {pct}", got == (ap, pct), f"{got}")
    check(f"2 result C: every baseline and quantized pair on the Pi 5 is one of the records' {len(want)}", len(pairs) == len(want) == 23,
          f"{len(pairs)} pairs, {len(want)} in the records")


def test_real_int8():
    """Result D against session 7's per-channel table (all 10 rows: AP, its change and AP-small, both recipes, and both latencies),
    session 6's stock table (the speed-ups and the AP-small change) and the phone's 2.09x."""
    rows = st.int8_rows(real())
    pi5 = {(r["checkpoint"], r["imgsz"], r["head"]): r for r in rows if r["platform"] == "pi5"}
    for name, imgsz, fp32, stock, pc, _, small, ms in record_table("pi5_matrix_7_record.md", "| Checkpoint | imgsz | AP, FP32 |"):
        r = pi5[(name.split(",")[0], int(imgsz), "NMS-free" if "NMS-free" in name else "one-to-many")]
        got = (r4(r["fp32"]["ap"]), f"{r['stock']['ap']:.4f} ({r['stock_ap_diff_pct']:.1f} %)", f"{r['per_channel']['ap']:.4f} ({r['per_channel_ap_diff_pct']:.1f} %)",
               f"{r['fp32']['ap_small']:.4f} -> {r['stock']['ap_small']:.4f} -> {r['per_channel']['ap_small']:.4f}",
               f"{r['stock']['latency_ms']:.1f} -> {r['per_channel']['latency_ms']:.1f}", r["per_channel_spans_session_6"])
        want = (float(fp32), stock, pc, small.replace("→", "->"), ms.replace(" s6", "").replace("→", "->"), "s6" in ms)
        check(f"2 result D, session 7's per-channel table, {name} {imgsz}", got == want, f"got {got}, want {want}")
    for name, imgsz, _, ap_lost, _, small_lost, _, speedup in record_table("pi5_matrix_6_record.md", "| Checkpoint | imgsz | AP, FP32 → INT8 |"):
        r = pi5[(name.split(",")[0], int(imgsz), "one-to-many")]
        got = (f"{r['stock_ap_diff']:.4f} ({r['stock_ap_diff_pct']:.1f} %)", f"{r['stock_ap_small_diff']:.4f} ({r['stock_ap_small_diff_pct']:.1f} %)",
               f"{r['stock_speedup']:.2f}×", r["stock_spans_session_6"])
        check(f"2 result D, session 6's stock table, {name} {imgsz}: {ap_lost}, {small_lost}, {speedup}", got == (ap_lost, small_lost, speedup, False), f"{got}")
    m34 = [r for r in rows if r["platform"] == "m34"]
    check("2 result D, m34_fullcell_int8_record.md: the phone's A1 640 INT8 is 2.09x faster (304.8 / 636.6)",
          [(r["checkpoint"], round(r["stock_speedup"], 2)) for r in m34] == [("A1", 2.09)], f"{m34}")
    check("2 result D: C1 and C2 have INT8 rows with no FP32 cell", all(pi5[(c, 640, "one-to-many")]["fp32"] is None for c in ("C1", "C2")))


def test_real_runtimes():
    """Result D's runtime table against session 6's two resolution ladders: per checkpoint and input resolution, ONNX Runtime FP32,
    OpenVINO FP32 and NCNN latency, NCNN's AP, and the stock INT8 cell's AP and latency."""
    rows = {(r["checkpoint"], r["imgsz"]): r for r in st.runtime_rows(real())}
    for nth in (0, 1):
        for imgsz, ck, ap, _, ort, ov, ncnn, int8 in record_table("pi5_matrix_6_record.md", "| imgsz | Checkpoint | AP, FP32 |", nth):
            r = rows[(ck, int(imgsz))]
            got = (r4(r["onnxruntime_fp32"]["ap"]), r1(r["onnxruntime_fp32"]["latency_ms"]), r1(r["openvino_fp32"]["latency_ms"]),
                   f"{r['ncnn_fp16']['latency_ms']:.1f} ({r['ncnn_fp16']['ap']:.4f})",
                   f"{r['onnxruntime_int8_stock']['ap']:.4f} / {r['onnxruntime_int8_stock']['latency_ms']:.1f}" if r["onnxruntime_int8_stock"] else "—")
            want = (float(ap), float(ort.split()[0]), float(ov.split()[0]), ncnn, int8)
            check(f"2 runtimes, session 6's ladders, {ck} {imgsz}", got == want, f"got {got}, want {want}")


def test_real_small_objects():
    """The supporting section against the records' drops (sessions 6 and 7) and #47's comment of 1 November (1.43x and 1.44x
    per-channel, 1.45x and 1.47x stock), and the pruned drops against session 2's (P1 -0.0302, P3 -0.0360 AP)."""
    rows = {(r["cell"]["checkpoint"], r["cell"]["configuration"], r["cell"]["int8_recipe"]): r for r in st.small_object_rows(real())}
    got = {k: (round(rows[k]["ap_drop_pct"], 1), round(rows[k]["ap_small_drop_pct"], 1)) for k in (("A1", "quantized", "stock"), ("A2", "quantized", "stock"))}
    check("2 small objects, pi5_matrix_6_record.md: A1 stock drops 10.3 % AP and 15.0 % AP-small, A2 stock 20.3 % and 29.9 %",
          got == {("A1", "quantized", "stock"): (10.3, 15.0), ("A2", "quantized", "stock"): (20.3, 29.9)}, f"{got}")
    got = [round(rows[(ck, "quantized", recipe)]["small_over_all"], 2) for recipe in ("per-channel", "stock") for ck in ("A1", "A2")]
    check("2 small objects, #47's comment: 1.43x (A1) and 1.44x (A2) per-channel, 1.45x and 1.47x stock, all weak by #9's bar",
          got == [1.43, 1.44, 1.45, 1.47] and all(rows[(ck, "quantized", rc)]["band"] == "weak" for ck in ("A1", "A2") for rc in ("per-channel", "stock")),
          f"{got}")
    got = [r4(rows[(ck, "pruned", "")]["reference"]["ap"] - rows[(ck, "pruned", "")]["cell"]["ap"]) for ck in ("P1", "P3")]
    check("2 small objects, pi5_matrix_2_record.md: P1 loses 0.0302 AP against A1, P3 0.0360 against A2", got == [0.0302, 0.0360], f"{got}")
    check("2 small objects: 10 rows, 5 per model (quantized and combined in both recipes, pruned)", len(rows) == 10, f"{list(rows)}")


def test_session_7_swap():
    """Re-running after #37 scored session 7 swaps in the per-channel cells with no code change: without session 7's cells the stock
    INT8 cells are the headline, with them the per-channel ones are."""
    for label, cells, recipe in (("without session 7", real(True), "stock"), ("with session 7", real(), "per-channel")):
        acc = [r["cell"]["int8_recipe"] for r in st.accuracy_rows(cells) if r["headline"] and r["cell"]["precision"] == "INT8"]
        pairs = [p["int8_recipe"] for p in st.model_pairs(cells) if p["headline"] and p["precision"] == "INT8" and p["platform"] == "pi5"]
        small = [r["cell"]["int8_recipe"] for r in st.small_object_rows(cells) if r["headline"] and r["cell"]["precision"] == "INT8"]
        check(f"2 {label}: the INT8 headline rows of the accuracy table, result C and the small-object section are all {recipe}",
              acc and pairs and small and set(acc) == set(pairs) == set(small) == {recipe}, f"{acc}, {pairs}, {small}")


# ---------------------------------------------------------------- 3: the figure's data

def test_figure_lines():
    """Result C's figure: YOLO26s's latency over YOLO11s's against input resolution on the Pi 5, one line per runtime and precision
    (FP32 and INT8 apart), baselines and quantized cells, one-to-many head. The stock INT8 line stays, behind the per-channel one."""
    lines = st.figure_lines(st.model_pairs(real()))
    got = [(ln["runtime"], ln["precision"], ln["int8_recipe"], [p["imgsz"] for p in ln["points"]], ln["headline"]) for ln in lines]
    want = [("ncnn", "FP16", "", [640, 800, 960, 1120, 1280], True), ("openvino", "FP32", "", [640, 800, 960, 1120, 1280], True),
            ("onnxruntime", "FP32", "", [640, 800, 960, 1120, 1280], True), ("onnxruntime", "INT8", "per-channel", [640, 800, 960, 1120], True),
            ("onnxruntime", "INT8", "stock", [640, 800, 960, 1120], False)]
    check("3 figure: NCNN FP16, OpenVINO FP32, ONNX Runtime FP32 and INT8 (per-channel, then stock), each through its input resolutions",
          got == want, f"{got}")
    ncnn = [round(p["latency_pct"], 1) for p in lines[0]["points"]]
    check("3 figure: the NCNN line is +1.3, +3.3, +4.0, +6.1 and +9.0 % (pi5_matrix_6_record.md)", ncnn == [1.3, 3.3, 4.0, 6.1, 9.0], f"{ncnn}")
    check("3 figure: every NCNN and INT8 point is clean (no image below 2000 MHz); the ONNX Runtime FP32 line's are not",
          all(p["clean"] for ln in lines if ln["runtime"] == "ncnn" or ln["precision"] == "INT8" for p in ln["points"])
          and not any(p["clean"] for p in lines[2]["points"]))
    before = [(ln["int8_recipe"], ln["headline"]) for ln in st.figure_lines(st.model_pairs(real(True))) if ln["precision"] == "INT8"]
    check("3 figure: without session 7 the one INT8 line is stock, as the headline", before == [("stock", True)], f"{before}")


if __name__ == "__main__":
    test_figure_lines()
    test_accuracy_rows()
    test_int8_rows()
    test_model_pairs()
    test_small_object_rows()
    test_runtime_rows()
    test_latency_rows()
    test_real_accuracy()
    test_real_latency()
    test_real_model_pairs()
    test_real_int8()
    test_real_runtimes()
    test_real_small_objects()
    test_session_7_swap()
    print(f"\n{len(FAILED)} failed" if FAILED else "\nall passed")
    sys.exit(1 if FAILED else 0)
