"""Check studynms.py (#45, result B: NMS time against objects per image) at its three seams, on a tiny fake session
built in a temporary folder with known literal values, and its table against the session 2 record:

  python study/scripts/tests/test_studynms.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import studynms as nms  # noqa: E402

FAILED = []


def check(label, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        FAILED.append(label)
        if detail:
            print(f"     {detail}")


NL = chr(10)
BS = chr(92)
TMP = Path(tempfile.mkdtemp(prefix="studynms-test-"))

# ---------------------------------------------------------------- 1: ground-truth objects per image file
gt = {"files": ["a.jpg", "b.jpg", "c.jpg"],
      "gdict": {"images": [{"id": 0}, {"id": 1}, {"id": 2}],
                "annotations": [{"image_id": 0}, {"image_id": 2}, {"image_id": 2}, {"image_id": 2}, {"image_id": 0}]}}
(TMP / "gt.json").write_text(json.dumps(gt), encoding="utf-8")
check("1 objects per image file: image id i is files[i], and an image with no annotation has 0",
      nms.gt_counts(TMP / "gt.json") == {"a.jpg": 2, "b.jpg": 0, "c.jpg": 3}, nms.gt_counts(TMP / "gt.json"))

# ---------------------------------------------------------------- 2: images in bins, with the median and the interquartile range
bins = nms.bin_stats([1, 2, 3, 10, 11, 12, 13, 99], [1, 2, 3, 10, 20, 30, 40, 7], [1, 10, 20])
check("2 two bins [1, 10) and [10, 20]: an image above the last edge is left out, not put in the last bin", len(bins) == 2, bins)
check("2 first bin: 3 images, x median 2, y median 2, quartiles 1.5 and 2.5",
      (bins[0]["n"], bins[0]["x_median"], bins[0]["median"], bins[0]["q1"], bins[0]["q3"]) == (3, 2.0, 2.0, 1.5, 2.5), bins[0])
check("2 second bin (upper edge included): 4 images, x median 11.5, y median 25, quartiles 17.5 and 32.5",
      (bins[1]["lo"], bins[1]["hi"], bins[1]["n"], bins[1]["x_median"], bins[1]["median"], bins[1]["q1"], bins[1]["q3"])
      == (10, 20, 4, 11.5, 25.0, 17.5, 32.5), bins[1])
check("2 a bin with no image is dropped", [b["lo"] for b in nms.bin_stats([1, 2], [5, 6], [1, 3, 5, 9])] == [1], "")
check("2 quantile edges: 8 objects counts 1..8 in 2 bins split at 4", nms.quantile_edges(list(range(1, 9)), 2) == [1, 4, 8],
      nms.quantile_edges(list(range(1, 9)), 2))
check("2 quantile edges drop repeats when many images share a count", nms.quantile_edges([5] * 9 + [6], 4) == [5, 6],
      nms.quantile_edges([5] * 9 + [6], 4))

# ---------------------------------------------------------------- 3: a session directory -> head pairs and their table rows
HEADER = "cell,runtime,precision,imgsz,images,attempts,outcome,latency_ms,latency_source,throttle,drift,run_dir,attempt_log"
TIMINGS = "index,file,t_pre_ms,t_net_ms,t_post_ms,n_cand,n_dets,temp_c,cpu_mhz,throttled,img_crc32,session,gpu_temp_c,gpu_mhz,t_post_deploy_ms,n_dets_deploy"


def attempt(session, cell, n, timing, ap, ap_small, ap_large, head, precision, rows):
    """cell/attempt-n with the four files a scored attempt has; rows are (file, t_post, n_cand, t_post_deploy)."""
    d = session / "runs" / cell / f"attempt-{n}"
    d.mkdir(parents=True)
    med = lambda v: {"median": v, "p95": v}
    (d / "meta.json").write_text(json.dumps({"timing_ms": {
        "t_pre_ms": med(timing[0]), "t_net_ms": med(timing[1]), "t_post_ms": med(timing[2]), "post_deploy": med(timing[3]),
        "e2e_deploy": med(timing[4])}}), encoding="utf-8")
    (d / "score.json").write_text(json.dumps({"head": head, "precision": precision, "stats": {
        "coco/AP_all": ap, "coco/AP_small": ap_small, "coco/AP_large": ap_large}}), encoding="utf-8")
    lines = [TIMINGS] + [f"{i},{f},3,100,{tp},{nc},5,70,2000,,x,0,,,{tpd},5" for i, (f, tp, nc, tpd) in enumerate(rows)]
    (d / "timings.csv").write_text(NL.join(lines) + NL, encoding="utf-8")


S = TMP / "pi5-matrix-9-fake"
S.mkdir()
IMG = [("a.jpg", 10.0, 40, 1.0), ("b.jpg", 20.0, 80, 2.0)]
FREE = [("a.jpg", 0.2, 500, 0.1), ("b.jpg", 0.4, 500, 0.3)]
attempt(S, "m-640-s0/640/onnxruntime-fp32", 1, (9, 9, 9, 9, 9), 0.9, 0.9, 0.9, "raw", "FP32", IMG)  # the attempt that does not count
attempt(S, "m-640-s0/640/onnxruntime-fp32", 2, (3.0, 100.0, 15.0, 1.0, 104.0), 0.2, 0.1, 0.4, "raw", "INT8", IMG)  # precision from score.json
attempt(S, "m-640-s0/640/onnxruntime-fp32-e2e", 1, (3.5, 105.0, 0.25, 0.5, 109.0), 0.19, 0.11, 0.3, "end2end", "INT8", FREE)
attempt(S, "m-640-s0/640/openvino-fp32", 1, (3, 3, 3, 3, 3), 0.1, 0.1, 0.1, "raw", "FP32", IMG)  # no NMS-free partner
attempt(S, "m-640-s0/640/ncnn-fp32-e2e", 1, (3, 3, 3, 3, 3), 0.1, 0.1, 0.1, "end2end", "FP16", FREE)  # no one-to-many partner
(S / "summary.csv").write_text(NL.join([
    HEADER,
    "drift-start,onnxruntime,FP32,640,200,1,ok,388.0,e2e_deploy,none,ok,runs/drift-start/attempt-1,a1",
    "m-640-s0/640/onnxruntime-fp32,onnxruntime,FP32,640,all,2,throttled,104.0,e2e_deploy,throttled,ok,runs/m-640-s0/640/onnxruntime-fp32/attempt-2,a1; a2",
    "m-640-s0/640/onnxruntime-fp32-e2e,onnxruntime,FP32,640,all,1,ok,109.0,e2e_deploy,none,ok,runs/m-640-s0/640/onnxruntime-fp32-e2e/attempt-1,a1",
    "m-640-s0/640/openvino-fp32,openvino,FP32,640,all,1,ok,3.0,e2e_deploy,none,ok,runs/m-640-s0/640/openvino-fp32/attempt-1,a1",
    "m-640-s0/640/ncnn-fp32-e2e,ncnn,FP16,640,all,1,ok,3.0,e2e_deploy,none,ok,runs/m-640-s0/640/ncnn-fp32-e2e/attempt-1,a1",
]) + NL, encoding="utf-8")

pairs = nms.head_pairs(S)
check("3 one pair: the cell ending -e2e with its partner in the same session; drift rows and unpaired cells are no pair",
      len(pairs) == 1 and pairs[0]["run"] == "m-640-s0" and pairs[0]["imgsz"] == 640 and pairs[0]["variant"] == "onnxruntime-fp32", pairs)
row = nms.pair_row(pairs[0])
check("3 one-to-many head: medians from meta.json of the attempt in run_dir (attempt 2, not 1)",
      (row["one_to_many_pre_ms"], row["one_to_many_net_ms"], row["one_to_many_post_025_ms"], row["one_to_many_post_0001_ms"],
       row["one_to_many_latency_ms"]) == (3.0, 100.0, 1.0, 15.0, 104.0), row)
check("3 NMS-free head medians", (row["nms_free_pre_ms"], row["nms_free_net_ms"], row["nms_free_post_025_ms"],
                                  row["nms_free_post_0001_ms"], row["nms_free_latency_ms"]) == (3.5, 105.0, 0.5, 0.25, 109.0), row)
check("3 AP and AP-small from score.json for both heads", (row["one_to_many_ap"], row["one_to_many_ap_small"], row["nms_free_ap"],
                                                       row["nms_free_ap_small"]) == (0.2, 0.1, 0.19, 0.11), row)
check("3 precision is score.json's (INT8), not the folder name's (fp32)", row["precision"] == "INT8" and row["runtime"] == "onnxruntime", row)
check("3 post-processing at 0.25 as a share of the latency, in percent: 1 / 104 and 0.5 / 109",
      (row["one_to_many_post_share_pct"], row["nms_free_post_share_pct"]) == (round(100 / 104, 3), round(50 / 109, 3)), row)
check("3 difference of the heads (NMS-free minus one-to-many): net +5, post at 0.25 -0.5, post at 0.001 -14.75, latency +5",
      (row["diff_net_ms"], row["diff_post_025_ms"], row["diff_post_0001_ms"], row["diff_latency_ms"]) == (5.0, -0.5, -14.75, 5.0), row)
check("3 difference of the heads in AP and AP-small", (row["diff_ap"], row["diff_ap_small"]) == (-0.01, 0.01), row)
check("3 summary.csv's throttle verdict of each head's cell is in the row (throttled / none)",
      (row["one_to_many_throttle"], row["nms_free_throttle"]) == ("throttled", "none"), row)
check("3 the session's name is in the row", row["session"] == "pi5-matrix-9-fake", row)

imgs = nms.per_image(pairs[0])
check("3 per-image points: for each image, objects joined by file name, candidates at 0.001 from the one-to-many head, "
      "and the post-processing times of both heads",
      imgs["a.jpg"]["one_to_many_post_0001_ms"] == 10.0 and imgs["a.jpg"]["one_to_many_post_025_ms"] == 1.0
      and imgs["a.jpg"]["n_cand"] == 40 and imgs["b.jpg"]["nms_free_post_0001_ms"] == 0.4 and imgs["b.jpg"]["nms_free_post_025_ms"] == 0.3
      and imgs["b.jpg"]["n_cand"] == 80, imgs)

# ---------------------------------------------------------------- 4: the table against the session 2 record (the real, tracked data)
# pi5_matrix_2_record.md, "The NMS-free head (result B)", to the record's rounding: AP to 4 decimals, ms to 1 (post-processing to 2).
BENCH = Path(__file__).resolve().parents[2] / "bench" / "pi5"
RECORD = {  # (runtime, precision): (AP, AP-small, AP-large, latency, network, post at 0.25, post at 0.001), one-to-many then NMS-free
    ("onnxruntime", "FP32"): ((0.1943, 0.0995, 0.3972, 383.1, 378.4, 1.11, 15.35), (0.1891, 0.0991, 0.3674, 386.9, 383.1, 0.17, 0.20)),
    ("openvino", "FP32"): ((0.1943, 0.0995, 0.3972, 302.2, 297.8, 1.12, 15.59), (0.1891, 0.0991, 0.3674, 310.5, 307.0, 0.15, 0.16)),
    ("onnxruntime", "INT8"): ((0.1549, 0.0698, 0.3306, 275.9, 271.3, 1.10, 15.27), (0.1540, 0.0770, 0.3278, 279.1, 275.3, 0.18, 0.20)),
}


def rounded(row, head):
    return (round(row[f"{head}_ap"], 4), round(row[f"{head}_ap_small"], 4), round(row[f"{head}_ap_large"], 4),
            round(row[f"{head}_latency_ms"], 1), round(row[f"{head}_net_ms"], 1), round(row[f"{head}_post_025_ms"], 2),
            round(row[f"{head}_post_0001_ms"], 2))


rows2 = [nms.pair_row(p) for p in nms.head_pairs(next(BENCH.glob("pi5-matrix-2-*")))]
check("4 session 2 gives its three head pairs, all A2 at 640",
      sorted((r["runtime"], r["precision"]) for r in rows2) == sorted(RECORD) and {(r["run"], r["imgsz"]) for r in rows2} == {("yolo26s-640-s0", 640)},
      [(r["runtime"], r["precision"], r["run"], r["imgsz"]) for r in rows2])
for r in rows2:
    want = RECORD[(r["runtime"], r["precision"])]
    check(f"4 {r['runtime']} {r['precision']}, one-to-many head: medians, AP, AP-small and AP-large equal the record's", rounded(r, "one_to_many") == want[0],
          (rounded(r, "one_to_many"), want[0]))
    check(f"4 {r['runtime']} {r['precision']}, NMS-free head: medians, AP, AP-small and AP-large equal the record's", rounded(r, "nms_free") == want[1],
          (rounded(r, "nms_free"), want[1]))
rows7 = [nms.pair_row(p) for p in nms.head_pairs(next(BENCH.glob("pi5-matrix-7-*")))]
check("4 session 7 adds the per-channel INT8 pair (onnxruntime-int8pc): NMS-free AP 0.1739, AP-small 0.0881, latency 279.5 ms",
      [(r["variant"], round(r["nms_free_ap"], 4), round(r["nms_free_ap_small"], 4), round(r["nms_free_latency_ms"], 1))
       for r in rows7] == [("onnxruntime-int8pc", 0.1739, 0.0881, 279.5)], rows7)

# ---------------------------------------------------------------- 5: every one-to-many cell at one input resolution, over all sessions
B = TMP / "bench"
for sess, cells in (("pi5-matrix-1-x", [("r1/640/onnxruntime-fp32", 640, "raw"), ("r1/800/onnxruntime-fp32", 800, "raw")]),
                    ("pi5-matrix-2-y", [("r2/640/openvino-fp32", 640, "raw"), ("r2/640/openvino-fp32-e2e", 640, "end2end")])):
    d = B / sess
    d.mkdir(parents=True)
    for cell, imgsz, head in cells:
        attempt(d, cell, 1, (1, 2, 3, 4, 5), 0.1, 0.1, 0.1, head, "FP32", IMG)
    (d / "summary.csv").write_text(NL.join([HEADER, "drift-start,onnxruntime,FP32,640,200,1,ok,1,e2e_deploy,none,ok,runs/drift-start/attempt-1,a1"]
        + [f"{c},rt,FP32,{z},all,1,ok,1,e2e_deploy,none,ok,runs/{c}/attempt-1,a1" for c, z, _ in cells]) + NL, encoding="utf-8")
(B / "pi5-energy-check").mkdir()  # not a matrix session
check("5 pooled cells at 640: the one-to-many cells of every pi5-matrix session, not the NMS-free head's, not 800, not a non-matrix folder",
      [(c["session"], c["cell"]) for c in nms.one_to_many_cells(B, 640)] == [("pi5-matrix-1-x", "r1/640/onnxruntime-fp32"),
                                                                            ("pi5-matrix-2-y", "r2/640/openvino-fp32")],
      [(c["session"], c["cell"]) for c in nms.one_to_many_cells(B, 640)])

# ---------------------------------------------------------------- 6: a straight line through the images (ms per object, ms per candidate)
fit = nms.line_fit([0, 1, 2, 3], [1, 3, 5, 7])
check("6 line through (0,1) (1,3) (2,5) (3,7): slope 2, intercept 1, correlation 1",
      (round(fit["slope"], 9), round(fit["intercept"], 9), round(fit["r"], 9)) == (2.0, 1.0, 1.0), fit)

# ---------------------------------------------------------------- 7: the table files
OUT = TMP / "out"
nms.write_table([row], OUT)
text = (OUT / "result_b_heads.csv").read_text(encoding="utf-8").splitlines()
check("7 CSV: a header and one row, the row's values as the pair_row gave them",
      len(text) == 2 and "one_to_many_latency_ms" in text[0] and "104.0" in text[1] and "pi5-matrix-9-fake" in text[1], text)
tex = (OUT / "result_b_heads.tex").read_text(encoding="utf-8")
check("7 LaTeX: booktabs, the pair's three rows (one-to-many head, NMS-free head, difference), a throttled cell marked t, no 'e2e'",
      all(k in tex for k in (BS + "toprule", BS + "midrule", BS + "bottomrule", "One-to-many head", "NMS-free head", "Difference", "104.0 t", "109.0"))
      and "e2e" not in tex.lower() and "109.0 t" not in tex, tex)

if FAILED:
    sys.exit(f"{len(FAILED)} failed: {FAILED}")
print("all passed")
