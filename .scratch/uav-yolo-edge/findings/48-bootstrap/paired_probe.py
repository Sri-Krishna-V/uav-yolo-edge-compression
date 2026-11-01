"""Paired bootstrap of AP and AP-small differences between two cells, from their saved detections (the #48 method).

  paired_probe.py GT_JSON CELL_A CELL_B [--resamples N] [--seed S] [--out file.json]

Each cell is evaluated once (COCOeval_faster(separate_eval=True) keeps the per-image results as a list of C++
ImageEvaluation objects, laid out [category][area range][image position]); every resample draws image positions with
replacement, once for both cells, rebuilds that list from the kept objects and calls the C++ accumulate. Run under studycap
(run_capped.py). Only one cell's detections are in memory at a time; the first cell's per-image results are kept.
Difference = A minus B. The interval is the 2.5 and 97.5 percentiles of the resampled differences.
"""
import argparse, copy, gc, json, time
import numpy as np, psutil
from pathlib import Path
import faster_coco_eval.faster_eval_api_cpp as _C
from faster_coco_eval import COCO, COCOeval_faster

ap = argparse.ArgumentParser()
ap.add_argument("gt"); ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--resamples", type=int, default=200); ap.add_argument("--seed", type=int, default=48)
ap.add_argument("--out", default=None)
x = ap.parse_args()
T = time.perf_counter; proc = psutil.Process()
def peak():
    m = proc.memory_info(); return {"peak_commit_gb": round(m.peak_pagefile / 2**30, 2), "rss_gb": round(m.rss / 2**30, 2)}
rec = json.loads(Path(x.gt).read_text(encoding="utf-8"))
index = {f: i for i, f in enumerate(rec["files"])}
log = {"resamples": x.resamples}

def evaluate(cell):
    """One full evaluation keeping per-image results; returns (per-image list, params, stats, seconds)."""
    t = T(); cell = Path(cell)
    preds = json.loads((cell / "predictions.json").read_text(encoding="utf-8"))
    for p in preds: p["image_id"] = index[p.pop("file_name")]
    preds.sort(key=lambda p: p["image_id"])
    G = COCO(rec["gdict"]); D = G.loadRes(preds)
    v = COCOeval_faster(G, D, iouType="bbox", lvis_style=False, print_function=lambda *a: None, separate_eval=True)
    v.params.imgIds = G.getImgIds(); v.params.maxDets = [1, 10, 100, 500]
    v.evaluate(); e, P = v._evalImgs_cpp, v._paramsEval
    v.accumulate(); v.summarize(); s = dict(v.stats_as_dict)
    score = json.loads((cell / "score.json").read_text(encoding="utf-8"))["stats"]
    ok = s["AP_all"] == score["coco/AP_all"] and s["AP_small"] == score["coco/AP_small"]
    return e, P, s, round(T() - t, 1), ok

def ap_stats(ev):
    pr = ev["precision"]; o = []
    for ai in (0, 1):
        y = pr[:, :, :, ai, -1]; o.append(float(np.mean(y[y > -1])))
    return o  # AP_all, AP_small

ea, P, sa, ta, oka = evaluate(x.a); gc.collect(); log["a"] = {"evaluate_s": ta, "AP_all": sa["AP_all"], "AP_small": sa["AP_small"], "matches_score_json": oka, **peak()}
eb, Pb, sb, tb, okb = evaluate(x.b); gc.collect(); log["b"] = {"evaluate_s": tb, "AP_all": sb["AP_all"], "AP_small": sb["AP_small"], "matches_score_json": okb, **peak()}
assert list(P.imgIds) == list(Pb.imgIds), "the two cells do not share an image list"
I, K, A = len(P.imgIds), len(P.catIds), len(P.areaRng)

P2 = copy.deepcopy(P); P2.areaRng = P.areaRng[:2]; P2.areaRngLbl = P.areaRngLbl[:2]; P2.maxDets = [500]  # all + small, maxDets 500 only: exact, 2.3x faster

def accumulate(e, P_, idx):
    out = []
    for k in range(K):
        for a in range(2):
            base = (k * A + a) * I; out.extend(e[base + i] for i in idx)
    return ap_stats(_C.COCOevalAccumulate(P2, out))

# identity: every image once
ia, ib = accumulate(ea, P, list(range(I))), accumulate(eb, Pb, list(range(I)))
log["identity"] = {"a_exact": ia == [sa["AP_all"], sa["AP_small"]], "b_exact": ib == [sb["AP_all"], sb["AP_small"]]}
rng = np.random.default_rng(x.seed); t = T(); d = np.empty((x.resamples, 2)); each = []
for n in range(x.resamples):
    t1 = T(); idx = [int(i) for i in rng.integers(0, I, I)]
    ra, rb = accumulate(ea, P, idx), accumulate(eb, Pb, idx)
    d[n] = np.subtract(ra, rb); each.append(T() - t1)
log["resample_loop_s"] = round(T() - t, 1); log["per_resample_pair_s"] = round(float(np.mean(each)), 2)
for j, name in enumerate(("AP_all", "AP_small")):
    lo, hi = np.percentile(d[:, j], [2.5, 97.5])
    log[f"diff_{name}"] = {"point": (sa, sb)[0][name] - sb[name], "ci95": [float(lo), float(hi)], "excludes_0": bool(lo > 0 or hi < 0), "boot_mean": float(d[:, j].mean()), "boot_sd": float(d[:, j].std(ddof=1))}
log["mem_final"] = peak()
if x.out: Path(x.out).write_text(json.dumps(log, indent=1), encoding="utf-8")
print(json.dumps(log, indent=1))
