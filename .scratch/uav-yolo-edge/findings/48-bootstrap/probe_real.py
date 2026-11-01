"""Probe #48: evaluate one real cell once (separate_eval=True keeps per-image results), then re-accumulate on resampled
image lists. Run it under the memory cap through run_capped.py, not directly.

  probe_real.py CELL_DIR GT_JSON [--resamples N] [--out result.json] [--verify-remap]
"""
import argparse, copy, json, sys, time
import numpy as np
import psutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("cell"); ap.add_argument("gt")
ap.add_argument("--resamples", type=int, default=5)
ap.add_argument("--out", default=None)
ap.add_argument("--verify-remap", action="store_true", help="one resample both ways: re-accumulate vs remapped-id full evaluate")
a = ap.parse_args()
proc = psutil.Process()
def mem():
    m = proc.memory_info()
    return {"rss_gb": round(m.rss / 2**30, 2), "peak_wset_gb": round(m.peak_wset / 2**30, 2), "peak_commit_gb": round(m.peak_pagefile / 2**30, 2)}
log = {}
T = time.perf_counter
cell = Path(a.cell)
import faster_coco_eval.faster_eval_api_cpp as _C
from faster_coco_eval import COCO, COCOeval_faster

# ---- load, as studyscore.score() does
t = T()
rec = json.loads(Path(a.gt).read_text(encoding="utf-8"))
index = {f: i for i, f in enumerate(rec["files"])}
gt = rec["gdict"]
preds = json.loads((cell / "predictions.json").read_text(encoding="utf-8"))
for p in preds:
    p["image_id"] = index[p.pop("file_name")]
preds.sort(key=lambda p: p["image_id"])
log["load_s"] = round(T() - t, 1); log["n_dets"] = len(preds); log["mem_after_load"] = mem()
MAXDETS = [1, 10, 100, 500]

def make(gt, preds, separate):
    G = COCO(gt); D = G.loadRes(preds)
    v = COCOeval_faster(G, D, iouType="bbox", lvis_style=False, print_function=lambda *x: None, separate_eval=separate)
    v.params.imgIds = G.getImgIds(); v.params.maxDets = list(MAXDETS)
    return v

# ---- one full evaluate, keeping per-image results
t = T(); v = make(gt, preds, True); log["coco_load_s"] = round(T() - t, 1); log["mem_after_coco"] = mem()
t = T(); v.evaluate(); log["evaluate_s"] = round(T() - t, 1); log["mem_after_evaluate"] = mem()
e = v._evalImgs_cpp
P = v._paramsEval
I, K, A = len(P.imgIds), len(P.catIds), len(P.areaRng)
log["evalImgs"] = {"type": type(e).__name__, "len": len(e), "I": I, "K": K, "A": A, "elem": type(e[0]).__name__, "len_is_KAI": len(e) == K * A * I}

# ---- the production accumulate+summarize, and the identity check against score.json
t = T(); v.accumulate(); v.summarize(); log["accumulate_summarize_s"] = round(T() - t, 2)
s = dict(v.stats_as_dict)
score = json.loads((cell / "score.json").read_text(encoding="utf-8"))["stats"]
log["identity_production_path"] = {k: [s[k[5:]], score[k]] for k in score if k.startswith("coco/")}
log["identity_production_path_exact"] = all(s[k[5:]] == score[k] for k in score if k.startswith("coco/"))

def stat(ev, params):
    """AP_all and AP_small straight from the accumulated precision [T,R,K,A,M], as _summarize does (maxDets 500, last M)."""
    prec = ev["precision"]
    out = {}
    for name, ai in (("AP_all", 0), ("AP_small", 1)):
        x = prec[:, :, :, ai, -1]
        out[name] = float(np.mean(x[x > -1]))
    return out

def reaccumulate(idx):
    """Re-accumulate on image positions idx (a draw with replacement). The C++ list is laid out [k][a][image position]."""
    out = []
    for blk in range(K * A):
        base = blk * I
        out.extend(e[base + i] for i in idx)
    return _C.COCOevalAccumulate(P, out)

# identity through the re-accumulate path (every image once)
t = T(); ev = reaccumulate(list(range(I))); ident_s = T() - t
r = stat(ev, P)
log["identity_reaccumulate"] = {**r, "score_AP_all": score["coco/AP_all"], "score_AP_small": score["coco/AP_small"],
                                "exact": r["AP_all"] == score["coco/AP_all"] and r["AP_small"] == score["coco/AP_small"]}
log["identity_reaccumulate_s"] = round(ident_s, 2); log["mem_after_identity"] = mem()

# ---- timed resamples
rng = np.random.default_rng(48)
rows = []
for n in range(a.resamples):
    idx = rng.integers(0, I, I)
    t0 = T(); out = []
    for blk in range(K * A):
        base = blk * I
        out.extend(e[base + int(i)] for i in idx)
    t1 = T(); ev = _C.COCOevalAccumulate(P, out); t2 = T(); r = stat(ev, P); t3 = T()
    del out
    rows.append({"reindex_s": round(t1 - t0, 3), "cpp_accumulate_s": round(t2 - t1, 2), "stat_s": round(t3 - t2, 3), **r, "mem": mem()})
log["resamples"] = rows
log["mem_final"] = mem()

# ---- optional: the same resample by remapping ids and running a full evaluate (second process-sized allocation)
if a.verify_remap:
    idx = np.random.default_rng(7).integers(0, I, I)
    ref_re = stat(reaccumulate(idx), P)
    imgids = list(P.imgIds)
    by_gt, by_dt = {}, {}
    for x in gt["annotations"]: by_gt.setdefault(x["image_id"], []).append(x)
    for x in preds: by_dt.setdefault(x["image_id"], []).append(x)
    del v, e
    ngt, ndt, nim, aid = [], [], [], 1
    for j, i in enumerate(idx):
        src = imgids[int(i)]
        nim.append({"id": j})
        for x in by_gt.get(src, []): ngt.append({**x, "id": aid, "image_id": j}); aid += 1
        for x in by_dt.get(src, []): ndt.append({**x, "image_id": j})
    gt2 = {**gt, "images": nim, "annotations": ngt}
    ndt.sort(key=lambda p: p["image_id"])
    t = T(); v2 = make(gt2, ndt, False); v2.evaluate(); v2.accumulate(); v2.summarize(); full_s = T() - t
    s2 = dict(v2.stats_as_dict)
    log["verify_remap"] = {"reaccumulate": ref_re, "remapped_full_evaluate": {"AP_all": s2["AP_all"], "AP_small": s2["AP_small"]},
                           "exact": ref_re["AP_all"] == s2["AP_all"] and ref_re["AP_small"] == s2["AP_small"],
                           "full_remap_evaluate_s": round(full_s, 1), "mem": mem()}
if a.out:
    Path(a.out).write_text(json.dumps(log, indent=1), encoding="utf-8")
print(json.dumps(log, indent=1))
