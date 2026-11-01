"""Synthetic tiny probe of faster-coco-eval 1.8.0: what does separate_eval keep, can accumulate take a resampled image list?"""
import copy, sys, numpy as np
from faster_coco_eval import COCO, COCOeval_faster
rng = np.random.default_rng(0)
NI, NC = 30, 3
images = [{"id": i, "width": 640, "height": 480} for i in range(NI)]
anns, dets = [], []
aid = 1
def box():
    w, h = rng.uniform(8, 150, 2); x, y = rng.uniform(0, 400), rng.uniform(0, 300)
    return [x, y, w, h]
for i in range(NI):
    for _ in range(rng.integers(0, 12)):
        b = box(); anns.append({"id": aid, "image_id": i, "category_id": int(rng.integers(0, NC)), "bbox": b, "area": b[2]*b[3], "iscrowd": 0}); aid += 1
        if rng.random() < .7:
            bb = [b[0]+rng.normal(0,3), b[1]+rng.normal(0,3), b[2]*rng.uniform(.8,1.2), b[3]*rng.uniform(.8,1.2)]
            dets.append({"image_id": i, "category_id": anns[-1]["category_id"], "bbox": bb, "score": float(rng.random())})
    for _ in range(rng.integers(0, 10)):
        dets.append({"image_id": i, "category_id": int(rng.integers(0, NC)), "bbox": box(), "score": float(rng.random()*.6)})
gt = {"images": images, "annotations": anns, "categories": [{"id": c, "name": str(c)} for c in range(NC)]}

def run(gt, dets, sep, imgids=None):
    G = COCO(gt); D = G.loadRes(copy.deepcopy(dets))
    v = COCOeval_faster(G, D, iouType="bbox", lvis_style=False, print_function=lambda *a: None, separate_eval=sep)
    v.params.imgIds = G.getImgIds() if imgids is None else imgids
    v.params.maxDets = [1, 10, 100, 500]
    v.evaluate()
    return v
a = run(gt, dets, False); a.accumulate(); a.summarize()
b = run(gt, dets, True)
print("separate_eval -> hasattr evalImgs_cpp:", hasattr(b, "_evalImgs_cpp"), type(b._evalImgs_cpp), "eval(non-sep) keys:", list(a.eval.keys()))
e = b._evalImgs_cpp
print("len", len(e), "expected K*A*I =", NC*4*NI)
try: print("elem type", type(e[0]), [x for x in dir(e[0]) if not x.startswith("_")])
except Exception as ex: print("index err", ex)
print("paramsEval type", type(b._paramsEval))
b.accumulate(); b.summarize()
print("sep vs nonsep AP_small/AP_all", b.stats_as_dict["AP_small"], a.stats_as_dict["AP_small"], b.stats_as_dict["AP_all"], a.stats_as_dict["AP_all"])
# resample via remapped ids (ground truth for the experiment)
idx = rng.integers(0, NI, NI)
new_images, new_anns, new_dets = [], [], []
aid = 1
by_i = {}
for x in anns: by_i.setdefault(x["image_id"], []).append(x)
dby = {}
for x in dets: dby.setdefault(x["image_id"], []).append(x)
for j, i in enumerate(idx):
    new_images.append({"id": j, "width": 640, "height": 480})
    for x in by_i.get(int(i), []): new_anns.append({**x, "id": aid, "image_id": j}); aid += 1
    for x in dby.get(int(i), []): new_dets.append({**x, "image_id": j})
gt2 = {**gt, "images": new_images, "annotations": new_anns}
r = run(gt2, new_dets, False); r.accumulate(); r.summarize(); ref = r.stats_as_dict
print("remap-per-resample AP_small/AP_all", ref["AP_small"], ref["AP_all"])
# re-accumulate on the position-indexed C++ list
pe = b._paramsEval
print("paramsEval imgIds head", list(pe["imgIds"])[:5] if isinstance(pe, dict) else pe.imgIds[:5])

import faster_coco_eval.faster_eval_api_cpp as _C
K, A, I = NC, 4, NI   # areaRng count: all/small/medium/large
A = len(pe.areaRng)
print("A", A, "K", K, "I", I, "len", len(e))
def reindex(e, idx, order):
    out = []
    if order == "kai":
        for k in range(K):
            for a in range(A):
                base = (k*A + a)*I
                out.extend(e[base + int(i)] for i in idx)
    elif order == "aki":
        for a in range(A):
            for k in range(K):
                base = (a*K + k)*I
                out.extend(e[base + int(i)] for i in idx)
    elif order == "ika":
        for i in idx:
            for k in range(K):
                for a in range(A):
                    out.append(e[int(i)*K*A + k*A + a])
    return out
for order in ("kai", "aki", "ika"):
    newe = reindex(e, idx, order)
    pe2 = copy.deepcopy(pe)
    pe2.imgIds = [int(i) for i in idx]
    try:
        ev = _C.COCOevalAccumulate(pe2, newe)
    except Exception as ex:
        print(order, "ERR", ex); continue
    c = copy.copy(b); c.eval = ev; c.params = pe2; c.matched = False
    c.summarize(); s = c.stats_as_dict
    print(order, "AP_small", s["AP_small"], "AP_all", s["AP_all"], "match ref:", s["AP_small"] == ref["AP_small"] and s["AP_all"] == ref["AP_all"])
# identity, and passing duplicates in imgIds with the unmodified list
pe3 = copy.deepcopy(pe); pe3.imgIds = [int(i) for i in idx]
try:
    ev = _C.COCOevalAccumulate(pe3, e); print("original list + dup imgIds: ran; shape", ev["counts"])
except Exception as ex: print("dup imgIds with original list ERR", ex)
