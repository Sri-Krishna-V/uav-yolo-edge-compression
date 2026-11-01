"""Does trimming accumulate to areaRng [all, small] and maxDets [500] stay exact and get faster? (#48)"""
import copy, json, sys, time
import numpy as np
from pathlib import Path
import faster_coco_eval.faster_eval_api_cpp as _C
from faster_coco_eval import COCO, COCOeval_faster
gtp, cell = sys.argv[1:3]
rec = json.loads(Path(gtp).read_text(encoding="utf-8")); index = {f: i for i, f in enumerate(rec["files"])}
preds = json.loads((Path(cell) / "predictions.json").read_text(encoding="utf-8"))
for p in preds: p["image_id"] = index[p.pop("file_name")]
preds.sort(key=lambda p: p["image_id"])
G = COCO(rec["gdict"]); D = G.loadRes(preds)
v = COCOeval_faster(G, D, iouType="bbox", lvis_style=False, print_function=lambda *a: None, separate_eval=True)
v.params.imgIds = G.getImgIds(); v.params.maxDets = [1, 10, 100, 500]; v.evaluate()
e, P = v._evalImgs_cpp, v._paramsEval; I, K, A = len(P.imgIds), len(P.catIds), len(P.areaRng)
print("areaRng", P.areaRng, P.areaRngLbl)
def stats(ev):
    pr = ev["precision"]; return [float(np.mean(pr[:, :, :, a, -1][pr[:, :, :, a, -1] > -1])) for a in range(pr.shape[3])]
def full(idx):
    out = []
    for blk in range(K * A): out.extend(e[blk * I + i] for i in idx)
    return _C.COCOevalAccumulate(P, out)
P2 = copy.deepcopy(P); P2.areaRng = P.areaRng[:2]; P2.areaRngLbl = P.areaRngLbl[:2]; P2.maxDets = [500]
def trim(idx):
    out = []
    for k in range(K):
        for a in range(2):
            blk = k * A + a; out.extend(e[blk * I + i] for i in idx)
    return _C.COCOevalAccumulate(P2, out)
rng = np.random.default_rng(1)
for idx in (list(range(I)), [int(i) for i in rng.integers(0, I, I)], [int(i) for i in rng.integers(0, I, I)]):
    t = time.perf_counter(); f = full(idx); t1 = time.perf_counter(); r = trim(idx); t2 = time.perf_counter()
    sf, sr = stats(f), stats(r)
    print(f"full {t1-t:.2f}s trim {t2-t1:.2f}s  all/small equal: {sf[0]==sr[0] and sf[1]==sr[1]}  {sf[:2]} {sr[:2]}")
