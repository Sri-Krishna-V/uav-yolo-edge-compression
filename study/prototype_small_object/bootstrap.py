"""PROTOTYPE - throwaway, lives only on branch prototype/small-object-effect. Map ticket: Confirm the small-object effect.

Paired bootstrap over test-dev images for a (reference, compressed) pair of quant_arm.py evaluations, and the
pre-registered verdict. The same resampled image set scores both artifacts in every replicate.

  .venv/Scripts/python prototype_small_object/bootstrap.py onnx-fp32 onnx-int8 [--split test] [--B 1000]

Re-running faster-coco-eval per replicate takes minutes per replicate at 500 dets/image, so this re-implements
COCO bbox matching once per image (evaluateImg semantics: area-range gt ignores, non-ignored gts preferred,
unmatched out-of-range dets ignored) and re-accumulates per replicate with image multiplicities as weights.
The unweighted point estimate must reproduce studyeval's coco/* stats; main() checks it to 5e-4.

Relative drop per bucket = (ref - comp) / ref. Verdict (pre-registered on the ticket, 2026-10-15):
  confirmed  small drop >= 2x all drop AND the 95% CI of (small drop - all drop) excludes 0
  weak       ratio in [1.3, 2) or the CI touches 0
  absent     ratio < 1.3
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import studyenv  # noqa: E402

WORK = studyenv.ROOT / "runs" / "prototype-small-object"
OUT = Path(__file__).resolve().parent / "results"
IOU_T = np.linspace(0.5, 0.95, 10)
REC_T = np.linspace(0.0, 1.0, 101)
AREAS = {"AP_all": (0, 1e10), "AP_small": (0, 32**2), "AP_medium": (32**2, 96**2), "AP_large": (96**2, 1e10)}


def iou(d, g):
    """COCO bbox IoU, xywh boxes, no crowd."""
    dx2, dy2, gx2, gy2 = d[:, 0] + d[:, 2], d[:, 1] + d[:, 3], g[:, 0] + g[:, 2], g[:, 1] + g[:, 3]
    iw = np.clip(np.minimum(dx2[:, None], gx2[None]) - np.maximum(d[:, None, 0], g[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(dy2[:, None], gy2[None]) - np.maximum(d[:, None, 1], g[None, :, 1]), 0, None)
    inter = iw * ih
    return inter / (d[:, 2:].prod(1)[:, None] + g[:, 2:].prod(1)[None] - inter)


def match(gb, ga, db, da):
    """Per area range: (tp[T, D], fp[T, D], n_non_ignored_gt) for dets already sorted by score."""
    ious = iou(db, gb) if len(gb) and len(db) else np.zeros((len(db), len(gb)))
    cand = np.flatnonzero(ious.max(1, initial=0) >= IOU_T[0]) if len(gb) else []
    out = {}
    for key, (lo, hi) in AREAS.items():
        g_ig = (ga < lo) | (ga > hi)
        dtm = np.zeros((len(IOU_T), len(db)), bool)
        dt_ig = np.zeros_like(dtm)
        gtm = np.zeros((len(IOU_T), len(gb)), bool)
        for d in cand:
            row = ious[d]
            for ti, t in enumerate(IOU_T):
                ok = ~gtm[ti] & (row >= min(t, 1 - 1e-10))
                for pool in (ok & ~g_ig, ok & g_ig):  # non-ignored gts first, as COCO's sorted loop does
                    if pool.any():
                        m = np.flatnonzero(pool)[np.argmax(row[pool])]
                        gtm[ti, m], dtm[ti, d], dt_ig[ti, d] = True, True, g_ig[m]
                        break
        dt_ig |= ~dtm & ((da < lo) | (da > hi))[None]
        out[key] = (dtm & ~dt_ig, ~dtm & ~dt_ig, int((~g_ig).sum()))
    return out


def prepare(name, split):
    """Match every (image, category) once. Returns image ids, npig[area][img, cat], per-area/cat sorted det arrays."""
    d = json.loads((WORK / f"{name}-{split}.coco.json").read_text(encoding="utf-8"))
    imgs = sorted(i["id"] for i in d["anno"]["images"])
    cats = sorted(c["id"] for c in d["anno"]["categories"])
    ii, ci = {k: n for n, k in enumerate(imgs)}, {k: n for n, k in enumerate(cats)}
    gt, dt = defaultdict(list), defaultdict(list)
    for a in d["anno"]["annotations"]:
        gt[a["image_id"], a["category_id"]].append((a["bbox"], a["area"]))
    for p in d["pred"]:
        dt[p["image_id"], p["category_id"]].append((p["bbox"], p["score"]))
    del d
    npig = {k: np.zeros((len(imgs), len(cats))) for k in AREAS}
    dets = {k: defaultdict(list) for k in AREAS}  # area -> cat -> [(scores, img, tp, fp)] in image order
    for img in imgs:
        for cat in cats:
            g, p = gt.get((img, cat), []), dt.get((img, cat), [])
            if not g and not p:
                continue
            gb = np.array([x[0] for x in g], float).reshape(-1, 4)
            ga = np.array([x[1] for x in g], float)
            sc = np.array([x[1] for x in p], float)
            order = np.argsort(-sc, kind="mergesort")
            db = np.array([x[0] for x in p], float).reshape(-1, 4)[order]
            sc = sc[order]
            for key, (tp, fp, n) in match(gb, ga, db, db[:, 2] * db[:, 3]).items():
                npig[key][ii[img], ci[cat]] = n
                if len(sc):
                    dets[key][ci[cat]].append((sc, np.full(len(sc), ii[img]), tp, fp))
    flat = {}
    for key in AREAS:
        flat[key] = {}
        for c, parts in dets[key].items():
            sc = np.concatenate([x[0] for x in parts])
            order = np.argsort(-sc, kind="mergesort")  # COCO accumulate: stable sort over image-ordered dets
            flat[key][c] = (np.concatenate([x[1] for x in parts])[order],
                            np.concatenate([x[2] for x in parts], 1)[:, order], np.concatenate([x[3] for x in parts], 1)[:, order])
    return imgs, len(cats), npig, flat


def accumulate(prep, w):
    """COCO AP (101-point, mean over IoU thresholds and categories with gts) per area, image weights w."""
    _, ncat, npig, flat = prep
    out = {}
    for key in AREAS:
        aps = []
        for c in range(ncat):
            n = float(w @ npig[key][:, c])
            if n == 0:
                continue
            if c not in flat[key]:
                aps.append(0.0)
                continue
            img, tp, fp = flat[key][c]
            ww = w[img]
            tps, fps = np.cumsum(tp * ww, 1), np.cumsum(fp * ww, 1)
            rc, pr = tps / n, tps / (tps + fps + np.spacing(1))
            pr = np.maximum.accumulate(pr[:, ::-1], 1)[:, ::-1]
            q = np.zeros((len(IOU_T), len(REC_T)))
            for t in range(len(IOU_T)):
                idx = np.searchsorted(rc[t], REC_T, side="left")
                ok = idx < pr.shape[1]
                q[t, ok] = pr[t, idx[ok]]
            aps.append(q.mean())
        out[key] = float(np.mean(aps))
    return out


def drops(r, c):
    return {k: (r[k] - c[k]) / r[k] for k in AREAS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ref")
    p.add_argument("comp")
    p.add_argument("--split", default="test")
    p.add_argument("--B", type=int, default=1000)
    a = p.parse_args()
    ref, comp = prepare(a.ref, a.split), prepare(a.comp, a.split)
    assert ref[0] == comp[0], "the two evaluations scored different images"
    n = len(ref[0])

    point = accumulate(ref, np.ones(n)), accumulate(comp, np.ones(n))
    for name, pt in zip((a.ref, a.comp), point):  # the re-implementation must reproduce studyeval
        s = json.loads((OUT / f"{name}-{a.split}.json").read_text(encoding="utf-8"))["stats"]
        gap = max(abs(pt[k] - s[f"coco/{k}"]) for k in AREAS)
        print(f"{name}: max |reimpl - studyeval| = {gap:.2e}")
        assert gap < 5e-4, f"re-implementation disagrees with studyeval for {name}: {pt} vs coco/* stats"

    rng = np.random.default_rng(0)
    reps = []
    for _ in range(a.B):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        reps.append(drops(accumulate(ref, w), accumulate(comp, w)))
    d0 = drops(*point)
    diff = np.array([d["AP_small"] - d["AP_all"] for d in reps])
    ratio = np.array([d["AP_small"] / d["AP_all"] for d in reps])
    ci = np.percentile(diff, [2.5, 97.5]).tolist()
    r0 = d0["AP_small"] / d0["AP_all"]
    excludes0 = ci[0] > 0 or ci[1] < 0
    verdict = "confirmed" if r0 >= 2 and excludes0 else "absent" if r0 < 1.3 else "weak"
    out = {
        "ref": a.ref, "comp": a.comp, "split": a.split, "B": a.B,
        "ref_stats": point[0], "comp_stats": point[1],
        "relative_drop": d0,
        "relative_drop_ci95": {k: np.percentile([d[k] for d in reps], [2.5, 97.5]).tolist() for k in AREAS},
        "small_over_all_ratio": r0, "ratio_ci95": np.percentile(ratio, [2.5, 97.5]).tolist(),
        "small_minus_all_drop_ci95": ci, "verdict": verdict,
    }
    (OUT / f"bootstrap-{a.ref}-vs-{a.comp}-{a.split}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
