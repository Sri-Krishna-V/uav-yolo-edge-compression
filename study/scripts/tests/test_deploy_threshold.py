"""Check the two post-processing passes (#22): the evaluation threshold's detections are saved, the deployment
threshold's are timed. Runs in the study venv (studyscore needs studyenv):

  python study/scripts/tests/test_deploy_threshold.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import studybench as sb  # noqa: E402

FAILED = []


def check(label, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        FAILED.append(label)
        if detail:
            print(f"     {detail}")


def same(a, b):
    """Two post-processing results (boxes, scores, classes) hold the same detections, in the same order, bit for bit."""
    return all(x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y) for x, y in zip(a[:3], b[:3]))


def above(result, conf):
    """The detections of a result that score above conf, in its order."""
    keep = result[1] > np.float32(conf)
    return tuple(x[keep] for x in result[:3])


NO_SCALE = ((1.0, 1.0), (0, 0))  # ratio_pad of an image already at imgsz: scale_boxes only clips


# ---------------------------------------------------------------- 1: a worked example on the one-to-many head
def raw_head(rows, nc=2):
    """A raw head output (4 + nc, anchors) from rows of (cx, cy, w, h, class scores...)."""
    return np.array(rows, dtype=np.float32).reshape(len(rows), 4 + nc).T.copy()


# A: 0.9, class 0. B: 0.3, class 0, IoU 0.82 with A: suppressed. C: 0.2, class 0, far away: kept only at 0.001.
# D: 0.5 class 0 and 0.4 class 1, multi-label: two detections. E: 0.0005, below both thresholds.
WORKED = raw_head([[100, 100, 40, 40, 0.9, 0.0], [102, 101, 40, 40, 0.3, 0.0], [400, 400, 20, 20, 0.2, 0.0],
                   [250, 100, 30, 30, 0.5, 0.4], [500, 500, 10, 10, 0.0005, 0.0]])
saved, t_saved, deploy, t_deploy = sb.two_passes(WORKED.copy(), "raw", NO_SCALE, (640, 640), deploy_first=False)
check("1 raw: the evaluation pass keeps A, D (both classes) and C, and suppresses B",
      saved[1].tolist() == [np.float32(0.9), np.float32(0.5), np.float32(0.4), np.float32(0.2)] and saved[2].tolist() == [0, 0, 1, 0]
      and saved[3] == 5, saved)
check("1 raw: the deployment pass keeps A and D (both classes) only, B being a candidate NMS suppresses",
      deploy[1].tolist() == [np.float32(0.9), np.float32(0.5), np.float32(0.4)] and deploy[2].tolist() == [0, 0, 1] and deploy[3] == 4, deploy)
check("1 raw: both passes are timed", t_saved > 0 and t_deploy > 0, (t_saved, t_deploy))


# ---------------------------------------------------------------- 2: a busy fake output, both heads, both orders
def busy_raw(seed, nc=10, per=60):
    """Overlapping boxes around cluster centres, with scores spread across both thresholds and a second class on some.
    Seed 0 has enough clusters that max_det (500) binds at the evaluation threshold; seeds 1 and 2 stay under it."""
    rng = np.random.default_rng(seed)
    clusters, per = (40, per) if seed == 0 else (12, 20)
    n = clusters * per
    centres = rng.uniform(40, 600, (clusters, 2)).repeat(per, 0)
    xy = centres + rng.normal(0, 4, (n, 2))
    wh = rng.uniform(8, 60, (clusters, 2)).repeat(per, 0) * rng.uniform(0.9, 1.1, (n, 2))
    scores = rng.uniform(0, 0.02 if seed == 0 else 0.0012, (n, nc))  # low-score noise on every class
    main = rng.integers(0, nc, clusters).repeat(per)
    scores[np.arange(n), main] = rng.beta(0.7, 1.5, n)
    second = rng.random(n) < 0.2
    scores[second, (main[second] + 1) % nc] = rng.uniform(0, 0.6, second.sum())
    return np.concatenate((xy, wh, scores), 1).astype(np.float32).T.copy()


def busy_e2e(seed, k=500):
    """A one-to-one head output (k, 6), top-k sorted by score, as the NMS-free export emits it."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 600, (k, 2))
    boxes = np.concatenate((xy, xy + rng.uniform(5, 60, (k, 2))), 1)
    out = np.concatenate((boxes, -np.sort(-rng.beta(0.5, 2.0, k))[:, None], rng.integers(0, 10, (k, 1))), 1).astype(np.float32)
    return out


RATIO_PAD = ((0.5, 0.5), (0, 80))  # a 1280 x 960 image letterboxed to 640: scale and pad both matter
for head, make in (("raw", busy_raw), ("end2end", busy_e2e)):
    for seed in (0, 1, 2):
        y = make(seed)
        before = y.copy()
        alone = sb.postprocess(y, head, sb.CONF, RATIO_PAD, (960, 1280))
        eval_first = sb.two_passes(y, head, RATIO_PAD, (960, 1280), deploy_first=False)
        deploy_first = sb.two_passes(y, head, RATIO_PAD, (960, 1280), deploy_first=True)
        tag = f"2 {head} seed {seed}"
        n_saved, n_deploy = len(alone[1]), len(eval_first[2][1])
        check(f"{tag}: the case is not trivial ({n_saved} saved, {n_deploy} at 0.25)", 0 < n_deploy < n_saved, (n_saved, n_deploy))
        if head == "raw":
            check(f"{tag}: NMS suppressed boxes above 0.25", int((y[4:] > 0.25).sum()) > n_deploy)
        check(f"{tag}: the 0.25 pass gives exactly the saved detections above 0.25 (evaluation pass first)",
              same(eval_first[2], above(eval_first[0], sb.DEPLOY_CONF)))
        check(f"{tag}: the 0.25 pass gives exactly the saved detections above 0.25 (deployment pass first)",
              same(deploy_first[2], above(deploy_first[0], sb.DEPLOY_CONF)))
        check(f"{tag}: the saved detections are those of a pass at 0.001 alone, in either order",
              same(eval_first[0], alone) and same(deploy_first[0], alone) and eval_first[0][3] == alone[3] == deploy_first[0][3])
        check(f"{tag}: neither pass changes the model output", np.array_equal(y, before))


# ---------------------------------------------------------------- 3: the stage summary
COLS = ["t_pre_ms", "t_net_ms", "t_post_ms", "t_post_deploy_ms"]
ROWS = [(2.0, 50.0, 9.0, 1.0), (3.0, 60.0, 11.0, 2.0), (2.5, 55.0, 10.0, 1.5)]
out = Path(tempfile.mkdtemp())
with open(out / "timings.csv", "w", newline="", encoding="utf-8") as f:
    f.write(",".join(sb.COLUMNS) + "\n")
    for i, r in enumerate(ROWS):
        row = {**dict.fromkeys(sb.COLUMNS, ""), "index": i, "file": f"{i}.jpg", "n_cand": 10, "n_dets": 5, "n_dets_deploy": 2,
               "session": 0, **dict(zip(COLS, map(str, r)))}
        f.write(",".join(str(row[c]) for c in sb.COLUMNS) + "\n")
_, stats, _ = sb.stage_stats(out)
check("3 the stage summary has post_deploy: the deployment pass's own time", stats.get("post_deploy", {}).get("median") == 1.5, stats.get("post_deploy"))
check("3 the stage summary has e2e_deploy: pre + net + the deployment pass", stats.get("e2e_deploy", {}).get("median") == 59.0
      and stats["e2e_deploy"]["max"] == 65.0, stats.get("e2e_deploy"))
check("3 the evaluation threshold's end to end time is still there", stats["t_total_ms"]["median"] == 67.5, stats["t_total_ms"])
check("3 the per-image columns are named as agreed", {"t_post_deploy_ms", "n_dets_deploy"} <= set(sb.COLUMNS), sb.COLUMNS)


# ---------------------------------------------------------------- 4: the scoring summary
import studyscore  # noqa: E402  (after the bench checks: it imports studyenv)

META = {"deploy_conf": 0.25, "postprocess": {"conf": 0.001}, "timing_ms": {
    "t_total_ms": {"median": 67.5, "p95": 70.0}, "t_post_ms": {"median": 10.0, "p95": 11.0},
    "e2e_deploy": {"median": 57.5, "p95": 60.0}, "post_deploy": {"median": 1.5, "p95": 2.0}}}
lat = studyscore.latency(META)
check("4 the scoring summary names e2e_deploy as the latency", lat["latency_ms"] == 57.5 and lat["latency_is"] == "e2e_deploy", lat)
check("4 it reports both thresholds", lat["at_deployment_threshold"] == {"conf": 0.25, "e2e_median_ms": 57.5, "e2e_p95_ms": 60.0, "post_median_ms": 1.5}
      and lat["at_evaluation_threshold"] == {"conf": 0.001, "e2e_median_ms": 67.5, "e2e_p95_ms": 70.0, "post_median_ms": 10.0}, lat)
OLD = {"postprocess": {"conf": 0.001}, "timing_ms": {"t_total_ms": {"median": 67.5, "p95": 70.0}, "t_post_ms": {"median": 10.0, "p95": 11.0}}}
lat = studyscore.latency(OLD)
check("4 a run from before #22 has no latency, only the evaluation threshold's time", lat["latency_ms"] is None
      and lat["at_deployment_threshold"] is None and lat["at_evaluation_threshold"]["e2e_median_ms"] == 67.5, lat)

if FAILED:
    sys.exit(f"{len(FAILED)} failed: {FAILED}")
print("all passed")
