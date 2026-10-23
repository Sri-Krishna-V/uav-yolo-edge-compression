"""Score studybench.py runs on the laptop, with exactly studyeval's COCO settings. Writes score.json per run.

  python studyscore.py bench/pi5/yolo11s-640-s0/640/onnxruntime-int8 [more run directories...]

The platform saved its detections in DetectionValidator.pred_to_json form; this scores them the way evaluate() scores
the same static export, in two halves, because importing torch alone costs 1.35 GB on the laptop and a 710k-detection
evaluation at maxDets 500 needs about as much again:
  ground truth  built once per (split, imgsz) by studyeval.ground_truth(), i.e. by DetectionValidator's own code at
                rect=False, batch 1, in a child process; cached in runs/gt/ and rebuilt when the label cache changes
  evaluation    faster-coco-eval called exactly as StudyValidator.coco_evaluate calls it (detect/val.py coco_evaluate:
                COCO(gt), loadRes, COCOeval_faster bbox, imgIds = every ground-truth image, maxDets from
                cfg/invariants.yaml), in this process, which never imports torch or ultralytics
The result carries the same stats keys evaluate() returns from coco_evaluate (every coco/* and metrics/mAP_*(B)).
It also carries the cell's runtime and the precision the platform recorded (what the runtime really computed in),
so the tables give precision a column of its own, and its time at both confidence thresholds: latency_ms is the median
of e2e_deploy, pre + net + post at the deployment threshold (0.25); at_evaluation_threshold is the same at 0.001, the
threshold the scored detections come from.
On Windows both halves run under studycap's memory cap and timeout (#20): each run is scored in its own capped child
process, which holds the ground-truth build too. A step that goes over 4 GB or 60 minutes (--mem-cap-gb, --timeout-min)
is stopped, and the scorer exits saying which run and which limit. Elsewhere nothing changes.
A --limit run is scored only on the images it ran and is marked partial: a plumbing check, never a reported number.
"""

import argparse
import csv
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

import studycap
import studyenv

GT_DIR = studyenv.ROOT / "runs" / "gt"


def label_cache_hash(split):
    """The hash Ultralytics stores in labels/<split>.cache: it changes whenever a label or image file changes."""
    cache = studyenv.VISDRONE_DIR / "labels" / f"{split}.cache"
    return np.load(str(cache), allow_pickle=True).item()["hash"] if cache.exists() else None


def ground_truth(split, imgsz):
    """The cached ground truth for (split, imgsz), built by studyeval.ground_truth() in a child process when missing or stale."""
    path = GT_DIR / f"visdrone-{split}-{imgsz}.json"
    want = {"label_cache_hash": label_cache_hash(split), "ultralytics": version("ultralytics")}
    if path.exists():
        rec = json.loads(path.read_text(encoding="utf-8"))
        if all(rec.get(k) == v for k, v in want.items()):
            return rec
    GT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    argv = [str(Path(__file__).with_name("studyeval.py")), "ground-truth", "--split", split, "--imgsz", str(imgsz), "--out", str(tmp)]
    code = studycap.run(argv, f"the {split} ground truth at {imgsz}")
    if code:
        raise subprocess.CalledProcessError(code, [sys.executable, *argv])
    rec = {**json.loads(tmp.read_text(encoding="utf-8")), **want, "label_cache_hash": label_cache_hash(split)}
    path.write_text(json.dumps(rec), encoding="utf-8")
    tmp.unlink()
    return rec


def coco_evaluate(anno_json, pred_json):
    """faster-coco-eval as StudyValidator.coco_evaluate runs it; returns the keys that method adds to the stats."""
    from faster_coco_eval import COCO, COCOeval_faster

    anno = COCO(anno_json)
    pred = anno.loadRes(pred_json)
    val = COCOeval_faster(anno, pred, iouType="bbox", lvis_style=False, print_function=print)
    val.params.imgIds = anno.getImgIds()
    val.params.maxDets = list(studyenv.COCO_MAX_DETS)  # what StudyValidator patches in at evaluate()
    val.evaluate()
    val.accumulate()
    val.summarize()
    s = dict(val.stats_as_dict)
    return {**{f"metrics/mAP_{k}(B)": s[f"AP_{k}"] for k in ("small", "medium", "large")}, **{f"coco/{k}": v for k, v in s.items()},
            "_max_dets": list(val.params.maxDets)}


def score(predictions, split, imgsz, images=None):
    """Score one predictions.json against the ground truth evaluate() would use for a static export at imgsz.

    `images` (file names) restricts the ground truth to the images the platform ran (a --limit check).
    """
    gt_rec = ground_truth(split, imgsz)
    index = {f: i for i, f in enumerate(gt_rec["files"])}
    gt = gt_rec["gdict"]
    if images is not None:
        keep = {index[f] for f in images}
        gt = {**gt, "images": [x for x in gt["images"] if x["id"] in keep], "annotations": [x for x in gt["annotations"] if x["image_id"] in keep]}
    with open(predictions, encoding="utf-8") as f:
        preds = json.load(f)
    for p in preds:
        p["image_id"] = index.get(p.pop("file_name"))
        if p["image_id"] is None or (images is not None and p["image_id"] not in keep):
            raise ValueError(f"{predictions}: a detection for an image outside the scored {split} images")
    preds.sort(key=lambda p: p["image_id"])  # dataset order, each image's detections in NMS order, as the validator emits them
    stats = coco_evaluate(gt, preds)
    return {
        "stats": {k: float(v) for k, v in stats.items() if not k.startswith("_")},
        "scoring_args": {**studyenv.EVAL_ARGS, "split": split, "imgsz": imgsz, "rect": False, "batch": 1},
        "coco_max_dets": stats["_max_dets"],
        "images": len(gt["images"]),
        "detections": len(preds),
        "partial": images is not None,
    }


def latency(meta):
    """A run's meta.json -> its latency (e2e_deploy's median, None for a run from before #22) and its time at both thresholds."""
    t = meta["timing_ms"]

    def at(conf, e2e, post):
        return {"conf": conf, "e2e_median_ms": t[e2e]["median"], "e2e_p95_ms": t[e2e]["p95"], "post_median_ms": t[post]["median"]}

    deploy = at(meta["deploy_conf"], "e2e_deploy", "post_deploy") if "e2e_deploy" in t else None
    return {"latency_ms": deploy and deploy["e2e_median_ms"], "latency_is": "e2e_deploy" if deploy else None,
            "at_deployment_threshold": deploy, "at_evaluation_threshold": at(meta["postprocess"]["conf"], "t_total_ms", "t_post_ms")}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("runs", nargs="+", help="studybench.py --out directories")
    p.add_argument("--split", default="test", help="the split the platform ran (test = test-dev)")
    studycap.add_arguments(p)
    a = p.parse_args()
    studycap.use(a)
    if studycap.applies():  # each run is scored in its own capped process, its ground truth and its evaluation both
        for run in a.runs:
            code = studycap.run([str(Path(__file__).resolve()), run, "--split", a.split], f"scoring {run}")
            if code:
                sys.exit(code)
        return
    for run in map(Path, a.runs):
        meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
        if meta.get("mode") == "energy":
            raise SystemExit(f"{run}: an energy run (#26) saves no detections and gives no latency: its numbers are in its session's summary.csv")
        if meta.get("status") != "complete":
            raise SystemExit(f"{run}: the run is not complete (resume it with studybench.py --resume)")
        drift = {k: (meta["postprocess"].get(k), studyenv.EVAL_ARGS[k]) for k in ("conf", "iou", "max_det") if meta["postprocess"].get(k) != studyenv.EVAL_ARGS[k]}
        if meta.get("deploy_conf", studyenv.DEPLOY_CONF) != studyenv.DEPLOY_CONF:
            drift["deploy_conf"] = (meta["deploy_conf"], studyenv.DEPLOY_CONF)
        if drift:
            raise SystemExit(f"{run}: post-processing differs from the invariants (platform, invariant): {drift}")
        with open(run / "timings.csv", newline="", encoding="utf-8") as f:
            ran = {r["file"] for r in csv.DictReader(f)}
        split_files = set(ground_truth(a.split, meta["imgsz"])["files"])
        if meta.get("complete_testdev") and ran != split_files:
            raise SystemExit(f"{run}: the platform ran {len(ran)} images, the {a.split} split has {len(split_files)}")
        result = score(run / "predictions.json", a.split, meta["imgsz"], images=None if ran == split_files else sorted(ran))
        result.update(run=str(run), platform_tag=meta.get("platform_tag"), runtime=meta["config"]["runtime"], precision=meta.get("precision"),
                      head=meta.get("head"), manifest=meta.get("manifest"), **latency(meta))
        (run / "score.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        s, ev, dep = result["stats"], result["at_evaluation_threshold"], result["at_deployment_threshold"]
        print(f"{run} ({result['runtime']} {result['precision'] or 'precision unrecorded'}): AP_all {s['coco/AP_all']:.4f}  AP_50 {s['coco/AP_50']:.4f}  AP_small {s['coco/AP_small']:.4f}  "
              f"AP_medium {s['coco/AP_medium']:.4f}  AP_large {s['coco/AP_large']:.4f}  AR_500 {s['coco/AR_500']:.4f}"
              + (f"  (PARTIAL: {result['images']} images)" if result["partial"] else ""))
        print(f"    latency (e2e_deploy, conf {dep['conf']}) {dep['e2e_median_ms']} ms, post {dep['post_median_ms']} ms; " if dep else
              "    no latency: this run is from before the deployment pass (#22); ",
              f"at the evaluation threshold (conf {ev['conf']}): pre + net + post {ev['e2e_median_ms']} ms, post {ev['post_median_ms']} ms", sep="")


if __name__ == "__main__":
    main()
