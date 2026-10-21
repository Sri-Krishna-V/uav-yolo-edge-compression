"""The only sanctioned accuracy evaluation in the study. Do not use model.val() for reported numbers.

Verified against ultralytics 8.4.149 / faster-coco-eval 1.8.0 source, three traps in the stock path:
  1. model.val() returns validator.metrics, whose results_dict has no area-split keys. The
     AP-small/medium/large live only in the stats dict the validator returns - and model.val()
     discards it (engine/model.py).
  2. For a custom dataset, Ultralytics writes only AP_small/medium/large into stats and leaves
     metrics/mAP50-95(B) on its native trapezoidal integrator (detect/val.py coco_evaluate), so
     the headline AP and the area splits would come from different integrators.
  3. faster-coco-eval evaluates at maxDets=[1, 10, 100] and computes EVERY AP at 100 detections
     per image per category, whatever max_det is. 1.1% of val images and 2.6% of test-dev images
     hold more than 100 objects of one class (19.5% / 7.9% hold more than 100 in total).
evaluate() fixes all three: every coco/* stat comes from one integrator at maxDets=500, matching
the official VisDrone AP@[.50:.95 | maxDets=500] and AR{1,10,100,500} family.

ground_truth() builds, with the validator's own code, the COCO ground truth that evaluate() scores a static export
against. studyscore.py caches it and scores studybench.py detections against it without loading torch:

  python studyeval.py ground-truth --split test --imgsz 640 --out gt.json
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import faster_coco_eval

import studyenv  # noqa: F401  must precede ultralytics

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.models.yolo.detect import DetectionValidator  # noqa: E402


class StudyValidator(DetectionValidator):
    """DetectionValidator that evaluates at the study's maxDets and keeps the whole COCO stat family."""

    def coco_evaluate(self, stats, pred_json, anno_json, iou_types="bbox", suffix="Box"):
        original = faster_coco_eval.COCOeval_faster
        captured = {}

        class Evaluator(original):
            def evaluate(inner, *args, **kwargs):
                inner.params.maxDets = list(studyenv.COCO_MAX_DETS)
                return super().evaluate(*args, **kwargs)

            def summarize(inner, *args, **kwargs):
                result = super().summarize(*args, **kwargs)
                captured["stats"] = dict(inner.stats_as_dict)
                captured["max_dets"] = list(inner.params.maxDets)
                return result

        # detect/val.py imports COCOeval_faster inside coco_evaluate, so it picks this up at call time.
        faster_coco_eval.COCOeval_faster = Evaluator
        try:
            stats = super().coco_evaluate(stats, pred_json, anno_json, iou_types, suffix)
        finally:
            faster_coco_eval.COCOeval_faster = original

        # Ultralytics swallows evaluator exceptions into a warning; an evaluation without area splits must fail loudly.
        if "stats" not in captured:
            raise RuntimeError("faster-coco-eval did not run: area-split metrics are missing. Check the log above.")
        self.coco_max_dets = captured["max_dets"]
        stats.update({f"coco/{k}": v for k, v in captured["stats"].items()})
        return stats


def evaluate(weights, split, imgsz, **overrides):
    """Evaluate a checkpoint on one split under the frozen invariants.

    Any override of an invariant is recorded in validator_args, so a deviation is always visible in the output.
    """
    model = weights if isinstance(weights, YOLO) else YOLO(weights)
    args = {**model.overrides, **studyenv.EVAL_ARGS, "data": studyenv.DATA, "split": split, "imgsz": imgsz, "mode": "val", **overrides}
    validator = StudyValidator(args=args, _callbacks=model.callbacks)
    stats = validator(model=model.model)
    return {
        "stats": {k: float(v) for k, v in stats.items()},
        "validator_args": {k: getattr(validator.args, k) for k in (*studyenv.EVAL_ARGS, "split", "imgsz", "quantize")},
        "coco_max_dets": validator.coco_max_dets,
        "end2end_head_scored": bool(validator.end2end),
        "save_dir": str(Path(validator.save_dir)),
    }


def ground_truth(split, imgsz):
    """The COCO ground truth evaluate() builds for a static export at imgsz (rect=False, batch 1), and the image files in order.

    The validator builds it while it validates, from labels letterboxed and scaled back through each image's ratio_pad,
    so it depends on imgsz and rect at float precision: an object near the 32^2 or 96^2 area bound can change bucket.
    This runs the validator's own code with empty predictions instead of re-deriving it.
    """
    args = {**studyenv.EVAL_ARGS, "data": studyenv.DATA, "split": split, "imgsz": imgsz, "rect": False, "batch": 1,
            "mode": "val", "plots": False, "workers": 0}
    v = StudyValidator(args=args)
    v.training, v.device, v.stride = False, torch.device("cpu"), 32
    v.data = check_det_dataset(v.args.data, split=split)
    v.dataloader = v.get_dataloader(v.data.get(split), 1)
    v.init_metrics(SimpleNamespace(names=v.data["names"], end2end=False))
    for batch in v.dataloader:
        v.update_metrics([{"bboxes": torch.zeros(0, 4), "conf": torch.zeros(0), "cls": torch.zeros(0), "extra": torch.zeros(0, 0)}], batch)
    return v.gdict, [Path(f).name for f in v.dataloader.dataset.im_files]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Write the ground truth evaluate() uses for a static export (studyscore.py calls this).")
    p.add_argument("command", choices=["ground-truth"])
    p.add_argument("--split", default="test")
    p.add_argument("--imgsz", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    gdict, files = ground_truth(a.split, a.imgsz)
    Path(a.out).write_text(json.dumps({"split": a.split, "imgsz": a.imgsz, "rect": False, "files": files, "gdict": gdict}), encoding="utf-8")
