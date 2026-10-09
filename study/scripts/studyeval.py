"""The only sanctioned accuracy evaluation in the study. Do not use model.val() for reported numbers.

Verified against ultralytics 8.4.149 / faster-coco-eval 1.8.0 source, three traps in the stock path:
  1. model.val() returns validator.metrics, whose results_dict has no area-split keys. The
     AP-small/medium/large live only in the stats dict the validator returns - and model.val()
     discards it (engine/model.py).
  2. For a custom dataset, Ultralytics writes only AP_small/medium/large into stats and leaves
     metrics/mAP50-95(B) on its native trapezoidal integrator (detect/val.py coco_evaluate), so
     the headline AP and the area splits would come from different integrators.
  3. faster-coco-eval evaluates at maxDets=[1, 10, 100] and computes EVERY AP at 100 detections
     per image, whatever max_det is. 19.5% of val images and 7.9% of test-dev images hold more
     than 100 objects.
evaluate() fixes all three: every coco/* stat comes from one integrator at maxDets=500, matching
the official VisDrone AP@[.50:.95 | maxDets=500] and AR{1,10,100,500} family.
"""

from pathlib import Path

import faster_coco_eval

import studyenv  # noqa: F401  must precede ultralytics

from ultralytics import YOLO  # noqa: E402
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
