"""PROTOTYPE - throwaway, lives only on branch prototype/small-object-effect. Map ticket: Confirm the small-object effect.

Pruning arm, scoring half: evaluate A1's .pt and the recovered P1 (`yolo11s-640-s0-prune30`, 30-epoch recovery on
Kaggle) on val and test-dev through studyeval, and dump predictions + GT for the paired bootstrap. Both sides are
.pt checkpoints, so no export confound (pre-registered reference for this arm: A1's .pt).

  .venv/Scripts/python prototype_small_object/prune_recovered.py
  .venv/Scripts/python prototype_small_object/bootstrap.py a1-pt prune30-recovered --split test
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import studyenv  # noqa: E402  must precede ultralytics
import studyeval  # noqa: E402

CKPT = Path(r"C:\Users\srikr\uav-yolo-checkpoints")
MODELS = [
    ("a1-pt", CKPT / "yolo11s-640-s0" / "best.pt"),
    ("prune30-recovered", CKPT / "yolo11s-640-s0-prune30" / "best.pt"),
]
WORK = studyenv.ROOT / "runs" / "prototype-small-object"
OUT = Path(__file__).resolve().parent / "results"
IMGSZ = 640

# Capture the in-memory COCO GT and predictions studyeval feeds faster-coco-eval, for the bootstrap.
CAPTURE = {}
_orig = studyeval.StudyValidator.coco_evaluate


def _capturing(self, stats, pred_json, anno_json, *args, **kwargs):
    CAPTURE["pred"], CAPTURE["anno"] = pred_json, anno_json
    return _orig(self, stats, pred_json, anno_json, *args, **kwargs)


studyeval.StudyValidator.coco_evaluate = _capturing


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    for name, weights in MODELS:
        for split in ("val", "test"):
            rec_file = OUT / f"{name}-{split}.json"
            if rec_file.exists():
                continue
            t = time.time()
            result = studyeval.evaluate(str(weights), split=split, imgsz=IMGSZ, device="0")
            result.update(artifact=name, weights=str(weights), split=split, eval_seconds=round(time.time() - t))
            rec_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
            (WORK / f"{name}-{split}.coco.json").write_text(json.dumps({"anno": CAPTURE["anno"], "pred": CAPTURE["pred"]}), encoding="utf-8")
            s = result["stats"]
            print(f"[{name} {split}] AP_all {s['coco/AP_all']:.4f} small {s['coco/AP_small']:.4f} medium {s['coco/AP_medium']:.4f} large {s['coco/AP_large']:.4f} ({result['eval_seconds']}s)")


if __name__ == "__main__":
    main()
