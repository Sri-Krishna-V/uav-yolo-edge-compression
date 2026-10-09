# Freeze the experimental invariants

Type: task
Status: resolved
Blocked by: none
Assignee: srikr

## Question

Pin, in writing and in a requirements file, the handful of settings that every later run inherits. Each of these is silently wrong by default, and each would invalidate results discovered weeks later.

Do this before any training run.

1. **`ultralytics>=8.4.133`** - THE MOST LOAD-BEARING PIN IN THE STUDY. AP-small / AP-medium / AP-large for a custom dataset arrived in 8.4.133 (PR #25981), and only with `save_json=True`. On any earlier version the call **silently returns nothing** and the paper central claim would need a hand-rolled pycocotools script. Pin it, record the exact version in the paper, and verify the metric keys actually appear before trusting a run. `faster-coco-eval>=1.6.7` is auto-installed alongside.
2. **`max_det=500`** - Ultralytics defaults to 300; the official VisDrone protocol uses 500; VisDrone averages ~70 objects per image with dense scenes far above that. At 300 the cap silences recall, and the loss gets mis-attributed to compression. Set it now, freeze it, apply it identically to every cell including baselines.
3. **Split discipline** - train on `train`, model-select on `val`, headline on `test-dev` (labels are public, authors recommend it, the challenge server has been dormant since 2024). Report both val and test-dev columns; the gap is a consistent ~6 mAP50. Never tune on test-dev. Never report test-challenge.
4. **Record the evaluation protocol caveat.** Ultralytics evaluation differs from the official MATLAB toolkit on five axes (no ignored-region concept, `max_det`, class averaging among them), all biasing Ultralytics mAP LOW. An Ultralytics `mAP50-95` is NOT the official `AP@[0.50:0.95 | maxDets=500]`. Never put the two in one table without a caveat.
5. **Verify the converter once.** The Ultralytics VisDrone converter has been checked as safe - `score==0` is exactly `category in {0, 11}`, so ignored regions and `others` are dropped correctly - but re-verify on your own download by counting annotation rows per split, because a silently truncated download would look identical to a converter bug.
6. **Pin and record everything else that moves**: torch, CUDA, the export toolchain versions per platform, and the exact dataset download date. A benchmark paper that cannot be rerun is not a benchmark paper.

Resolution is a committed `requirements.txt` (or lockfile), a short `INVARIANTS.md` recording each choice and its justification, and a verified smoke run showing the area-split metric keys present in the output.

## Answer

Resolved 2026-09-12. Everything lives in `study/` at the repo root. The detail is in `study/INVARIANTS.md`; this answer records only what later tickets depend on.

**Deliverables.**

- `study/requirements.txt` (portable pins) and `study/requirements.lock.txt` (full laptop freeze).
- `study/INVARIANTS.md` (every choice with its justification).
- `study/cfg/invariants.yaml` (machine-readable `shared` / `train` / `eval` blocks).
- `study/scripts/`: `studyenv.py`, `studyeval.py`, `fingerprint_visdrone.py`, `smoke.py`.
- Smoke run **passed with 0 failures**; record in `study/verification/smoke_results.json`.
- The workspace is not a git repo, so "committed" means these files on disk.

**Environment.** `ultralytics==8.4.149` (floor 8.4.133, enforced at import), `faster-coco-eval==1.8.0`, `torch 2.14.0+cu130`, Python 3.12.6, venv at `study/.venv`. `YOLO_AUTOINSTALL=false`. Ultralytics settings are isolated to `study/.ultralytics`, because the laptop's global `settings.json` points datasets and runs at an unrelated project, where they would otherwise have landed. Every machine that produces numbers (Kaggle, Colab) runs `smoke.py` once first.

**Dataset.** Downloaded 2026-09-12 14:49 UTC from the Ultralytics mirror. SHA-256 per zip is recorded. Every count matches the research download exactly, and the converted labels match the raw zips row for row. Items 5 and 6 are done.

**Items 1-4 frozen as asked, but the research recipe was wrong in three ways, found by reading the 8.4.149 / faster-coco-eval 1.8.0 source.** All three are silent:

1. **`model.val()` discards the area splits.** Its `results_dict` never contains `mAP_small/medium/large`, contrary to `findings/02-visdrone.md` §5.6, which now carries a correction.
2. **On a custom dataset, `metrics/mAP50-95(B)` stays on Ultralytics' trapezoidal integrator** while the area splits use COCO 101-point. Smoke demonstrated it: 0.0506 vs 0.0521 on the same predictions.
3. **faster-coco-eval computes every AP at `maxDets=100`**, whatever `max_det` is. 19.5% of val and 7.9% of test-dev images have more than 100 objects.

→ **`studyeval.evaluate()` is the only sanctioned evaluator.** Every reported number is a `coco/*` stat at maxDets=500.

**Additions beyond the ticket's list,** each found while verifying:

- `max_det=500` matters more than stated. Left at 300, Ultralytics *silently raises* it to each split's observed maximum (902 / 317 / 461), which gives a different protocol per split. Per-image maxima show 500 never binds at eval time.
- `iou=0.7` and `conf=0.001` are made explicit.
- Eval `rect=True, batch=16`, because rect batching makes batch size move mAP.
- Per-epoch training validation never computes area splits, by design in `validator.py`.
- **`nms=null`: the one-to-many head is scored for both families.** This was a judgment call: it is the head every INT8 edge path runs. It only changes which head is *scored*, never training, so it can be revisited by re-evaluating without retraining. One-to-one numbers appear only as labelled rows.
- **Quantized artifacts are compared against a same-format FP32 export, never the `.pt`.** For static exports the validator forces `rect=False` and batch 1. Noted on *Confirm the small-object effect*.

**Not frozen:** export toolchain versions (NCNN, LiteRT, TensorRT/JetPack, ORT). They depend on *Precision and export matrix*. The rule is frozen instead: pin each platform's toolchain at its first export and add a row to `INVARIANTS.md` §1.3 before recording any benchmark from that platform.

**Surfaced:** the official-toolkit A/B fog graduated into *Official-toolkit A/B: in or out* (`issues/16-official-toolkit-ab.md`), with a sharpened question. Does the protocol gap *move* under compression, since ignored regions sit exactly where AP-small lives?
