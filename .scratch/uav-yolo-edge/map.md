# Map: UAV YOLO edge-compression study

Label: wayfinder:map

## Destination

An **executable implementation spec** for the UAV YOLO11s/YOLO26s edge-compression study — training recipe, compression pipeline, deployment path per platform, and measurement protocol — together with a **revised experiment matrix** honestly sized to **two executors, one month** (from 2026-10-09), and a real GPU-hour budget. Reaching the destination means someone can start executing on day one with nothing left to decide.

**Plan-don't-do override (2026-10-09).** The map no longer withholds execution. The training recipe is locked, so **training runs start now and proceed in parallel with the remaining decision tickets** — training wall-clock is the long pole, and every remaining ticket is downstream of a trained baseline. Exports and benchmarks still wait on their own tickets. Run plan: `study/RUNS.md`.

## Notes

**Domain**: UAV/aerial object detection, model compression, embedded benchmarking. See `CONTEXT.md` at repo root for the glossary.

**Skills every session should consult**: `grilling` + `domain-modeling` by default. `research` for AFK fact-finding. `prototype` where a cheap concrete artifact beats more discussion.

**Settled during charting** (not ticket decisions — scope inputs):

- **Models**: YOLO11s and YOLO26s only. The n-scale variants are dropped by the dev's decision.
- **Teachers**: YOLO11m / YOLO26m, used for distillation only, never deployed.
- **Dataset**: VisDrone2019-DET (6,471 / 548 / 1,610, 10 classes).
- **Platforms**: laptop CPU (owned, RTX 4050 6GB for *training*), Raspberry Pi 5 (owned, primary focus), Samsung Galaxy M35 (owned), Jetson Nano (EEE dept access, tested last).
- **Training compute**: **Kaggle only, accelerator T4 ×2**, two accounts (one per executor) at ~30 GPU-h/wk each, so ~60 h/wk. Colab free is overflow. The RTX 4050 laptop **does not train** — 6 GB plus a Windows paging-file limit killed every probe above 640/batch 8. Kaggle's **P100 is unusable**: the pinned torch is a CUDA 13 build and CUDA 13 dropped Pascal. **GPU-hours are the scarce resource this map allocates.**
- **Timeline**: **one month from 2026-10-09** (deadline ~2026-11-09), **two executors** — the dev and a teammate. The teammate runs the YOLO26s training line and owns Jetson Nano lab access. Benchmarking on Pi/phone/Jetson is covered by the pair.
- **Venue**: journal *or* conference, confirmed acceptable by the supervisor. Venue shortlist is fog until results shape is known.
- **Contribution shape**: (c) *benchmark + a compression recipe specialized for small-object aerial imagery*, with (a) *pure benchmark + deployment guidelines* as the guaranteed floor. (c) is a superset of (a): the same matrix is run either way, so if the small-object effect proves weak, (a) still publishes.
- **(c) is REFRAMED, per the pruning research**: the claim is **not** "we discovered that pruning shallow layers harms small objects" - I-YOLOv11n says that in print. The claim is the **quantification**: a controlled per-pyramid-level sensitivity map with AP-small/medium/large at matched parameters *and matched measured device latency*, which nobody has published. Cite I-YOLOv11n up front; a reviewer in this field will find it, and citing it first reads as thorough rather than uninformed.
- **Report latency, never FLOPs.** TensorRT issue #3884: 20% structured pruning of YOLOv8s produced *zero* FPS gain on a Jetson Nano; NVIDIA ModelOpt reports 34% FLOPs reduction buying 16% latency. Any claim expressed in FLOPs or parameter counts alone is unfalsifiable on real hardware and will be challenged.
- **Proposal deck**: malleable. Research topic is approved; only results are owed. The deck gets revised to match the final matrix.

**Corrections to the deck, established by ticket findings** (the deck is malleable and must be revised):

- The deck calls YOLO26 "purpose-built for hardware-constrained devices" and implies it is lighter. **YOLO26s is larger than YOLO11s** — 10.0M params / 22.8 GFLOPs / 260 layers vs 9.5M / 21.7 / 181 — and two independent sources report it *slower* at equal scale. Any "lighter/faster" claim must go.
- The deck's ~40.1 / ~47.8 mAP50-95 figures (slide 5) are **COCO val2017**, not VisDrone. Expect roughly **40% of the COCO number**. Reset slide 5 before the baseline run, or a perfectly normal ~18 mAP50-95 will read as catastrophic failure:

  | Split | mAP50 | mAP50-95 |
  |---|---|---|
  | val (548) | 38-43 | 23-26 |
  | test-dev (1,610) | 31-35 | 18-21 |

  Concretely, at imgsz 640: YOLO26s test-dev **32.5 / 18.2**, YOLOv8s **31.5 / 17.9** (arXiv 2605.24831 Table 5); YOLO11s val **39.4 / 23.6**.
- **The YOLO11-vs-YOLO26 accuracy narrative is dead.** YOLO26s beats YOLOv8s by +1.0 mAP50 - inside run-to-run noise - and measures *slower* on both CPU and GPU. Combined with the NMS-free finding from ticket 01, the paper's YOLO26 story must be a **systems and deployment** story, never an accuracy one.

**Frozen experimental invariants** (see [Freeze the experimental invariants](issues/15-freeze-experimental-invariants.md); full rationale in `study/INVARIANTS.md`): `ultralytics==8.4.149` (floor 8.4.133), `max_det=500`, `iou=0.7`, `nms=null` (one-to-many head scored for both families), COCO `maxDets=500`; train on `train`, model-select on `val`, headline on `test-dev`, report both columns, never tune on test-dev, never report test-challenge. **Every reported accuracy number is a `coco/*` stat from `study/scripts/studyeval.py`, never from `model.val()`**, which silently drops the area splits.

**Budget GPU-hours to resolution, not epochs.** 100->600 epochs buys ~+4 mAP50 at 6x the cost; 640->960 buys ~+9. This inverts the intuitive allocation.

**Standing preference**: prefer cutting matrix *cells* over cutting *rigor*. A smaller, defensible matrix beats a large one with unmeasured variance.

**Standing preference - keep the phone.** The dev asked explicitly not to cut the Galaxy M35 target: it was in the approved proposal, and the dev is the only one who has the device. When cutting cells, cut *within* the phone's column (fewer configurations, fewer runtimes) before dropping the platform. If a later decision really does force it out, that is the dev's call - put it to them, don't cut it silently.

## Decisions so far

<!-- one line per closed ticket: gist + link -->

- [VisDrone training practice](issues/02-visdrone-training-practice.md): the deck's 40.1/47.8 are verbatim **COCO val2017** figures; real VisDrone s-scale @640 is **31-35 mAP50 / 18-21 mAP50-95 on test-dev** (~40% of the COCO number). AP-small/medium/large **do** come free, but only on **ultralytics >= 8.4.133** with `save_json=True` (PR #25981) - the single most load-bearing dependency in the study. **Resolution beats capacity**: yolov8s@960 (30.8 mAP50-95) beats yolov8l@640 (27.8). The converter is safe; the *evaluation* is not official-toolkit-comparable (`max_det` 300 vs 500, no ignored-region concept), all biasing Ultralytics mAP low.
- [Structured pruning tooling](issues/03-structured-pruning-tooling.md): the small-object claim is **partially taken** - I-YOLOv11n (Sensors 25(15):4857) states the P2-pruning-harms-small-objects hypothesis almost verbatim on UAV imagery, but as *motivation for a method, never as an isolated result*. Nobody has published a controlled per-pyramid-level pruning sweep with AP-small/medium/large at matched parameters. Contribution (c) reframes from **discovery to quantification**. Also: Torch-Pruning DepGraph is the only viable tool but its YOLOv8 example is broken and no YOLO11 example exists; `round_to=8` is mandatory or pruning yields zero latency gain.
- [YOLO26 reality check](issues/01-yolo26-reality-check.md): YOLO26 is a valid study target (ultralytics v8.4.0, AGPL-3.0), but its headline NMS-free head is **auto-disabled by the exporter on every INT8 edge path in this matrix** — NCNN at any precision, LiteRT INT8, TensorRT <8.5 and TRT 10.3+JetPack 6 INT8. Root cause is representational (the head emits a class *index* as a numeric channel), so no calibration fixes it. Also: YOLO26s is **larger** than YOLO11s (10.0M/22.8 GFLOPs vs 9.5M/21.7), first-party feature-based KD shipped v8.4.77, and pruning has **zero** published YOLO26 precedent.
- [Baseline training recipe](issues/05-baseline-training-recipe.md): locked in `study/cfg/train_baseline.yaml`, launched by `study/scripts/studytrain.py`, run plan in `study/RUNS.md`. **imgsz is a study axis only for baselines** (640/960/1280); every compressed configuration runs at the **matrix resolution, 640**. 100 epochs, early stopping off, **SGD pinned** — `optimizer=auto` silently picks MuSGD above 10k iterations and AdamW below (`trainer.py:1145`), which would have given the baseline and its pruning-recovery runs different optimizers. Stock augmentation, written out explicitly. `nbs=64` already accumulates to ~64 images per update, so **per-GPU batch** (not total) is the quantity held constant across machines. **3 seeds per study model at 640** plus free bootstrap resampling, because three tickets need a noise floor. 10 baseline runs total, split by family between the two executors.
- [Freeze the experimental invariants](issues/15-freeze-experimental-invariants.md): pinned and **smoke-verified** on the laptop (`study/`: requirements, `INVARIANTS.md`, `cfg/invariants.yaml`). The research recipe was wrong on evaluation: `model.val()` drops the area splits, the headline mAP uses a different integrator from the splits, and faster-coco-eval scores every AP at **maxDets=100** (19.5% of val images exceed 100 objects). All three are fixed by `study/scripts/studyeval.py`, now the only sanctioned evaluator. Also froze `nms=null` (one-to-many head scored for both families, revisitable without retraining) and the rule that quantized artifacts are compared against a same-format FP32 export, never the `.pt`.

## Not yet specified

- **Venue shortlist and paper structure.** Depends on what the results actually show; premature until the small-object effect is confirmed or denied.
- **Whether the NMS-free-unavailable-at-edge result becomes a contribution in its own right.** The YOLO26 reality check surfaced a code-verified, surprising, and apparently unpublished negative result: the architectural change YOLO26 advertises is precisely the one that quantization tooling rejects, on every INT8 edge path. This may be a *stronger* and more certain contribution than the small-object recipe. Cannot be phrased sharply until the small-object verdict lands and we know whether this is the primary or the secondary claim.
- **Whether YOLO26 stays a pruning target at all.** Two independent findings now point at dropping it: zero published YOLO26 pruning precedent, and ultralytics #23365 reporting the unpruned yolo26n NCNN export already failing. The fallback shape - YOLO11s as the pruning/recipe vehicle, YOLO26s as benchmark-only - would be a coherent study, but it changes what the paper claims about YOLO26. Sharpens once the feasibility trace runs.
- **SAHI / tiled inference as a latency-fair comparison point.** Training at 960-1280 in one forward pass will likely beat 640 + 6-tile SAHI on accuracy-per-millisecond, but that is an assertion until measured. Belongs in related work, or as one clearly-labelled upper-bound row - not in the main matrix.
- **What a fair YOLO11-vs-YOLO26 comparison even means** now that the e2e head is disabled at the edge. If both families run one-to-many heads on the Pi and the phone, the comparison reduces to backbone/neck differences plus training recipe — which needs saying explicitly rather than being discovered by a reviewer.
- **Reproducibility and release.** Repo layout, weights hosting, and the AGPL-3.0 implications of building on Ultralytics for a publicly released artifact.
- **Android harness design.** TFLite vs NCNN on the M35, which timing API, screen-off/thermal discipline, how to isolate inference from app overhead. Sharpens once the measurement protocol is fixed.
- **Statistical treatment — partially settled.** *Baseline training noise* is now decided: 3 seeds per study model at 640, plus bootstrap resampling of test-dev images on every reported evaluation (free, from the saved predictions). Still fog: repeat counts for **compressed** cells, how CIs are reported in the tables, and thermal drift across a long benchmark run. Sharpens once the baseline noise floor is an actual number.
- **QAT beyond PTQ.** The deck says "explored where beneficial" — whether that's worth GPU-hours is unknowable until PTQ damage is measured.
- **An n-scale literature anchor.** Whether to run a single YOLO11n/YOLO26n reference config purely so reviewers can align the numbers with published work, despite n being dropped as a study target.
- **Deck revision.** How and when the proposal slides get rewritten to match the final matrix.

## Out of scope

- **ESP32 / microcontroller targets** (deck refs [22]–[25]) — no board, and no credible path to running YOLO-scale detectors there.
- **Hailo-8 / Hailo-8L NPU on Pi 5** (deck refs [12], [14], [15]) — no accelerator owned. Cited as related work only.
- **YOLO11 segmentation / pose / OBB tasks** — detection only.
- **Deploying m-scale models** — teachers exist to distill from, never as deployment targets.
