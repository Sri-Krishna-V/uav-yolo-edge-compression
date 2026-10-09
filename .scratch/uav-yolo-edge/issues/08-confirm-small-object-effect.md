# Confirm the small-object effect

Type: prototype
Status: open
Blocked by: 05

## Question

Does naive INT8 quantization degrade SMALL-OBJECT mAP disproportionately on VisDrone? The paper central claim depends on the answer, so buy the evidence cheaply and early.

This is the highest-value ticket on the map. Contribution (c), a compression recipe specialized for small-object aerial imagery, is only worth framing the paper around if the effect is real and large. If it is weak, the effort falls back to contribution (a), the pure benchmark, and nothing is lost PROVIDED we learn this in week 2 rather than week 6.

The cheap experiment:

1. Take one trained YOLO11s VisDrone baseline from the Baseline training recipe ticket. **Concretely: run A1, `yolo11s-640-s0`, at imgsz 640** (`study/RUNS.md`). It is the first run launched precisely because this ticket is the critical path; nothing else in the baseline set is needed here.
2. Apply naive PTQ INT8 with default calibration.
3. Evaluate both FP32 and INT8 with COCO-style area-split metrics - AP-small, AP-medium, AP-large - not just aggregate mAP.
4. Compare the RELATIVE degradation per area bucket.

The signal we are looking for: aggregate mAP drops a few points, but AP-small drops far more, proportionally. That asymmetry is the opening of the paper.

The pruning arm is now **required, not optional**: run the same comparison after a naive uniform structured prune, with round_to=8. The pruning research showed the small-object hypothesis is already stated in print for PRUNING (I-YOLOv11n) but NOT for QUANTIZATION - so the quantization arm is the more novel of the two, and having both is what separates a measured phenomenon from a one-off.

Note the reframing: this ticket no longer has to prove the effect is UNKNOWN, since I-YOLOv11n asserts it for pruning on UAV imagery. It has to establish that the effect is **real in our setup and large enough to be worth quantifying** - which is what Scale-aware pruning recipe then does properly.

**Evaluation constraint, from *Freeze the experimental invariants*:** every number goes through `study/scripts/studyeval.py`, never `model.val()`, which drops the area splits. And **the FP32 reference for step 3 is an FP32 export of the same format at the same imgsz, not the `.pt` checkpoint**. For static exports the validator forces `rect=False` and batch 1, so `.pt`-vs-INT8 would confound letterboxing with quantization. See `study/INVARIANTS.md` §4.4.

Resolution records the numbers, the verdict (effect confirmed / weak / absent), and which contribution shape the paper commits to as a result. If confirmed, it also opens the follow-on question of WHICH LAYERS are responsible, which is the basis for the proposed recipe.
