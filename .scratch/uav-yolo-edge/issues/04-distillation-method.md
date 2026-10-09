# Distillation method

Type: grilling
Status: open
Blocked by: none

## Question

Is knowledge distillation worth its GPU-hours in this study, and if so, on what terms?

UNBLOCKED and substantially narrowed by the YOLO26 reality check, which settled most of what this ticket originally asked:

- Ultralytics shipped **first-party knowledge distillation in v8.4.77** (2026-06-25, PR #24894). It is **feature-based on the three neck layers**, therefore head-agnostic, therefore unaffected by the NMS-free question. Same-family teacher/student is already required, which matches the map.
- Expected gain is only about **+0.6 mAP**.
- `reg_max=1` in YOLO26 makes **LD / DFL-distribution distillation inapplicable** to it. That method would work on YOLO11 only, which would make "distilled" mean different things for the two families - a confound, not a comparison.

So the remaining decisions are:

1. **Is +0.6 mAP resolvable at all?** If run-to-run variance on VisDrone exceeds the effect size, the `distilled` and `combined` cells produce a null result dressed as a finding. Decide the repeat count needed to resolve +0.6 mAP, then decide whether that many runs is affordable. **If it is not affordable, the honest move is to cut distillation from the matrix entirely** and say why - that is a legitimate, defensible scope cut, not a failure.
2. **Use first-party KD only, or add a small-object-targeted variant?** The paper claim is about small objects. Feature-based KD on the high-resolution P3 neck layer is exactly where small-object information lives, so a P3-weighted distillation loss is the natural specialization. Decide whether that is in scope as part of contribution (c) or out.
3. **Ordering** - prune-then-distill, distill-then-prune, or joint, for the `combined` cell.
4. **Teacher cost** - YOLO11m and YOLO26m must themselves be fine-tuned on VisDrone first. That is 2 extra training runs of the most expensive models in the study, and it is pure overhead if decision 1 goes against distillation. Cost this before committing.
