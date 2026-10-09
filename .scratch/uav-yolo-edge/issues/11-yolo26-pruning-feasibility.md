# YOLO26 pruning feasibility

Type: prototype
Status: open
Blocked by: 03

## Question

Can YOLO26s be structurally pruned at all? De-risk this in week 1, because there is no precedent anywhere.

The YOLO26 reality check found **zero published YOLO26 pruning precedent**: Ultralytics has no official pruning (PR #21977 closed unmerged), and Torch-Pruning has no worked example past YOLOv8. Every other ticket assumes the `pruned` and `combined` configurations exist for both families. If YOLO26 cannot be pruned, a quarter of the matrix vanishes and the study becomes YOLO11-only for two of its five configurations - which changes the paper.

The pruning research raised the risk further:

- The Torch-Pruning YOLOv8 example is **broken** (crashed due to an ultralytics upgrade) and there is **no YOLO11 example at all**. The C2f-to-C2f_v2 surgery must be re-derived for C3k2 and C2PSA. Budget engineering days, not a config change.
- **ultralytics #23365** reports the UNPRUNED yolo26n NCNN export already failing on current ultralytics. Verify the unpruned YOLO26s-to-NCNN path works before spending anything on pruning YOLO26s; if the baseline export is broken, pruning it is moot.
- Use **round_to=8** in every trace and prune from the first experiment, so feasibility is judged on a configuration that could actually yield latency gains.

Cheap experiment, hours not days:

1. Run a Torch-Pruning **DepGraph trace** on an untrained `yolo26s` and see whether the dependency graph resolves across the dual-head fork.
2. Prune a trivial amount (say 5%) and confirm the model still forward-passes and still exports to ONNX.
3. If the fork defeats DepGraph, try the documented fallback: call `fuse()` with `end2end=False` first, which collapses the dual-head fork and should make the graph identical to YOLO11 - then retrace.

Resolution records: does it work, via which path, and what the fallback costs. If neither path works, this becomes a scope decision on *Final matrix and schedule* rather than an engineering problem to grind on.

**Order the checks by cost**: unpruned YOLO26s NCNN export first (minutes, and #23365 says it may already fail), then the DepGraph trace, then the fallback. A failure at step one settles the question without touching Torch-Pruning at all.

The live alternative if this fails: **YOLO11s as the pruning and recipe vehicle, YOLO26s as benchmark-only**. That is a coherent study, not a retreat, but it changes what the paper can claim about YOLO26, so it is a scope decision for the map rather than a silent substitution.

Do this before any expensive training run depends on the answer.
