# Scale-aware pruning recipe

Type: grilling
Status: open
Blocked by: 08, 11

## Question

Design the experiment that IS contribution (c): a controlled per-pyramid-level pruning sensitivity study, with the controls that make it a result rather than an anecdote.

The pruning research reframed this ticket into existence. I-YOLOv11n (Sensors 25(15):4857) already states in print, on UAV imagery, that LAMP pruning "mistakenly deletes" shallow P2 channels and attenuates small-target response - but states it as motivation for a method, never isolating it. The unoccupied ground is the MEASUREMENT.

Decisions to reach:

1. **The sweep.** Pruning ratio per pyramid level (P2/P3/P4/P5) against AP-small / AP-medium / AP-large, at **matched parameter count** so the comparison is fair. This sensitivity map is the artifact the paper is built on.
2. **Two mandatory controls.** The scale-aware recipe must beat (a) uniform-ratio pruning and (b) **plain LAMP**, which is already layer-adaptive without being told anything about scale. Beating uniform but not LAMP would mean the recipe adds nothing over an off-the-shelf method - so LAMP is the control that actually decides whether contribution (c) lives.
3. **Matched latency, not just matched params.** Both controls must be compared at matched MEASURED DEVICE latency as well. See the map note on FLOPs versus latency.
4. **round_to=8 from day one**, in every cell including the controls. Without it, pruned channel counts fall off the parallelism granularity of the hardware and the latency gain is zero - which would make the entire study report a null result for reasons having nothing to do with the hypothesis.
5. **Recovery budget - this is the GPU-hour bomb.** The stock iterative Torch-Pruning recipe is 16 steps x 10 epochs = 160 epochs per variant, unaffordable across a sweep. Decide a one-shot prune plus a fixed 30-50 epoch recovery, held IDENTICAL across every sweep cell, so differences are attributable to the pruning pattern and not to unequal recovery.
6. **Which model carries the recipe** - YOLO11s alone, or both families. See the fog note on whether YOLO26 survives as a pruning target.

Resolution is an experiment design precise enough to execute without further decisions, plus its GPU-hour cost, which feeds Final matrix and schedule.
