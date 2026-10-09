# Structured pruning tooling

Type: research
Status: resolved
Blocked by: none

## Question

How do we actually perform structured channel pruning on a modern YOLO (the C3k2 / C2PSA / SPPF blocks in YOLO11), and what tool should we use?

Structured pruning is not a one-liner here. YOLO backbones and necks are full of concatenations, residual adds, and split paths, so removing a channel in one layer forces coupled removals elsewhere. Naive magnitude pruning produces a model that will not forward-pass.

Establish, with sources:

1. **Tooling.** Compare Torch-Pruning (DepGraph dependency-graph pruning) against any Ultralytics-native support and against hand-rolled masking. Which has working, published examples on YOLOv8 or YOLO11?
2. **Importance criteria.** Magnitude (L1/L2) vs gradient/Taylor vs BN-scale (network slimming). What do the YOLO pruning papers actually use?
3. **Reported results.** What parameter reduction is achievable at what mAP cost after recovery fine-tuning? The deck claims up to 50% params at under 5% mAP degradation (refs [1], [4]) - verify against primary sources and note whether it was measured on COCO or on something small-object-heavy.
4. **Layer sensitivity.** Is there published evidence that pruning the high-resolution or early detection paths (P2/P3) disproportionately harms small-object AP? THIS IS THE LITERATURE BASIS FOR THE PAPER CORE CLAIM - search hard here, and report honestly if it does not exist.
5. **Does the pruned model still export?** Pruned YOLO graphs sometimes break ONNX export or NCNN conversion due to irregular channel counts. Note any known issues.
6. **Recovery fine-tuning cost** - how many epochs do published works use to recover after pruning.

Write findings to `.scratch/uav-yolo-edge/findings/03-pruning.md` and summarise under an `## Answer` heading here.

## Answer

Full detail: `findings/03-pruning.md`.

**1. Tooling.** No native Ultralytics support: its only pruning doc is a YOLOv5 *unstructured* tutorial ("inference time remains unchanged"); PR #21977 closed unmerged Feb 2026; issues #19476, #17733 closed *not planned*. **Torch-Pruning (DepGraph) is the only option**, but its `yolov8` example is marked "crashed due to ultralytics upgrade" and there is **no YOLO11 example** — the C2f→C2f_v2 rewrite must be re-derived for C3k2/C2PSA. Start from `heyongxin233/YOLO-Pruning-RKNN` or `JasonSloan/yolov11-prune`. Budget engineering days.

**2. Criteria.** BN-scale sparse training in ~60% of surveyed YOLO pruning papers, L1/L2 in most of the rest; ~85% channel-granularity (arXiv 2307.11904). LAMP is current fashion. BN-scale cannot reach C2PSA internals.

**3. Results.** "50% params / <5% mAP" is inside the published envelope (30–90% params, 0–4 mAP50 pts). The best VisDrone number — 73.5% params, −2.7 AP50 (arXiv 2509.12918) — **pairs pruning with distillation**; cite it as such. Most other numbers are COCO.

**4. Layer sensitivity — PARTIALLY TAKEN.** I-YOLOv11n (Sensors 25(15):4857, Aug 2025) says it outright on UAV data: LAMP "mistakenly deletes" shallow P2 channels, "the small target feature response is attenuated", so they protect them. The *hypothesis* is published; the **controlled evidence is not** — nobody has swept pruning ratio per pyramid level reporting AP_s/AP_m/AP_l. Reframe from discovery to **quantification**, cite I-YOLOv11n, and control against uniform-ratio *and* plain LAMP at matched params and latency.

**5. Export.** Not the problem — no reports of pruned YOLO failing ONNX/NCNN/TFLite on channel counts. Real issues: pruned checkpoints need `torch.save(model)`, breaking Ultralytics' YAML reload; and **no speedup** (TensorRT #3884: 20% pruning, zero FPS gain on Jetson Nano). **Use `round_to=8`; report latency, not FLOPs.** Verify YOLO26s→NCNN first (#23365).

**6. Recovery.** Reference recipe is 16 steps × 10 epochs = 160 epochs — unaffordable. Use one-shot prune + fixed short recovery (30–50 epochs), identical across all sweep cells.
