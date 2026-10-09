# Findings: Structured pruning tooling for YOLO11 / YOLO26

Ticket: `issues/03-structured-pruning-tooling.md`
Researched: 2026-09-12
Method: primary sources only — Torch-Pruning repo/source, Ultralytics repo issues/PRs/docs, arXiv/MDPI papers, NVIDIA issue tracker. Secondary write-ups are marked as such.

---

## 1. Tooling

### 1.1 Ultralytics has no native structured pruning. Confirmed.

The only pruning page in Ultralytics' own docs is a **YOLOv5** tutorial, and it is **unstructured**:

> "Unstructured pruning sets weights to zero but leaves the tensor shapes unchanged."
> "Despite pruning, the processing speed is essentially the same" / "Inference time remains unchanged."
> Results: baseline 0.507 mAP -> 30% sparsity 0.489 mAP.

Source: [Model Pruning and Sparsity in YOLOv5 — Ultralytics Docs](https://docs.ultralytics.com/yolov5/tutorials/model_pruning_and_sparsity). It uses `torch_utils.prune()` (a wrapper over `torch.nn.utils.prune`). It is useless for our purpose: no parameter reduction, no latency reduction, no export benefit. **Hand-rolled masking is the same dead end** — this is the published proof of that.

There is no equivalent page for YOLOv8, YOLO11, or YOLO26.

Attempts to add native structured pruning have not landed:

- [PR #21977 — "Add optional structured pruning for YOLOv8 detection models"](https://github.com/ultralytics/ultralytics/pull/21977). True structured channel removal (the author explicitly rejected `torch.nn.utils.prune.ln_structured` "due to lack of groupwise pruning support"). Glenn Jocher reviewed and asked for docs + tests but **did not commit to supporting other YOLO versions or tasks**. **Closed 2026-02-14 due to inactivity. Never merged.**
- [Issue #19476 — YOLOv10/YOLO11 pruning, masking and fine-tuning](https://github.com/ultralytics/ultralytics/issues/19476): closed as *not planned*, stale, no maintainer guidance.
- [Issue #17733 — Structured pruning of yolov8 model](https://github.com/ultralytics/ultralytics/issues/17733) (opened 2024-11-23, Raspberry Pi 4B deployment): closed as *not planned*, stale, no maintainer answer.

**Conclusion: as of 2026-09, Ultralytics will not help. Whatever we do, we own it.**

### 1.2 Torch-Pruning (DepGraph) is the only serious option — but the YOLO example is broken

[VainF/Torch-Pruning](https://github.com/VainF/Torch-Pruning), CVPR 2023 *DepGraph: Towards Any Structural Pruning*. It builds a dependency graph so that removing a channel propagates the coupled removals through concats, residual adds and splits — exactly the YOLO problem the ticket describes.

Example inventory (from [`examples/`](https://github.com/VainF/Torch-Pruning/tree/master/examples)): `yolov5`, `yolov7`, `yolov8`, plus LLMs / timm / torchvision / transformers.
**There is no `yolo11`, `yolov12` or generic `ultralytics` example.**

The `yolov8` example is flagged in the repo's own listing as **"(Crashed due to repo upgrade)"** and its [readme](https://github.com/VainF/Torch-Pruning/blob/master/examples/yolov8/readme.md) opens with *"This example is crashed due to ultralytics upgrade."* It is pinned to ultralytics commit `44c7c3514d87a5e05cfb14dba5a3eeb6eb860e70`.

Concrete configuration from the example source ([`yolov8_pruning.py`](https://github.com/VainF/Torch-Pruning/blob/master/examples/yolov8/yolov8_pruning.py)) — this is our reference recipe:

| Setting | Value |
|---|---|
| Importance | `tp.importance.GroupMagnitudeImportance()` (group L2) |
| Pruner | `tp.pruner.GroupNormPruner` |
| `--iterative-steps` | **16** |
| `--target-prune-rate` | **0.5** |
| Epochs per step | `pruning_cfg['epochs'] = 10` |
| Ignored layers | `if isinstance(m, (Detect,)): ignored_layers.append(m)` — **detection head is excluded from pruning** |
| Required surgery | `replace_c2f_with_c2f_v2(model.model)` — C2f must be rewritten before the graph can be traced |
| Concat/Split | no special-case code; handled by DepGraph generically |
| Ultralytics patches | patches `train`, `save_model`, `final_eval` to stop model reassignment from YAML and to stop half-precision saving |

Two things matter for us. First, **the C2f -> C2f_v2 rewrite is the crux**: YOLO11 uses `C3k2` (a C2f variant whose sub-branch is either a Bottleneck or a C3k) and `C2PSA`, so the equivalent surgery has to be written from scratch. Second, **16 x 10 = 160 epochs of fine-tuning** is the stock recipe — see §6.

The repo README also carries the deployment-critical note:

> "It's recommended to round dims/channels to 4x or 8x for acceleration" (`round_to`)

and offers `isomorphic=True` (Isomorphic Pruning, ECCV 2024) to stop global pruning from "overly pruning specific layers".

The [Torch-Pruning FAQ](https://github.com/VainF/Torch-Pruning/wiki/Frequently-Asked-Questions) adds three operational facts:

> "store the entire model object using `torch.save(model, PATH)` rather than solely utilizing `torch.save(model.state_dict(), PATH)`"
> "After pruning, it's necessary to establish a fresh optimizer."
> "TP relies on AutoGrad for tracing the computational graph. Therefore, please ensure that all layers are executed in the forward pass."

### 1.3 Third-party YOLO11 pruning repos (unreviewed, but working)

- **[heyongxin233/YOLO-Pruning-RKNN](https://github.com/heyongxin233/YOLO-Pruning-RKNN)** — Torch-Pruning based, claims support "from YOLOv3 to the latest YOLO12", explicitly including YOLO11. Two modes: *quick* (train a few epochs -> prune -> finetune in one pass) and *normal* (train -> prune -> retrain). Optional sparse training, documented as possibly "better or worse". Claims up to 75% parameter reduction "without losing accuracy" but publishes **no benchmark table**. Includes RKNN export. Notes that pruning a pretrained model directly "generally results in lower accuracy and larger models".
- **[JasonSloan/yolov11-prune](https://github.com/JasonSloan/yolov11-prune)** — BN-gamma L1 sparsity (network slimming) for YOLO11. Its most useful contribution is an explicit **non-prunable list**: *residual structures in C3k2 modules, most structures inside PSABlock, and the final convolutions / depthwise-separable convs in the detection head*. Five-stage pipeline: normal train (`sr=0`) -> sparse train -> visualise BN gamma -> prune -> finetune. Known limitation: "sparse training currently does not achieve the desired sparsity under multi-GPU DDP mode". No published metrics.
- **NVIDIA ModelOpt / FastNAS route** — [tutorial by y-t-g](https://y-t-g.github.io/tutorials/yolo-prune/) using `modelopt.torch.prune` with constraint `{'flops': '66%'}`. Requires the `ultralytics@qat-nvidia` branch. YOLOv8m: 25.9M -> 16.4M params, 79.3 -> 52.3 GFLOPs, 99MB -> 62.7MB, **6.4ms -> 5.4ms on T4 TensorRT FP16**. Fine-tuned 50 epochs on COCO128. Secondary source (personal site), but the numbers are self-consistent and the latency figure is the honest kind — see §5.

### 1.4 Recommendation

Torch-Pruning/DepGraph, seeded from `heyongxin233/YOLO-Pruning-RKNN` or the upstream `yolov8` example, with the C3k2/C2PSA surgery ported by hand and validated by a forward-pass + ONNX-export smoke test after every prune. Budget real engineering days for this; it is not a config change.

---

## 2. Importance criteria — what the literature actually uses

The one systematic count is [*Model Compression Methods for YOLOv5: A Review* (arXiv:2307.11904)](https://arxiv.org/abs/2307.11904):

> "The main saliency criterion used for pruning is the BNSF sparse training method which is employed in around **60% of the surveyed papers** in our scope, while the rest employed l1-norm or l2-norm or proposed a new saliency criterion."

with **~85% of surveyed work using channel-level granularity**.

So: **BN-scale / network slimming is the field default** for YOLO, magnitude norms second, Taylor/gradient a distant third.

Current practice by source:

| Source | Criterion |
|---|---|
| Torch-Pruning yolov8 example | `GroupMagnitudeImportance` (group L2) + `GroupNormPruner` |
| [arXiv 2509.12918](https://arxiv.org/abs/2509.12918) (VisDrone) | BN scaling factors + sparsity-aware training |
| JasonSloan/yolov11-prune | BN gamma L1 (network slimming) |
| I-YOLOv11n (Sensors 2025) | **LAMP + Taylor hybrid** |
| Recent YOLO11 lightweighting papers | LAMP (layer-adaptive magnitude) is the current fashion |
| NVIDIA ModelOpt | FastNAS score-function search, not a norm at all |

Practical wrinkle for YOLO11 specifically: BN-scale pruning only reaches channels that have a BN after them. `C2PSA`/`PSABlock` attention internals largely do not qualify (JasonSloan lists them as non-prunable), so a pure network-slimming recipe will silently leave the attention block untouched. Group-magnitude/LAMP via DepGraph reaches more of the graph.

LAMP's behaviour is worth knowing because it is relevant to §4: LAMP computes per-layer importance, ranks globally, and **adaptively chooses each layer's retention ratio** — pruning redundant layers hard and sparing critical ones. That means LAMP already produces a non-uniform, layer-adaptive ratio schedule *without* being told anything about object scale. Any scale-aware recipe we propose has to beat LAMP, not just uniform pruning.

---

## 3. Reported results — verifying the deck's "50% params at <5% mAP"

**Verdict: the claim is inside the published envelope, but the deck's framing is optimistic and the numbers must be re-cited to sources we have actually read.** Every headline number below is confounded in some way; note which.

| Source | Model / data | Compression | Accuracy cost | Confound |
|---|---|---|---|---|
| [arXiv 2509.12918](https://arxiv.org/abs/2509.12918) — *A Novel Compression Framework for YOLOv8... Aerial Object Detection on Edge Devices* | YOLOv8m, **VisDrone** | 25.85M -> **6.85M params (73.51%)**; FLOPs 49.6G -> 13.3G; MACs 101G -> 34.5G | **AP50 -2.7 pts -> 47.9**; 26 -> 45 FPS; TensorRT 68 FPS at 47.6 AP50 | **Not pruning alone** — three-stage pipeline: sparsity-aware training + BN-scale structured pruning + **Channel-Wise Distillation**. The distillation is doing an unknown share of the recovery. |
| arXiv 2307.11904 (review of the YOLOv5 pruning literature) | YOLOv5, mostly COCO/VOC | channel pruning typically **30-90% params**, 11-87% FLOPs | **0-4 mAP50 points** drop | Aggregated across heterogeneous papers; inference speedup only 1-65%, often far below the FLOP cut. |
| NVIDIA ModelOpt FastNAS (y-t-g) | YOLOv8m, COCO128 | 25.9M -> 16.4M (37%), FLOPs to 66% | not meaningfully measurable on COCO128 | Toy dataset. |
| Torch-Pruning `yolov8` example | YOLOv8x half, coco128 | 68.2M -> 20.8M params; MACs 129.1G -> 41.7G | not reported | Toy dataset; example is broken. |
| [DUST-YOLO](https://www.mdpi.com/2079-9292/15/12/2579) (Electronics 15(12):2579, 2026-06) | UAV, **VisDrone** | multi-dimensional structured pruning + mixed-precision QAT | **AP_s 16.2 -> 15.3** | Pruning + QAT jointly; Swin-based architecture, not stock YOLO. |
| [MPCS](https://link.springer.com/chapter/10.1007/978-981-95-3052-6_14) (Springer LNCS) | YOLOv12, **VisDrone + TinyPerson** | +10% pruning ratio over the baseline method | "stable detection accuracy" | Abstract only (paywalled); numbers not verified. |

**How to state it in the paper:** "50% parameter reduction at <5% relative mAP loss after recovery fine-tuning" is defensible *as a target*, citing 2307.11904's 30-90% / 0-4 mAP50 range and 2509.12918's VisDrone result, **provided we say plainly that 2509.12918 pairs pruning with distillation**. Do not cite it as a pure-pruning number.

**Measured on COCO or on small-object data?** Mostly COCO/VOC. The genuinely small-object-heavy pruning results are 2509.12918 (VisDrone, AP50 only) and DUST-YOLO (VisDrone, reports AP_s). That scarcity is itself a defensible motivation for our study.

---

## 4. Layer sensitivity — the core claim. Honest verdict: **PARTIALLY TAKEN**

### 4.1 The claim is already in print, on UAV imagery. This is the threat.

**[I-YOLOv11n: A Lightweight and Efficient Small Target Detection Framework for UAV Aerial Images](https://doi.org/10.3390/s25154857)** — *Sensors* 25(15):4857, published 2025-08-07. Verbatim, from the full text ([Europe PMC full-text XML, PMC12349104](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12349104/fullTextXML)):

> "The LAMP method is pruned according to the weight amplitude, and **the shallow key channels (such as the P2 layer of 160 x 160 resolution) are mistakenly deleted. The small target feature response is attenuated.**"

> "the importance of the channel is evaluated by combining LAMP and Taylor scoring mechanisms, and the pruning process is guided to maintain the integrity of the key semantic path, **especially to retain the modeling ability of the shallow features of P2**."

> "the LAMP-Taylor channel pruning mechanism is used to compress the model redundancy, **mainly to protect the key channels sensitive to shallow small targets**."

That is, almost word for word, the hypothesis our paper wants to claim — asserted, on UAV aerial imagery, on a YOLO11 derivative, thirteen months ago.

### 4.2 But the *evidence* is not there. This is the opening.

I could not find, in I-YOLOv11n or anywhere else, **a controlled experiment that isolates the effect**. Specifically, nobody appears to have published:

> a sweep of structured channel-pruning ratio applied *independently per pyramid level* (P2/P3 high-resolution paths vs P4/P5 low-resolution paths), reporting AP_small / AP_medium / AP_large as a function of *where* the channels were removed, at matched total parameter count.

In I-YOLOv11n the statement is **motivation for a method, not a result**. Its headline comparison is I-YOLOv11n vs YOLOv11n (+7.1 mAP@0.5, +4.9 mAP@0.5:0.95, 3.87M params, 14.7 GFLOPs, 24 FPS on Jetson TX2) — a delta dominated by four architectural contributions (RFCBAMConv, STCMSP, DFPC pyramid, Transformer-DyHead head). The pruning choice is one term in a large bundle and is not separately ablated in anything I could retrieve.

*Caveat on my own evidence:* Europe PMC's full text truncated before the experiments section, and MDPI, PMC HTML and DOAJ all block automated fetch. **Someone should read the PDF by hand before related work is written.** If they do have an isolating ablation, §4.5 changes.

Several independent search framings for the general version of the claim (AP_S degradation vs pruning ratio, scale-wise pruning sensitivity, shallow-vs-deep pruning effect on small objects) returned nothing. The negative result looks robust.

### 4.3 What else is adjacent, and how close it gets

- **[PAGCP](https://arxiv.org/abs/2303.11923)** — *Performance-aware Approximation of Global Channel Pruning for Multitask CNNs*. The nearest thing to a general theory. It characterises sensitivity as **task**-dependent, not scale-dependent:
  > "most filters in shallow layers focus on extracting more detailed and spatial information that benefits the localization task. Thereby, shallow layers are more robust to the localization task and more sensitive to the classification task."
  > "most filters in deep layers are robust to the classification task and sensitive to the localization task due to rich semantic information extracted by filters."

  It also reports the **emergent** observation that *the pruning ratio of layers sensitive to large-scale prediction tasks is on average larger than that of layers sensitive to small-scale tasks* — i.e. their algorithm spontaneously spares small-scale-relevant layers. That is an output of the method, not a designed constraint, and it is on VOC/COCO with SSD/YOLOv5 (VGG16-SSD: 75.6% mAP, 74.3% FLOPs reduction, 85.6% params reduction, -1.9 pts), **not on a UAV small-object set, and with no AP_s/AP_m/AP_l breakdown**. This is the best citable prior for "scale sensitivity is real but unquantified".
- **DUST-YOLO** (Electronics 2026): applies "**asymmetric** channel pruning to convolutional and feature-fusion modules while compressing the Swin Transformer prediction heads and bottleneck stacks, thereby reducing parameters and computation **with limited impact on multi-scale representation**"; VisDrone AP_s 16.2 -> 15.3. The same instinct, non-uniform per-module ratios, no isolating ablation, confounded with QAT.
- **MPCS** (Springer LNCS, YOLOv12, VisDrone + TinyPerson): cosine-annealed gamma scheduling for "balanced layer-wise regularization". Layer-*balance*-aware, not scale-aware.
- **[Iterative Filter Pruning for Concatenation-based CNN Architectures](https://arxiv.org/abs/2405.03715)** (YOLOv7, FPGA + Jetson Xavier AGX): automates sensitivity analysis + prune + finetune over concat graphs. Generic layer sensitivity, no scale analysis. Useful methodological citation for *how* to run a sensitivity sweep.

### 4.4 The thing that is NOT our claim, and must not be confused with it

There is a large, mature literature on **architectural** scale surgery for UAV detection — add a P2 head, delete the P5 head:

- [LDSNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC13207737/) "discards the redundant P4 and P5 heads and focuses instead on dual-scale branches at the P2 and P3 levels."
- [DroneScan-YOLO](https://arxiv.org/abs/2604.13278) "features a redesigned neck with a high-resolution P2 detection head and prunes the P5 head." Its own pruning is *unstructured* cosine-similarity masking; the authors concede it "does not physically remove parameters, limiting real-world speed gains on embedded hardware." No scale-sensitivity study.
- [arXiv 2606.09081](https://arxiv.org/abs/2606.09081) reports "+P2 increases APsmall by 31.10% over baseline" on VisDrone; explicitly does **no** pruning sensitivity analysis.
- The standard justification, widely repeated: "For an input of 640 x 640, a target occupying 8 x 8 pixels is reduced to a 1 x 1 feature response at P3 — insufficient for reliable localization."

This is *branch-level* surgery and it is done to death. Our claim is *channel-level within a branch*. Reviewers will conflate the two unless we separate them explicitly in related work.

### 4.5 Verdict for the paper's framing

| Framing | Status |
|---|---|
| "We are the first to observe that pruning P2/P3 harms small objects" | **Dead.** I-YOLOv11n says it in print, on UAV data. |
| "We are the first to *propose* protecting shallow channels during pruning" | **Dead.** I-YOLOv11n's LAMP-Taylor does exactly that; DUST-YOLO's asymmetric pruning is adjacent. |
| "We **quantify** the effect — a controlled per-level pruning-ratio sweep with AP_s/AP_m/AP_l on a UAV dataset — and derive a recipe from the measurement" | **Open, as far as I can find.** Nobody has published the sweep. |
| "...and carry it through to measured latency on four real edge platforms" | **Open**, and this is the stronger half. |

**Recommended reframe: from *discovery* to *measurement*.** Cite I-YOLOv11n as prior support for the hypothesis rather than pretending it does not exist — a reviewer who knows this field will find it, and citing it first makes us look thorough instead of uninformed. The contribution becomes: the sensitivity map, the recipe derived from it, and the demonstration that it survives to real device latency.

**Mandatory controls.** The scale-aware recipe must beat, at matched parameter count *and* matched measured latency:
1. a **uniform-ratio** structured prune, and
2. a **plain LAMP / global-magnitude** prune (which is already layer-adaptive, per §2).

If it beats neither, there is no contribution (c), and the map's floor (a) — benchmark + deployment guidelines — is what publishes. The map already anticipates this; running these controls is what makes the fallback honest rather than a retreat.

---

## 5. Does the pruned model still export?

**Short answer: yes, export is largely a non-issue. The real problem is that the pruned model is not faster.** The ticket's worry is aimed at the wrong failure mode.

### 5.1 Export itself

Torch-Pruning physically shrinks weight tensors, so the result is an ordinary conv graph with unusual channel counts — not a special op, not a sparse tensor. Nothing in ONNX, NCNN, TFLite or TensorRT requires channel counts to be powers of two. The Torch-Pruning FAQ even suggests ONNX export as a way to *verify* the size reduction. I found **no** GitHub issue reporting "pruned YOLO fails to export to ONNX because of channel counts".

What does break:

- **Checkpoint reload.** The FAQ is explicit: after pruning "your model's architecture will be altered, rendering the original structure defined in your `.py` file incompatible"; you must `torch.save(model, PATH)` (whole object) rather than the `state_dict`. This collides head-on with Ultralytics' checkpoint/resume/`YOLO('best.pt')` path, which reconstructs the model from YAML. **This is the actual practical blocker, and it is an engineering problem rather than an export problem.** It is precisely what the upstream example patches `save_model`/`final_eval` to work around, and what the (closed) PR #21977 tried to solve with an `ckpt["is_pruned"]` metadata flag.
- **Toolchain-version breakage, unrelated to pruning.** Ultralytics NCNN/TFLite export has its own ongoing failures: [#19091 NCNN export broken on Pi 4 with 8.3.71 + torch 2.6](https://github.com/ultralytics/ultralytics/issues/19091), [#15624 NCNN export broke a pretrained yolov8](https://github.com/ultralytics/ultralytics/issues/15624), [#20639 TFLite conversion fails](https://github.com/ultralytics/ultralytics/issues/20639), [#13111 RTDETR fails to TFLite/NCNN for Pi 4, PNNX SIGABRT](https://github.com/ultralytics/ultralytics/issues/13111). NCNN goes through PNNX, a separate binary with its own version coupling. Pin versions early and record them.
- **YOLO26 specifically:** [#23365 — yolo26n ncnn export](https://github.com/ultralytics/ultralytics/issues/23365) reports warnings preventing successful end-to-end NCNN export on current ultralytics. **Verify the unpruned YOLO26s -> NCNN path works before committing GPU-hours to pruning YOLO26s.**

### 5.2 The real failure mode: pruning that does not buy latency

[**NVIDIA/TensorRT issue #3884**](https://github.com/NVIDIA/TensorRT/issues/3884) — *"Smaller pruned model yolov8s doesn't faster than original yolov8s on Tensor RT Jetson Nano"*:

- prune ratio **0.2 -> ~7.4 FPS, identical to the unpruned model**
- prune ratio **0.4 -> ~8.5 FPS**
- triaged and assigned by NVIDIA; no resolution posted.

Corroborating, from sources that measured rather than assumed:

- NVIDIA ModelOpt FastNAS on YOLOv8m: **34% of FLOPs removed -> 16% of latency removed** (6.4ms -> 5.4ms, T4, TensorRT FP16).
- arXiv 2307.11904's survey: FLOPs reductions of 11-87% but inference-time improvements of only **1-65%**.

The mechanism is channel alignment. Torch-Pruning's README says it directly:

> "It's recommended to round dims/channels to 4x or 8x for acceleration"

Kernels on every one of our targets vectorise over channels — TensorRT tensor cores, NCNN's ARM NEON packed-4/packed-8 layouts, XNNPACK behind TFLite. A conv with 37 output channels pads to the next multiple and does the same work as 40. Prune to arbitrary counts and you buy parameters (model size, memory bandwidth) but not FLOPs-in-wall-clock.

### 5.3 Actionable consequences for the plan

1. **Set `round_to=8`** (or 16 for TensorRT) in the Torch-Pruning config from day one. Not optional. Without it the benchmark half of the paper reports a null result for the wrong reason.
2. **Report measured latency per platform, never FLOPs, as the efficiency axis.** The gap between the two is itself a publishable finding for a deployment-guidelines paper and directly serves contribution (a).
3. **Smoke-test the full export chain (ONNX -> NCNN / TFLite / TensorRT) on a pruned model in week one**, at a throwaway pruning ratio, before any real training. Cheap, and it de-risks the entire compression arm.
4. [*Deployment-oriented pruning for real-time object detection: a survey*](https://link.springer.com/article/10.1007/s10489-026-07416-6), Applied Intelligence 56:380 (2026), is framed entirely around this gap — "how complexity reductions induced by model pruning can be reliably translated into end-to-end latency gains", analysing "static-graph export, operator coverage and fusion, **tensor-shape constraints**, and their alignment with hardware parallelism granularity". Paywalled; **get it through the university library**. It is the single best related-work anchor for our benchmark framing, and it confirms the FLOPs-vs-latency gap is a recognised open problem rather than a rookie mistake.

---

## 6. Recovery fine-tuning cost

This is the GPU-hour bomb. Published practice:

| Source | Recovery budget |
|---|---|
| Torch-Pruning `yolov8` example (reference config) | `iterative_steps=16` x `epochs=10` = **160 epochs** interleaved with pruning, to reach `target_prune_rate=0.5` |
| NVIDIA ModelOpt tutorial | 50 epochs on COCO128, with the author noting that "higher epoch counts with actual datasets" are needed |
| PAGCP (arXiv 2303.11923) | "the number of training epochs proportional to the compression rate of FLOPs" |
| Network-slimming pipelines (JasonSloan; arXiv 2509.12918) | **three** training phases: normal train -> sparsity train -> post-prune finetune, i.e. **2-3x a baseline training run per variant** |
| arXiv 2307.11904 survey | does not systematically report fine-tuning epochs — the review flags this as a reproducibility gap in the field |

**Implication for a 2-month solo study on an RTX 4050 6GB + Kaggle 30h/wk:** the stock 16-step iterative recipe is unaffordable. Budget as:

- **One-shot prune + short recovery** (e.g. 30-50 epochs on VisDrone) ~= 0.3-0.5x a full training run per pruned variant.
- **Iterative or sparsity-based** ~= 2-3x a full training run per variant. Off the table for a matrix of any width.

Because the P2/P3 sensitivity sweep in §4 requires *many* pruned variants (per level x per ratio), **one-shot pruning with a fixed short recovery schedule is the only viable protocol**, and the schedule must be held identical across all cells or the sensitivity comparison is confounded by recovery budget. State this explicitly in the methods section — it is a defensible design choice, and holding recovery fixed is more rigorous than letting each variant train to convergence.

A cheap accelerator worth considering: run the sensitivity sweep at **reduced recovery** (e.g. 15 epochs) to *locate* the effect, then re-run only the two or three winning configurations at full recovery for the headline table. Sensitivity *ordering* is usually stable under short recovery even when absolute mAP is not — but say so as an assumption, and validate it on one config.

---

## Sources

Primary — code and trackers:

- https://github.com/VainF/Torch-Pruning
- https://github.com/VainF/Torch-Pruning/tree/master/examples
- https://github.com/VainF/Torch-Pruning/blob/master/examples/yolov8/readme.md
- https://github.com/VainF/Torch-Pruning/blob/master/examples/yolov8/yolov8_pruning.py
- https://github.com/VainF/Torch-Pruning/wiki/Frequently-Asked-Questions
- https://github.com/ultralytics/ultralytics/pull/21977
- https://github.com/ultralytics/ultralytics/issues/19476
- https://github.com/ultralytics/ultralytics/issues/17733
- https://github.com/ultralytics/ultralytics/issues/23365
- https://github.com/ultralytics/ultralytics/issues/19091
- https://github.com/ultralytics/ultralytics/issues/15624
- https://github.com/ultralytics/ultralytics/issues/20639
- https://github.com/ultralytics/ultralytics/issues/13111
- https://github.com/NVIDIA/TensorRT/issues/3884
- https://github.com/heyongxin233/YOLO-Pruning-RKNN
- https://github.com/JasonSloan/yolov11-prune
- https://docs.ultralytics.com/yolov5/tutorials/model_pruning_and_sparsity

Primary — papers:

- DepGraph / Torch-Pruning, CVPR 2023 (repo above)
- https://arxiv.org/abs/2509.12918 — Compression framework for YOLOv8, VisDrone, structured pruning + CWD
- https://arxiv.org/abs/2303.11923 — PAGCP
- https://arxiv.org/abs/2307.11904 — Model Compression Methods for YOLOv5: A Review
- https://arxiv.org/abs/2405.03715 — Iterative Filter Pruning for Concatenation-based CNNs
- https://doi.org/10.3390/s25154857 — I-YOLOv11n (full text via https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12349104/fullTextXML)
- https://www.mdpi.com/2079-9292/15/12/2579 — DUST-YOLO
- https://link.springer.com/chapter/10.1007/978-981-95-3052-6_14 — MPCS (abstract only, paywalled)
- https://link.springer.com/article/10.1007/s10489-026-07416-6 — Deployment-oriented pruning survey (paywalled)
- https://arxiv.org/abs/2604.13278 — DroneScan-YOLO
- https://arxiv.org/abs/2606.09081 — Edge-constrained UAV small-object detection with P2 enhancement

Secondary (flagged as such):

- https://y-t-g.github.io/tutorials/yolo-prune/ — NVIDIA ModelOpt pruning tutorial for Ultralytics YOLO

Could not retrieve (blocked; needs manual or library access):

- I-YOLOv11n experiments and ablation sections (MDPI / PMC / DOAJ all 403 or CAPTCHA) — **read by hand before writing related work**
- Applied Intelligence deployment-oriented pruning survey (paywalled)
- MPCS full text (paywalled)
