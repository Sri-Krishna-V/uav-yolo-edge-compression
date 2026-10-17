# Findings: Novelty check for the candidate contributions

Researched: 2026-10-17. Primary sources only. All three search passes have now reported. The pass for angles 1
and 6 and the pass for angles 4, 5 and 7 arrived after the body of this file was written. Their results are in the
verdict table below and in the **Addendum** at the end. Where a body section still says "pending", the addendum
supersedes it.

**Verdict in one line:** **no single angle is unclaimed**, but four *measurements* are open. All four reuse assets
that already exist, and the literature names each one as future work:
- **YOLO11s vs YOLO26s on real edge devices** (Pi 5, phone, Jetson), with INT8 and pruning. GPU-only comparisons
  exist, and they contradict each other. Aero-YOLO26 and PFE-Det both leave onboard measurement to future work.
- **The iso-latency resolution trade on CPU-class devices.** The idea, quantization buying resolution for UAV
  small objects, is in print: Plastiras et al. AICAS 2020, and Vandersteegen et al. MVA 2019 on VisDrone/Xavier.
  A controlled, matched-latency study on a Pi 5 or phone with AP-small and CIs is not.
  - **Risk:** INT8 may be *slower* than FP32 on the Pi 5. Marasini & Zhou 2026 measured −43% FPS with OpenVINO/NNCF;
    Rey et al. 2025 could not get INT8 working with NCNN.
- **Runtime dependence of INT8 on the Pi 5.** Marasini & Zhou: "Runtime dependence is left for future work."
- **NMS cost against ground-truth density on an ARM CPU or phone**, and what YOLO26's fallback to NMS costs on
  dense UAV images. The mechanism is known (Overload, CVPR 2024); no edge-CPU measurement binned by density was found.

The "toolchain dominates, the ratio is stable" decomposition of the PTQ damage (B5) remains a supporting result.

---

## Verdict summary

| # | Angle | Verdict | Closest prior work |
|---|---|---|---|
| 1 | Compression budget spent on resolution, at matched measured edge latency, with AP-small | **Partially taken.** The idea is in print. Open: a controlled matched-latency study on CPU-class devices, VisDrone test-dev AP-small with CIs, YOLO11s/26s, two INT8 runtimes plus pruning. Risk: INT8 may not speed up on a Pi 5 | Plastiras et al. AICAS 2020 (INT8 + tiling, UAV, TX2 CPU); Vandersteegen et al. MVA 2019 (arXiv 1904.02024; VisDrone, Xavier, INT8@960 implicitly beats FP32@608); E³-UAV (resolution as an energy knob, no compression); MCUNetV2 (memory, not latency) |
| 2 | Size-bucket sensitivity to INT8 PTQ on aerial data | **Partially taken.** "AP_S drops more than AP under PTQ on VisDrone" is in print. Open: ≥2 real toolchains, the stable-ratio decomposition, CIs, on device | I-LW-DETR, arXiv 2607.24981 (Jul 2026), Table 3 |
| 3 | Size-bucket sensitivity to uniform structured pruning with recovery | **Partially taken.** COCO, driving and one VisDrone operating point all show small objects hit 1.4–2× harder. Open: a controlled DepGraph+recovery study with CIs, and the 1.17× counter-finding | Choi & Tian, IEEE IV 2023 (arXiv 2303.02512); Sabaghian et al. arXiv 2509.12918; Girish et al. CVPR 2021 |
| 4 | maxDets=100 under-reporting on VisDrone | **Not a contribution; a reproducibility note.** The effect is 0.3–0.5 AR points. pycocotools caps detections per image *per category*, so ">100 objects per image" overstates when the cap binds. 0 of 18 sampled VisDrone YOLO papers state maxDets. A three-way comparison (Ultralytics / COCO / toolkit) on identical predictions looks new for VisDrone | RF100-VL (arXiv 2505.20612 §3.2, maxDets=500 and Ultralytics over-estimation); SKU-110K (AR300); `evalDET.m` |
| 5a | YOLO26 NMS-free head disabled on INT8 edge exports | **Taken as a fact.** Ultralytics docs, exporter warnings, the official YOLO26 paper (arXiv 2606.03748 §3.5) and YOLOv10 issue #118 all record it. **Open as a measurement** of its cost on dense UAV images. Wording fixes are in the addendum | docs.ultralytics.com/guides/end2end-detection; `exporter.py` L682–713 (v8.4.149) |
| 5b | NMS latency vs objects per image on edge CPUs | **Mechanism known, measurement likely open.** No ARM-CPU or phone measurement on naturally dense scenes binned by ground-truth count | Overload (CVPR 2024, arXiv 2304.05370, Xavier NX); RT-DETR Table 1 (T4); YOLOv10 Table 2 (T4) |
| 6 | Energy per frame and thermally throttled throughput, Pi 5 or phone, UAV | **Partially taken.** A Pi 5 + YOLO + J/frame + 30-minute throttling study exists for road-damage data, FP32, at one resolution. Open: UAV data, resolution × compression, AP-small per joule, Pi 5 vs phone. No peer-reviewed phone study was found | Marasini & Zhou arXiv 2609.04705 (Sep 2026 preprint); Rey et al. Electronics 14:638 (Pi 5 J/inference, 60 s, FP32); Benoit-Cattin et al. 2020 (Pi 4 throttling method) |
| 7 | YOLO11 vs YOLO26 on VisDrone at the edge | **Taken on desktop GPU, where results conflict** (Aero-YOLO26: YOLO26s wins; PFE-Det: it loses). **Edge (Pi, phone, Jetson), INT8 or pruned: no precedent found.** Likely open | Aero-YOLO26, Sensors 26(19):6346; PFE-Det, Sensors 26(15):5003; MD-YOLO; arXiv 2605.24831 |

---

## 1. Spend the compression budget on resolution

**What is known (all already in `02-visdrone.md` §1):**
- Resolution beats capacity on VisDrone: yolov8s@960 scores 30.8 val mAP50-95 against 27.8 for yolov8l@640.
  That is on a GPU and uncompressed.
  ([xuanandsix/VisDrone-yolov8](https://github.com/xuanandsix/VisDrone-yolov8))
- MFP-YOLO's 640→1024 ablation gives +8 to +12 mAP50. ([PMC10346989](https://pmc.ncbi.nlm.nih.gov/articles/PMC10346989/), Table 5)
- SAHI and ASAHI show sliced inference trading latency for AP-small. ([arXiv 2202.06934](https://arxiv.org/abs/2202.06934),
  [arXiv 2604.19233](https://arxiv.org/html/2604.19233v1))

None of these holds measured device latency fixed while compressing.

**Leads I found but did not open (unverified: pending the angle-1 search pass):**
- **E³-UAV**, an "Edge-based Energy-Efficient Object Detection System for UAVs" ([arXiv 2308.04774](https://arxiv.org/pdf/2308.04774)).
  The title suggests it picks the model and its settings under energy constraints. It must be read before any
  "first iso-latency" claim.
- **"Analysis and Adaptation of YOLOv4 for Object Detection in Aerial Images"** ([arXiv 2203.10194](https://arxiv.org/abs/2203.10194)).
  The search snippet reports 8.7 fps at 416² falling to 2.3 fps at 1120² on VisDrone. That is a resolution-vs-speed
  curve with no compression in it.
- [arXiv 2608.23636](https://arxiv.org/pdf/2608.23636) (orchard, YOLO27/26/11/v8). The snippet compares latency at 640
  and 960; it is not UAV data.

**The literature names the gap without filling it.** The Applied Intelligence survey
(doi:[10.1007/s10489-026-07416-6](https://doi.org/10.1007/s10489-026-07416-6)) says:
- §2.7: "latency bottlenecks may differ across input resolutions and batch-size settings".
- §4: post-processing cost "becomes more pronounced under high-resolution inputs or dense-object scenes".

It treats resolution as a confound to report. It never treats resolution as the place to spend the compression saving.

**First-order arithmetic from our own assets.** GFLOPs scale with the pixel count. These are predictions, not
measurements.

- **Pruning.** The pruned model is 12.2 GFLOPs at 640. It matches the unpruned 640 model (21.7 GFLOPs) at about
  640·√(21.7/12.2) ≈ **864 px**.
  - Interpolating the unpruned 640→960 AP gives about 21.9 AP at 864.
  - Applying the measured −17% pruning loss gives about 18.2, against 18.8 for the unpruned 640 model. **Roughly
    break-even.**
  - Pruning at 960 (27.5 GFLOPs) predicts about 19.9 AP, but costs 27% more FLOPs than unpruned 640.
- **INT8.** OpenVINO INT8 at 960 should land near 23.5 AP (−2.3% of 24.0). It reaches iso-latency with FP32@640
  only if the device's INT8 speedup is at least (960/640)² = **2.25×**; at 1280 it needs 4×.
- So the answer **depends on the device's INT8 speedup and on NMS cost**. That dependence is exactly what an
  iso-latency measurement on a Pi 5 and a phone would show. This is a reason to measure, not a result.

**Verdict: unsure, leaning open.** My searches did not find a comparison of a compressed detector at higher
resolution against an uncompressed one at lower resolution, at matched *measured* edge latency, on UAV data, with
AP-small. These searches are listed in the search log. The angle-1 pass that covers compound scaling, adaptive
resolution, joint resolution/bit-width search and Pi/Jetson VisDrone Pareto studies is **pending**. Do not write
"first" until it reports and E³-UAV has been read.

---

## 2. Size-bucket sensitivity to INT8 PTQ

**The plain claim is taken.**

**I-LW-DETR**: Le, Szczepanski and Poreba, "Enabling Fully Integer-Only Inference for Lightweight Detection
Transformers", [arXiv 2607.24981](https://arxiv.org/html/2607.24981), submitted 2026-07-27. I verified Table 3, which
is VisDrone test-dev with 8/8/16 PTQ: per-channel weights, per-tensor activations and percentile calibration.

| Variant | QR-FP32 → PTQ mAP | AP_S | small/all ratio |
|---|---|---|---|
| Tiny | 15.0 → 7.8 (−48%) | 6.6 → 2.3 (−65%) | 1.36 |
| Small | 17.2 → 13.3 (−23%) | 7.7 → 5.5 (−29%) | 1.26 |
| Medium | 19.0 → 11.4 (−40%) | 8.8 → 4.2 (−52%) | 1.31 |

- The paper says: "APS exhibits the largest drop from the QR baseline to PTQ across all model variants."
- It also says: "it does not constitute a deployment study, and no compilation or on-device measurements are
  reported."
- It uses one quantizer, a DETR rather than a YOLO or CNN, and no statistics.

**Other prior work, reported by the angle-2 search pass (not re-opened by me):**
- **Luo et al.**, ISPRS Annals XI-2-2026, 697–704, published 2026-07-03
  (doi:[10.5194/isprs-annals-XI-2-2026-697-2026](https://doi.org/10.5194/isprs-annals-XI-2-2026-697-2026)).
  - Eq. 5 argues that quantization "acts as a scale-dependent filter", because small objects have a lower
    signal-to-quantization-error ratio.
  - It adds a scale-aware QAT regulariser to YOLOX on bird and building data, run on a KV260.
  - It reports **no per-size AP** and no PTQ. Its argument predicts the same size effect under every quantizer.
    **Our data tests that prediction directly.**
- **Johannsen**, LIACS bachelor thesis, 2026-08-04, not peer-reviewed
  ([pdf](https://theses.liacs.nl/pdf/2025-2026-JohannsenL.pdf)), Table 2.
  - YOLOv11n on XS-VID, ONNX Runtime INT8 through the Ultralytics export, MinMax calibration. That is nearly our
    ORT setup.
  - AP falls 10.5%; the extremely-small bin falls 35%.
  - Sec. 3.4 names the gap: "No disaggregated small-object quantization metrics."
- **COCO low-bit tables already show AP_S falling faster**, but at 4 bits. A reviewer can say "APs is known to be
  fragile":
  - AQD, CVPR 2021, Table 2 ([arXiv 2007.06919](https://arxiv.org/pdf/2007.06919)): 1.84×.
  - Q-DETR, CVPR 2023, Table 3 ([arXiv 2304.00253](https://arxiv.org/pdf/2304.00253)): 1.82×.
  - Q-YOLO, Table 1 ([arXiv 2307.04816](https://arxiv.org/html/2307.04816v1)): at 8 bits, APs barely moves.
- **Calibration and toolchain choice moves aggregate loss**, on COCO and aggregate only:
  - Wu et al. (NVIDIA), [arXiv 2004.09602](https://arxiv.org/pdf/2004.09602), Table 5. RetinaNet loses 0.3% under
    max calibration and about 10% under 99.9th-percentile calibration.
  - The Cloudflight engineering blog (not peer-reviewed) measures YOLOv8n under TFLite per-tensor (−11.8%) and
    OpenVINO (−4.3%). That mirrors our gap, with no size buckets.

**The confound reviewers will find in our own toolchain comparison.** I checked this in the code at the pinned
v8.4.149, and it is unchanged on main at v8.4.179 (2026-10-17).
- Ultralytics' **ORT INT8** path
  ([`utils/export/onnx.py` `onnx_int8_quantize`](https://github.com/ultralytics/ultralytics/blob/v8.4.149/ultralytics/utils/export/onnx.py))
  calls `quantize_static` with its defaults: per-tensor, MinMax. It excludes only nodes whose op type is *not*
  Conv/Gemm/MatMul, **so the head's convolutions are quantized**.
- The **OpenVINO** path
  ([`utils/export/openvino.py`](https://github.com/ultralytics/ultralytics/blob/v8.4.149/ultralytics/utils/export/openvino.py),
  `int8_detect`) uses NNCF's `MIXED` preset with an `IgnoredScope` covering the **whole Detect head plus every
  Sigmoid**. The head stays in float.
- So "ORT vs OpenVINO" bundles four differences: head in float or not, per-channel or per-tensor weights, the
  quantization preset, and NNCF's bias correction. The "4× toolchain effect" is real, but it is not yet *explained*.
  Job B item B5 below separates the four.

**Verdict: partially taken.**
- **Taken:** small objects lose more under INT8 PTQ on VisDrone (I-LW-DETR, ratios 1.26–1.36), and the
  scale-dependence argument (ISPRS 2026).
- **Open, as far as the angle-2 pass found** (36 searches, log below):
  - A YOLO/CNN on VisDrone with AP_S/M/L across **two or more real deployment toolchains**.
  - The decomposition **"toolchain moves aggregate loss ~4×, while the small/all ratio stays at ~1.3–1.5"**.
    I-LW-DETR's own 1.26–1.36 is independent support for the stable ratio.
  - Paired-bootstrap CIs.
  - On-device INT8 (Pi 5, phone).
- **Unsure:** van Manen et al. (SPIE, YOLOv8 on Jetson AGX Orin) claims in its abstract that INT8 costs nothing on
  small objects. It is paywalled, so a reviewer may raise it.

---

## 3. Size-bucket sensitivity to uniform structured pruning with recovery

**I-YOLOv11n, re-read from the local PDF** (`papers/sensors-25-04857.pdf`). This settles the open caveat in
`03-pruning.md` §4.2.
- The paper has **no isolating pruning ablation and no AP_S table at all**.
- Table 4 (p. 23) bundles "KD + Prune + anchor mechanism" into a single "Model E" row: 40.1 mAP50, 24.1 mAP50-95.
- Its conclusion names the gap: "the pruning granularity needs to be further optimized in the future" (p. 28).
- It is internally inconsistent: §4.2.2 says the testbed is a Jetson **Xavier NX**, while the abstract and
  conclusion say **TX2**; the conclusion gives 89 FPS, Table 4 gives 92.

The `03-pruning.md` verdict stands: the hypothesis is in print, the evidence is not.

**Prior work with size buckets, reported by the angle-3 search pass (not re-opened by me):**
- **Choi & Tian**, IEEE IV 2023 ([arXiv 2303.02512](https://arxiv.org/pdf/2303.02512)), Tables I–IV.
  - YOLOX-S and YOLOF, pruning rate swept 0.1–0.7 with fine-tuning, AP-s/m/l at IoU 0.5, on KITTI and COCO-traffic.
  - Small objects lose 1.64–1.77× more at rates 0.4–0.5. At a low rate, AP-s *rises*.
  - **This is the closest analogue.**
- **Sabaghian et al.** ([arXiv 2509.12918](https://arxiv.org/pdf/2509.12918), v3 2025-12-31, no venue), Table 6.
  - VisDrone YOLOv8m, uniform 50% BN-scale pruning, 100-epoch fine-tune.
  - AP −6.0%, APsmall −8.5% (about 1.41×), APlarge **+10.6%**.
  - Inconsistent: APsmall (41.3) is above AP (28.3), so its bucket or IoU definition is non-standard.
- **Girish et al.**, CVPR 2021 ([arXiv 2012.04643](https://arxiv.org/pdf/2012.04643v2)), §5 and Fig. 4.
  - Lottery-ticket pruning, unstructured. Small objects lose slightly more beyond about 80% sparsity.
- The aerial pruning papers the survey cites (Deng, RS 14:5347; L1RR, RS 16:2026) have **no per-size AP**. They argue
  pruning removes "weak small-object or texture-sensitive features" (survey §3.3) only as motivation.

**Verdict: partially taken.**
- **Taken:** "small objects lose more under pruning". There are COCO and driving sweeps, plus one VisDrone point.
- **Open:**
  - A controlled DepGraph + recovery study on VisDrone with per-size AP and paired CIs.
  - Above all, **our 1.17× counter-finding** (AP_all −17.1%, AP-small −20.0%) against the 1.4–2× reported elsewhere.
    It is publishable as a counter-finding *only if* the bootstrap CI on the ratio excludes about 1.4. It also has to
    be reconciled with Choi & Tian's IoU-0.5 metric and with recovery length.
  - The per-pyramid-level sensitivity map (contribution c). No paper found does it, and I-YOLOv11n is confirmed not
    to.

---

## 4. The dense-scene evaluation pitfall (maxDets)

**Our own records cap the size of the effect.** I read all 11 records in `study/verification/runs/*.json`. Each one
stores `coco/AR_100` and `coco/AR_500` from the same predictions.

| Split | AR@500 − AR@100, all 11 runs | Trend |
|---|---|---|
| test-dev | **+0.33 to +0.53 pts** | grows with resolution: YOLO11s 0.35 (640), 0.46 (960), 0.51 (1280); YOLO26s 0.36, 0.45, 0.53 |
| val | +0.26 to +0.38 pts | same trend |

Recall is the quantity that truncation at 100 detections can hurt. The AP gap should be of the same order or smaller.
`studyeval` can compute AP@100 exactly, from saved predictions, at zero cost.

**Consequence:** the maxDets=100 trap is real, but it is worth about 0.3–0.5 points over the whole dataset, close to
the seed noise floor (0.07 AP_all, 0.27 AP-small).
- **As a headline contribution it is weak.** It belongs in a methods box.
- **It can matter in a stratum.** On the 7.9% of test-dev images with more than 100 objects, the truncation falls
  entirely on that stratum, so the stratified gap will be several times larger (Job B item B3). It also grows with
  resolution, which makes it a small but systematic bias *against* the high-resolution arm of angle 1.

**What else I found:**
- MD-YOLO (*Algorithms* 19(9):737, published 2026-09-01) reports AP_S via "standard COCO-style size stratification"
  (§4.3, Table 3) and never states maxDets.
- I-YOLOv11n reports no area splits at all. Both are examples of the under-specified protocol.
- How common maxDets=100 or Ultralytics-default validation is across 2024–2026 VisDrone edge papers is
  **unverified: pending the angle-4 pass**. So is any prior quantification of the COCO-vs-`evalDET.m` gap. Issue #17
  states that the magnitude "has never been measured", and that is still the team's own claim.

**Verdict:**
- As a standalone pitfall: **low value**, because our own numbers make it small.
- Its prevalence in the literature: **unsure**, pending.
- The **official-toolkit gap**, and whether it *moves under compression* (issue #17 point 3), is the more promising
  evaluation angle. See B7.

---

## 5. YOLO26's NMS-free head at the edge, and NMS latency

### 5a. The fallback is vendor-documented, so "unpublished" is the wrong word

The Ultralytics end-to-end guide ([docs.ultralytics.com/guides/end2end-detection](https://docs.ultralytics.com/guides/end2end-detection/))
says so outright:
- "NCNN, RKNN, PaddlePaddle, ExecuTorch, IMX, Edge TPU and Qualcomm QNN fall back to the one-to-many path".
- "TensorRT versions before 8.5.0, TensorRT 10.3.0 INT8 on JetPack 6, and LiteRT INT8 or `w8a16` fall back to
  one-to-many outputs."

The exporter prints a warning each time. The logic is unchanged on main at
[v8.4.179](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/exporter.py): lines 724–734 for
the format and LiteRT-INT8 rules, 820–846 for the TensorRT rules. **A reviewer will call the fact itself "documented
behaviour".**

**The claim also needs narrowing.** Ultralytics does **not** disable end-to-end on its **ONNX (ONNX Runtime) INT8**
or **OpenVINO INT8** paths, and both runtimes run on aarch64, so on a Pi 5 as well. Both paths keep the head decode
in float (§2), which is the very thing the LiteRT comment says collapses ("Static activation quantization collapses
the end2end class-index output").
- "Unavailable on every INT8 edge path" is true only for the runtimes in *our* matrix: NCNN, LiteRT, TensorRT on
  JetPack 6.
- It is not true of the Pi 5 as a platform. Rephrase it as **"unavailable on the fastest, most-used edge runtimes:
  NCNN, LiteRT INT8, TensorRT INT8 on JetPack 6"**.

**What is open:** the *consequence*, measured.
- Does the one-to-one head keep its accuracy under ORT or OpenVINO INT8, where it is allowed?
- On the Pi 5, what is the latency of e2e INT8 (ORT/OpenVINO) against one-to-many INT8 plus NMS (NCNN or ORT),
  split into T_net and T_post?
- In other words, what is the head worth, in milliseconds and AP-small, on exactly the dense scenes where it should
  matter?

The YOLO26 docs give no latency numbers for the one-to-one vs one-to-many comparison (same page). Whether any paper
has measured this is **unverified: pending the angle-5 pass**. My own searches found no YOLO26 edge paper reporting
which head it ran.

### 5b. NMS latency against objects per image

- The survey (§4, p. 14) says the post-processing bottleneck "becomes more pronounced under high-resolution inputs or
  dense-object scenes, where the number of candidate boxes increases sharply and amplifies decoding and NMS cost".
  It cites Oh, Kwon & Lee, *Sensors* 25(5):1376 (2025), a multi-NPU system, which is not aerial and has not been
  opened.
- The survey's §6.3 also lists "high-percentile or worst-case latency constraints are still rarely integrated" as an
  open problem.
- RT-DETR ([arXiv 2304.08069](https://arxiv.org/abs/2304.08069)) says in its abstract that "the speed and accuracy of
  YOLOs are negatively affected by the NMS". I did not verify its body-text NMS analysis (pending).

**Verdict: unsure.** The mechanism is stated in print. **A measured T_post curve against GT objects per image, on an
edge CPU, on dense aerial data, with P95/P99** was not found by me; it is pending the angle-5 pass. It is cheap with
VisDrone test-dev, whose object count per image runs past 500.

---

## 6. Energy per frame and sustained, throttled throughput

**Unverified: pending the angle-6 pass.** What I have:
- Two drone-edge benchmarks cited by MD-YOLO §2.2, neither opened:
  - Bakirci, *Eng. Appl. Artif. Intell.* 159:111775 (2025): Jetson Xavier on a drone, "accuracy, inference speed, and
    on-board resource utilization".
  - Rey et al., *Electronics* 14:638 (2025): several edge platforms including Jetson Orin NX; reports that INT8 PTQ
    raises YOLOv8 throughput "without significant accuracy degradation".
- The study's own blocker (issue #7): the Pi 5 has no software power telemetry, and a USB-C PD meter is not owned.
  Energy is therefore only as available as that meter.

**Verdict: unsure.** If the pass finds Pi 5 energy work, the open part is likely the combination: INT8/pruned ×
VisDrone resolution × sustained throttling × AP-small. A combination is a weak novelty claim. Treat energy as
supporting data, not as a lead angle.

---

## 7. YOLO11 vs YOLO26 on VisDrone at the edge

**MD-YOLO** (Zhang et al., *Algorithms* 19(9):737, published 2026-09-01, doi:[10.3390/a19090737](https://doi.org/10.3390/a19090737)),
read from the local PDF.
- Setup: trained **from scratch** ("no pretrained weights were used"), 640², 250 epochs, batch 6, 3 runs, RTX 4090
  FP32 FPS.
- Table 2:
  - YOLO11s: 32.9 mAP50 / 19.2 mAP50-95, 148.3 FPS.
  - YOLO26s: 36.9 / 20.0, 158.8 FPS.
- Table 3, AP_S/M/L:
  - YOLO11s: 12.6 / 33.4 / 42.6.
  - YOLO26s: 13.4 / 32.8 / 42.8.
- **No edge device, no INT8, no pruning.** Its §1 says YOLO26 removes "the NMS bottleneck", but it measures only on
  a desktop GPU.
- Its YOLO11s→YOLO26s mAP50-95 gap (+0.8) has the same sign as ours (+0.6 at 640) and similar size. Its mAP50 gap
  (+4.0) is much larger than ours.

**arXiv 2605.24831** (YOLO26 vs YOLOv8, VisDrone; already in `02-visdrone.md` §2.3) has no YOLO11 baseline and no
edge device.

**YOLO27:**
- Announced on stage at YOLO Vision 2026 on 2026-09-13. The Ultralytics recap says it "will be available this year"
  ([blog](https://www.ultralytics.com/de/blog/key-highlights-from-ultralytics-yolo-vision-2026)).
- It is **not on PyPI** as of `ultralytics` 8.4.179 (2026-10-17).
- Reviewer risk: if it ships before submission, say in one sentence why the study pins 8.4.149.

**Verdict: partially taken** for the desktop accuracy comparison. **An edge comparison was not found by me**; full
pass pending. Either way, keep the YOLO26 story about systems, as issue #1 already decided.

---

## Job B: additions that would make the paper novel, ranked

Cost assumptions come from our records:
- A 640 run takes about 3.6 T4-h for YOLO11s and 4.4 for YOLO26s. The 30-epoch prune recovery took 0.8 T4-h.
- 960 takes about 2× and 1280 about 3.1× the 640 time.
- The GPU is idle. **None of the top five needs more than about 5 T4-h.**
- `predictions.json` files are *not* in `C:\Users\srikr\uav-yolo-checkpoints\`. Where a Job B item says
  "saved predictions", budget 1–3 laptop CPU-h to regenerate them through `studyeval`, or pull them from the
  Kaggle outputs.

| Rank | Addition | What to run | Reuses | Cost | Must differ from | Novelty (honest) |
|---|---|---|---|---|---|---|
| **B1** | **Iso-latency frontier: is compression's saving better spent on pixels?** | Pi 5 (and the M35 in a reduced column): FP32/INT8 at 640/960/1280 for YOLO11s and YOLO26s; pruned30@640; **one new prune30 at 960 with a 30-epoch recovery**; inference-only intermediate sizes (e.g. 864) from the nearest trained weights. Plot AP-small against *measured* end-to-end latency, NMS included. Paired bootstrap at matched-latency pairs | All 10 baselines, INT8 export pipeline, `studyprune.py`, `studyeval`, bootstrap | **~1.6–2 T4-h** (prune@960 recovery); optional prune@1280 ~3.5 T4-h. **~6 Pi-h + ~5 phone-h** of benchmarking; ~1 day of CPU evaluation | xuanandsix (GPU, uncompressed); SAHI/ASAHI (tiling); E³-UAV (unread) | **Probably open in this exact form**; pending the angle-1 pass. The highest-value item |
| **B2** | **What YOLO26's NMS-free head is worth at the edge** | (a) Laptop: YOLO26s e2e under ORT INT8 and OpenVINO INT8 (not disabled), plus LiteRT INT8 with the disable bypassed to *show* the collapse. Score the one-to-one vs one-to-many heads for AP and AP-small. (b) Pi 5: T_net and T_post for e2e (ORT/OpenVINO) vs one-to-many + NMS (NCNN FP16, ORT INT8), with P50/P95/P99, **stratified by GT objects per image** | YOLO26s baselines (teammate's weights), export pipeline, `studyeval`, the `01-yolo26.md` code analysis | **0 T4-h**; ~1 laptop day; **~3–4 Pi-h** | Ultralytics docs (the fact); survey §4 and ref [157] (the mechanism) | **High** for the measured consequence; the fact itself is vendor-documented. Pending the angle-5 pass |
| **B3** | **Density-stratified compression damage** | Bin test-dev images by GT count (e.g. ≤50, 50–100, 100–200, >200). Per bin: AP and AP-small for FP32 vs ORT INT8, OpenVINO INT8 and prune30, with paired bootstrap; AP@100 vs AP@500 per bin | Saved or regenerated predictions, `studyeval`, bootstrap | **0 T4-h**, ~0.5 person-day | Nothing found (one empty search) | **Moderate, likely open.** Also gives the maxDets result a stratum where it is not negligible |
| **B4** | **Resolution × INT8 × size interaction** | ORT and OpenVINO INT8 of the 960 and 1280 baselines, both families; size-bucket ratios with CIs. Does the INT8 small-object penalty shrink when small objects cover more pixels? Luo et al. (ISPRS 2026) Eq. 5 predicts that it should | 960/1280 weights, INT8 pipeline, bootstrap | **0 T4-h**; ~6–8 laptop CPU-h | I-LW-DETR (640 only, one quantizer); Luo et al. (theory, no per-size AP) | **Moderate, likely open.** A clean empirical test of a published theoretical claim |
| **B5** | **Decompose the toolchain gap** | ORT `quantize_static` toggles: (i) head in float (matching OpenVINO's scope), (ii) per-channel, (iii) calibration MinMax/Entropy/Percentile, (iv) calibration-set composition (dense small-object images vs random). OpenVINO with and without the head ignored-scope | ORT/OpenVINO artifacts and pipeline, bootstrap | **0 T4-h**, ~1 laptop day | Wu et al. 2004.09602 (COCO, aggregate only); Cloudflight blog | **Moderate.** It is also **defensively necessary**: without it, "the quantizer matters 4×" is attackable as "you left the head in float on one arm" |
| **B6** | **Per-pyramid-level pruning sensitivity map** (planned contribution c) plus a ratio curve | Prune only the backbone-shallow, P3, P4 or P5 paths at matched parameters, 4–5 runs; uniform ratios of 10/20/40/50%, 4 runs; each with a 30-epoch recovery | `studyprune.py`, recipe, bootstrap | **~7–8 T4-h** | Choi & Tian 2023 (whole-network rates, IoU 0.5); I-YOLOv11n (asserted only) | **Open** per `03-pruning.md`, confirmed for I-YOLOv11n. The uniform result (1.17×) lowers the prior of a large effect, but a measured null is still a result |
| B7 | **Official-toolkit A/B and whether the protocol gap moves under compression** (issue #17) | `evalDET.m` under Octave on baseline and prune30/INT8 predictions | Predictions, issue #17 plan | 0 T4-h; ~1 day tool setup | Pending the angle-4 pass | Moderate. It protects every ΔAP-small claim |
| B8 | **Sustained throughput and J/frame on Pi 5** for the B1 frontier configurations | 10-minute runs with and without the active cooler, with a USB-C PD meter | B1 artifacts | 0 T4-h; ~5 Pi-h; **needs the meter** (issue #7) | Pending the angle-6 pass | Low to moderate. Supporting data |
| B9 | YOLO26s PTQ size buckets (second family) | ORT and OpenVINO INT8 of `yolo26s-640-s0` | Pipeline | 0 T4-h; ~3 CPU-h | `01-yolo26.md` §6 found no YOLO26 INT8 VisDrone result | Low to moderate. It falls out of B2 or B4 anyway |

Total for B1–B6: **about 10–14 T4-h** (well under one week's 60), **about 10 Pi-h plus 5 phone-h**, and about 4–5
laptop days. It fits in three weeks for two people, with the phone column reduced.

---

## Recommended paper shape

> **Revised after the angle 1/6 and 4/5/7 passes (see Addendum).**
> - Angle 1 is *partially taken*: cite Plastiras 2020 and Vandersteegen 2019 first, and claim only the controlled,
>   CPU-class, matched-latency measurement.
> - **Before committing to Lead 1, run a one-hour INT8 speed probe on the Pi 5.** Time YOLO11s 640 FP32 against ORT
>   INT8 and OpenVINO INT8. If INT8 is slower, as Marasini & Zhou found, the resolution budget has to come from
>   pruning. That runtime dependence then becomes a headline result in its own right, and the paper fills
>   Marasini's stated future work.
> - **Add a third lead:** the YOLO11s vs YOLO26s comparison on edge devices, which no paper has made.
> - Lead 2 gains the density-binned NMS-cost curve on the Pi 5 (angle 5b, likely open).

**Title direction:** *"Where should the compression budget go? Iso-latency evaluation of compressed YOLO11/YOLO26
detectors for small objects on a Raspberry Pi 5."*

**Lead 1 (B1): the iso-latency resolution trade.**
- One figure: AP-small against measured Pi 5 latency, for FP32, INT8 and pruned models at 640, 864, 960 and 1280.
- One table: pairs at matched latency, with paired-bootstrap ΔAP-small.
- The answer depends on the device's INT8 speedup and on NMS cost. That makes it a deployment guideline, not a
  tautology.
- Minimum to run: YOLO11s FP32 and INT8 at 640/960/1280 plus prune30@640 on the Pi 5, the one prune@960 recovery
  (about 2 T4-h), and YOLO26s at 640 and 960.

**Lead 2 (B2): YOLO26's NMS-free head at the edge.**
- Cite the Ultralytics docs for the fallback, then measure what it costs:
  - Does the one-to-one head survive ORT/OpenVINO INT8?
  - What T_post and P99 does NMS add on the Pi 5, as a function of objects per image?
- Minimum: the laptop e2e INT8 scoring plus about 3 Pi-h.

**Supporting section: calibrated compression sensitivity.**
- Report the PTQ and pruning size-bucket results as **measured, modest effects** (INT8 1.4–1.5×, pruning 1.17×)
  with pre-registered bars and CIs.
- Cite I-LW-DETR, Choi & Tian and I-YOLOv11n first.
- Add B5 (essential) and B3 (cheap) so that "the toolchain dominates object size" becomes an explained result.
- B6 goes in only if the GPU-hours are spent anyway.

**Methods box, not a contribution:** maxDets=500 (worth about 0.3–0.5 points here, more on dense images), area
splits outside `model.val()`, and the official-toolkit caveat (B7 if time allows).

**Drop as headline claims:**
- "first to show small objects suffer under INT8 or pruning": taken.
- "first to report the NMS-free fallback": vendor-documented.
- "maxDets pitfall": too small.

---

## Search log

**Searches run by me, nothing directly relevant:**
- "object detection accuracy stratified by object density per image VisDrone crowd density compression quantization".
  Nothing stratifies by density under compression. Leads only: ISPRS 2026 (QAT, scale), a chatslide.ai page
  (generated summary, not citable).
- "TIDE error analysis quantized pruned object detector which errors increase compression". Only TIDE itself
  ([arXiv 2008.08115](https://arxiv.org/abs/2008.08115)); no TIDE analysis of compressed detectors.
- "test-time input resolution different from training resolution YOLO VisDrone accuracy latency trade-off same
  weights". No same-weights resolution sweep on VisDrone with latency. Leads: arXiv 2308.04774, 2203.10194,
  2608.23636 (unread).
- "Ultralytics YOLO27 release". Announcement only; no package.

**Angle-2/3 search pass (36 web searches plus 3 Europe PMC queries), queries with nothing relevant:**
- "PTQ YOLO VisDrone AP small medium large OpenVINO NNCF"
- "ONNX Runtime static vs OpenVINO NNCF INT8 YOLO paper"
- "YOLO INT8 calibration entropy percentile minmax per-channel per-tensor study"
- "benchmark PTQ remote sensing DOTA DIOR small objects"
- "TensorRT INT8 drone VisDrone APs FP16 vs INT8"
- "small-object-aware PTQ 2026 foreground calibration"
- "what do compressed / pruning identified exemplars detection"
- "uniform vs global layer-adaptive pruning YOLO small AP P2/P3"
- "empirical pruning COCO AP_S/M/L sparsity Faster R-CNN RetinaNet"
- "Torch-Pruning DepGraph uniform VisDrone mAP_small"
- "structured pruning aerial AP_S sweep 2026"
- "compression fairness object sizes"
- Europe PMC: VisDrone + INT8/PTQ/QAT (18 hits, none with per-size INT8 results)
- Europe PMC: pruning + VisDrone + APs (7 hits, only I-YOLOv11n relevant)

Set aside after checking:
- BRECQ and QDrop: no APs column.
- InlierQ (arXiv 2602.03472): abstract only.
- MCAQ-YOLO (arXiv 2511.12976): full text not checked.
- van Manen et al. (SPIE): paywalled, **unsure**.

**Formerly pending:** these searches were run by the two later passes. Their results and their empty-query logs are
in the Addendum below.

---

## Addendum: the angle 1/6 and angle 4/5/7 passes (2026-10-17)

Two search passes, about 27 and 45 queries. Each read full texts where it could get them and marks "abstract
only" where it could not. The main session spot-checked three claims against primary text:
- Marasini & Zhou §VI-B: −43% FPS and +69.4% J/frame, confirmed in the PDF text.
- The exporter end-to-end conditions at L682–713: confirmed in the installed `ultralytics` 8.4.149.
- test-dev has 1,610 images, so it is the correct split and not test-challenge (1,580): confirmed.

### A1. Angle 1: compression spent on resolution, partially taken

Ranked closest prior work:
1. **Plastiras et al., "Joint Quantization and Tiling", AICAS 2020** (doi:10.1109/AICAS48895.2020.9073885;
   zenodo.org/records/4442387).
   - INT8 (DP4A) plus selective tiling keeps native resolution.
   - Table I, Jetson TX2 CPU: quantized selective tiling scores 98.6% "accuracy" in 0.098 s; the FP32 detector on a
     resized image scores 79.6% in 0.089 s.
   - Not covered: AP or AP-small (it uses a detection rate on a custom 1,500-image one-class pedestrian set), formal
     latency matching, pruning.
   - **A reviewer will cite this.**
2. **Vandersteegen et al., MVA 2019** (arXiv 1904.02024).
   - YOLOv3 on VisDrone2018 val: mAP50 45.3 at 608 and 56.8 at 960 (Table 1).
   - TensorRT INT8 on a Xavier: about 12 ms at 608 and about 26 ms at 960 (§4.2).
   - So INT8@960 implicitly beats FP32@608, but the paper never frames it as matched latency, reports no AP-small,
     does no pruning and runs no CPU device.
3. **E³-UAV, Suo et al., IEEE IoT J. 2023** (doi:10.1109/JIOT.2023.3301623; arXiv 2308.04774).
   - YOLOv3 on VisDrone at 416/608/800/1024, TensorRT FP16, Xavier NX.
   - Resolution is chosen jointly with altitude, speed and sampling rate for mission energy.
   - No compressed-vs-uncompressed comparison and no AP-small.
4. **MCUNetV2, NeurIPS 2021** (arXiv 2110.15352): INT8 plus patch-based inference frees memory for a higher
   resolution. Constrained by memory, not latency, and on microcontrollers.
5. **Kaymakçı et al., INISTA 2023** (doi:10.1109/INISTA59065.2023.10310562), abstract only: YOLOv5/7/8 at 640 and
   1280 on VisDrone, Jetson Nano and Orin; TensorRT about doubles FPS for at most 6% mAP loss.
6. **Remix, MobiCom 2021** (doi:10.1145/3447993.3483274): non-uniform partitioning of 4K images under a latency
   budget; no compression.
7. **Adjacent work:**
   - Messmer et al. ICPR 2022 (resizing driven by altitude); AdaScale; ApproxDet.
   - AFWD-Net, Sensors 26(19) (learned foveated crops, no INT8).
   - arXiv 2603.02142 (high resolution with small models is Pareto-optimal, solar panels, no compression).
   - arXiv 2203.10194 (YOLOv4 416→1120 on a K80; no edge device or INT8).
   - arXiv 2608.23636 (orchard, accuracy only).
   - **arXiv 2609.12920 "Input Resolution Matters"** (YOLO11n latency on Orin NX, COCO, no accuracy). Avoid
     confusion with its title.
   - Shakya et al. IEEE Access 2025, abstract only (Orin Nano, FP16, energy).
   - HF card steven0226/uav-traffic-vision, not peer-reviewed: YOLO26s VisDrone val AP-small 0.184 at 640 vs 0.260
     at 1024, on a 4090.

**Open:** controlled matched-latency on CPU-class devices (Pi 5, phone, laptop CPU); VisDrone test-dev AP-small with
bootstrap CIs; YOLO11s/26s at 640/960/1280; two INT8 runtimes plus structured pruning. No precedent spends a pruning
budget on resolution.

**Risk to test first:**
- Marasini & Zhou (arXiv 2609.04705 §VI-B): OpenVINO/NNCF INT8 on the Pi 5 Cortex-A76 gave **−43% FPS and +69.4%
  J/frame** versus FP32. NNCF inserted convert ops at every quantized boundary, so the DotProd path was largely
  bypassed. The A76 lacks I8MM.
- Rey et al. 2025 (§4.1) could not get FP16 or INT8 working on a Pi 5 with NCNN.
- On the Pi 5, INT8 may buy no latency at all. That would itself be a reportable runtime-dependence result.

### A2. Angle 6: energy and thermal behaviour, partially taken

1. **Marasini & Zhou, arXiv 2609.04705** (4 Sep 2026, preprint; full text read).
   - Setup: Pi 5 with passive vs active cooling, YOLOv8n on RDD2022 road damage, OpenVINO FP32. 30-minute runs, n=3,
     at 23 ± 2 °C.
   - Instruments: POWER-Z KM003C at 1 kS/s; `vcgencmd get_throttled`; bootstrap CIs.
   - Fanless at full speed: 1,823 throttle events, plateau at 84.8 °C.
   - Their DVFS scheduler: 0.531 J/frame vs 0.563 for the actively cooled reference.
   - Not covered: UAV data, multiple resolutions, AP-small, phones, ONNX Runtime INT8.
2. **Rey et al., Electronics 14(3):638, 2025** (arXiv 2502.15737; full text read).
   - Indoor drone dataset at 640; YOLOv8n/s.
   - Pi 5 with NCNN FP32 only; YOLOv8s ran at 7.32 FPS and 5.42 W, 1.50 J per inference (Table 4).
   - A 60-second window, no thermal measurement, no AP-small.
3. **Ghoniya et al., IJERT 15(3), 2026** (low-tier): TFLite INT8 on a Pi 5 throttles at about 120 s and 84 °C, and
   FPS falls 2.6→2.1. Wattages are assumed, not measured.
4. **Alqahtani et al., arXiv 2409.16808v3**: Pi 3/4/5 and Jetsons on COCO; mWh per request from a UM25C. No thermal,
   no Pi-CPU INT8.
5. **Others:**
   - Minott et al. SoutheastCon 2025, abstract only (Pi 5 + Coral, about 8.3 W, 80 °C).
   - Ranjan et al. AI 7:354, 2026 (arXiv 2607.09835): Pi 5 ORT FP32 throttles within minutes without a cooler.
   - Boddu & Mukherjee, arXiv 2506.09300: internally inconsistent latency figures; do not lean on it.
   - Benoit-Cattin et al. Electronics 9:2106, 2020: the Pi 4 methodology precedent, with throttling up to about −90%.
6. **Bakirci, EAAI 159:111775, 2025** (doi:10.1016/j.engappai.2025.111775): paywalled, so **unverified**. The indexed
   abstract describes YOLOv7-tiny/v9-tiny/v10-n/s on a drone-mounted Jetson Xavier with power figures.
7. **Phones:** no peer-reviewed sustained YOLO throughput + J/frame study was found. One abstract-only thesis:
   Abdullah, Univ. Helsinki 2026.

**Open:** sustained J/frame and throttled throughput on UAV data as a function of imgsz × compression (ORT vs
OpenVINO INT8, pruned); AP-small per joule; Pi 5 vs phone. Phone energy without root is hard, so state the method.

### A3. Angle 4: the maxDets pitfall, a reproducibility note

- **Prevalence.** 18 VisDrone YOLO papers from 2024–2026 were regex-searched in full text (MDPI Sensors,
  Sci. Rep., arXiv). **None states maxDets, none uses the official toolkit, and most visibly use Ultralytics `val()`.**
  arXiv 2605.24831 §4.5 names "the built-in evaluation pipeline provided by the Ultralytics YOLO framework".
- **Closest precedent:** RF100-VL (arXiv 2505.20612).
  - §3.2 uses pycocotools with maxDets=500 "because there are many images with more than 100 objects".
  - App. B / Table 4: Ultralytics `val()` over-estimates by 1.7–2.6 mAP, which they attribute to its PR integration.
  - Different dataset, and it does not separate the maxDets effect from the integrator effect.
  - SKU-110K (arXiv 1904.00853 §4) reports AR300. TIDE defaults `max_dets=100`.
- **Why our gap is small:** pycocotools applies maxDets **per image per category** (`evaluateImg`). The count that
  matters is objects per image per class, so `INVARIANTS.md` trap 3's ">100 objects per image" overstates how often
  the cap binds.
- **Toolkit quirks (code-verified in `evalDET.m` / `calcAccuracy.m`):**
  - It truncates detections per image across all classes **in file order, unsorted by score**. The pass infers that
    this drives the large AR100/AR500 gaps in TPAMI Table 2.
  - VOC all-point interpolation.
  - Class APs are weighted by the number of images containing the class. This is undocumented; verify it before
    claiming it.
  - Objects more than 50% inside ignored regions are dropped.
  - AP@maxDets=500 is defined only in the code. The TPAMI paper (arXiv 2001.06303 §4.2) defines only AR1/10/100/500.
- **Reproducibility trap:** the VisDrone-Dataset README has the test-dev and test-challenge Baidu links swapped
  (issues #45, #52, #56). Our test-dev has 1,610 images, so it is correct.
- **Verdict:** the general pitfall is not novel. A VisDrone three-way comparison (Ultralytics / COCO / toolkit) on
  identical predictions appears new, but the effect is small. It belongs in an appendix or methods box (B7).

### A4. Angle 5: YOLO26's end-to-end head and NMS cost

**Wording corrections for the map and `01-yolo26.md`** (verified against `exporter.py` v8.4.149):
- `model.end2end = self.args.nms is False` (L683). Since PR #26066 ("Unify `end2end` under `nms`", merged
  2026-09-05, v8.4.142), YOLO26 predict, val and export **default to one-to-many + NMS** (`nms=None`). This matches
  our frozen `nms=null`. Any one-to-one measurement must pass `nms=False` explicitly.
- TensorRT INT8 fallback: `trt >=10.3.0,<10.4.0 and is_jetson(jetpack=6)` only (L705–711), not "TRT 10.3+". A
  JetPack 6 board with TRT 10.7 keeps end-to-end under INT8 (arXiv 2609.13551 did exactly this).
- **TensorRT <8.5.0 disables end-to-end at any precision** (Mod operator, L697; issue #24607). An original Jetson
  Nano (JetPack 4.x, TRT 8.2) therefore falls back **even in FP16**. This is relevant to *Confirm the Jetson board*.
- LiteRT falls back only for `quantize ∈ {8, "w8a16"}`; `w8a32` keeps end-to-end. The main session did not
  independently check that ORT and OpenVINO INT8 keep it.

**Prior record of the fallback:**
- The Ultralytics end-to-end guide (since 2026-03; LiteRT INT8 added 2026-08-28, PR #25971) and the Jetson guide FAQ.
- Issues #23841, #24607, #23753 (RKNN) and #24668 (YOLO26 TensorRT INT8 confidence compressed to about 0.45).
- The official YOLO26 paper (arXiv 2606.03748 §3.5).
- YOLOv10 issue #118 (2024, NCNN `topk`).
- Two papers ignore the fallback: Sapkota et al. (arXiv 2509.25164 §4.2) claim YOLO26 "maintains stability" under
  INT8 without numbers, and arXiv 2607.09835 credits the Pi 5 speed to NMS removal.

**NMS cost:**
- Overload (CVPR 2024, arXiv 2304.05370 §3.2, Xavier NX): NMS time is dominated by the object count.
- Phantom Sponges (arXiv 2205.13618); RT-DETR Table 1 (T4, does not vary density).
- YOLOv10 Table 2 (T4: 7.07→2.44 ms without NMS); Fast NMS; Matrix NMS (GPU); TrackFlood (arXiv 2609.33948).
- Oh, Kwon & Lee (Sensors 25(5):1376) measure 0.46 ms of post-processing on an NPU with COCO and never vary
  density. The survey mischaracterises it, so **do not cite it for dense-scene NMS cost**.
- **Likely open:** NMS latency on an ARM CPU or phone for naturally dense scenes binned by ground-truth count.
  arXiv 2610.19369 §V asks for the NMS vs end-to-end comparison on CPU as future work.

### A5. Angle 7: YOLO11 vs YOLO26 on UAV data

| Paper | Data | Devices | YOLO11s vs YOLO26s |
|---|---|---|---|
| Aero-YOLO26, Sensors 26(19):6346 (Oct 2026) | VisDrone2019 | RTX 4060 laptop, ORT | mAP50-95 0.233 vs **0.245**; ONNX FPS 149.2 vs 128.9 |
| PFE-Det, Sensors 26(15):5003 (Aug 2026) | VisDrone test-dev, COCO-style | RTX 4090 D | AP **0.176** vs 0.160; APs 0.080 vs 0.082 |
| GMD-YOLO26, Research Square (Sep 2026) | VisDrone, UAVDT | — | mAP50 35.3% |
| arXiv 2607.09835 (Jul 2026) | fish, not UAV | Pi 5, ORT FP32, UL 8.4.23 | 2.79 vs 2.90 FPS |
| arXiv 2610.19369 (Oct 2026) | MOT17/20 | RTX 4070 Ti | NMS vs end-to-end ablation |
| arXiv 2609.13551 (Sep 2026) | fruit | Orin Nano Super, TRT 10.7 INT8 | YOLO26s only |

The two GPU papers contradict each other on VisDrone. **No peer-reviewed YOLO11-vs-YOLO26 VisDrone comparison was
found on a Pi, phone or Jetson, with INT8, or with pruning: likely open.**

### A6. Gaps the literature names, which this study can fill

- **Aero-YOLO26** (end of §4): its FPS was measured "using a laptop GPU and ONNX Runtime, rather than an embedded UAV
  computing platform".
- **PFE-Det:** "onboard measurements on embedded UAV platforms (e.g., Jetson-class devices) are left for future
  work."
- **Marasini & Zhou** §VII: "Runtime dependence is left for future work." §VIII lists runtime-specific INT8
  evaluation.
- **arXiv 2610.19369 §V:** repeat the NMS vs end-to-end comparison on CPU.
- **arXiv 2607.09835**, limitations: ONNX Runtime only; TFLite and OpenVINO not assessed.
- **Habash et al., Sensors 25(24):7563** §6.1: few studies measure the FPS and energy effect of quantization and
  pruning on embedded devices; pre- and post-processing latency often goes unreported. §5.3.1: the effect of
  quantization depends on resolution (cf. B4).
- **Rey et al.** §4.1: INT8 does not work on a Pi 5 with NCNN. §7: evaluate YOLO11-era models on drones.
- **Ultralytics guide:** the one-to-many head's +0.6–0.8 COCO mAP gain does "not guarantee a gain on every dataset",
  which our saved predictions can test on VisDrone. Checkpoints were "trained with max_det=300", which matters for
  dense VisDrone images.

### A7. Empty queries from the two passes

- **Angles 1/6:**
  - joint resolution × quantization latency Pareto for detection (classification only)
  - VisDrone YOLO Pi INT8 640/960/1280 AP-small
  - SAHI on a Jetson or Pi with measured FPS
  - once-for-all elastic resolution for detection
  - "INT8" "higher resolution" "same latency" YOLO
  - iso-latency resolution vs model size on a Pi for drones
  - pruned YOLO at higher resolution with the same inference time
  - VisDrone Jetson INT8 multi-resolution AP_S Pareto
  - ClusDet/DMNet/UFPMP edge latency
  - Pi 5 YOLO11 NCNN/OpenVINO INT8 J/frame, peer-reviewed
  - smartphone YOLO J/inference with thermal
  - MLPerf Mobile throttling for detection
  - Pi 5 VisDrone power and temperature in flight
  - smartphone DNN throttling with energy per frame (LLM papers only)
- **Angles 4/5/7:**
  - VisDrone evaluation protocol discrepancy COCO API toolkit
  - VisDrone toolkit calcAccuracy evalClass weighting
  - YOLO26 pruning UAV
  - YOLO26 vs YOLO11 UAV small object (vendor pages only)
  - NMS latency Raspberry Pi dense scenes
  - NMS runtime VisDrone edge
  - YOLOv10 INT8 one-to-one head accuracy drop
  - YOLO26 Jetson Orin Nano VisDrone
  - YOLO26 Android LiteRT INT8 small object
  - arXiv API `YOLOv10 AND INT8` / `NMS AND latency AND "Raspberry Pi"` (0 hits)
  - ultralytics GitHub issues "VisDrone maxDets" / "VisDrone official toolkit" (0 hits)
