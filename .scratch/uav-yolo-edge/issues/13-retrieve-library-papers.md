# Retrieve the library papers

Type: task
Status: open
Blocked by: none

## Question

Pull three papers that automated retrieval could not reach. You have university library access; the research agents did not.

**1. I-YOLOv11n** - Sensors 25(15):4857, doi 10.3390/s25154857. THE MOST IMPORTANT OF THE THREE.

This paper states our core hypothesis almost verbatim, on UAV imagery, on a YOLO11 derivative. The full text was readable via Europe PMC (PMC12349104), but the EXPERIMENTS AND ABLATION SECTIONS could not be retrieved (MDPI, PMC and DOAJ all returned 403 or CAPTCHA).

Read the ablation section by hand and answer one question: **do they isolate the pruning choice in a controlled ablation, or is it bundled into their +7.1 mAP50 headline alongside their four architectural contributions (RFCBAMConv, STCMSP, DFPC pyramid, Transformer-DyHead)?**

If it is bundled, the quantification gap is real and contribution (c) stands as reframed. **If they DO have an isolating ablation, the framing changes and the contribution question reopens.** Do not skip this check.

**2. Deployment-oriented pruning for real-time object detection: a survey** - Applied Intelligence 56:380 (2026), doi 10.1007/s10489-026-07416-6. Paywalled.

Framed entirely around the FLOPs-versus-latency gap: static-graph export, operator coverage and fusion, tensor-shape constraints, hardware parallelism granularity. This is the single best related-work anchor for the benchmark half of the paper, and it establishes that the FLOPs/latency gap is a recognised open problem rather than our rookie mistake.

**3. MD-YOLO** - Algorithms 19(9):737 (MDPI). mdpi.com returns 403 to automated fetch.

One of very few published YOLO26-on-VisDrone baselines, so it is a direct sanity check on our own numbers before spending GPU-hours. **Read it with suspicion**: its abstract claims YOLO26 uses DFL, which is wrong, since YOLO26 sets reg_max=1 and removes DFL. Extract numbers and setup; do not inherit its architectural claims. Also check whether it already makes a small-object-specific compression claim.

For each: record the reported metrics, the split, the imgsz, the training schedule, and whether it is citable as a comparison point or only a rough sanity check.
