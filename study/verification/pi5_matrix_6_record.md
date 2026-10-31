# Raspberry Pi 5 matrix, session 6 (B3 and B4 at 1120): session record

Written on 2026-11-01, after the run. Vidhyarth ran the session with `pi/wizard.sh`, pushed the small files (`880ddd7`)
and sent the detections; they were scored on the laptop with `scripts/studyscore.py` under its 4 GB cap. The cells
and their order of priority are #37's.

## Question

What accuracy and latency do B3 (YOLO11s) and B4 (YOLO26s), both trained at 1280 (RUNS.md), give at 1120 on the
Raspberry Pi 5, as FP32 baselines and as ONNX Runtime INT8? The 1120 cells were added to the matrix by the dev's
decision of 28 October (#37, open question 1), with no new training (#10), to give more equal-latency pairs. B4's INT8
cell is the only YOLO26s INT8 point above 960: B4's at 1280 was cut from session 2 (`4058740`), its calibration needing
more than 8 GB.

## Plan and code

- Plan: `plans/pi5-matrix-6-1120.json` (`plan_sha256` `365a2727…59250`, which the repo's plan gives through
  `studysession.load_plan`). 8 full cells, all 1,610 test-dev images, batch 1. The reference runs, first and last:
  ONNX Runtime FP32, A1 at 640, on the first 200 images, as in session 1.
- The rules of #8: 4 threads, 20 untimed warm-up images, the 50 °C start gate with its 3-minute timeout, cell order
  shuffled with seed 0, the drift check at ±3 %.
- **The first session with no throttle re-run** (INVARIANTS.md §4.7, `16221bb`, "from matrix sessions 2 and 6 on";
  session 2 started before it). A throttled cell keeps its one attempt and its flag, so every cell here ran once.
- **The CPU locked at 2000 MHz** (INVARIANTS.md §4.7). The setup checks passed: governor `performance`, no desktop, the
  apt timers off, ncnn 1.0.20260526, ONNX Runtime 1.31.0, OpenVINO 2026.4.1, `vcgencmd` clock 2000.
- Room 28 °C, as entered at the start, fan 2,038 rpm, idle CPU 57.3 °C: the warmest idle and the fastest fan of the
  matrix so far (51.2–56.8 °C, 1,907–2,035 rpm). The session started 30 minutes after session 2 ended.
- Both INT8 exports were calibrated on the same 548 VisDrone val images (`images_sha256` `1189ee95…`) at 1120, with the
  stock recipe: per-tensor weights, which #41 has since replaced (INVARIANTS.md §8).
- The detections came as `pi5-matrix-6-1120.tgz` (sha256 `b65035bb…03a86`, whole by `gzip -t`), with a
  `predictions.json` for every cell and both reference runs (10). Its other files match the pushed ones except for
  line endings (and `runner.pid`, which is not pushed). The ground truth at 1120 was built by this scoring (`runs/gt/`).

## Results

Session `bench/pi5/pi5-matrix-6-1120`, 31 October, 18:51 to 23:08 IST (4.3 h, against the plan's 3.7 h: see below).
**Drift: 409.1 → 403.0 ms, −1.5 %: ok**, but both reference runs were slower than in any earlier session (below).
Every cell completed.

| Cell | Precision recorded | AP | AP50 | AP-small | AR@500 | Median latency, ms | Attempts | Throttle |
|---|---|---|---|---|---|---|---|---|
| B3 1120, ONNX Runtime | FP32 | 0.2541 | 0.4309 | 0.1495 | 0.4460 | 1315.4 | 1 | throttled |
| B3 1120, OpenVINO | FP32 | 0.2541 | 0.4309 | 0.1495 | 0.4460 | 1012.5 | 1 | throttled |
| B3 1120, NCNN | FP16 | 0.2528 | 0.4301 | 0.1482 | 0.4447 | 596.8 | 1 | none |
| B3 1120, ONNX Runtime | INT8 | 0.2401 | 0.4138 | 0.1390 | 0.4311 | 883.5 | 1 | none |
| B4 1120, ONNX Runtime | FP32 | 0.2610 | 0.4409 | 0.1536 | 0.4568 | 1397.8 | 1 | none |
| B4 1120, OpenVINO | FP32 | 0.2610 | 0.4409 | 0.1536 | 0.4568 | 1082.7 | 1 | throttled |
| B4 1120, NCNN | FP16 | 0.2591 | 0.4402 | 0.1512 | 0.4545 | 633.4 | 1 | none |
| B4 1120, ONNX Runtime | INT8 | 0.2235 | 0.3973 | 0.1251 | 0.4158 | 943.1 | 1 | none |

AP is COCO AP@[.50:.95] at maxDets 500 (INVARIANTS.md), from detections at conf 0.001; every AP here is a `coco/*`
stat (§4.2). Latency is the median of `e2e_deploy`: pre + net + post at the deployment threshold, 0.25. Each cell ran
once (§4.7).

- **OpenVINO FP32 gives ONNX Runtime FP32's AP to 4 digits** in both pairs, as in sessions 1 to 5. NCNN's FP16 costs
  0.0014 (B3) and 0.0019 (B4) AP. B4's is the largest of the matrix so far (0.0003–0.0017 before, the 0.0017 being
  B4's at 1280).
- **YOLO26s loses 0.0374 AP (14.3 %) to INT8 at 1120, YOLO11s 0.0140 (5.5 %)**: 2.7 times as much, the widest gap
  of the four sizes with a YOLO26s INT8 cell. See *INT8 against FP32*.
- **Both reference runs were slower than in any earlier session, the first by 4.6–5.5 %, and the cells likely carry
  it.** See *The reference runs*.
- **Every attempt held a median clock of 2000 MHz.** See *Throttle and clock*.
- These are the session's numbers, not the paper's equal-latency analysis (#18).

### Throttle and clock (INVARIANTS.md §4.7, #40, #42)

Three FP32 cells hit the soft temperature limit (`get_throttled` bit 3): B3's ONNX Runtime and OpenVINO cells and
B4's OpenVINO cell. Under the no-re-run rule each kept its one attempt with its flag. No NCNN or INT8 cell was
flagged. The clock and temperature per attempt come from `timings.csv`, counted as in session 1: the clock and
temperature are read at every image, `get_throttled` at every 50th (32 samples in a full cell, 4 in a reference run).
The throttle column counts the samples with bit 3 set. The paper reports throttling by the clock (§4.7, #42): the
images below 2000 MHz and the median clock.

| Cell | Attempt | Median latency, ms | Throttle during the run | Images below 2000 MHz | Clock min / median, MHz | Peak CPU |
|---|---|---|---|---|---|---|
| reference run, first | 1 | 409.1 | none (0 of 4) | 0 of 200 | 2000 / 2000 | 81.5 °C |
| reference run, last | 1 | 403.0 | none (0 of 4) | 1 of 200 | 1966 / 2000 | 82.0 °C |
| B3 1120, ONNX Runtime FP32 | 1 | 1315.4 | soft limit (9 of 32) | 513 of 1,610 | 1856 / 2000 | 83.2 °C |
| B3 1120, OpenVINO FP32 | 1 | 1012.5 | soft limit (5 of 32) | 193 of 1,610 | 1801 / 2000 | 83.7 °C |
| B3 1120, NCNN | 1 | 596.8 | none | 0 | 2000 / 2000 | 77.7 °C |
| B3 1120, ONNX Runtime INT8 | 1 | 883.5 | none | 0 | 2000 / 2000 | 75.5 °C |
| B4 1120, ONNX Runtime FP32 | 1 | 1397.8 | none (0 of 32) | 19 of 1,610 | 1856 / 2000 | 83.2 °C |
| B4 1120, OpenVINO FP32 | 1 | 1082.7 | soft limit (9 of 32) | 264 of 1,610 | 1801 / 2000 | 83.7 °C |
| B4 1120, NCNN | 1 | 633.4 | none | 0 | 2000 / 2000 | 77.7 °C |
| B4 1120, ONNX Runtime INT8 | 1 | 943.1 | none | 0 | 2000 / 2000 | 75.5 °C |

- **Every attempt held a median clock of 2000 MHz**, as every attempt used did in sessions 1 and 5 (session 1's
  discarded first attempts did not all). The three flagged cells had 193–513 of
  1,610 images below 2000 MHz, down to 1801 MHz. Against them, session 4's attempts at 1280 had 974–1,367 images below
  and medians of 1911–1966 MHz, and B2's at 960 (session 2) 1,262–1,377 and 1911 MHz.
- **B4's ONNX Runtime FP32 cell was not flagged, but 19 of its images ran below 2000 MHz**, down to 1856 MHz. The flag
  reads `get_throttled` at every 50th image only and misses short dips (§4.7, #42).
- **No re-runs.** Under the old rule the three flagged cells would have run again: about 1.8 h more, by their first
  attempts' times with their gates. Across sessions 1 to 5, 1 of 22 re-runs cleared the flag.
- **For latency, the clean cells are NCNN and INT8.** No NCNN or INT8 attempt has read below 2000 MHz in any session
  (28 attempts in sessions 1 to 6). Their peaks here were 75.5–77.7 °C, against 83.2–83.7 °C for the FP32 ONNX Runtime
  and OpenVINO cells.
- Neither reference run was flagged. The first held 2000 MHz on all 200 images, and the last had one image at 1966 MHz.
  Their throttle word was `0xe0000` throughout: the "has occurred" bits 17–19, already set when the session started.

### The reference runs: slower than in any earlier session

The first reference run gave 409.1 ms, 4.6–5.5 % above the first runs of sessions 1 to 5 (387.7–391.2 ms) and 5.3 %
above the pilot's 388.5 ms (§4.7). The last gave 403.0 ms, 1.8–3.2 % above the other sessions' last runs
(390.4–395.9 ms), with 1 image below 2000 MHz where theirs had 2–123. The drift check compares the two ends of one
session, so it passes at −1.5 %.

| Session, in run order | First reference run, ms | Network, median (fastest image), ms | Pre-processing, ms | Post-processing at 0.25, ms | Images below 2000 MHz | Throttle word at the start |
|---|---|---|---|---|---|---|
| 1 | 390.0 | 385.1 (382.4) | 3.24 | 1.66 | 4 of 200 | `0x80000` |
| 3 | 391.2 | 386.0 (383.1) | 3.27 | 1.66 | 19 of 200 | `0x0` |
| 4 | 391.2 | 385.6 (382.7) | 3.24 | 1.67 | 73 of 200 | `0xe0000` |
| 5 | 387.7 | 382.6 (380.8) | 3.25 | 1.65 | 0 of 200 | `0x0` |
| 2 | 388.3 | 383.1 (380.9) | 3.23 | 1.67 | 1 of 200 | `0x80000` |
| **6** | **409.1** | **403.6 (399.5)** | **3.46** | **1.65** | **0 of 200** | `0xe0000` |
| 6, the last run | 403.0 | 397.6 (395.1) | 3.40 | 1.67 | 1 of 200 | `0xe0000` |

- **The time is in the network and the pre-processing, on almost every image.** The runs use the same 200 images in the
  same order. Image by image against session 2's first run, the network took 1.2–10.1 % longer on all 200 (median
  +5.3 %), and pre-processing a median 6.8 % longer; post-processing at 0.25 took no longer (−1.1 %). Against the other
  first runs, the network was slower on 193–200 of the 200 images (median +4.5 to +5.5 %). Of the 20.8 ms over session
  2, 20.5 ms is the network's. The two steps that slowed are the ones run on four threads (ONNX Runtime, and OpenCV
  with `cv2_threads` 4); post-processing is `studybench.py`'s own numpy NMS.
- **What the records show did not change**: the kernel (6.12.109+rpt-rpi-2712, the same build), the OS image, Python,
  numpy, OpenCV and the three runtimes' versions, the model file (sha256 `22e9560f…`), the 200 images
  (`images_sha1` `9ecdc307…`), the thread settings and the clock settings. The run gave the same 90,219 detections. The
  clock read 2000 MHz on all 200 images, and no throttle read caught a current bit. The CPU ran as warm as in the other
  first runs (median 79.3 °C, peak 81.5, inside their 78.2–81.25 and 81.0–84.2 °C), and reading the clock took as long
  (12.8 ms). The throttle word was the one session 4's first run started with, at 391.2 ms.
- **What the records do not hold:** the other processes on the Pi and its load, its uptime or last reboot, and the core
  and SDRAM clocks. So they cannot say what took the time. Session 2 started 53 minutes after session 5 and ran at the
  usual speed; this session started 30 minutes after session 2.
- **The cells likely carry it too.**
  - The last reference run, after all eight cells, was still slow: its fastest image's network time is 395.1 ms,
    against 381.2–383.2 ms in the other sessions' last runs (+3.1 to +3.7 %).
  - Pre-processing is the same code for every model and runtime, so its time depends on imgsz alone. A straight line in
    pixel count through the median of the other sessions' attempts at each size (640, 800, 960 and 1280; 76 attempts)
    passes within 0.7 % of each median. Single attempts at 800, 960 and 1280 lie within −1.8 to +2.9 % of it (at 640
    up to +12 %, session 2's NMS-free cells). All 8 cells at 1120 lie above it, by 2.8–13.4 % (median +5.5 %).
  - On NCNN, each step up the ladder below 960 costs 3.2–5.4 % more than its pixel count. The step into 1120 costs
    6.6 % more (YOLO11s) and 8.8 % (YOLO26s), and the step out of it −0.5 % and +2.2 %. A 1120 latency about 3–4 % too
    high would give that.
- **So the session's own comparisons hold, and its latencies against other sessions' carry an offset, likely 3–5 %.**
  YOLO26s against YOLO11s and INT8 against FP32 set cells of this session against each other. The ladders below, and
  any equal-latency pair that sets a 1120 cell against another session's (#18), use 1120 latencies that are likely
  3–5 % high. The accuracy is unaffected.

### The start gate and the session's length

All 10 gates opened at the 3-minute timeout, with the CPU at 54.0–56.2 °C; 1 of the matrix's 96 gates so far has
opened on temperature (session 5's first). The session took 4.3 h against the plan's "about 3.7 h", with no re-runs:

| Part | Time | Of which |
|---|---|---|
| The 8 cells, one attempt each, and both reference runs | 4.3 h (18:51–23:08 IST) | cells 3.7 h, 10 gates 30 min, reference runs 3 min |

- **The cells ran 0.6 h over the estimate.** #37's 3.7 h is #25's latencies at 640 scaled by pixel count (3.06×) over
  1,610 images, with YOLO26s as fast as YOLO11s, plus a 3-minute gate for each of the 10 runs and about 8 minutes of
  reference runs:
  about 3.1 h of cells. The cells took 3.7 h:
  - 1,610 × each median latency is 211 minutes, against the estimate's 185. B3 at 1120 ran 3.24–3.48× A1 at 640
    (session 1) on each runtime, not 3.06×, and YOLO26s was 6.1–6.9 % slower than YOLO11s. The slowdown above is
    likely part of that.
  - each cell spent 1.4–1.7 minutes beyond 1,610 × its median latency, 12 minutes in all, as in sessions 4 and 5
    (1.3–1.8 minutes): the warm-up, the second post-processing pass at conf 0.001, and the untimed part of each image's
    loop. The estimate leaves it out.
- The gates and reference runs took 33 minutes.

## Against what we already have

### The training records

B3's and B4's training records (`verification/runs/yolo11s-1280-s0.json`, `yolo26s-1280-s0.json`) scored test-dev at
1280, their training size. There is no record at 1120, so these cells are not set against them: the difference would be
the input size, not the export. No laptop CPU score of these exports exists either. What checks them here is the
runtimes agreeing with each other (above) and the ladder below, as for session 5's cells at 800. At 1280, session 4's
B3 and B4 FP32 exports were within +0.0002 and −0.0014 AP of their records.

### INT8 against FP32

Each INT8 cell against the ONNX Runtime FP32 export of the same checkpoint at the same imgsz (INVARIANTS.md §4.4).
Differences are from the unrounded values.

| Checkpoint | imgsz | AP, FP32 → INT8 | AP lost | AP-small, FP32 → INT8 | AP-small lost | Latency, FP32 → INT8, ms | Speed-up |
|---|---|---|---|---|---|---|---|
| A1, YOLO11s | 640 | 0.1875 → 0.1682 | −0.0193 (−10.3 %) | 0.0904 → 0.0769 | −0.0135 (−15.0 %) | 389.8 t → 272.5 | 1.43× |
| A2, YOLO26s | 640 | 0.1943 → 0.1549 | −0.0395 (−20.3 %) | 0.0995 → 0.0698 | −0.0297 (−29.9 %) | 383.1 t → 275.9 | 1.39× |
| B1, YOLO11s | 800 | 0.2190 → 0.2040 | −0.0150 (−6.8 %) | 0.1163 → 0.1041 | −0.0122 (−10.5 %) | 629.9 t → 416.7 | 1.51× |
| B2, YOLO26s | 800 | 0.2218 → 0.1857 | −0.0361 (−16.3 %) | 0.1184 → 0.0952 | −0.0232 (−19.6 %) | 611.9 t → 431.2 | 1.42× |
| B1, YOLO11s | 960 | 0.2392 → 0.2226 | −0.0166 (−6.9 %) | 0.1365 → 0.1258 | −0.0107 (−7.8 %) | 909.7 t → 614.0 | 1.48× |
| B2, YOLO26s | 960 | 0.2408 → 0.2069 | −0.0339 (−14.1 %) | 0.1391 → 0.1192 | −0.0199 (−14.3 %) | 926.5 t → 644.4 | 1.44× |
| **B3, YOLO11s** | **1120** | 0.2541 → 0.2401 | −0.0140 (−5.5 %) | 0.1495 → 0.1390 | −0.0104 (−7.0 %) | 1315.4 t → 883.5 | 1.49× |
| **B4, YOLO26s** | **1120** | 0.2610 → 0.2235 | **−0.0374 (−14.3 %)** | 0.1536 → 0.1251 | **−0.0285 (−18.6 %)** | 1397.8 → 943.1 | 1.48× |
| B3, YOLO11s | 1280 | 0.2652 → 0.2548 | −0.0104 (−3.9 %) | 0.1619 → 0.1560 | −0.0060 (−3.7 %) | 1695.1 t → 1152.3 | 1.47× |

t: the cell was flagged throttled; B4's FP32 cell at 1120 was not, but had 19 images below 2000 MHz. Bold: this
session; 640 and 960 are sessions 1 and 2, 800 session 5, B3 at 1280 sessions 4 and 1. B4 has no INT8 cell at 1280.

- **B4 at 1120 is the only YOLO26s INT8 point above 960, and its gap to YOLO11s is the widest yet**: 2.7× in AP
  (0.0374 against 0.0140) and in AP-small (0.0285 against 0.0104), against 2.0–2.4× and 1.9–2.2× at 640, 800 and 960.
- **YOLO11s's INT8 loss falls as the input grows (flat from 800 to 960); YOLO26s's stops falling above 960.** YOLO11s
  loses 10.3, 6.8, 6.9, 5.5 and 3.9 % of its AP from 640 to 1280. YOLO26s loses 20.3, 16.3, 14.1 and 14.3 %, and its AP-small
  loss rises again at 1120 (14.3 % at 960, 18.6 % here). 960 → 1120 changes the checkpoint as well as the size (B2 to
  B4), so this does not say which of the two did it. B3 alone, from 1120 to 1280, goes from −5.5 to −3.9 %.
- It loses in every size range: AP-medium −12.4 % (B3 −5.0 %), AP-large −15.6 % (B3 −6.5 %), AR@500 −0.041 (B3
  −0.015).
- **The cause is #41's**, found on the laptop at 800: the stock recipe's one weight scale per layer, on YOLO26s's
  32-channel box branch (INVARIANTS.md §8). This session's INT8 cells are the stock recipe and stay as recorded.
  Session 7 runs B3's and B4's per-channel cells at 1120 (`onnxruntime-int8pc`) beside them.
- **At INT8, YOLO26s falls behind YOLO11s at 1120 too**: 0.2235 against 0.2401 AP and 0.1251 against 0.1390 AP-small,
  at 943.1 against 883.5 ms. In FP32 it is ahead (below). It is behind at INT8 at every size with a YOLO26s INT8 cell.
- **B4's INT8 export saves 14.0 % more detections at conf 0.001 than its FP32 export** (695,575 against 609,939), the
  most of any pair; B3's saves 2.6 % more (666,287 against 649,704). B2 at 800 stays the only pair that saves fewer.
- **The speed-up is 1.49× (B3) and 1.48× (B4)**, within the 1.39–1.51× of the other sizes, and YOLO26s's largest. Both
  sides of each pair ran in this session, so the slowdown above does not bear on it.
- **NCNN FP16 beats ONNX Runtime INT8 again**: 32.5 % less latency for B3 (596.8 against 883.5 ms) and 32.8 % for B4
  (633.4 against 943.1), with 0.0127 and 0.0356 more AP.

### The resolution ladders

**YOLO11s:**

| imgsz | Checkpoint | AP, FP32 | AP-small | ONNX Runtime FP32, ms | OpenVINO FP32, ms | NCNN, ms (AP) | ONNX Runtime INT8, AP / ms |
|---|---|---|---|---|---|---|---|
| 640 | A1 | 0.1875 | 0.0904 | 389.8 t | 294.0 | 171.5 (0.1866) | 0.1682 / 272.5 |
| 800 | B1 | 0.2190 | 0.1163 | 629.9 t | 474.6 | 276.8 (0.2179) | 0.2040 / 416.7 |
| 960 | B1 | 0.2392 | 0.1365 | 909.7 t | 712.6 t | 411.3 (0.2381) | 0.2226 / 614.0 |
| 1120 | **B3** | **0.2541** | **0.1495** | **1315.4 t** | **1012.5 t** | **596.8 (0.2528)** | **0.2401 / 883.5** |
| 1280 | B3 | 0.2652 | 0.1619 | 1695.1 t | 1308.2 t | 775.7 (0.2638) | 0.2548 / 1152.3 |

**YOLO26s:**

| imgsz | Checkpoint | AP, FP32 | AP-small | ONNX Runtime FP32, ms | OpenVINO FP32, ms | NCNN, ms (AP) | ONNX Runtime INT8, AP / ms |
|---|---|---|---|---|---|---|---|
| 640 | A2 | 0.1943 | 0.0995 | 383.1 t | 302.2 t | 173.8 (0.1932) | 0.1549 / 275.9 |
| 800 | B2 | 0.2218 | 0.1184 | 611.9 t | 489.6 | 286.0 (0.2208) | 0.1857 / 431.2 |
| 960 | B2 | 0.2408 | 0.1391 | 926.5 t | 766.3 t | 427.9 (0.2399) | 0.2069 / 644.4 |
| 1120 | **B4** | **0.2610** | **0.1536** | **1397.8** | **1082.7 t** | **633.4 (0.2591)** | **0.2235 / 943.1** |
| 1280 | B4 | 0.2690 | 0.1659 | 1675.1 t | 1400.7 t | 845.3 (0.2673) | — |

t: the cell was flagged throttled. Bold: this session; 640 and 960 are sessions 1 (YOLO11s) and 2 (YOLO26s), 800
session 5, 1280 session 4 (B3's INT8: session 1; B4 has none). The 1120 latencies are likely 3–5 % high against the
other sessions' (*The reference runs*).

- **AP rises at every step of both ladders, and 1120 lands between 960 and 1280.** For YOLO11s, 1120 gives 58 % of the
  960 → 1280 AP gain (+0.0150 of +0.0260) and 51 % of the AP-small gain. For YOLO26s it gives 71 % (+0.0202 of
  +0.0282) and 54 %. Each costs about half the latency step on NCNN (51 % and 49 %) and OpenVINO (50 %). 960 → 1120
  changes the checkpoint as well as the size (B1 to B3, B2 to B4); 1120 → 1280 is B3 or B4 alone.
- **On the same checkpoint, 1280 → 1120 costs B3 0.0111 AP and B4 0.0080**, and 0.0125 and 0.0122 AP-small, for 23 %
  and 25 % less NCNN latency.
- **Only NCNN and INT8 give clean latency comparisons** along the ladders: no attempt of either has read below 2000 MHz.
  The FP32 ONNX Runtime and OpenVINO columns mix flagged and unflagged cells, at median clocks from 1911 to 2000 MHz.

### YOLO11s against YOLO26s at 1120

| Runtime | AP, B3 / B4 | AP-small, B3 / B4 | Median latency, ms, B3 / B4 | B4 against B3 |
|---|---|---|---|---|
| ONNX Runtime FP32 | 0.2541 / 0.2610 | 0.1495 / 0.1536 | 1315.4 t / 1397.8 | +6.3 % |
| OpenVINO FP32 | 0.2541 / 0.2610 | 0.1495 / 0.1536 | 1012.5 t / 1082.7 t | +6.9 % |
| NCNN FP16 | 0.2528 / 0.2591 | 0.1482 / 0.1512 | 596.8 / 633.4 | +6.1 % |
| ONNX Runtime INT8 | 0.2401 / 0.2235 | 0.1390 / 0.1251 | 883.5 / 943.1 | +6.7 % |

- **In FP32, YOLO26s is 0.0068 AP and 0.0042 AP-small ahead**, with AR@500 +0.011 and AP-large +0.016. B4 led B3 by
  0.0038 and 0.0039 at 1280 (session 4), and by 0.0054 and 0.0043 in their training records. In INT8 the order flips
  (above).
- **On the clean runtimes, YOLO26s's latency gap keeps growing with the input size**: on NCNN +1.3 % at 640, +3.3 % at
  800, +4.0 % at 960, +6.1 % at 1120 and +9.0 % at 1280; on ONNX Runtime INT8 +1.2 %, +3.5 %, +4.9 % and +6.7 %. Both
  cells of each pair at 1120 ran in this session, so the slowdown above does not bear on the gap. Nothing in the
  records explains the growth.
- **On ONNX Runtime FP32 YOLO26s is slower too, by 6.3 %**, where at 640, 800 and 1280 it was faster (−1.7, −2.9 and
  −1.2 %) and at 960 1.9 % slower. B3's attempt here had 513 images below 2000 MHz and B4's 19, so the throttle narrows
  the gap if anything. On OpenVINO, both flagged (193 and 264 images below), it is +6.9 %.

## Known limitations

- One session, one seed's order; each cell ran once.
- The latencies are at 2000 MHz, not the Pi 5's stock 2400 MHz.
- Three FP32 cells' latencies carry a soft throttle, and B4's ONNX Runtime FP32 cell dipped below 2000 MHz without a
  flag (above); every attempt held a median of 2000 MHz. Their accuracy is unaffected: the detections do not depend on
  the clock.
- Both reference runs were slower than in any earlier session, for a reason the records do not show, and the cells
  likely carry it. Comparisons within the session stand; its latencies set against other sessions' are likely 3–5 %
  high.
- The INT8 cells are the stock recipe. The per-channel cells the matrix now uses are session 7's.
- There is no training record and no laptop CPU score at 1120, so the exports' accuracy is checked only by the runtimes
  agreeing and by the ladder.
- B4 has no INT8 cell at 1280, so the YOLO26s INT8 ladder stops at 1120.
