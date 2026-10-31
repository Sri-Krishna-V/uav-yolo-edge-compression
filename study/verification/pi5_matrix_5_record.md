# Raspberry Pi 5 matrix, session 5 (B1 and B2 at 800): session record

Written on 2026-10-31, after the run. Vidhyarth ran the session with `pi/wizard.sh`, pushed the small files (`bcc51f3`)
and sent the detections; they were scored on the laptop with `scripts/studyscore.py` under its 4 GB cap. The cells
and their order of priority are #37's.

## Question

What accuracy and latency do B1 (YOLO11s) and B2 (YOLO26s), both trained at 960 (RUNS.md), give at 800 on the
Raspberry Pi 5, as FP32 baselines and as ONNX Runtime INT8? The 800 cells were added to the matrix by the dev's
decision of 28 October (#37, open question 1), with no new training (#10), to give more equal-latency pairs. B2's INT8
cell is the first YOLO26s baseline quantized on the Pi 5 so far: A2's (640) and B2's (960) are in session 2, still to
run, and B4's was cut from it.

## Plan and code

- Plan: `plans/pi5-matrix-5-800.json` (`plan_sha256` `24fbaa06…3084c`). 8 full cells, all 1,610 test-dev images,
  batch 1. The reference runs, first and last: ONNX Runtime FP32, A1 at 640, on the first 200 images, as in session 1.
- The rules of #8: 4 threads, 20 untimed warm-up images, the 50 °C start gate with its 3-minute timeout, cell order
  shuffled with seed 0, the drift check at ±3 %, one re-run of a crashed or throttled cell.
- **The CPU locked at 2000 MHz** (INVARIANTS.md §4.7). The setup checks passed: governor `performance`, no desktop, the
  apt timers off, ncnn 1.0.20260526, ONNX Runtime 1.31.0, OpenVINO 2026.4.1, `vcgencmd` clock 2000.
- Room 27 °C, as entered at the start, fan 1,907 rpm, idle CPU 51.2 °C: a degree cooler and the coolest idle of the
  matrix so far (sessions 1, 3 and 4: 28 °C, 53.5–56.8 °C).
- Both INT8 exports were calibrated on the same 548 VisDrone val images at 800, with the same export arguments and
  toolchain (their manifests).
- The detections came as `pi5-matrix-5-800.tgz` (sha256 `1f2d1c00…71f13`), with a `predictions.json` for every attempt
  of every cell and both reference runs (12). Its other files match the pushed ones except for line endings (and
  `runner.pid`, which is not pushed). The ground truth at 800 was built by this scoring (`runs/gt/`).

## Results

Session `bench/pi5/pi5-matrix-5-800`, 31 October, 09:32 to 12:33 IST (3.0 h, against the plan's 2.1 h: see below).
**Drift: 387.7 → 390.4 ms, +0.7 %: ok**, with both reference runs flagged throttled (below). Every cell completed.

| Cell | Precision recorded | AP | AP50 | AP-small | AR@500 | Median latency, ms | Attempts | Throttle |
|---|---|---|---|---|---|---|---|---|
| B1 800, ONNX Runtime | FP32 | 0.2190 | 0.3753 | 0.1163 | 0.3953 | 629.9 | 2 | throttled |
| B1 800, OpenVINO | FP32 | 0.2190 | 0.3753 | 0.1163 | 0.3953 | 474.6 | 1 | none |
| B1 800, NCNN | FP16 | 0.2179 | 0.3749 | 0.1153 | 0.3941 | 276.8 | 1 | none |
| B1 800, ONNX Runtime | INT8 | 0.2040 | 0.3584 | 0.1041 | 0.3751 | 416.7 | 1 | none |
| B2 800, ONNX Runtime | FP32 | 0.2218 | 0.3811 | 0.1184 | 0.4075 | 611.9 | 2 | throttled |
| B2 800, OpenVINO | FP32 | 0.2218 | 0.3811 | 0.1184 | 0.4075 | 489.6 | 1 | none |
| B2 800, NCNN | FP16 | 0.2208 | 0.3802 | 0.1173 | 0.4062 | 286.0 | 1 | none |
| B2 800, ONNX Runtime | INT8 | 0.1857 | 0.3417 | 0.0952 | 0.3694 | 431.2 | 1 | none |

AP is COCO AP@[.50:.95] at maxDets 500 (INVARIANTS.md), from detections at conf 0.001; every AP here is a `coco/*`
stat (§4.2). Latency is the median of `e2e_deploy`: pre + net + post at the deployment threshold, 0.25. A throttled
cell's row is its second attempt.

- **OpenVINO FP32 gives ONNX Runtime FP32's AP to 4 digits** in both pairs, as in sessions 1, 3 and 4. NCNN's FP16
  costs 0.0012 (B1) and 0.0010 (B2) AP, within the 0.0003–0.0017 of sessions 1, 3 and 4.
- **B2's INT8 loses 0.0361 AP (−16.3 %)** against its FP32 export, 2.4 times B1's 0.0150 (−6.8 %). See *INT8 against
  FP32 at 800*.
- These are the session's numbers, not the paper's analysis: the equal-latency comparisons wait for sessions 2 and 6.

### Throttle and clock (INVARIANTS.md §4.7, #40)

Two cells, both ONNX Runtime FP32 (`summary.csv`), hit the soft temperature limit (`get_throttled` bit 3) and were
re-run once, as #8 says; both throttled again. By the decision of 29 October (#40), they are used with their flag. No
OpenVINO, NCNN or INT8 cell was flagged. The clock and temperature per attempt, from `timings.csv`, counted as in
session 1: the clock and temperature are read at every image, `get_throttled` every 50th (32 samples in a full cell, 4
in a reference run), and the throttle column counts the samples with bit 3 set. **Bold** is the attempt the table above
uses.

| Cell | Attempt | Median latency, ms | Throttle during the run | Images below 2000 MHz | Clock min / median, MHz | Peak CPU |
|---|---|---|---|---|---|---|
| reference run, first | 1 | 387.7 | bit 19 only (0 of 4) | 0 of 200 | 2000 / 2000 | 81.0 °C |
| reference run, last | 1 | 390.4 | soft limit (1 of 4) | 2 of 200 | 1966 / 2000 | 82.0 °C |
| B1 800, ONNX Runtime FP32 | 1 | 627.5 | soft limit (3 of 32) | 309 of 1,610 | 1856 / 2000 | 83.2 °C |
| **B1 800, ONNX Runtime FP32** | 2 | 629.9 | soft limit (7 of 32) | 351 of 1,610 | 1856 / 2000 | 83.2 °C |
| **B1 800, OpenVINO FP32** | 1 | 474.6 | none | 5 of 1,610 | 1966 / 2000 | 82.0 °C |
| **B1 800, NCNN** | 1 | 276.8 | none | 0 | 2000 / 2000 | 77.1 °C |
| **B1 800, ONNX Runtime INT8** | 1 | 416.7 | none | 0 | 2000 / 2000 | 77.1 °C |
| B2 800, ONNX Runtime FP32 | 1 | 609.1 | soft limit (1 of 32) | 27 of 1,610 | 1856 / 2000 | 83.2 °C |
| **B2 800, ONNX Runtime FP32** | 2 | 611.9 | soft limit (5 of 32) | 388 of 1,610 | 1801 / 2000 | 83.7 °C |
| **B2 800, OpenVINO FP32** | 1 | 489.6 | none | 37 of 1,610 | 1911 / 2000 | 82.6 °C |
| **B2 800, NCNN** | 1 | 286.0 | none | 0 | 2000 / 2000 | 75.5 °C |
| **B2 800, ONNX Runtime INT8** | 1 | 431.2 | none | 0 | 2000 / 2000 | 76.0 °C |

- **The re-runs did not clear the throttle; both throttled more heavily than the first attempts** (3 → 7 and 1 → 5 of
  32 samples; 309 → 351 and 27 → 388 images below 2000 MHz), and the rule takes the second attempt. The median latency
  moved by +0.4 % (B1) and +0.5 % (B2). Across sessions 1, 3, 4 and 5, 1 of 16 re-runs has cleared the flag: session
  1's D1 OpenVINO.
- **The cost to latency is not measured at 800**: no ONNX Runtime FP32 attempt there ran unflagged. Between B2's two
  attempts, 27 against 388 images below 2000 MHz, the median moved 0.5 % (609.1 to 611.9 ms), and every attempt has a
  median clock of 2000 MHz.
- **Neither OpenVINO cell was flagged, but both dipped below 2000 MHz** between the throttle reads: 5 images (B1) and 37
  (B2, down to 1911 MHz). `get_throttled` is read every 50th image, and its "has occurred" bit 19 has been set since the
  first reference run, so only a read that lands during the soft limit flags a run. B2's 37 images are the most of any
  unflagged attempt so far; the most before were 16, session 1's P1 ONNX Runtime FP32. At 960 and 1280, every
  baseline's OpenVINO FP32 cell was flagged (B1, B3, B4).
- **The first reference run was flagged by bit 19 alone**: it started at `0x0` and ended at `0x80000`, so the soft
  limit occurred during it, but no throttle read caught bit 3 and all 200 per-image clock reads gave 2000 MHz. It gave
  387.7 ms, 0.2 % below the pilot's unthrottled 388.5 ms (INVARIANTS.md §4.7) and below sessions 3 and 4's throttled
  first runs (391.2 ms). The last run caught bit 3 in 1 of 4 reads, with 2 images at 1966 MHz.
- **No NCNN or INT8 attempt has read below 2000 MHz in any session** (19 attempts in sessions 1, 3, 4 and 5). Their
  peaks here were 75.5–77.1 °C, against 82.0–83.7 °C for the FP32 ONNX Runtime and OpenVINO cells.

### The start gate and the session's length

The first reference run's gate opened on temperature, at 49.6 °C after 25 s: the first of the matrix's 67 gates to do
so. The other 11 opened at the 3-minute timeout, with the CPU at 53.5–59.0 °C. The session took 3.0 h against the
plan's "about 2.1 h":

| Part | Time | Of which |
|---|---|---|
| First attempts of the 8 cells, and both reference runs | 2.3 h (09:32–11:46 IST, and 12:28–12:33) | cells 1.8 h, 10 gates 27 min, reference runs 3 min |
| The two throttle re-runs | 0.7 h (11:46–12:28 IST) | cells 36 min, 2 gates 6 min |

- **The first attempts and reference runs ran 0.2 h over the estimate.** #37's 2.1 h is #25's latencies at 640 scaled
  by pixel count (1.56×) over 1,610 images, 94.5 minutes of cells, plus a 3-minute gate per cell and 8 minutes of
  reference runs. The cells took 107.8 minutes:
  - the latencies themselves added 2.4 minutes (1,610 × each median latency is 96.9 minutes). B1 at 800 ran 1.61× A1
    at 640 on the FP32 runtimes and NCNN, more than the pixel count, and 1.53× on INT8; YOLO26s, which #37 took to be
    as fast as YOLO11s, was 3 % slower on OpenVINO and NCNN.
  - each cell spent 1.3–1.5 minutes beyond 1,610 × its median latency, 10.9 minutes in all: the 20 warm-up images and
    the model load (6–13 s), the second post-processing pass at conf 0.001 that the scored detections come from
    (about 13 ms an image, 0.35 minutes a cell), and the untimed part of each image's loop (27–31 ms an image:
    decoding the image, reading the clock and temperature). Session 4 measured 1.4–1.8 minutes at 1280.
- **The re-runs took 0.7 h, 23 % of the session**, about 21 minutes each with its gate.
- #37's table leaves that per-cell overhead out of every session: about 0.2–0.3 h each for sessions 2 (11 cells) and 6
  (8 cells), before any re-runs.

## Against what we already have

### The training records

B1's and B2's training records (`verification/runs/yolo11s-960-s0.json`, `yolo26s-960-s0.json`) scored test-dev at
960, their training size. There is no record at 800, so these cells are not set against them: the difference would be
the input size, not the export. No laptop CPU score of these exports exists either. What checks them here is the
runtimes agreeing with each other (above) and the ladder below. At 960, session 1's B1 FP32 export was within 0.0010
AP of its record (session 4's record).

### YOLO11s by resolution

| imgsz | Checkpoint | AP, FP32 | AP-small | ONNX Runtime FP32, ms | OpenVINO FP32, ms | NCNN, ms (AP) | ONNX Runtime INT8, AP / ms |
|---|---|---|---|---|---|---|---|
| 640 | A1 | 0.1875 | 0.0904 | 389.8 t | 294.0 | 171.5 (0.1866) | 0.1682 / 272.5 |
| 800 | **B1** | **0.2190** | **0.1163** | **629.9 t** | **474.6** | **276.8 (0.2179)** | **0.2040 / 416.7** |
| 960 | B1 | 0.2392 | 0.1365 | 909.7 t | 712.6 t | 411.3 (0.2381) | 0.2226 / 614.0 |
| 1280 | B3 | 0.2652 | 0.1619 | 1695.1 t | 1308.2 t | 775.7 (0.2638) | 0.2548 / 1152.3 |

t: the cell throttled. Bold: this session; 640 and 960 are session 1's, 1280 session 4's (B3's INT8: session 1).

- **B1 at 800 gets about 60 % of the AP that 640 → 960 gives, for under half the latency**: +0.0315 AP and +0.0259
  AP-small over A1 at 640 (61 % and 56 % of the step to B1 at 960), for 42–46 % of the latency step on every runtime.
  800 → 960 then adds +0.0202 AP and +0.0202 AP-small for 1.44–1.50× the latency. The 640 → 800 step changes the
  checkpoint as well as the size (A1 is trained at 640, B1 at 960); 800 → 960 is B1 alone.
- **Where B1 at 800 lands in latency**, for the equal-latency analysis after sessions 2 and 6: within 3 % of P2 at 960
  on OpenVINO (474.6 against 474.7 ms) and NCNN (276.8 against 271.0), and its ONNX Runtime FP32 within 3 % of B1's own
  INT8 at 960 (629.9 against 614.0 ms).

### INT8 against FP32 at 800

Each INT8 cell against the ONNX Runtime FP32 export of the same checkpoint at the same imgsz (INVARIANTS.md §4.4).
Differences are from the unrounded values.

| Checkpoint | imgsz | AP, FP32 → INT8 | AP lost | AP-small, FP32 → INT8 | AP-small lost | Latency, FP32 → INT8, ms | Speed-up |
|---|---|---|---|---|---|---|---|
| A1, YOLO11s | 640 | 0.1875 → 0.1682 | −0.0193 (−10.3 %) | 0.0904 → 0.0769 | −0.0135 (−15.0 %) | 389.8 t → 272.5 | 1.43× |
| **B1, YOLO11s** | **800** | 0.2190 → 0.2040 | −0.0150 (−6.8 %) | 0.1163 → 0.1041 | −0.0122 (−10.5 %) | 629.9 t → 416.7 | 1.51× |
| B1, YOLO11s | 960 | 0.2392 → 0.2226 | −0.0166 (−6.9 %) | 0.1365 → 0.1258 | −0.0107 (−7.8 %) | 909.7 t → 614.0 | 1.48× |
| B3, YOLO11s | 1280 | 0.2652 → 0.2548 | −0.0104 (−3.9 %) | 0.1619 → 0.1560 | −0.0060 (−3.7 %) | 1695.1 t → 1152.3 | 1.47× |
| **B2, YOLO26s** | **800** | 0.2218 → 0.1857 | **−0.0361 (−16.3 %)** | 0.1184 → 0.0952 | **−0.0232 (−19.6 %)** | 611.9 t → 431.2 | 1.42× |

- **YOLO26s loses much more to INT8 than YOLO11s at the same size**: 0.0361 AP and 0.0232 AP-small for B2, against
  0.0150 and 0.0122 for B1 (2.4 and 1.9 times), from the same calibration images and export settings. It loses in every size range:
  AP-medium −13.7 % (B1 −5.8 %), AP-large −17.7 % (B1 −1.7 %), AR@500 −0.038 (B1 −0.020). B2's loss is beyond all of
  YOLO11s's on the ladder (−3.9 to −10.3 % AP, −3.7 to −15.0 % AP-small), and its speed-up is the smallest (1.42×).
- **This answers session 3's open question for the baselines at 800: YOLO26s does lose more to INT8.** Session 3 saw it
  in C2 against C1: −16.8 % against −10.3 % AP, against their training records (session 4's like-for-like figures). That
  was only a pointer, since no FP32 export of C1 or C2 was run (§4.4). B2 against B1 is the first like-for-like pair,
  and it agrees. Session 2's pairs, A2 at 640 and B2 at 960, will show whether it holds at other sizes.
- **At INT8, YOLO26s falls behind YOLO11s**: 0.1857 against 0.2040 AP, at 431.2 against 416.7 ms.
- **For the same checkpoint, 960 → 800 leaves B1's INT8 AP loss where it was** (−6.9 → −6.8 %) and raises its AP-small
  loss (−7.8 → −10.5 %). Along the YOLO11s ladder the loss shrinks with the input size, but each size there is a
  different checkpoint (A1, B1, B3); B1 alone does not show the trend in AP.
- **NCNN FP16 beats ONNX Runtime INT8 again**, as at 640, 960 and 1280: 34 % less latency in both pairs (276.8 against
  416.7 ms, 286.0 against 431.2), with 0.0138 (B1) and 0.0350 (B2) more AP.
- Unexplained: B2's INT8 saves fewer detections at conf 0.001 than its FP32 (659,388 against 661,098), where each
  YOLO11s INT8 cell with an FP32 export beside it saves 1–6 % more (B1 at 800: 688,517 against 659,719).

### YOLO11s against YOLO26s at 800

| Runtime | AP, B1 / B2 | AP-small, B1 / B2 | Median latency, ms, B1 / B2 | B2 against B1 |
|---|---|---|---|---|
| ONNX Runtime FP32 | 0.2190 / 0.2218 | 0.1163 / 0.1184 | 629.9 t / 611.9 t | −2.9 % |
| OpenVINO FP32 | 0.2190 / 0.2218 | 0.1163 / 0.1184 | 474.6 / 489.6 | +3.2 % |
| NCNN FP16 | 0.2179 / 0.2208 | 0.1153 / 0.1173 | 276.8 / 286.0 | +3.3 % |
| ONNX Runtime INT8 | 0.2040 / 0.1857 | 0.1041 / 0.0952 | 416.7 / 431.2 | +3.5 % |

- **In FP32, YOLO26s is 0.0028 AP and 0.0021 AP-small ahead**, with AR@500 +0.012; session 4 had 0.0038 and 0.0039 at
  1280. AP-large goes YOLO26s's way here (0.4976 against 0.4623, +0.035), where at 1280 it went YOLO11s's way (−0.017).
  In INT8 the order flips (above).
- **YOLO26s is 3.3 % slower on NCNN**, the clean comparison (both cells at 2000 MHz throughout), and 3.2 % on OpenVINO;
  at 1280 the gaps were 9.0 % and 7.1 %. Its GFLOPs are 5.0 % more (22.8 against 21.7 at 640; session 4's record).
- **On ONNX Runtime FP32, YOLO26s is faster again**, by 2.9 % (1.2 % at 1280). The throttle does not explain it: B2's
  first attempt, with 27 images below 2000 MHz, gave 609.1 ms against B1's 627.5 (309 below), and the difference is in
  the network's time (median 605.6 against 623.7 ms, second attempts). Nothing in the session's records explains it.

## Known limitations

- One session, one seed's order; each cell ran once (twice when throttled).
- The latencies are at 2000 MHz, not the Pi 5's stock 2400 MHz.
- Both ONNX Runtime FP32 cells' latencies carry a soft throttle (above), not cleared by the re-run; both reference
  runs were flagged; and the OpenVINO cells dipped below 2000 MHz without a flag. Their accuracy is unaffected: the
  detections do not depend on the clock.
- There is no training record and no laptop CPU score at 800, so the exports' accuracy is checked only by the runtimes
  agreeing and by the ladder.
- One INT8 pair per model, at one size. Why YOLO26s quantizes worse is not examined here.
