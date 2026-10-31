# Raspberry Pi 5 matrix, session 4 (FP32 baselines at 1280): session record

Written on 2026-10-31, after the run. Vidhyarth ran the session with `pi/wizard.sh`, pushed the small files (`2393cb9`)
and sent the detections; they were scored on the laptop with `scripts/studyscore.py` under its 4 GB cap. The cells
and their order of priority are #37's.

## Question

What accuracy and latency do the FP32 baselines at 1280 give on the Raspberry Pi 5? The checkpoints (RUNS.md): B3
(YOLO11s) and B4 (YOLO26s), the 1280 rows of the resolution study, added to the matrix by the dev's decision of 28
October (#37, open question 2). B3's INT8 cell is session 1's; B4's was cut from session 2, its calibration needing
more than the laptop's free memory.

## Plan and code

- Plan: `plans/pi5-matrix-4-1280-baselines.json` (`plan_sha256` `6eea09ed…0a3bd`). 6 full cells, all 1,610 test-dev
  images, batch 1. The reference runs, first and last: ONNX Runtime FP32, A1 at 640, on the first 200 images, as in
  session 1.
- The rules of #8: 4 threads, 20 untimed warm-up images, the 50 °C start gate with its 3-minute timeout, cell order
  shuffled with seed 0, the drift check at ±3 %, one re-run of a crashed or throttled cell.
- **The CPU locked at 2000 MHz** (INVARIANTS.md §4.7). The setup checks passed: governor `performance`, no desktop, the
  apt timers off, ncnn 1.0.20260526, ONNX Runtime 1.31.0, OpenVINO 2026.4.1, `vcgencmd` clock 2000.
- Room 28 °C, as entered at the start, fan 2,013 rpm, idle CPU 56.2 °C.
- The detections came as `pi5-matrix-4-1280-baselines.tgz` (sha256 `57c33347…c6fee`), with a `predictions.json` for
  every attempt of every cell and both reference runs. Its other files match the pushed ones except for line endings
  (and `runner.pid`, which is not pushed). The first download of it was corrupt (the gzip stream broke about 54 MB in);
  the second was whole.

## Results

Session `bench/pi5/pi5-matrix-4-1280-baselines`, 30 October, 23:52 IST, to 31 October, 06:59 IST (7.1 h, against the
plan's 3.5 h: see below). **Drift: 391.2 → 394.3 ms, +0.8 %: ok**, with both reference runs throttled (below). Every
cell completed.

| Cell | Precision recorded | AP | AP50 | AP-small | AR@500 | Median latency, ms | Throttle |
|---|---|---|---|---|---|---|---|
| B3 1280, ONNX Runtime | FP32 | 0.2652 | 0.4482 | 0.1619 | 0.4581 | 1695.1 | throttled |
| B3 1280, OpenVINO | FP32 | 0.2652 | 0.4482 | 0.1619 | 0.4581 | 1308.2 | throttled |
| B3 1280, NCNN | FP16 | 0.2638 | 0.4474 | 0.1603 | 0.4563 | 775.7 | none |
| B4 1280, ONNX Runtime | FP32 | 0.2690 | 0.4550 | 0.1659 | 0.4691 | 1675.1 | throttled |
| B4 1280, OpenVINO | FP32 | 0.2690 | 0.4550 | 0.1659 | 0.4691 | 1400.7 | throttled |
| B4 1280, NCNN | FP16 | 0.2673 | 0.4542 | 0.1634 | 0.4667 | 845.3 | none |

AP is COCO AP@[.50:.95] at maxDets 500 (INVARIANTS.md), from detections at conf 0.001. Latency is the median of
`e2e_deploy`: pre + net + post at the deployment threshold, 0.25. A throttled cell's row is its second attempt.

- **OpenVINO FP32 gives ONNX Runtime FP32's AP to 4 digits** in both pairs, as in sessions 1 and 3. NCNN's FP16 costs
  0.0014 (B3) and 0.0017 (B4) AP, a little more than the 0.0003–0.0011 of sessions 1 and 3 at 640 and 960.
- These are the session's numbers, not the paper's analysis: the equal-latency comparisons wait for the other sessions.

### Throttle and clock (INVARIANTS.md §4.7, #40)

Four cells, every ONNX Runtime and OpenVINO FP32 cell (`summary.csv`), hit the soft temperature limit (`get_throttled`
bit 3) and were re-run once, as #8 says; all four throttled again. By the decision of 29 October (#40), they are used
with their flag. Neither NCNN cell throttled. The clock and temperature per attempt, from `timings.csv`, counted as in
session 1: the clock and temperature are read at every image, `get_throttled` every 50th (32 samples in a full cell, 4
in a reference run), and the throttle column counts the samples with bit 3 set. **Bold** is the attempt the table above
uses.

| Cell | Attempt | Median latency, ms | Throttle during the run | Images below 2000 MHz | Clock min / median, MHz | Peak CPU |
|---|---|---|---|---|---|---|
| reference run, first | 1 | 391.2 | soft limit (3 of 4) | 73 of 200 | 1746 / 2000 | 84.2 °C |
| reference run, last | 1 | 394.3 | soft limit (2 of 4) | 123 of 200 | 1746 / 1966 | 84.2 °C |
| B3 1280, ONNX Runtime FP32 | 1 | 1708.7 | soft limit (28 of 32) | 1,375 of 1,610 | 1746 / 1911 | 84.2 °C |
| **B3 1280, ONNX Runtime FP32** | 2 | 1695.1 | soft limit (25 of 32) | 1,220 of 1,610 | 1746 / 1966 | 84.2 °C |
| B3 1280, OpenVINO FP32 | 1 | 1316.2 | soft limit (29 of 32) | 1,070 of 1,610 | 1500 / 1966 | 84.8 °C |
| **B3 1280, OpenVINO FP32** | 2 | 1308.2 | soft limit (21 of 32) | 974 of 1,610 | 1746 / 1966 | 84.2 °C |
| **B3 1280, NCNN** | 1 | 775.7 | none | 0 | 2000 / 2000 | 79.8 °C |
| B4 1280, ONNX Runtime FP32 | 1 | 1702.3 | soft limit (32 of 32) | 1,512 of 1,610 | 1500 / 1856 | 85.3 °C |
| **B4 1280, ONNX Runtime FP32** | 2 | 1675.1 | soft limit (28 of 32) | 1,367 of 1,610 | 1746 / 1911 | 84.2 °C |
| B4 1280, OpenVINO FP32 | 1 | 1400.2 | soft limit (20 of 32) | 1,016 of 1,610 | 1746 / 1966 | 84.2 °C |
| **B4 1280, OpenVINO FP32** | 2 | 1400.7 | soft limit (23 of 32) | 1,022 of 1,610 | 1746 / 1966 | 84.2 °C |
| **B4 1280, NCNN** | 1 | 845.3 | none | 0 | 2000 / 2000 | 79.8 °C |

- **The re-runs did not clear the throttle.** Three of the four throttled more lightly (28 → 25, 29 → 21 and 32 → 28
  of 32 samples), B4's OpenVINO more (20 → 23). The median latency moved by −1.6 % to +0.04 %. Across sessions 1, 3
  and 4, 1 of 14 re-runs has cleared the flag: session 1's D1 OpenVINO, from 6 of 32 samples to none.
- **B4's first ONNX Runtime attempt is the heaviest throttle of the matrix so far**: bit 3 in all 32 samples and 1,512
  images below 2000 MHz, both the most of any attempt; its median clock, 1856 MHz, equals the heaviest attempts of
  sessions 1 and 3.
- **The reference runs throttled again**, as in session 3: 3 of 4 and 2 of 4 samples here, 1 and 3 there. The first
  gave 391.2 ms, as session 3's did. The last ran at a lower median clock (1966 against 2000 MHz, 123 images below it
  against 73), so the +0.8 % drift is at least partly its throttle.
- **The cost to latency is not measured at 1280**: no attempt there ran unthrottled. The one measured bound is A1 at
  640 in session 1, whose heavily throttled attempt (median 1856 MHz) cost +4.1 % against the unthrottled pilot
  (404.4 against 388.5 ms). Every attempt the table uses has a median clock of 1911 MHz or more. The NCNN cells held
  2000 MHz throughout, peaking at 79.8 °C.

### The start gate and the session's length

All 12 gates opened at the 3-minute timeout, with the CPU at 54.0–58.4 °C, never below 50 °C. The session took 7.1 h
against the plan's "about 3.5 h":

| Part | Time | Of which |
|---|---|---|
| First attempts of the 6 cells, and both reference runs | 4.1 h (23:52–03:52 IST, and 06:54–06:59) | cells 3.6 h, 8 gates 0.4 h, reference runs 3 min |
| The four throttle re-runs | 3.0 h (03:52–06:54 IST) | cells 2.8 h, 4 gates 12 min |

- **The re-runs are the larger part: 3.0 h, 43 % of the session**, about 45 minutes each with its gate. #37's estimate
  counts first attempts only, as for sessions 1 and 3; session 1's record put a re-run at about 30 minutes, which holds
  up to 960, not at 1280.
- **The first attempts ran 0.6 h over the estimate.** The 1280 cells cost more than 4× the 640 ones that #37 scaled by:
  B3 against session 1's A1, ONNX Runtime 4.35×, OpenVINO 4.45×, NCNN 4.52× (both NCNN cells unthrottled). And each full
  cell spent 1.4–1.8 minutes beyond 1,610 × its median latency: the 20 warm-up images, the second post-processing pass at
  conf 0.001 that the scored detections come from (about 15 ms an image), and the untimed part of each image's loop
  (decoding the image, reading the clock and temperature). Loading the model took under a second.
- Session 6 runs the same checkpoints at 1120, with four ONNX Runtime and OpenVINO FP32 cells. Scaled by pixel count,
  re-runs like these would add about 2.4 h to its 3.7 h.

## Against what we already have

### The training records

No laptop CPU score of these exports exists. The other test-dev score of the same checkpoints is the training record's
(`verification/runs/<run>.json`, `eval_test`): PyTorch on Kaggle, the same one-to-many head (`end2end_head_scored`
false), but `rect` batches of 16 where the exports run a fixed square input. Both sides here are `coco/AP_all` and
`coco/AP_small` (INVARIANTS.md §4.2). Session 1's baselines are listed with them. Differences are from the unrounded
values.

| Checkpoint | Training record AP | Pi 5 AP (FP32) | Difference | Training record AP-small | Pi 5 AP-small | Difference |
|---|---|---|---|---|---|---|
| A1 640 | 0.1877 | 0.1875 | −0.0003 | 0.0905 | 0.0904 | −0.0001 |
| B1 960 | 0.2402 | 0.2392 | −0.0010 | 0.1387 | 0.1365 | −0.0022 |
| **B3 1280** | 0.2650 | 0.2652 | +0.0002 | 0.1629 | 0.1619 | −0.0010 |
| **B4 1280** | 0.2704 | 0.2690 | −0.0014 | 0.1671 | 0.1659 | −0.0013 |

- **The FP32 exports match their training records.** Across the 10 FP32 checkpoints scored so far (sessions 1, 3 and
  4), the Pi 5's AP is within −0.0014 (B4) to +0.0017 (P1) of the record's, and its AP-small within −0.0022 (B1) to
  +0.0015 (P1), with no consistent sign (4 of 10 above in AP).
- **This corrects session 3's record.** Its training-record table took the record's `metrics/mAP50-95(B)`,
  Ultralytics' own integrator, which reads 0.002–0.003 below `coco/AP_all` on the same detections, and set it against
  the Pi 5's `coco/AP_all`. Its "0.002–0.005 AP above their records" was that difference between integrators, not the
  exports. Its AP-small column used one key on both sides and stands. Like for like, the INT8 cells' loss against their
  records is larger than it says: C1 −0.0162 AP (−10.3 %), C2 −0.0260 (−16.8 %). (INVARIANTS.md §4.4 rules that
  comparison out for the paper, since no FP32 export of C1 or C2 was run; it stands only as a pointer.)
- **AP-large moves more than the rest**: from −0.028 (A1) to +0.009 (D1) between record and export across the 10
  checkpoints, with large objects few in test-dev. B4's is −0.021 (0.4957 against 0.5163).

### Session 1's YOLO11s cells: the resolution ladder and B3's INT8

| imgsz | Checkpoint | AP, FP32 | AP-small | ONNX Runtime FP32, ms | OpenVINO FP32, ms | NCNN, ms (AP) | ONNX Runtime INT8, AP / ms |
|---|---|---|---|---|---|---|---|
| 640 | A1 | 0.1875 | 0.0904 | 389.8 t | 294.0 | 171.5 (0.1866) | 0.1682 / 272.5 |
| 960 | B1 | 0.2392 | 0.1365 | 909.7 t | 712.6 t | 411.3 (0.2381) | 0.2226 / 614.0 |
| 1280 | **B3** | **0.2652** | **0.1619** | **1695.1 t** | **1308.2 t** | **775.7 (0.2638)** | 0.2548 / 1152.3 |

t: the cell throttled. Bold: this session; the rest, and B3's INT8, are session 1's.

- **960 → 1280 gives half the gain of 640 → 960**: +0.0260 AP and +0.0254 AP-small, against +0.0517 and +0.0461, for a
  latency step of 1.89× (NCNN) against 2.40×.
- **B3's INT8 costs 0.0104 AP (−3.9 %) and 0.0059 AP-small (−3.7 %)** against its FP32 export (INVARIANTS.md §4.4),
  for a 1.47× speed-up (1695.1 → 1152.3 ms, the FP32 side throttled). The loss shrinks with resolution, from −10.3 %
  (AP) and −15.0 % (AP-small) at 640 and −6.9 % and −7.8 % at 960; the speed-up stays at 1.43–1.48×.
- **NCNN FP16 beats ONNX Runtime INT8 again**, at 1280 as at 640 and 960: 775.7 against 1152.3 ms (−33 %), with
  0.0090 more AP.

### YOLO11s against YOLO26s at 1280

| Runtime | AP, B3 / B4 | AP-small, B3 / B4 | Median latency, ms, B3 / B4 | B4 against B3 |
|---|---|---|---|---|
| ONNX Runtime FP32 | 0.2652 / 0.2690 | 0.1619 / 0.1659 | 1695.1 t / 1675.1 t | −1.2 % |
| OpenVINO FP32 | 0.2652 / 0.2690 | 0.1619 / 0.1659 | 1308.2 t / 1400.7 t | +7.1 % |
| NCNN FP16 | 0.2638 / 0.2673 | 0.1603 / 0.1634 | 775.7 / 845.3 | +9.0 % |

- **YOLO26s is 0.0038 AP and 0.0039 AP-small ahead** (FP32), against 0.0054 and 0.0043 in the training records, with
  AR@500 +0.011. AP-large goes the other way on the Pi 5 (0.4957 against 0.5128, −0.017), where the records have
  YOLO26s ahead (+0.012). Most of the flip is B4's own move against its record (−0.021, against B3's +0.009).
- **YOLO26s is 9.0 % slower on NCNN**, the clean comparison (both cells at 2000 MHz throughout), for 5.6 % more
  parameters (9.96 M against 9.43 M) and 5.0 % more GFLOPs (22.8 against 21.7 at 640; the pruning records'
  `gflops_before`). OpenVINO shows +7.1 %, both attempts throttled alike (median 1966 MHz).
- **On ONNX Runtime the gap does not show**: B4 is 1.2 % faster, though its attempt throttled more (median 1911
  against 1966 MHz; first attempts 1702.3 against 1708.7 ms). Nothing in the session's records explains it.
- The YOLO26s ladder at 640 and 960 (A2, B2) is session 2.

## Known limitations

- One session, one seed's order; each cell ran once (twice when throttled).
- The latencies are at 2000 MHz, not the Pi 5's stock 2400 MHz.
- All four ONNX Runtime and OpenVINO cells' latencies carry a soft throttle (above), not cleared by the re-run, and
  both reference runs throttled. Their accuracy is unaffected: the detections do not depend on the clock.
- No laptop CPU score of these exports exists, so their accuracy is checked against the training records only.
- B4 has no INT8 cell (cut from session 2), so the INT8 comparison at 1280 is YOLO11s's alone.
