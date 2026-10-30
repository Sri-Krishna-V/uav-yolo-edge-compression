# Raspberry Pi 5 matrix, session 3 (YOLO26s compressed): session record

Written on 2026-10-31, after the run. Vidhyarth ran the session with `pi/wizard.sh`, pushed the small files (`bd9d056`)
and sent the detections; they were scored on the laptop with `scripts/studyscore.py` under its 4 GB cap. The cells
and their order of priority are #37's.

## Question

What accuracy and latency do the compressed YOLO26s cells of the matrix give on the Raspberry Pi 5? The checkpoints
(RUNS.md), the teammate's: P3 and P4 (A2 and B2 pruned 30 %, at 640 and 960), D2 (distilled from T2) and C2 (pruned and
distilled, INT8). They are the YOLO26s counterparts of session 1's P1, P2, D1 and C1. The YOLO26s baselines are
session 2.

## Plan and code

- Plan: `plans/pi5-matrix-3-yolo26s-compressed.json` (`plan_sha256` `7ad4a6d6…639f4`). 10 full cells, all 1,610
  test-dev images, batch 1. The reference runs, first and last: ONNX Runtime FP32, A1 at 640, on the first 200 images,
  as in session 1.
- The rules of #8: 4 threads, 20 untimed warm-up images, the 50 °C start gate with its 3-minute timeout, cell order
  shuffled with seed 0, the drift check at ±3 %, one re-run of a crashed or throttled cell.
- **The CPU locked at 2000 MHz** (INVARIANTS.md §4.7). The setup checks passed: governor `performance`, no desktop, the
  apt timers off, ncnn 1.0.20260526, ONNX Runtime 1.31.0, OpenVINO 2026.4.1, `vcgencmd` clock 2000.
- Room 28 °C, as entered at the start, fan 2,012 rpm, idle CPU 53.5 °C.
- The detections came as `pi5-matrix-3-yolo26s-compressed.tgz`, with a `predictions.json` for every attempt of every
  cell and both reference runs. Its other files match the pushed ones except for line endings.

## Results

Session `bench/pi5/pi5-matrix-3-yolo26s-compressed`, 30 October, 19:34 to 23:03 IST (3.5 h). **Drift: 391.2 →
395.9 ms, +1.2 %: ok**, with both reference runs throttled (below). Every cell completed.

| Cell | Precision recorded | AP | AP50 | AP-small | AR@500 | Median latency, ms | Throttle |
|---|---|---|---|---|---|---|---|
| P3 640, ONNX Runtime | FP32 | 0.1583 | 0.2856 | 0.0754 | 0.3266 | 232.9 | throttled |
| P3 640, OpenVINO | FP32 | 0.1583 | 0.2856 | 0.0754 | 0.3266 | 193.1 | none |
| P3 640, NCNN | FP16 | 0.1580 | 0.2853 | 0.0749 | 0.3251 | 113.8 | none |
| P4 960, ONNX Runtime | FP32 | 0.2100 | 0.3691 | 0.1157 | 0.3961 | 571.7 | throttled |
| P4 960, OpenVINO | FP32 | 0.2100 | 0.3691 | 0.1157 | 0.3961 | 509.4 | throttled |
| P4 960, NCNN | FP16 | 0.2090 | 0.3685 | 0.1140 | 0.3951 | 289.3 | none |
| D2 640, ONNX Runtime | FP32 | 0.1979 | 0.3514 | 0.1027 | 0.3734 | 388.4 | throttled |
| D2 640, OpenVINO | FP32 | 0.1979 | 0.3514 | 0.1027 | 0.3734 | 306.8 | throttled |
| D2 640, NCNN | FP16 | 0.1973 | 0.3504 | 0.1019 | 0.3724 | 174.3 | none |
| C2 640, ONNX Runtime | INT8 | 0.1285 | 0.2533 | 0.0538 | 0.2883 | 193.2 | none |

AP is COCO AP@[.50:.95] at maxDets 500 (INVARIANTS.md), from detections at conf 0.001. Latency is the median of
`e2e_deploy`: pre + net + post at the deployment threshold, 0.25. A throttled cell's row is its second attempt.

- **OpenVINO FP32 gives ONNX Runtime FP32's AP to 4 digits** in all three pairs, as in session 1. NCNN's FP16 costs
  0.0003–0.0010 AP.
- **C2's INT8 loses more than C1's.** See *Against what we already have*.
- These are the session's numbers, not the paper's analysis: the equal-latency comparisons wait for the other sessions.

### Throttle and clock (INVARIANTS.md §4.7, #40)

Five FP32 cells hit the soft temperature limit (`get_throttled` bit 3) and were re-run once, as #8 says; all five
throttled again, and unlike session 1, not more lightly (below). By the decision of 29 October (#40), they are used with
their flag. No INT8 or NCNN cell throttled. The clock and temperature per attempt, from `timings.csv`, counted as in
session 1: the clock and temperature are read at every image, `get_throttled` every 50th (32 samples in a full cell, 4
in a reference run), and the throttle column counts the samples with bit 3 set. **Bold** is the attempt the table above
uses.

| Cell | Attempt | Median latency, ms | Throttle during the run | Images below 2000 MHz | Clock min / median, MHz | Peak CPU |
|---|---|---|---|---|---|---|
| reference run, first | 1 | 391.2 | soft limit (1 of 4) | 19 of 200 | 1911 / 2000 | 82.6 °C |
| reference run, last | 1 | 395.9 | soft limit (3 of 4) | 108 of 200 | 1746 / 1966 | 84.2 °C |
| P3 640, ONNX Runtime FP32 | 1 | 233.1 | soft limit (28 of 32) | 1,249 of 1,610 | 1500 / 1911 | 84.8 °C |
| **P3 640, ONNX Runtime FP32** | 2 | 232.9 | soft limit (23 of 32) | 1,101 of 1,610 | 1746 / 1966 | 84.2 °C |
| **P3 640, OpenVINO FP32** | 1 | 193.1 | none | 15 of 1,610 | 1911 / 2000 | 82.6 °C |
| **P3 640, NCNN** | 1 | 113.8 | none | 0 | 2000 / 2000 | 75.5 °C |
| P4 960, ONNX Runtime FP32 | 1 | 570.2 | soft limit (27 of 32) | 1,350 of 1,610 | 1500 / 1911 | 85.3 °C |
| **P4 960, ONNX Runtime FP32** | 2 | 571.7 | soft limit (29 of 32) | 1,381 of 1,610 | 1500 / 1911 | 84.8 °C |
| P4 960, OpenVINO FP32 | 1 | 511.8 | soft limit (13 of 32) | 637 of 1,610 | 1746 / 2000 | 84.2 °C |
| **P4 960, OpenVINO FP32** | 2 | 509.4 | soft limit (10 of 32) | 425 of 1,610 | 1801 / 2000 | 83.7 °C |
| **P4 960, NCNN** | 1 | 289.3 | none | 0 | 2000 / 2000 | 77.1 °C |
| D2 640, ONNX Runtime FP32 | 1 | 385.3 | soft limit (29 of 32) | 1,432 of 1,610 | 1500 / 1856 | 84.8 °C |
| **D2 640, ONNX Runtime FP32** | 2 | 388.4 | soft limit (29 of 32) | 1,418 of 1,610 | 1500 / 1911 | 85.3 °C |
| D2 640, OpenVINO FP32 | 1 | 301.1 | soft limit (15 of 32) | 495 of 1,610 | 1746 / 2000 | 84.2 °C |
| **D2 640, OpenVINO FP32** | 2 | 306.8 | soft limit (19 of 32) | 965 of 1,610 | 1746 / 1966 | 84.2 °C |
| **D2 640, NCNN** | 1 | 174.3 | none | 0 | 2000 / 2000 | 77.7 °C |
| **C2 640, ONNX Runtime INT8** | 1 | 193.2 | none | 0 | 2000 / 2000 | 77.7 °C |

- **The re-runs did not clear the throttle.** In session 1 every re-run throttled less than its first attempt; here
  P4's and D2's ONNX Runtime cells throttled as much or more (27 → 29 and 29 → 29 of 32), and D2's OpenVINO more
  (15 → 19, 495 → 965 images below 2000 MHz). The table still uses the second attempt, as #8 says. For D2 that is the
  slower one: 388.4 against 385.3 ms (ONNX Runtime), 306.8 against 301.1 ms (OpenVINO, +1.9 %).
- **The reference runs throttled too**, which they did not in session 1, so the +1.2 % drift is at least partly the
  last run's throttle (3 of 4 samples, median clock 1966 MHz), not a change in the Pi. The first reference run, lightly
  throttled, gave 391.2 ms against session 1's 390.0–390.8 ms (+0.3 %).
- **The effect on latency is a few percent at most.** No cell here has an unthrottled attempt to compare with. In
  session 1, A1's heavily throttled attempt (median clock 1856 MHz) cost +3.7 % against the unthrottled pilot; every
  attempt this table uses has a median clock of 1911 MHz or more.
- **The same room, a cooler start, more throttling.** Room 28 °C as in session 1, idle CPU 53.5 °C against 56.8 °C.
  The share of ONNX Runtime and OpenVINO FP32 cells, the ones that throttle, is close (6 of 10, against 10 of 19), so
  the cell mix does not explain it, and nothing in the session's records does.

### The start gate and the session's length

All 17 gates opened at the 3-minute timeout, with the CPU at 54.5–57.9 °C, never below 50 °C. The first reference run
and the 10 first attempts took 2.1 h (19:37–21:42 IST), as #37's estimate of 2.2 h said; the five re-runs added 1.2 h
(21:45–22:58 IST), about 15 minutes each with their gate.

## Against what we already have

### The training records

No laptop CPU score of these exports exists. The other test-dev score of the same checkpoints is the training record's
(`verification/runs/<run>.json`, `eval_test`): PyTorch on Kaggle, the same one-to-many head (`end2end_head_scored`
false), but `rect` batches of 16 where the exports run a fixed square input. Session 1's checkpoints are listed with
them for the pattern.

| Checkpoint | Training record AP | Pi 5 AP | Difference | Training record AP-small | Pi 5 AP-small |
|---|---|---|---|---|---|
| A1 640 | 0.1850 | 0.1875 (FP32) | +0.0025 | 0.0905 | 0.0904 |
| P1 640 | 0.1526 | 0.1573 (FP32) | +0.0047 | 0.0724 | 0.0739 |
| P2 960 | 0.2025 | 0.2055 (FP32) | +0.0030 | 0.1137 | 0.1121 |
| D1 640 | 0.1933 | 0.1950 (FP32) | +0.0017 | 0.0959 | 0.0963 |
| **P3 640** | 0.1555 | 0.1583 (FP32) | +0.0028 | 0.0746 | 0.0754 |
| **P4 960** | 0.2070 | 0.2100 (FP32) | +0.0030 | 0.1158 | 0.1157 |
| **D2 640** | 0.1955 | 0.1979 (FP32) | +0.0024 | 0.1021 | 0.1027 |
| C1 640 | 0.1555 | 0.1420 (INT8) | −0.0135 | 0.0717 | 0.0607 |
| **C2 640** | 0.1517 | 0.1285 (INT8) | **−0.0232** | 0.0709 | **0.0538** |

- **Every FP32 export scores 0.002–0.005 AP above its training record**, in both families: the square input against
  the record's `rect` batches. P3, P4 and D2 fall in the band session 1's checkpoints set.
- **C2's INT8 costs about 15 % of its AP and 24 % of its AP-small** against its record, where C1's cost 9 % and 15 %.
  Against an FP32 export of C2, which no session has, the loss would be larger by the band above. The INT8 export
  followed #9's recipe (548 val images, no chunked calibration). Whether YOLO26s quantizes worse than YOLO11s in general
  is for session 2: it has A2 at 640 in both FP32 and INT8.

### Session 1's YOLO11s cells

| Configuration | AP, YOLO11s / YOLO26s | AP-small | ONNX Runtime, ms | OpenVINO, ms | NCNN, ms |
|---|---|---|---|---|---|
| Pruned 640 (P1 / P3) | 0.1573 / 0.1583 | 0.0739 / 0.0754 | 239.7 / 232.9 t | 193.1 / 193.1 | 112.2 / 113.8 |
| Pruned 960 (P2 / P4) | 0.2055 / 0.2100 | 0.1121 / 0.1157 | 574.6 / 571.7 t | 474.7 / 509.4 t | 271.0 / 289.3 |
| Distilled 640 (D1 / D2) | 0.1950 / 0.1979 | 0.0963 / 0.1027 | 388.1 t / 388.4 t | 293.7 / 306.8 t | 171.6 / 174.3 |
| Pruned and distilled, INT8 (C1 / C2) | 0.1420 / 0.1285 | 0.0607 / 0.0538 | 192.8 / 193.2 | — | — |

t: the cell throttled (its flag in the tables above).

- **The FP32 and FP16 YOLO26s cells are slightly more accurate** (+0.001 to +0.005 AP, +0.002 to +0.006 AP-small) at
  much the same latency. The exception is C2, 0.0135 AP below C1.
- **P4 is slower than P2 on the unthrottled runtime**: NCNN 289.3 against 271.0 ms (+6.8 %), both at 2000 MHz
  throughout. Pruned YOLO26s keeps more of the network: the pruner skips its attention block (RUNS.md), leaving 6.72 M
  parameters against 5.64 M, and 31.1 GFLOPs at 960 against 27.8. At 640 the gap does not show in latency (NCNN 113.8
  against 112.2 ms, 13.5 against 12.2 GFLOPs). P4's OpenVINO gap (+7.3 %) also carries its throttle.

## Known limitations

- One session, one seed's order; each cell ran once (twice when throttled).
- The latencies are at 2000 MHz, not the Pi 5's stock 2400 MHz.
- Five cells' latencies carry a soft throttle (above), heavier than session 1's and not cleared by the re-run. Their
  accuracy is unaffected: the detections do not depend on the clock.
- No FP32 export of C2 was run, so its INT8 loss is measured against the training record, not against an FP32 export.
