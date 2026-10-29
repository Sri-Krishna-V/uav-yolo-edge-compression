# Raspberry Pi 5 matrix, session 1 (YOLO11s): session record

Written on 2026-10-29, after the run. Vidhyarth ran the session with `pi/wizard.sh`, pushed the small files (`82f72a1`)
and sent the detections; they were scored on the laptop with `scripts/studyscore.py` under its 4 GB cap. The cells
and their order of priority are #37's.

## Question

What accuracy and latency do the YOLO11s cells of the matrix give on the Raspberry Pi 5? The checkpoints (RUNS.md): A1
(640) and B1 (960) baseline and quantized, B3 quantized at 1280, P1 and P2 (pruned 30 % at 640 and 960), D1
(distilled) and C1 (pruned and distilled, INT8). The 1280 baselines and the 800 and 1120 cells are later sessions.

## Plan and code

- Plan: `plans/pi5-matrix-1-yolo11s.json` (`plan_sha256` `f6172b88…19eae`). 19 full cells, all 1,610 test-dev images,
  batch 1. The reference runs, first and last: ONNX Runtime FP32, A1 at 640, on the first 200 images.
- The rules of #8: 4 threads, 20 untimed warm-up images, the 50 °C start gate with its 3-minute timeout, cell order
  shuffled with seed 0, the drift check at ±3 %, one re-run of a crashed or throttled cell.
- **The CPU locked at 2000 MHz** (INVARIANTS.md §4.7). The setup checks passed: governor `performance`, no desktop, the
  apt timers off, ncnn 1.0.20260526, ONNX Runtime 1.31.0, OpenVINO 2026.4.1, `vcgencmd` clock 2000.
- Room 28 °C, as entered at the start, fan 2,035 rpm, idle CPU 56.8 °C.

## Results

Session `bench/pi5/pi5-matrix-1-yolo11s`, 28 October, 18:01 IST to 29 October, 00:40 IST (6.6 h). **Drift: 390.0 →
390.8 ms, +0.2 %: ok.** Every cell completed.

| Cell | Precision recorded | AP | AP50 | AP-small | AR@500 | Median latency, ms | Throttle |
|---|---|---|---|---|---|---|---|
| A1 640, ONNX Runtime | FP32 | 0.1875 | 0.3265 | 0.0904 | 0.3563 | 389.8 | throttled |
| A1 640, OpenVINO | FP32 | 0.1875 | 0.3265 | 0.0904 | 0.3563 | 294.0 | none |
| A1 640, NCNN | FP16 | 0.1866 | 0.3257 | 0.0895 | 0.3551 | 171.5 | none |
| A1 640, ONNX Runtime | INT8 | 0.1682 | 0.3056 | 0.0769 | 0.3330 | 272.5 | none |
| B1 960, ONNX Runtime | FP32 | 0.2392 | 0.4067 | 0.1365 | 0.4211 | 909.7 | throttled |
| B1 960, OpenVINO | FP32 | 0.2392 | 0.4067 | 0.1365 | 0.4211 | 712.6 | throttled |
| B1 960, NCNN | FP16 | 0.2381 | 0.4058 | 0.1359 | 0.4201 | 411.3 | none |
| B1 960, ONNX Runtime | INT8 | 0.2226 | 0.3870 | 0.1258 | 0.3999 | 614.0 | none |
| B3 1280, ONNX Runtime | INT8 | 0.2548 | 0.4347 | 0.1560 | 0.4506 | 1152.3 | none |
| P1 640, ONNX Runtime | FP32 | 0.1573 | 0.2808 | 0.0739 | 0.3201 | 239.7 | none |
| P1 640, OpenVINO | FP32 | 0.1573 | 0.2808 | 0.0739 | 0.3201 | 193.1 | none |
| P1 640, NCNN | FP16 | 0.1565 | 0.2809 | 0.0732 | 0.3185 | 112.2 | none |
| P2 960, ONNX Runtime | FP32 | 0.2055 | 0.3583 | 0.1121 | 0.3876 | 574.6 | none |
| P2 960, OpenVINO | FP32 | 0.2055 | 0.3583 | 0.1121 | 0.3876 | 474.7 | none |
| P2 960, NCNN | FP16 | 0.2047 | 0.3577 | 0.1119 | 0.3866 | 271.0 | none |
| D1 640, ONNX Runtime | FP32 | 0.1950 | 0.3404 | 0.0963 | 0.3623 | 388.1 | throttled |
| D1 640, OpenVINO | FP32 | 0.1950 | 0.3404 | 0.0963 | 0.3623 | 293.7 | none |
| D1 640, NCNN | FP16 | 0.1943 | 0.3398 | 0.0955 | 0.3610 | 171.6 | none |
| C1 640, ONNX Runtime | INT8 | 0.1420 | 0.2636 | 0.0607 | 0.2986 | 192.8 | none |

AP is COCO AP@[.50:.95] at maxDets 500 (INVARIANTS.md), from detections at conf 0.001. Latency is the median of
`e2e_deploy`: pre + net + post at the deployment threshold, 0.25. A throttled cell's row is its second attempt.

- **The scores match the earlier full cells.** A1's ONNX Runtime FP32 gives AP 0.1875, as in the pilot
  ([pi5_pilot_record.md](pi5_pilot_record.md)); its INT8 gives 0.1682, as on the Galaxy M34
  ([m34_fullcell_int8_record.md](m34_fullcell_int8_record.md)).
- **OpenVINO FP32 gives ONNX Runtime FP32's AP to 4 digits** in all five pairs, as INVARIANTS.md §8's FP32 rule
  intends. NCNN's FP16 costs 0.0008–0.0011 AP.
- These are the session's numbers, not the paper's analysis: the equal-latency comparisons wait for the other sessions.

### Throttle and clock (INVARIANTS.md §4.7, #40)

Five FP32 cells hit the soft temperature limit (`get_throttled` bit 3) and were re-run once, as #8 says; all five
throttled again, more lightly. By the decision of 29 October (#40), they are used with their flag. No INT8 or NCNN cell
throttled. The clock and temperature per attempt, from `timings.csv`: the clock and temperature are read at every
image, `get_throttled` every 50th (32 samples in a full cell, 4 in a reference run), and the throttle column counts
the samples with bit 3 set. **Bold** is the attempt the table above uses.

| Cell | Attempt | Median latency, ms | Throttle during the run | Images below 2000 MHz | Clock min / median, MHz | Peak CPU |
|---|---|---|---|---|---|---|
| reference run, first | 1 | 390.0 | none (of 4) | 4 of 200 | 1966 / 2000 | 82.0 °C |
| reference run, last | 1 | 390.8 | none | 4 of 200 | 1966 / 2000 | 82.0 °C |
| A1 640, ONNX Runtime FP32 | 1 | 404.4 | soft limit (30 of 32) | 1,473 of 1,610 | 1500 / 1856 | 85.3 °C |
| **A1 640, ONNX Runtime FP32** | 2 | 389.8 | soft limit (4 of 32) | 381 of 1,610 | 1801 / 2000 | 83.2 °C |
| **A1 640, OpenVINO FP32** | 1 | 294.0 | none | 1 of 1,610 | 1966 / 2000 | 82.0 °C |
| **A1 640, NCNN** | 1 | 171.5 | none | 0 | 2000 / 2000 | 76.0 °C |
| **A1 640, ONNX Runtime INT8** | 1 | 272.5 | none | 0 | 2000 / 2000 | 76.5 °C |
| B1 960, ONNX Runtime FP32 | 1 | 918.8 | soft limit (20 of 32) | 964 of 1,610 | 1746 / 1966 | 84.2 °C |
| **B1 960, ONNX Runtime FP32** | 2 | 909.7 | soft limit (12 of 32) | 680 of 1,610 | 1856 / 2000 | 83.2 °C |
| B1 960, OpenVINO FP32 | 1 | 699.8 | soft limit (4 of 32) | 150 of 1,610 | 1801 / 2000 | 83.7 °C |
| **B1 960, OpenVINO FP32** | 2 | 712.6 | soft limit (1 of 32) | 54 of 1,610 | 1856 / 2000 | 83.2 °C |
| **B1 960, NCNN** | 1 | 411.3 | none | 0 | 2000 / 2000 | 81.0 °C |
| **B1 960, ONNX Runtime INT8** | 1 | 614.0 | none | 0 | 2000 / 2000 | 79.8 °C |
| **B3 1280, ONNX Runtime INT8** | 1 | 1152.3 | none | 0 | 2000 / 2000 | 76.0 °C |
| **P1 640, ONNX Runtime FP32** | 1 | 239.7 | none | 16 of 1,610 | 1911 / 2000 | 82.6 °C |
| **P1 640, OpenVINO FP32** | 1 | 193.1 | none | 0 | 2000 / 2000 | 78.2 °C |
| **P1 640, NCNN** | 1 | 112.2 | none | 0 | 2000 / 2000 | 75.5 °C |
| **P2 960, ONNX Runtime FP32** | 1 | 574.6 | none | 6 of 1,610 | 1966 / 2000 | 82.0 °C |
| **P2 960, OpenVINO FP32** | 1 | 474.7 | none | 0 | 2000 / 2000 | 79.8 °C |
| **P2 960, NCNN** | 1 | 271.0 | none | 0 | 2000 / 2000 | 77.1 °C |
| D1 640, ONNX Runtime FP32 | 1 | 394.1 | soft limit (21 of 32) | 1,138 of 1,610 | 1746 / 1966 | 84.2 °C |
| **D1 640, ONNX Runtime FP32** | 2 | 388.1 | soft limit (4 of 32) | 368 of 1,610 | 1801 / 2000 | 83.7 °C |
| D1 640, OpenVINO FP32 | 1 | 294.1 | soft limit (6 of 32) | 77 of 1,610 | 1856 / 2000 | 83.2 °C |
| **D1 640, OpenVINO FP32** | 2 | 293.7 | none | 2 of 1,610 | 1966 / 2000 | 82.0 °C |
| **D1 640, NCNN** | 1 | 171.6 | none | 0 | 2000 / 2000 | 75.5 °C |
| **C1 640, ONNX Runtime INT8** | 1 | 192.8 | none | 0 | 2000 / 2000 | 78.2 °C |

- **The effect on latency is small.** A1's ONNX Runtime FP32 cell gives 389.8 ms on its second attempt, against the
  pilot's unthrottled 388.5 ms at the same lock (+0.3 %) and the reference runs' 390.0–390.8 ms. Its heavily throttled
  first attempt (median clock 1856 MHz) gave 404.4 ms, +3.7 %.
- **The clock dips without the bit.** Some cells with no throttle bit in any sample still had images below 2000 MHz
  (P1 ONNX Runtime FP32: 16, down to 1911). The clock columns are the better measure: the clock is read at every
  image, the bit at every 50th.
- **Hotter than the pilot.** The room was 28 °C against the pilot's 26 °C, and the idle CPU 56.8 °C against 50.1 °C.

### The start gate and the session's length

All 26 gates opened at the 3-minute timeout, with the CPU at 54.0–58.4 °C, never below 50 °C. The 19 first attempts
took 5.0 h (18:04–23:03 IST), close to #37's estimate of 4.8 h; the five re-runs added 1.6 h. Sessions with more FP32
cells at 960 and above (4 and 6) will take longer than #37's table says, by about 30 minutes per throttled cell.

## Known limitations

- One session, one seed's order; each cell ran once (twice when throttled).
- The latencies are at 2000 MHz, not the Pi 5's stock 2400 MHz.
- Five cells' latencies carry a light soft throttle (above). Their accuracy is unaffected: the detections do not
  depend on the clock.
