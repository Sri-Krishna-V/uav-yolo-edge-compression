# Raspberry Pi 5 matrix, session 2 (YOLO26s baselines at 640 and 960): session record

Written on 2026-10-31, after the run. Vidhyarth ran the session with `pi/wizard.sh`, pushed the small files (`8f450e8`)
and sent the detections; they were scored on the laptop with `scripts/studyscore.py` under its 4 GB cap. The cells
and their order of priority are #37's.

## Question

What accuracy and latency do A2 and B2 (YOLO26s, trained at 640 and 960; RUNS.md) give on the Raspberry Pi 5 at their
training sizes, as FP32 baselines and as ONNX Runtime INT8? And what does A2's NMS-free head cost or save against its
one-to-many head (#37's result B)? A2 and B2 are the YOLO26s counterparts of session 1's A1 and B1, and the baselines
of session 3's compressed YOLO26s cells. B4's INT8 cell at 1280 was cut from the session (`4058740`): its calibration
needs more than 8 GB.

## Plan and code

- Plan: `plans/pi5-matrix-2-yolo26s.json` (`plan_sha256` `c34165a6…076f3`, which the repo's plan gives through
  `studysession.load_plan`). 11 full cells, all 1,610 test-dev images, batch 1. The reference runs, first and last:
  ONNX Runtime FP32, A1 at 640, on the first 200 images, as in session 1.
- The rules of #8: 4 threads, 20 untimed warm-up images, the 50 °C start gate with its 3-minute timeout, cell order
  shuffled with seed 0, the drift check at ±3 %, one re-run of a crashed or throttled cell.
- **The session ran under the re-run rule.** It started at 13:26 IST, two hours before `16221bb` (15:33 IST) stopped
  Pi 5 throttle re-runs "from matrix sessions 2 and 6 on" (INVARIANTS.md §4.7). Its six throttled cells ran twice, and
  this record uses the second attempts, as `summary.csv` and sessions 1, 3, 4 and 5 do. The first attempts are in the
  clock table below.
- **The CPU locked at 2000 MHz** (INVARIANTS.md §4.7). The setup checks passed: governor `performance`, no desktop, the
  apt timers off, ncnn 1.0.20260526, ONNX Runtime 1.31.0, OpenVINO 2026.4.1, `vcgencmd` clock 2000.
- Room 28 °C, as entered at the start, fan 1,972 rpm, idle CPU 55.1 °C. The session started 53 minutes after session 5
  ended.
- The three INT8 exports were calibrated on the same 548 VisDrone val images (`images_sha256` `1189ee95…`), each at its
  own imgsz, with the stock recipe: per-tensor weights, which #41 has since replaced (INVARIANTS.md §8).
- The NMS-free cells (`*-e2e`) are A2 exported with `nms=False`: the graph selects 500 boxes per image itself (output
  500 × 6), and the Pi runs no NMS. Their `score.json` records `head: end2end`, and they are labelled apart from the
  one-to-many cells (INVARIANTS.md §5).
- The detections came as `pi5-matrix-2-yolo26s.tgz` (sha256 `ca54539a…0a865`), with a `predictions.json` for every
  attempt of every cell and both reference runs (19). Its other files match the pushed ones except for line endings
  (and `runner.pid`, which is not pushed).

## Results

Session `bench/pi5/pi5-matrix-2-yolo26s`, 31 October, 13:26 to 18:21 IST (4.9 h, against the plan's 3.1 h: see below).
**Drift: 388.3 → 393.2 ms, +1.3 %: ok**, with the last reference run flagged throttled. Every cell completed.

| Cell | Precision recorded | AP | AP50 | AP-small | AR@500 | Median latency, ms | Attempts | Throttle |
|---|---|---|---|---|---|---|---|---|
| A2 640, ONNX Runtime | FP32 | 0.1943 | 0.3394 | 0.0995 | 0.3694 | 383.1 | 2 | throttled |
| A2 640, OpenVINO | FP32 | 0.1943 | 0.3394 | 0.0995 | 0.3694 | 302.2 | 2 | throttled |
| A2 640, NCNN | FP16 | 0.1932 | 0.3386 | 0.0987 | 0.3681 | 173.8 | 1 | none |
| A2 640, ONNX Runtime | INT8 | 0.1549 | 0.2933 | 0.0698 | 0.3248 | 275.9 | 1 | none |
| B2 960, ONNX Runtime | FP32 | 0.2408 | 0.4119 | 0.1391 | 0.4326 | 926.5 | 2 | throttled |
| B2 960, OpenVINO | FP32 | 0.2408 | 0.4119 | 0.1391 | 0.4326 | 766.3 | 2 | throttled |
| B2 960, NCNN | FP16 | 0.2399 | 0.4112 | 0.1383 | 0.4311 | 427.9 | 1 | none |
| B2 960, ONNX Runtime | INT8 | 0.2069 | 0.3756 | 0.1192 | 0.3931 | 644.4 | 1 | none |
| A2 640 NMS-free head, ONNX Runtime | FP32 | 0.1891 | 0.3331 | 0.0991 | 0.3710 | 386.9 | 2 | throttled |
| A2 640 NMS-free head, OpenVINO | FP32 | 0.1891 | 0.3331 | 0.0991 | 0.3710 | 310.5 | 2 | throttled |
| A2 640 NMS-free head, ONNX Runtime | INT8 | 0.1540 | 0.2911 | 0.0770 | 0.3309 | 279.1 | 1 | none |

AP is COCO AP@[.50:.95] at maxDets 500 (INVARIANTS.md), from detections at conf 0.001; every AP here is a `coco/*`
stat (§4.2). Latency is the median of `e2e_deploy`: pre + net + post at the deployment threshold, 0.25. A throttled
cell's row is its second attempt.

- **OpenVINO FP32 gives ONNX Runtime FP32's AP to 4 digits** in all three pairs, as in sessions 1, 3, 4 and 5. NCNN's
  FP16 costs 0.0011 (A2) and 0.0009 (B2) AP, within the 0.0003–0.0017 of the earlier sessions.
- **YOLO26s loses 20.3 % (A2) and 14.1 % (B2) of its AP to INT8**, twice YOLO11s's loss at the same sizes. See *INT8
  against FP32*.
- **The NMS-free head scores 0.0052 AP below the one-to-many head in FP32, and is slower, not faster.** See *The
  NMS-free head*.
- **Five of the six FP32 ONNX Runtime and OpenVINO cells ran at a median clock of 1911 MHz.** See *Throttle and clock*.
- These are the session's numbers, not the paper's analysis: the equal-latency comparisons wait for session 6.

### Throttle and clock (INVARIANTS.md §4.7, #40, #42)

Every ONNX Runtime and OpenVINO FP32 cell, six in all, hit the soft temperature limit (`get_throttled` bit 3). Each was
re-run once, as the session's rules said, and all six throttled again. No NCNN or INT8 cell was flagged. The clock and
temperature per attempt come from `timings.csv`, counted as in session 1: the clock and temperature are read at every
image, `get_throttled` at every 50th (32 samples in a full cell, 4 in a reference run). The throttle column counts the
samples with bit 3 set. The paper reports throttling by the clock (§4.7, #42): the images below 2000 MHz and the
median clock. **Bold** is the attempt the table above uses.

| Cell | Attempt | Median latency, ms | Throttle during the run | Images below 2000 MHz | Clock min / median, MHz | Peak CPU |
|---|---|---|---|---|---|---|
| reference run, first | 1 | 388.3 | none (0 of 4) | 1 of 200 | 1966 / 2000 | 82.0 °C |
| reference run, last | 1 | 393.2 | soft limit (2 of 4) | 93 of 200 | 1801 / 2000 | 83.7 °C |
| A2 640, ONNX Runtime FP32 | 1 | 393.0 | soft limit (29 of 32) | 1,516 of 1,610 | 1500 / 1856 | 85.3 °C |
| **A2 640, ONNX Runtime FP32** | 2 | 383.1 | soft limit (31 of 32) | 1,373 of 1,610 | 1746 / 1911 | 85.3 °C |
| A2 640, OpenVINO FP32 | 1 | 296.7 | soft limit (6 of 32) | 176 of 1,610 | 1801 / 2000 | 83.7 °C |
| **A2 640, OpenVINO FP32** | 2 | 302.2 | soft limit (19 of 32) | 624 of 1,610 | 1746 / 2000 | 84.2 °C |
| **A2 640, NCNN** | 1 | 173.8 | none | 0 | 2000 / 2000 | 77.7 °C |
| **A2 640, ONNX Runtime INT8** | 1 | 275.9 | none | 0 | 2000 / 2000 | 79.3 °C |
| B2 960, ONNX Runtime FP32 | 1 | 931.0 | soft limit (30 of 32) | 1,540 of 1,610 | 1500 / 1856 | 85.3 °C |
| **B2 960, ONNX Runtime FP32** | 2 | 926.5 | soft limit (29 of 32) | 1,377 of 1,610 | 1746 / 1911 | 84.2 °C |
| B2 960, OpenVINO FP32 | 1 | 735.2 | soft limit (7 of 32) | 350 of 1,610 | 1801 / 2000 | 83.7 °C |
| **B2 960, OpenVINO FP32** | 2 | 766.3 | soft limit (30 of 32) | 1,262 of 1,610 | 1500 / 1911 | 84.8 °C |
| **B2 960, NCNN** | 1 | 427.9 | none | 0 | 2000 / 2000 | 77.7 °C |
| **B2 960, ONNX Runtime INT8** | 1 | 644.4 | none | 0 | 2000 / 2000 | 80.4 °C |
| A2 640 NMS-free, ONNX Runtime FP32 | 1 | 400.2 | soft limit (30 of 32) | 1,542 of 1,610 | 1500 / 1801 | 85.9 °C |
| **A2 640 NMS-free, ONNX Runtime FP32** | 2 | 386.9 | soft limit (27 of 32) | 1,408 of 1,610 | 1746 / 1911 | 84.8 °C |
| A2 640 NMS-free, OpenVINO FP32 | 1 | 320.6 | soft limit (31 of 32) | 1,444 of 1,610 | 1500 / 1856 | 85.3 °C |
| **A2 640 NMS-free, OpenVINO FP32** | 2 | 310.5 | soft limit (28 of 32) | 1,255 of 1,610 | 1746 / 1911 | 84.2 °C |
| **A2 640 NMS-free, ONNX Runtime INT8** | 1 | 279.1 | none | 0 | 2000 / 2000 | 79.8 °C |

- **The throttle was the heaviest of the matrix so far.** Five of the six attempts used ran at a median clock of 1911
  MHz, with 1,255–1,408 of 1,610 images below 2000 MHz. A2's OpenVINO attempt held a median of 2000 MHz, with 624 images
  below. In sessions 1 and 5 every attempt used held a median of 2000 MHz. In session 3, 2 of its 6 FP32 ONNX Runtime and
  OpenVINO attempts used ran at a median of 1911 MHz, and in session 4, 1 of 4. Their others ran at 1966–2000 MHz.
- **The re-runs did not clear the flag** (0 of 6). Across the matrix, 1 of 22 re-runs has cleared it: session 1's D1
  OpenVINO. Here the re-runs moved the median latency by −3.3 to +4.2 %: down for the three ONNX Runtime cells and the
  NMS-free OpenVINO cell, up for A2's and B2's one-to-many OpenVINO cells. B2 OpenVINO's +4.2 % is outside the −3.6 to
  +1.9 % that §4.7 cites from the earlier sessions: its second attempt had 1,262 images below 2000 MHz, against 350 in
  its first.
- **For latency, the clean cells are NCNN and INT8.** No NCNN or INT8 attempt has read below 2000 MHz in any session
  (24 attempts in sessions 1 to 5). Their peaks here were 77.7–80.4 °C, against 83.7–85.9 °C for the FP32 ONNX Runtime
  and OpenVINO attempts.
- The first reference run had no flag and one image at 1966 MHz. It gave 388.3 ms, against the pilot's unthrottled
  388.5 ms (§4.7). The last reference run caught bit 3 in 2 of 4 reads, with 93 of its 200 images below 2000 MHz.

### The start gate and the session's length

All 19 gates opened at the 3-minute timeout, with the CPU at 52.9–58.4 °C; 1 of the matrix's 86 gates so far has
opened on temperature (session 5's first). The session took 4.9 h against the plan's
"about 3.1 h":

| Part | Time | Of which |
|---|---|---|
| First attempts of the 11 cells, and both reference runs | 3.1 h (13:26–16:28 IST, and 18:17–18:21) | cells 2.4 h, 13 gates 39 min, reference runs 3 min |
| The six throttle re-runs | 1.8 h (16:28–18:17 IST) | cells 1.5 h, 6 gates 18 min |

- **The first attempts and reference runs took the plan's 3.1 h.** Each cell spent 0.9–1.6 minutes beyond 1,610 × its
  median latency, against 1.3–1.8 minutes in sessions 4 and 5. The NMS-free cells spent the least (0.9–1.1 minutes): with
  no NMS, the second post-processing pass at conf 0.001 takes 0.2 ms an image against 15 ms.
- **The re-runs took 1.8 h, 37 % of the session**, about 18 minutes each with its gate. Session 6 runs without them
  (`16221bb`).

## Against what we already have

### The training records

No laptop CPU score of these exports exists. The other test-dev score of the same checkpoints is the training record's
(`verification/runs/<run>.json`, `eval_test`): PyTorch on Kaggle, the same one-to-many head, `rect` batches of 16.
Both sides are `coco/AP_all` and `coco/AP_small`, as in session 4's record. Differences are from the unrounded values.

| Checkpoint | Training record AP | Pi 5 AP (FP32) | Difference | Training record AP-small | Pi 5 AP-small | Difference |
|---|---|---|---|---|---|---|
| A1 640 | 0.1877 | 0.1875 | −0.0003 | 0.0905 | 0.0904 | −0.0001 |
| B1 960 | 0.2402 | 0.2392 | −0.0010 | 0.1387 | 0.1365 | −0.0022 |
| **A2 640** | 0.1939 | 0.1943 | +0.0005 | 0.0994 | 0.0995 | +0.0002 |
| **B2 960** | 0.2426 | 0.2408 | −0.0018 | 0.1407 | 0.1391 | −0.0016 |

- **The FP32 exports match their training records.** Across the 12 FP32 checkpoints scored so far (sessions 1 to 4),
  the Pi 5's AP is within −0.0018 (B2) to +0.0017 (P1) of the record's, and its AP-small within −0.0022 (B1) to +0.0015
  (P1). 5 of the 12 are above the record in AP. B2's −0.0018 is the largest shortfall so far; B4's was −0.0014.
- AP-large moves the most again: A2 −0.0104, B2 −0.0189, inside the −0.028 to +0.009 of the earlier 10.

### INT8 against FP32

Each INT8 cell against the ONNX Runtime FP32 export of the same checkpoint at the same imgsz (INVARIANTS.md §4.4).
Differences are from the unrounded values.

| Checkpoint | imgsz | AP, FP32 → INT8 | AP lost | AP-small, FP32 → INT8 | AP-small lost | Latency, FP32 → INT8, ms | Speed-up |
|---|---|---|---|---|---|---|---|
| A1, YOLO11s | 640 | 0.1875 → 0.1682 | −0.0193 (−10.3 %) | 0.0904 → 0.0769 | −0.0135 (−15.0 %) | 389.8 t → 272.5 | 1.43× |
| **A2, YOLO26s** | **640** | 0.1943 → 0.1549 | **−0.0395 (−20.3 %)** | 0.0995 → 0.0698 | **−0.0297 (−29.9 %)** | 383.1 t → 275.9 | 1.39× |
| B1, YOLO11s | 800 | 0.2190 → 0.2040 | −0.0150 (−6.8 %) | 0.1163 → 0.1041 | −0.0122 (−10.5 %) | 629.9 t → 416.7 | 1.51× |
| B2, YOLO26s | 800 | 0.2218 → 0.1857 | −0.0361 (−16.3 %) | 0.1184 → 0.0952 | −0.0232 (−19.6 %) | 611.9 t → 431.2 | 1.42× |
| B1, YOLO11s | 960 | 0.2392 → 0.2226 | −0.0166 (−6.9 %) | 0.1365 → 0.1258 | −0.0107 (−7.8 %) | 909.7 t → 614.0 | 1.48× |
| **B2, YOLO26s** | **960** | 0.2408 → 0.2069 | **−0.0339 (−14.1 %)** | 0.1391 → 0.1192 | **−0.0199 (−14.3 %)** | 926.5 t → 644.4 | 1.44× |
| B3, YOLO11s | 1280 | 0.2652 → 0.2548 | −0.0104 (−3.9 %) | 0.1619 → 0.1560 | −0.0060 (−3.7 %) | 1695.1 t → 1152.3 | 1.47× |
| **A2, NMS-free head** | **640** | 0.1891 → 0.1540 | −0.0351 (−18.5 %) | 0.0991 → 0.0770 | −0.0221 (−22.3 %) | 386.9 t → 279.1 | 1.39× |

t: the cell throttled. Bold: this session; 640 and 960 YOLO11s are session 1's, 800 session 5's, B3 sessions 4 and 1.

- **YOLO26s loses about twice YOLO11s's AP to INT8 at every size measured**: 2.0× at 640 (0.0395 against 0.0193), 2.4×
  at 800 and 2.0× at 960, and 2.2×, 1.9× and 1.9× in AP-small. Session 5's open question is answered: the gap it found
  at 800 holds at 640 and 960. It loses in every size range here too: AP-large 16.8 % at 640 and 17.0 % at 960, where
  YOLO11s loses 4.5 % and 1.4 %.
- **The cause is #41's**, found on the laptop at 800: the stock recipe's one weight scale per layer, on YOLO26s's
  32-channel box branch. Per-channel weights there gave B2 −6.6 % against −16.6 % (INVARIANTS.md §8). This session's
  INT8 cells are the stock recipe and stay as recorded. Session 7 measures the per-channel cells (`onnxruntime-int8pc`)
  beside them on the Pi 5, including A2's two and B2's at 960.
- **YOLO26s's INT8 loss shrinks as the input grows, within one checkpoint too**: B2 loses 16.3 % at 800 and 14.1 % at
  960, where B1 loses 6.8 % and 6.9 % (sessions 5 and 1). A2 at 640 loses the most, 20.3 %.
- **At INT8, YOLO26s falls behind YOLO11s at every size**: 0.1549 against 0.1682 AP at 640, 0.1857 against 0.2040 at
  800, 0.2069 against 0.2226 at 960. In FP32 it is ahead at all three (below).
- **The NMS-free head loses a little less to INT8** than the one-to-many head (−18.5 % against −20.3 % AP, −22.3 %
  against −29.9 % AP-small). See *The NMS-free head*.
- **A2's and B2's INT8 exports save more detections at conf 0.001 than their FP32 exports**: 740,365 against 690,807
  (+7.2 %) and 669,139 against 649,210 (+3.1 %), as every YOLO11s pair does. So B2 at 800 (session 5: fewer) is
  the only pair that saves fewer, and that is still unexplained.
- **NCNN FP16 beats ONNX Runtime INT8 again**: 37 % less latency for A2 (173.8 against 275.9 ms) and 34 % for B2
  (427.9 against 644.4), with 0.0383 and 0.0330 more AP.

### The NMS-free head (result B)

A2 at 640, the one-to-many head with NMS against the NMS-free head, on the same runtime and precision. Network and
post-processing are medians from `meta.json`.

| Runtime | AP, one-to-many / NMS-free | AP-small | AP-large | Median latency, ms | Network, ms | Post-processing at 0.25, ms | Post-processing at 0.001, ms | Clock median, MHz (images below 2000) |
|---|---|---|---|---|---|---|---|---|
| ONNX Runtime FP32 | 0.1943 / 0.1891 | 0.0995 / 0.0991 | 0.3972 / 0.3674 | 383.1 t / 386.9 t | 378.4 / 383.1 | 1.11 / 0.17 | 15.35 / 0.20 | 1911 (1,373) / 1911 (1,408) |
| OpenVINO FP32 | 0.1943 / 0.1891 | 0.0995 / 0.0991 | 0.3972 / 0.3674 | 302.2 t / 310.5 t | 297.8 / 307.0 | 1.12 / 0.15 | 15.59 / 0.16 | 2000 (624) / 1911 (1,255) |
| ONNX Runtime INT8 | 0.1549 / 0.1540 | 0.0698 / 0.0770 | 0.3306 / 0.3278 | 275.9 / 279.1 | 271.3 / 275.3 | 1.10 / 0.18 | 15.27 / 0.20 | 2000 (0) / 2000 (0) |

The one-to-many head's post-processing is NMS; the NMS-free head's is only the conversion of its 500 boxes.

- **At the deployment threshold, NMS costs about 1 ms on the Pi 5**: 0.92–0.97 ms more post-processing than the
  NMS-free head, 0.2–0.3 % of the latency.
- **The NMS-free head's network costs more than that.** On the clean pair (INT8, 2000 MHz throughout), its network
  takes 4.0 ms more (275.3 against 271.3), and its latency is 3.2 ms (1.2 %) longer. ONNX Runtime FP32, both attempts
  at a median of 1911 MHz, agrees: +4.7 ms of network, +3.9 ms (+1.0 %) in all. OpenVINO's +8.4 ms (+2.8 %) also carries
  a heavier throttle on the NMS-free attempt (1,255 against 624 images below 2000 MHz).
- At the evaluation threshold, conf 0.001, NMS takes 15.3–15.6 ms an image against the NMS-free head's 0.2 ms. That
  is the scoring pass, not the deployed one.
- **In FP32 the NMS-free head is 0.0052 AP (−2.7 %) below the one-to-many head, almost all of it in AP-large**
  (−0.0297, −7.5 %). AP-small is −0.0004 and AR@500 +0.0016. In INT8 the two heads are within 0.0008 AP, and the
  NMS-free head is ahead in AP-small (0.0770 against 0.0698).
- So on the Pi 5's CPU at conf 0.25, A2's NMS-free head saves no latency, and in FP32 it costs AP.

### YOLO26s by resolution

| imgsz | Checkpoint | AP, FP32 | AP-small | ONNX Runtime FP32, ms | OpenVINO FP32, ms | NCNN, ms (AP) | ONNX Runtime INT8, AP / ms |
|---|---|---|---|---|---|---|---|
| 640 | **A2** | **0.1943** | **0.0995** | **383.1 t** | **302.2 t** | **173.8 (0.1932)** | **0.1549 / 275.9** |
| 800 | B2 | 0.2218 | 0.1184 | 611.9 t | 489.6 | 286.0 (0.2208) | 0.1857 / 431.2 |
| 960 | **B2** | **0.2408** | **0.1391** | **926.5 t** | **766.3 t** | **427.9 (0.2399)** | **0.2069 / 644.4** |
| 1280 | B4 | 0.2690 | 0.1659 | 1675.1 t | 1400.7 t | 845.3 (0.2673) | — |

t: the cell throttled. Bold: this session; 800 is session 5's, 1280 session 4's (B4 has no INT8 cell).

- **The YOLO26s ladder climbs like YOLO11s's.** A2 at 640 → B2 at 800 gives +0.0275 AP and +0.0189 AP-small, for 1.65×
  the NCNN latency (YOLO11s, A1 → B1: +0.0315 and +0.0259, 1.61×). B2 at 800 → 960 gives +0.0190 and +0.0208 for 1.50×
  (B1: +0.0202 and +0.0202, 1.49×). The 640 → 800 step changes the checkpoint as well as the size.

### YOLO11s against YOLO26s at 640 and 960

| Runtime | AP, A1 / A2 | Latency, ms, A1 / A2 | A2 against A1 | AP, B1 / B2 | Latency, ms, B1 / B2 | B2 against B1 |
|---|---|---|---|---|---|---|
| ONNX Runtime FP32 | 0.1875 / 0.1943 | 389.8 t / 383.1 t | −1.7 % | 0.2392 / 0.2408 | 909.7 t / 926.5 t | +1.9 % |
| OpenVINO FP32 | 0.1875 / 0.1943 | 294.0 / 302.2 t | +2.8 % | 0.2392 / 0.2408 | 712.6 t / 766.3 t | +7.5 % |
| NCNN FP16 | 0.1866 / 0.1932 | 171.5 / 173.8 | +1.3 % | 0.2381 / 0.2399 | 411.3 / 427.9 | +4.0 % |
| ONNX Runtime INT8 | 0.1682 / 0.1549 | 272.5 / 275.9 | +1.2 % | 0.2226 / 0.2069 | 614.0 / 644.4 | +4.9 % |

- **In FP32, YOLO26s is 0.0069 AP and 0.0091 AP-small ahead at 640, and 0.0016 and 0.0026 at 960**, as in the training
  records (+0.0061 and +0.0089, +0.0024 and +0.0020). With sessions 4 and 5, its FP32 lead runs from 0.0016 (960) to
  0.0069 (640) AP. AP-large goes YOLO11s's way at 640 (−0.023) and YOLO26s's at 960 (+0.016), as the records do.
- **On the clean runtimes, YOLO26s's latency gap grows with the input size.** On NCNN it is +1.3 % at 640, +3.3 % at
  800, +4.0 % at 960 and +9.0 % at 1280; on ONNX Runtime INT8, +1.2 %, +3.5 % and +4.9 %. Its GFLOPs are 5.0 % more at
  640 (22.8 against 21.7) and 5.7 % more at 960, from the pruning records' `gflops_before` (none exist at 800 or 1280).
  The latency gap stays below the FLOP gap up to 960, then reaches 9.0 % at 1280. Nothing in the session's records
  explains the growth.
- **The FP32 runtimes carry the clock**: A2's and B2's ONNX Runtime attempts ran at a median of 1911 MHz, against 2000
  for A1's and B1's. Even so, A2 is 1.7 % faster than A1 on ONNX Runtime FP32, as YOLO26s was at 800 (−2.9 %) and 1280
  (−1.2 %). At 960 it is 1.9 % slower. Behind B2's OpenVINO +7.5 % are 1,262 images below 2000 MHz, against B1's 54.

### Session 3's compressed YOLO26s cells against their baselines

Session 3 had no YOLO26s baseline to set its cells against. Here they are against A2 and B2, with session 1's YOLO11s
cells against A1 and B1. AP is from the ONNX Runtime export of each cell; latency is NCNN's, the clean runtime, except
for the INT8 rows.

| Configuration | AP, YOLO11s / YOLO26s | Against the baseline, YOLO11s / YOLO26s | Latency, ms, YOLO11s / YOLO26s | Latency against the baseline |
|---|---|---|---|---|
| Pruned 640 (P1 / P3), against A1 / A2 | 0.1573 / 0.1583 | −0.0302 / −0.0360 | 112.2 / 113.8 | −34.6 % / −34.5 % |
| Distilled 640 (D1 / D2), against A1 / A2 | 0.1950 / 0.1979 | +0.0075 / +0.0036 | 171.6 / 174.3 | +0.1 % / +0.3 % |
| Pruned 960 (P2 / P4), against B1 / B2 | 0.2055 / 0.2100 | −0.0337 / −0.0308 | 271.0 / 289.3 | −34.1 % / −32.4 % |
| Pruned and distilled, INT8 (C1 / C2), against A1's / A2's INT8 | 0.1420 / 0.1285 | −0.0262 / −0.0264 | 192.8 / 193.2 (ONNX Runtime INT8) | −29.3 % / −30.0 % |

- **C2's deficit against C1 is A2's INT8 deficit carried through.** Against its own baseline's INT8, C2 loses 0.0264
  AP and C1 0.0262. C2's 0.0135 below C1 (session 3) is A2's INT8 0.0133 below A1's. Session 7 has C2's per-channel
  cell too.
- **Pruning costs YOLO26s more AP at 640** (−0.0360 against −0.0302) for the same NCNN saving (−34.5 %). At 960 it
  costs less (−0.0308 against −0.0337) and saves less (−32.4 % against −34.1 %). Session 3 noted that the pruner skips
  YOLO26s's attention block (RUNS.md).
- **Distillation gains YOLO26s half of what it gains YOLO11s**: +0.0036 AP for D2 against +0.0075 for D1. A2 is the
  best of YOLO26s's three seeds in the training records (0.1913–0.1939 AP), so in the records D2 gains +0.0058 over the
  seeds' mean against +0.0042 over A2 (D1: +0.0080 and +0.0081). Weighing that is for the analysis.

## Known limitations

- One session, one seed's order; each cell ran once (twice when throttled).
- The latencies are at 2000 MHz, not the Pi 5's stock 2400 MHz.
- All six FP32 ONNX Runtime and OpenVINO cells' latencies carry a soft throttle (above), five of them at a median clock
  of 1911 MHz, and the re-run did not clear it. Set against sessions 1 and 5, whose attempts held a median of 2000 MHz,
  they compare different clocks. Their accuracy is unaffected: the detections do not depend on the clock.
- The attempts used are the second ones, under the re-run rule this session ran with. §4.7 now keeps one attempt, and
  the first attempts are in the clock table.
- The INT8 cells are the stock recipe. The per-channel cells the matrix now uses are session 7's.
- No laptop CPU score of these exports exists, so their accuracy is checked only against the training records and by
  the runtimes agreeing.
- B4 (YOLO26s at 1280) has no INT8 cell, so the YOLO26s INT8 ladder stops at 960.
