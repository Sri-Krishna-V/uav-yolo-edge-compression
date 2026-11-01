# Raspberry Pi 5 matrix, session 7 (per-channel ONNX Runtime INT8): session record

Written on 2026-11-01, after the run. Vidhyarth ran the session with `pi/wizard.sh`, pushed the small files (`c01b8c4`)
and sent the detections; they were scored on the laptop with `scripts/studyscore.py` under its 4 GB cap. The cells
are #41's fix, measured on the Pi 5; the session is the last of #37's matrix.

## Question

What accuracy and latency does the per-channel ONNX Runtime INT8 recipe (`onnxruntime-int8pc`, INVARIANTS.md §8,
adopted on 31 October from #41) give on the Raspberry Pi 5, beside each stock ONNX Runtime INT8 cell of sessions 1–6?
On the laptop at 800 it took B2 (YOLO26s) from −16.6 % to −6.6 % AP against FP32 and left B1 (YOLO11s) at −6.4 %.
Its Pi 5 latency was not known.

## Plan and code

- Plan: `plans/pi5-matrix-7-int8pc.json` (`plan_sha256` `04940e7c…74f606`, which the repo's plan gives through
  `studysession.load_plan`). 12 full cells, all 1,610 test-dev images, batch 1: one beside each stock ONNX Runtime INT8
  cell of sessions 1–6. The reference runs, first and last: ONNX Runtime FP32, A1 at 640, on the first 200 images.
- The rules of #8: 4 threads, 20 untimed warm-up images, the 50 °C start gate with its 3-minute timeout, the drift
  check at ±3 %. Cell order shuffled with seed 65, chosen so that B2 and B1 at 800 ran first. No throttle re-run
  (INVARIANTS.md §4.7).
- **The CPU locked at 2000 MHz** (§4.7). The setup checks passed at both starts (see *The drift flag*): governor
  `performance`, no desktop, the apt timers off, numpy 2.4.6, OpenCV 5.0.0.93, ONNX Runtime 1.31.0, `vcgencmd`
  clock 2000. Kernel 6.12.109+rpt-rpi-2712, as in sessions 1–6.
- Room 28 °C, fan 2,015 rpm, idle CPU 54.5 °C. The session started 29 minutes after session 6 ended.
- **The exports** were built on the laptop by `studyexport.py --only onnxruntime-int8pc` (`scripts/studyqdq.py`,
  `a480ad3`) from each cell's FP32 and stock INT8 exports, with no new calibration: the stock export's nodes and
  activation scales, with each Conv's weights per output channel. Each manifest names both source models by sha256.
  Every cell's `meta.json` gives the model's sha256, and all 12 match the laptop's manifests. All 12 recorded INT8.
- The detections came as `pi5-matrix-7-int8pc.tgz` (sha256 `037928fd…e99bb`, whole by `gzip -t`), with a
  `predictions.json` for every cell and both reference runs (14), and the interrupted attempt's partial
  `predictions.jsonl`. Its other files match the pushed ones except for line endings (and `runner.pid`, which is not
  pushed). The ground truth at every size was already built (`runs/gt/`).

## Results

Session `bench/pi5/pi5-matrix-7-int8pc`, 31 October 23:37 IST to 1 November 03:37 IST (4.0 h, the plan's "about
4 h"). **Drift: 401.7 → 389.1 ms, −3.1 %: flagged, and every cell with it.** The Pi restarted 11 minutes in, before
any attempt that counts. The slowdown that session 6 carried ended at the restart, and the first reference run was
its last measurement. The cells were kept by the dev's decision of 1 November: see *The drift flag*. Every cell
completed.

| Cell | Precision recorded | AP | AP50 | AP-small | AR@500 | Median latency, ms | Attempts | Throttle |
|---|---|---|---|---|---|---|---|---|
| A1 640 (YOLO11s) | INT8 | 0.1696 | 0.3068 | 0.0781 | 0.3367 | 272.3 | 1 | none |
| B1 800 | INT8 | 0.2053 | 0.3607 | 0.1049 | 0.3790 | 416.2 | 1 | none |
| B1 960 | INT8 | 0.2239 | 0.3892 | 0.1259 | 0.4042 | 617.2 | 1 | none |
| B3 1120 | INT8 | 0.2418 | 0.4172 | 0.1414 | 0.4334 | 832.0 | 1 | none |
| B3 1280 | INT8 | 0.2593 | 0.4404 | 0.1579 | 0.4525 | 1153.2 | 1 | none |
| A2 640 (YOLO26s) | INT8 | 0.1795 | 0.3227 | 0.0886 | 0.3525 | 277.4 | 1 | none |
| A2 640, NMS-free head | INT8 | 0.1739 | 0.3147 | 0.0881 | 0.3556 | 279.5 | 1 | none |
| B2 800 | INT8 | 0.2074 | 0.3630 | 0.1083 | 0.3888 | 430.3 | 2 (the first interrupted) | none |
| B2 960 | INT8 | 0.2295 | 0.3980 | 0.1309 | 0.4186 | 646.0 | 1 | none |
| B4 1120 | INT8 | 0.2471 | 0.4252 | 0.1436 | 0.4411 | 880.6 | 1 | none |
| C1 640 (YOLO11s, pruned and distilled) | INT8 | 0.1442 | 0.2678 | 0.0631 | 0.3027 | 193.2 | 1 | none |
| C2 640 (YOLO26s, pruned and distilled) | INT8 | 0.1446 | 0.2703 | 0.0645 | 0.3052 | 194.1 | 1 | none |

AP is COCO AP@[.50:.95] at maxDets 500 (INVARIANTS.md), from detections at conf 0.001; every AP here is a `coco/*`
stat (§4.2). Latency is the median of `e2e_deploy`: pre + net + post at the deployment threshold, 0.25. All cells
are `onnxruntime-int8pc`; the NMS-free cell is `onnxruntime-int8pc-e2e`.

- **The fix holds on the Pi 5. YOLO26s's INT8 loss falls from 14.1–20.3 % to 4.7–8.0 % of its AP** at every size,
  recovering 57–67 % of the stock loss. YOLO11s's barely moves (+0.0013 to +0.0017 AP below 1280, +0.0045 at 1280).
  See *Per-channel against stock*.
- **At INT8, YOLO26s is now ahead of YOLO11s at every size**, as in FP32, where with the stock recipe it was behind at
  every size.
- **B2 at 800 gives the laptop's AP to 4 digits** (0.2074 on both), and B1 at 800 is within 0.0004 (0.2053 against
  0.2049).
- **The recipe costs no latency.** Each per-channel cell runs within −0.2 to +0.6 % of its stock cell, for the 10
  cells whose stock cell ran outside session 6.
- **Session 6's latencies are about 6 % high, now measured**: at 1120 the per-channel cells are 5.8 % and 6.6 %
  faster than session 6's stock cells, on every one of the 1,610 images. See *Session 6's latencies*.
- No cell throttled, and no image ran below 2000 MHz.
- These are the session's numbers, not the paper's equal-latency analysis (#46).

### The drift flag, and a restart that ended session 6's slowdown

The runner flagged every cell for a re-run: the first reference run gave 401.7 ms, the last 389.1 ms (−3.1 %). The
records show that the Pi restarted between the two, before any attempt that counts. The slow state ended with that
restart.

- **The restart.** B2 at 800, the first cell, started at 23:45 IST. Its `timings.csv` stops at image 400, and
  `runner.out` and the attempt's `bench.log` end in NUL bytes (110 and 106), which is what a power cut or a hard reset
  leaves. The runner was started again at 23:53 IST. As it does after an interruption, it ran the cell again from the
  start in a fresh directory. An interruption is not a crash.
- **The since-boot bits cleared.** Every throttle read before the restart (the first reference run and the
  interrupted attempt) gives `0xe0000`: bits 17–19, "has occurred since boot", set as in session 6. Every read after
  it gives `0x0`, in all 12 cells and the last reference run. Those bits clear only at a boot.
- **Before the restart, the Pi ran slow; after it, at the usual speed.**

  | Run | Median latency, ms | Network, median (fastest image), ms | Throttle word |
  |---|---|---|---|
  | first reference run, before the restart | 401.7 | 396.7 (393.7) | `0xe0000` |
  | B2 800, attempt 1, before the restart (400 images) | 457.1 | 450.7 (446.7) | `0xe0000` |
  | B2 800, attempt 2, after it | 430.3 | 424.1 (421.3) | `0x0` |
  | last reference run, after it | 389.1 | 383.9 (381.5) | `0x0` |

  - The first reference run's network took a median 3.2 % longer than the last's, image by image, slower on 199 of
    the 200 images. Against the first runs of sessions 1–5 it was +2.7 to +3.6 %, slower on 191–199 images.
    Pre-processing was +4.0 %.
  - The interrupted attempt's network took 6.3 % longer than attempt 2's on the same 400 images, longer on every
    one of them (+3.0 to +9.5 %).
  - The last reference run is the fastest last run of the matrix (sessions 1–5: 390.4–395.9 ms), and it held 2000
    MHz on all 200 images. Its network was within −1.6 to −0.3 % of their last runs.
  - Every cell after the restart runs as its stock cell did in sessions 1, 2, 3 and 5 (*Per-channel against
    stock*). The two recipes have the same nodes and the same activation scales and differ only in their weights'
    scales (§8).
- **Kept** (the dev, 1 November). The drift check is there to catch the Pi changing while a session's cells run.
  Here the change came before the first attempt that counts, and the reference run that registered it ran before the
  restart. The summary's drift column stays `re-run` as recorded. A re-run would have taken about 4 h.
- **What the records do not show**: what slowed the Pi, what restarted it, and whether the restart was meant. The
  `get_throttled` bits, the timings and the NUL bytes are the evidence. A restart ended the slowdown, so it was a state
  of the running system, not the hardware, the room, the versions or the files. Vidhyarth may know what ran on the Pi
  between sessions 2 and 6, and what happened at about 23:48 IST.

### Session 6's latencies: about 6 % high, measured

Session 6's record put its 1120 latencies "likely 3–5 % high" against the other sessions', from its slow reference
runs. Session 7 measures it, from the cells at 1120 that have the same checkpoint and size in both sessions:

| Cell | Session 6, stock INT8, ms | Session 7, per-channel INT8, ms | Session 6 above session 7 | Network, image by image | Pre-processing |
|---|---|---|---|---|---|
| B3 1120 | 883.5 | 832.0 | +6.2 % | session 7 faster on 1,610 of 1,610 (median −5.9 %) | −5.1 % |
| B4 1120 | 943.1 | 880.6 | +7.1 % | session 7 faster on 1,610 of 1,610 (median −6.7 %) | −3.8 % |
| the 10 other cells: sessions 1, 2, 3 and 5's stock against session 7 | | | −0.6 to +0.2 % | | |

- Pre-processing is the same code for both recipes, so the slowdown is not the recipe's.
- **It started between sessions 2 and 6.** Session 2's last reference run (31 October, 18:21 IST) ran at the usual
  speed; session 6's first (18:51 IST) was 5 % slow. The since-boot bits stayed set from session 2's end to session
  6's start, so the Pi did not restart in between. The slowdown lasted until the restart in session 7, about 5 hours.
- **For the paper** (the dev, 1 November): session 6's latencies stay as measured, flagged about 6 % high against the
  other sessions'. Session 7's per-channel cells at 1120 are the clean latency points at 1120. Comparisons within
  session 6 stand, and its accuracy is unaffected.

### Throttle and clock (INVARIANTS.md §4.7, #40, #42)

**No cell throttled, and no image in the session ran below 2000 MHz.** Counted as in sessions 1–6, from
`timings.csv`: the clock and temperature are read at every image, `get_throttled` at every 50th.

| Attempts | Throttle during the run | Images below 2000 MHz | Clock min / median, MHz | Peak CPU |
|---|---|---|---|---|
| The 12 cells, one attempt each | none (0 of 32 in each) | 0 | 2000 / 2000 | 75.5–77.1 °C |
| B2 800, the interrupted attempt (400 images) | none (0 of 8) | 0 | 2000 / 2000 | 76.5 °C |
| reference run, first / last | none (0 of 4) | 0 / 0 | 2000 / 2000 | 81.0 / 79.8 °C |

- No NCNN or INT8 attempt has now read below 2000 MHz in any session: 40 complete attempts in sessions 1–7.
- No read caught the soft-limit bit (bit 3) anywhere in the session. After the restart no "has occurred" bit was set
  either: the INT8 cells never reached the limit.

### The session's length

| Part | Time | Of which |
|---|---|---|
| The whole session | 4.0 h (23:37–03:37 IST) | |
| The 12 cells | 3.05 h | 1,610 × each median latency: 2.77 h |
| 15 gates | 45 min | all opened at the 3-minute timeout, at 52.4–55.6 °C |
| The interrupted attempt and the restart | 8 min | |
| The 2 reference runs | 3 min | |

- Each cell ran 1.1–1.6 minutes beyond 1,610 × its median latency, as in sessions 4–6: the warm-up, the second
  post-processing pass at conf 0.001, and the untimed part of each image's loop.
- The restart cost one extra gate and the interrupted attempt's 3.5 minutes.

## Against what we already have

### Per-channel against stock

Each INT8 cell against the ONNX Runtime FP32 export of the same checkpoint at the same imgsz (INVARIANTS.md §4.4).
Differences are from the unrounded values. *Recovered*: the share of the stock recipe's AP loss that per-channel takes
back.

| Checkpoint | imgsz | AP, FP32 | AP, stock INT8 (lost) | AP, per-channel INT8 (lost) | Recovered | AP-small: FP32 → stock → per-channel | Latency, stock → per-channel, ms |
|---|---|---|---|---|---|---|---|
| A1, YOLO11s | 640 | 0.1875 | 0.1682 (−10.3 %) | 0.1696 (−9.5 %) | 7 % | 0.0904 → 0.0769 → 0.0781 | 272.5 → 272.3 |
| B1, YOLO11s | 800 | 0.2190 | 0.2040 (−6.8 %) | 0.2053 (−6.3 %) | 8 % | 0.1163 → 0.1041 → 0.1049 | 416.7 → 416.2 |
| B1, YOLO11s | 960 | 0.2392 | 0.2226 (−6.9 %) | 0.2239 (−6.4 %) | 8 % | 0.1365 → 0.1258 → 0.1259 | 614.0 → 617.2 |
| B3, YOLO11s | 1120 | 0.2541 | 0.2401 (−5.5 %) | 0.2418 (−4.8 %) | 12 % | 0.1495 → 0.1390 → 0.1414 | 883.5 s6 → 832.0 |
| B3, YOLO11s | 1280 | 0.2652 | 0.2548 (−3.9 %) | 0.2593 (−2.2 %) | 43 % | 0.1619 → 0.1560 → 0.1579 | 1152.3 → 1153.2 |
| **A2, YOLO26s** | 640 | 0.1943 | 0.1549 (−20.3 %) | **0.1795 (−7.6 %)** | **62 %** | 0.0995 → 0.0698 → **0.0886** | 275.9 → 277.4 |
| **A2, NMS-free head** | 640 | 0.1891 | 0.1540 (−18.5 %) | **0.1739 (−8.0 %)** | **57 %** | 0.0991 → 0.0770 → **0.0881** | 279.1 → 279.5 |
| **B2, YOLO26s** | 800 | 0.2218 | 0.1857 (−16.3 %) | **0.2074 (−6.5 %)** | **60 %** | 0.1184 → 0.0952 → **0.1083** | 431.2 → 430.3 |
| **B2, YOLO26s** | 960 | 0.2408 | 0.2069 (−14.1 %) | **0.2295 (−4.7 %)** | **67 %** | 0.1391 → 0.1192 → **0.1309** | 644.4 → 646.0 |
| **B4, YOLO26s** | 1120 | 0.2610 | 0.2235 (−14.3 %) | **0.2471 (−5.3 %)** | **63 %** | 0.1536 → 0.1251 → **0.1436** | 943.1 s6 → 880.6 |

The FP32 cells are sessions 1, 2, 4, 5 and 6's; the stock cells sessions 1, 2, 5 and 6's. s6: session 6, about 6 %
high (above). B4 has no INT8 cell at 1280 (no stock export, §8).

- **YOLO26s's loss is now YOLO11s's.** Per-channel, YOLO26s loses 0.83×, 1.05×, 0.74× and 1.13× YOLO11s's AP at 640,
  800, 960 and 1120, against 2.0×, 2.4×, 2.0× and 2.7× with the stock recipe. Its AP-small loss falls from 14.3–29.9 %
  to 5.9–11.2 %, and its AP-large loss from 10.8–17.7 % to at most 6.4 % (the NMS-free head gains 0.2 %).
- **YOLO11s barely changes**: +0.0013 to +0.0017 AP from 640 to 1120, +0.0045 at 1280. As on the laptop (§8), most of
  YOLO11s's INT8 loss is not the weight scales.
- **The loss that remains falls as the input grows, in both models**: YOLO11s −9.5, −6.3, −6.4, −4.8 and −2.2 % from
  640 to 1280; YOLO26s −7.6, −6.5, −4.7 and −5.3 % from 640 to 1120. Its cause is not examined (#41).
- **The speed-up over FP32 is unchanged**: 1.38–1.51×, as with the stock recipe (1.39–1.51×), where both cells ran
  outside session 6. At 1120 the per-channel cells give 1.58× and 1.59× against session 6's FP32 cells, but that sets
  them against the slow session; within session 6 the stock cells gave 1.48–1.49×.
- **C1 and C2 now lose the same.** C2's per-channel cell gains 0.0161 AP over its stock cell and C1's 0.0022, so C2's
  0.0135 AP gap to C1 (session 3) closes: 0.1446 against 0.1442 AP, 0.0645 against 0.0631 AP-small. No FP32 export of
  C1 or C2 was run. Against their training records, a pointer only (§4.4), C1 loses −0.0140 AP (−8.8 %) and C2
  −0.0099 (−6.4 %), where the stock cells lost −0.0162 and −0.0260.
- **NCNN FP16 still beats ONNX Runtime INT8** on both counts at every size: 33–37 % less latency (28 % at 1120, where
  the NCNN cells are session 6's) and 0.0045–0.0169 more AP.
- AP-large moves the most and both ways: B1 at 800 gains +3.7 % over FP32, and B3 at 1280 loses 5.7 %. Large objects
  are few in test-dev.
- Detections at conf 0.001: per-channel saves fewer than stock in every YOLO26s cell but C2 (B4 at 1120: +8.9 % over
  FP32, against the stock cell's +14.0 %). B2's per-channel cells at 800 and 960 save 0.8 % and 2.3 % fewer than FP32.
  Of the stock cells, only B2 at 800's saved fewer than FP32 (#41).

### YOLO11s against YOLO26s at INT8

| imgsz | AP, YOLO11s / YOLO26s | AP-small, YOLO11s / YOLO26s | Latency, ms, YOLO11s / YOLO26s | YOLO26s's latency | YOLO26s's AP lead, FP32 |
|---|---|---|---|---|---|
| 640 | 0.1696 / 0.1795 | 0.0781 / 0.0886 | 272.3 / 277.4 | +1.9 % | +0.0069 |
| 800 | 0.2053 / 0.2074 | 0.1049 / 0.1083 | 416.2 / 430.3 | +3.4 % | +0.0028 |
| 960 | 0.2239 / 0.2295 | 0.1259 / 0.1309 | 617.2 / 646.0 | +4.7 % | +0.0016 |
| 1120 | 0.2418 / 0.2471 | 0.1414 / 0.1436 | 832.0 / 880.6 | +5.8 % | +0.0068 |

- **YOLO26s leads at INT8 at every size**: +0.0099, +0.0021, +0.0056 and +0.0052 AP, and +0.0021 to +0.0105 AP-small.
  With the stock recipe it trailed by 0.0133–0.0183 AP. At 640 and 960 its INT8 lead is larger than its FP32 lead,
  because YOLO11s loses more there.
- **Its latency gap grows with the input size**, as on NCNN and stock INT8: +1.9 % at 640 to +5.8 % at 1120. Both
  cells of each pair ran in this session.

### What it changes for the equal-latency pairs (#46)

The per-channel cells have the stock cells' latency, so #46's INT8 pairs stay the same pairs; their accuracy changes.
Against the ONNX Runtime FP32 baseline one step down:

| Compressed cell | ms | Baseline | ms | Apart | AP-small difference, per-channel (stock) |
|---|---|---|---|---|---|
| B1 INT8 at 800 | 416.2 | A1 FP32 at 640 | 389.8 t | 6.8 % | +0.0145 (+0.0137) |
| B1 INT8 at 960 | 617.2 | B1 FP32 at 800 | 629.9 t | 2.1 % | +0.0096 (+0.0095) |
| B3 INT8 at 1120 | 832.0 | B1 FP32 at 960 | 909.7 t | 9.3 % | +0.0049 (+0.0025) |
| **B2 INT8 at 960** | 646.0 | B2 FP32 at 800 | 611.9 t | 5.6 % | **+0.0125 (+0.0008)** |
| **B4 INT8 at 1120** | 880.6 | B2 FP32 at 960 | 926.5 t | 5.2 % | **+0.0044 (−0.0140)** |

t: the baseline was flagged throttled. Apart: the larger latency over the smaller.

- **The YOLO26s pairs turn from a loss or a tie into a gain** in AP-small; the YOLO11s pairs barely move.
- B2 INT8 at 800 is 12.3 % above A2 FP32 at 640 (430.3 against 383.1 ms), so it is not a pair.
- B3 INT8 at 1280 (1153.2 ms) and B3 FP32 at 1120 (1315.4 ms) are 14.1 % apart, but that FP32 cell is session 6's.
  Take 6 % out of it and they are about 7 % apart. #46 decides.

## Known limitations

- One session, one seed's order; each cell ran once (B2 at 800 twice, the first interrupted).
- The latencies are at 2000 MHz, not the Pi 5's stock 2400 MHz.
- The drift check flagged the session (−3.1 %). The cells were kept because the Pi restarted before the first attempt
  that counts, and the slowdown ended with it (above). What slowed the Pi, and what restarted it, is not recorded.
- No FP32 export of C1 or C2 was run, so their INT8 loss is measured against their stock cells and, as a pointer
  only, their training records.
- The per-channel recipe keeps the stock calibration and activation scales; only the weights' scales change (§8). The
  loss it leaves, 2–10 % of AP, is not examined.
- B4 has no INT8 cell at 1280, so the YOLO26s INT8 ladder stops at 1120.
