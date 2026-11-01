# Galaxy M34 matrix, session 2: session record

Written on 2026-11-01 at about 15:55 IST, after the session ended. No record was written before it ran: the rules
are #8's and the plan's, and the analysis below is session 1's ([m34_matrix_1_record.md](m34_matrix_1_record.md)), run
on this session's files with the same code.

## Question

- What are the latency and accuracy, on the Galaxy M34 with ONNX Runtime, of: YOLO26s at 960 in per-channel INT8
  (B2); the combined configuration of both study models at 640 in per-channel INT8 (C1, C2); and the pruned (P1, P3)
  and distilled (D1, D2) configurations of both study models at 640 in FP32, on all 1,610 test-dev images?
- Is B2 at 960 in per-channel INT8 an equal-latency partner (#8: medians within 10 % of the smaller) for A2 at 640 in
  FP32, which session 1 measured?

## Plan and code

- Plan: `plans/m34-matrix-2.json` (committed in 979ded4), `plan_sha256` in `session.json`
  `661c7443b6faaa069d08e673702916d105126390af8a1be64bab6201d37b3bab`. Seven cells, seed 376. The order, shuffled once:
  B2 960 INT8, C1 640 INT8, C2 640 INT8, P3 640 FP32, P1 640 FP32, D2 640 FP32, D1 640 FP32. The plan's file on the
  phone is the committed one byte for byte.
- The reference runs, first and last: YOLO11s, ONNX Runtime FP32, 640, the first 200 test-dev images (#8).
- The code is unchanged since 31 October (`studybench.py` and `studysession.py`, last changed in 16221bb).
- The software on the phone is session 1's: Python 3.14.8, numpy 2.5.4, OpenCV 5.0.0, ONNX Runtime 1.31.1. Each cell's
  `meta.json` carries its versions.
- The cpuset log: `bench/m34/cpuset-m34-matrix-2.log`, a sample every 10 s, copied home after the session.
- Run from the laptop with the wizard (`ENV_FILE=pi/.env.m34 bash pi/wizard.sh plans/m34-matrix-2.json`), the phone
  plugged in, Termux kept on screen.

## Results

**The drift check failed: +3.4 %, past the ±3 % limit, so the runner flagged every cell for a re-run.** The cpuset
held `/top-app` for the whole session and no cell failed. The user had decided before the session to keep it on
session 1's terms if it failed (see [The decision](#the-decision)). The phone's latency stays provisional (#34).

| Run (in order) | Run starts (gate opens), IST | Battery at its gate | Images | Median latency, ms |
|---|---|---|---|---|
| reference run, first | 13:23:31 | 31.7 °C | 200 | 630.5 |
| B2, YOLO26s 960, INT8 per-channel | 13:29:07 | 31.7 °C | 1,610 | 763.0 |
| C1, YOLO11s 640, combined, INT8 per-channel | 13:55:00 | 33.7 °C | 1,610 | 246.5 |
| C2, YOLO26s 640, combined, INT8 per-channel | 14:06:59 | 33.6 °C | 1,610 | 247.2 |
| P3, YOLO26s 640, pruned, FP32 | 14:18:56 | 33.6 °C | 1,610 | 393.2 |
| P1, YOLO11s 640, pruned, FP32 | 14:34:50 | 33.3 °C | 1,610 | 389.3 |
| D2, YOLO26s 640, distilled, FP32 | 14:50:36 | 33.2 °C | 1,610 | 657.9 |
| D1, YOLO11s 640, distilled, FP32 | 15:13:35 | 33.5 °C | 1,610 | 675.2 |
| reference run, last | 15:37:17 | 33.5 °C | 200 | 652.0 |

- **Drift:** 630.5 → 652.0 ms, +3.4 %: outside ±3 %.
- **Cpuset:** all 824 samples, from 13:20:29 to 15:39:48, read `/top-app` with CPUs 0–7.
- **No cell failed.** Each run took one attempt. The session ran from 13:20:29 to 15:39:58 IST (2 h 19 min).
- **Battery:** 31.7–33.7 °C at the gates, cooler than session 1 (33.5–35.0 °C). It read 100 %, full, plugged in (AC),
  33.1 °C, at about 15:47. Its charge at the start was not recorded.

### Latency and accuracy

All seven cells use the one-to-many head. Latency is the median `e2e_deploy`, at the deployment threshold 0.25.

| Cell | Median, ms | p10, ms | p90, ms | Time of the cell | AP | AP50 | AP-small | AP-medium | AP-large | AR500 |
|---|---|---|---|---|---|---|---|---|---|---|
| B2 960 INT8 | 763.0 | 689.7 | 882.3 | 22 min 52 s | 0.2295 | 0.3980 | 0.1309 | 0.3385 | 0.4786 | 0.4186 |
| C1 640 INT8 | 246.5 | 219.0 | 315.5 | 8 min 57 s | 0.1442 | 0.2678 | 0.0631 | 0.2227 | 0.3675 | 0.3027 |
| C2 640 INT8 | 247.2 | 217.9 | 307.4 | 8 min 55 s | 0.1446 | 0.2703 | 0.0645 | 0.2206 | 0.3387 | 0.3052 |
| P3 640 FP32 | 393.2 | 370.2 | 452.8 | 12 min 52 s | 0.1583 | 0.2856 | 0.0754 | 0.2399 | 0.3431 | 0.3266 |
| P1 640 FP32 | 389.3 | 369.1 | 447.3 | 12 min 45 s | 0.1573 | 0.2808 | 0.0739 | 0.2395 | 0.3640 | 0.3201 |
| D2 640 FP32 | 657.9 | 624.9 | 719.9 | 19 min 57 s | 0.1979 | 0.3514 | 0.1027 | 0.2919 | 0.4093 | 0.3734 |
| D1 640 FP32 | 675.2 | 647.8 | 749.2 | 20 min 40 s | 0.1950 | 0.3404 | 0.0963 | 0.2979 | 0.4617 | 0.3623 |

- **The accuracy matches the Pi 5's.** The three INT8 cells give exactly the AP, AP50, AP75, AP-small, AP-medium,
  AP-large and AR500 of the Pi 5's session 7 cells. The four FP32 cells differ from the Pi 5's sessions 1 and 3 by less
  than 2 × 10⁻⁶ (the pruned cells by less than 5 × 10⁻⁷).
- **B2 does not form an equal-latency pair with A2:** B2 960 INT8 763.0 ms against session 1's A2 640 FP32 667.4 ms,
  14.3 % apart (outside 10 %). They come from different sessions, so the gap also holds the change between sessions,
  which neither drift check measures. That cannot close it: session 2's first reference ran 2.7 % faster than session
  1's (630.5 against 648.3 ms), and scaling B2 by it widens the gap to 17.6 %. B2 has +0.0352 AP and +0.0314 AP-small
  over A2, at 14.3 % more latency.
- **YOLO11s against YOLO26s at 640, within the session:** combined 246.5 against 247.2 ms (−0.3 %), pruned 389.3
  against 393.2 ms (−1.0 %), distilled 675.2 against 657.9 ms (+2.6 %). All are inside this session's drift, so the two
  models' latency cannot be told apart on the phone, as in session 1.
- **Pruned against distilled, within the session** (the distilled models keep the baseline's architecture): YOLO11s
  389.3 / 675.2 = 0.577 (1.73× faster); YOLO26s 393.2 / 657.9 = 0.598 (1.67×). The Pi 5's ONNX Runtime FP32 cells give
  0.618 and 0.600.
- **Combined (INT8) against pruned (FP32), within the session:** YOLO11s 246.5 / 389.3 = 0.633 (1.58×); YOLO26s
  247.2 / 393.2 = 0.629 (1.59×). On the Pi 5 the same pairs give 0.806 and 0.833: INT8 gains more on the phone, as
  session 1's A1 and A2 already showed (about 2× on the phone).
- **Against session 1, across sessions:** D1 675.2 ms against A1 675.2 ms (same architecture, 0.0 %); D2 657.9 against
  A2 667.4 ms (−1.4 %); B2 763.0 against B1 733.5 ms (+4.0 %). All are inside session 1's drift (4.9 %). The Pi 5 put
  B2 at 960 in INT8 4.7 % above B1 within one session (646.0 against 617.2 ms, session 7).

### Where the drift is

- Over the reference's first 50 images: 588.3 → 668.4 ms. Over images 51–200: 633.8 → 649.1 ms, **+2.4 %**.
- The first reference run started fast and had not settled by its end: its four blocks of 50 images had median net
  times of 579, 602, 621 and 636 ms. The last reference run did not start fast: 656, 639, 635 and 636 ms. This is
  session 1's pattern.
- D1, which has the reference's architecture and ran just before the last reference run, has a median of 647.6 ms over
  the reference's 200 images, against the last reference run's 652.0 ms.
- The medians of 10 equal parts of each cell (about 161 images each), in ms:
  - B2 960 INT8: 677.7, 697.4, 766.9, 770.5, 772.1, 765.8, 763.6, 760.7, 762.2, 771.4.
  - C1 640 INT8: 230.5, 224.3, 226.1, 228.4, 235.6, 262.2, 251.8, 269.4, 252.0, 250.5.
  - C2 640 INT8: 228.6, 223.5, 221.2, 220.9, 258.2, 255.2, 257.9, 255.7, 247.5, 253.7.
  - P3 640 FP32: 380.6, 378.7, 376.4, 395.7, 403.1, 406.9, 393.9, 399.7, 392.2, 394.6.
  - P1 640 FP32: 380.9, 377.5, 381.2, 406.5, 394.1, 397.5, 388.8, 379.2, 375.0, 406.5.
  - D2 640 FP32: 630.9, 632.9, 658.8, 661.3, 653.5, 660.6, 666.2, 671.6, 657.8, 663.3.
  - D1 640 FP32: 648.8, 646.4, 682.3, 675.8, 677.1, 678.7, 684.1, 674.5, 681.5, 683.5.
- So every cell starts fast after its 3-minute gate and steps up 3.6–4.6 minutes into the run: by about 5 % for the
  FP32 cells and 10–15 % for the INT8 cells, then holds that level. In session 1 the step came after 2–4 minutes. The
  reference's 200 images take about 2.6 minutes, so the first reference run sits wholly in the fast phase, and the drift
  check, as #8 defines it, measures the step. A full cell runs for 9–23 minutes and spends most of them at the settled
  level. The phone shows no CPU temperature or clock, so the cause is not confirmed (#33, #34).

### The slow stretches (#34)

Blocks of 10 images, slow when their median net time is at least 1.3× the run's median net time.

| Run | Blocks | Slow blocks |
|---|---|---|
| reference run, first | 20 | 0 |
| B2 960 INT8 | 161 | 2: images 101–110, 1181–1190 (1,099 and 1,035 ms against 739 ms) |
| C1 640 INT8 | 161 | 6, in three pairs: images 71–90, 811–830, 1131–1150 (314–817 ms against 234 ms) |
| C2 640 INT8 | 161 | 4: images 641–660 (a pair), 1381–1390, 1601–1610 (310–583 ms against 235 ms) |
| P3 640 FP32 | 161 | 3: images 921–930, 1591–1610 (a pair) (554–578 ms against 381 ms) |
| P1 640 FP32 | 161 | 5: images 211–220, 541–550, 861–870, 1531–1550 (a pair) (497–731 ms against 377 ms) |
| D2 640 FP32 | 161 | 1: images 301–310 (917 ms against 645 ms) |
| D1 640 FP32 | 161 | 2: images 871–880, 1361–1370 (1,159 and 917 ms against 663 ms) |
| reference run, last | 20 | 0 |

### Events

- No aborted start. The session started once, at 13:20:29, with Termux on screen; the cpuset logger was waiting for
  the runner and caught it from its first sample.
- The wizard's progress view polled the phone over SSH every 30 s for the whole session, as in session 1. The agents
  read the session log and the cpuset log over SSH a few times.
- The wizard copied the session home and scored the seven cells; their score files were written at 15:44–15:45.

## The decision

Recorded in session 1's record, at about 13:20 IST on 2026-11-01, before this session started: the user chose to run
session 2 on session 1's terms and to keep it if its drift check failed too. The reasons are session 1's:

- A re-run is 2.5 hours, and a session that long would likely fail ±3 % again.
- The accuracy does not depend on the drift.
- No conclusion drawn from the phone moves with a 3.4 % shift. B2 and A2 are 14.3 % apart. Pruning is about 1.7× faster
  and INT8 about 1.6× faster again. The YOLO11s and YOLO26s latencies are reported as not separable.

**How the cells are used:** they are reported with the failed drift check (`cells.csv` notes it), and the phone's
latency stays provisional (#34). A phone latency difference smaller than this session's drift (3.4 %) is not reported
as a difference; across sessions 1 and 2, the larger drift (4.9 %) applies.
