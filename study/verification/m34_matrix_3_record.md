# Galaxy M34 matrix, session 3: session record

Written on 2026-11-01 at about 21:45 IST, after the session ended. No record was written before it ran: the rules
are #8's and the plan's, and the analysis below is session 1's ([m34_matrix_1_record.md](m34_matrix_1_record.md)),
run on this session's files with the same code.

## Question

- What are the latency and accuracy, on the Galaxy M34 with ONNX Runtime, of: B1 and B2 in FP32 at 800; B3 and B4 in
  per-channel INT8 at 1120; B1 and B2 in FP32 at 960; and the pruned P2 and P4 in FP32 at 960, on all 1,610 test-dev
  images?
- Are B1 in FP32 at 800 against B3 in INT8 at 1120, and B2 in FP32 at 800 against B4 in INT8 at 1120, equal-latency
  pairs on the phone (#8: medians within 10 % of the smaller), each measured back to back in one session?

## Plan and code

- Plan: `plans/m34-matrix-3.json` (committed in 9e93bb2), `plan_sha256` in `session.json`. Eight cells, seed 1638. The
  order, shuffled once: B1 800 FP32, B3 1120 INT8, B2 800 FP32, B4 1120 INT8, B1 960 FP32, B2 960 FP32, P2 960 FP32,
  P4 960 FP32. Each pair runs back to back, the FP32 cell first.
- The reference runs, first and last: YOLO11s, ONNX Runtime FP32, 640, the first 200 test-dev images (#8).
- The code is unchanged since 31 October (`studybench.py` and `studysession.py`, last changed in 16221bb).
- The software on the phone is sessions 1 and 2's: Python 3.14.8, numpy 2.5.4, OpenCV 5.0.0, ONNX Runtime 1.31.1. Each
  cell's `meta.json` carries its versions (the 1.31.0 in its manifest is the laptop's export toolchain).
- The cpuset log: `bench/m34/cpuset-m34-matrix-3.log`, a sample every 10 s, copied home after the session.
- B1 and B2 run at 800 and 960 from their 960 checkpoints, and B3 and B4 at 1120 from their 1280 checkpoints, as on
  the Pi 5. At 800 and 1120 they run at an input resolution they were not trained at.
- Run from the laptop with the wizard (`ENV_FILE=pi/.env.m34 bash pi/wizard.sh plans/m34-matrix-3.json`), the phone
  plugged in, Termux kept on screen.

## Results

**The drift check failed: +5.9 %, past the ±3 % limit, so the runner flagged every cell for a re-run.** No cell
failed. **Termux left the screen once, for about 10 minutes during B1 at 800** (see [Events](#events)); the cpuset
held `/top-app` for the rest of the session. The user had decided to keep the phone's sessions on session 1's terms if
they failed the drift check (see [The decision](#the-decision)). The phone's latency stays provisional (#34).

| Run (in order) | Run starts (gate opens), IST | Battery at its gate | Images | Median latency, ms |
|---|---|---|---|---|
| reference run, first | 16:09:28 | 31.9 °C | 200 | 609.3 |
| B1, YOLO11s 800, FP32 | 16:15:00 | 31.9 °C | 1,610 | 1,043.7 |
| B3, YOLO11s 1120, INT8 per-channel | 16:55:54 | 34.6 °C | 1,610 | 987.0 |
| B2, YOLO26s 800, FP32 | 17:27:11 | 34.6 °C | 1,610 | 1,007.3 |
| B4, YOLO26s 1120, INT8 per-channel | 18:00:05 | 33.6 °C | 1,610 | 1,045.1 |
| B1, YOLO11s 960, FP32 | 18:34:17 | 33.2 °C | 1,610 | 1,397.6 |
| B2, YOLO26s 960, FP32 | 19:17:05 | 34.6 °C | 1,610 | 1,411.9 |
| P2, YOLO11s 960, pruned, FP32 | 20:01:06 | 33.3 °C | 1,610 | 858.9 |
| P4, YOLO26s 960, pruned, FP32 | 20:29:25 | 33.5 °C | 1,610 | 862.3 |
| reference run, last | 20:58:18 | 33.6 °C | 200 | 645.5 |

- **Drift:** 609.3 → 645.5 ms, +5.9 %: outside ±3 %.
- **Cpuset:** 1,738 samples, from 16:06:27 to 21:00:56, with no gap over 15 s. 1,678 read `/top-app` with CPUs 0–7.
  From 16:34:31 to about 16:44:20, 4 read `/foreground` or `/foreground-boost` and 56 read `/moderate` with CPUs 0–2.
  One more read `/foreground` at 19:01:39.
- **No cell failed.** Each run took one attempt. The session ran from 16:06:26 to 21:00:57 IST (4 h 55 min).
- **Battery:** 31.9–34.6 °C at the gates. At 21:33, after the session, it read 100 %, unplugged, 31.2 °C.

### Latency and accuracy

All eight cells use the one-to-many head. Latency is the median `e2e_deploy`, at the deployment threshold 0.25.

| Cell | Median, ms | p10, ms | p90, ms | Time of the cell | AP | AP50 | AP-small | AP-medium | AP-large | AR500 |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 800 FP32 | 1,043.7 | 963.7 | 1,573.3 | 37 min 52 s | 0.2190 | 0.3753 | 0.1163 | 0.3271 | 0.4623 | 0.3953 |
| B3 1120 INT8 | 987.0 | 873.7 | 1,078.5 | 28 min 15 s | 0.2418 | 0.4172 | 0.1414 | 0.3523 | 0.4876 | 0.4334 |
| B2 800 FP32 | 1,007.3 | 991.3 | 1,099.1 | 29 min 52 s | 0.2218 | 0.3811 | 0.1184 | 0.3303 | 0.4976 | 0.4075 |
| B4 1120 INT8 | 1,045.1 | 1,030.3 | 1,198.6 | 31 min 10 s | 0.2471 | 0.4252 | 0.1436 | 0.3590 | 0.4749 | 0.4411 |
| B1 960 FP32 | 1,397.6 | 1,317.0 | 1,485.7 | 39 min 47 s | 0.2392 | 0.4067 | 0.1365 | 0.3496 | 0.4635 | 0.4211 |
| B2 960 FP32 | 1,411.9 | 1,394.4 | 1,509.7 | 40 min 59 s | 0.2408 | 0.4119 | 0.1391 | 0.3496 | 0.4797 | 0.4326 |
| P2 960 FP32 | 858.9 | 797.9 | 949.5 | 25 min 17 s | 0.2055 | 0.3583 | 0.1121 | 0.3027 | 0.3928 | 0.3876 |
| P4 960 FP32 | 862.3 | 846.4 | 943.8 | 25 min 51 s | 0.2100 | 0.3691 | 0.1157 | 0.3091 | 0.4463 | 0.3961 |

- **The accuracy matches the Pi 5's.** The two INT8 cells give exactly the AP, AP50, AP75, AP-small, AP-medium,
  AP-large and AR500 of the Pi 5's session 7 cells. The six FP32 cells differ from the Pi 5's sessions 1, 2, 3 and 5 by
  less than 5 × 10⁻⁶.
- **Both equal-latency pairs hold:**
  - **B3 1120 INT8 against B1 800 FP32:** 987.0 against 1,043.7 ms, 5.7 % apart. B3 has +0.0251 AP-small and +0.0228
    AP over B1. B1's median carries the 10 minutes off screen; without its images 1031–1200 it is 1,037.0 ms, and the
    pair is 5.1 % apart. B3 ran second and is the faster: if the phone slowed through the session, the true gap is
    wider than measured. The two ran back to back, about 35 minutes apart, so that share of a 5.9 % drift over 4 h 55 min
    is small.
  - **B4 1120 INT8 against B2 800 FP32:** 1,045.1 against 1,007.3 ms, 3.8 % apart. B4 ran second. If the phone slowed
    through the session, B4 is measured slow, and the true gap is at most 3.8 %. B4 has +0.0252 AP-small and +0.0253
    AP over B2.
  - On the Pi 5 the INT8 cells at 1120 pair with the FP32 cells at 960 instead, for +0.0044 to +0.0049 AP-small. INT8 is
    about 2× faster on the phone, against about 1.4× on the Pi 5, so on the phone it buys a larger step in input
    resolution, and a larger gain.
- **Pruned against baseline at 960, within the session:** YOLO11s 858.9 / 1,397.6 = 0.615 (1.63× faster); YOLO26s
  862.3 / 1,411.9 = 0.611 (1.64×). The Pi 5's ONNX Runtime FP32 cells give 0.632 and 0.617. P2 is 21.5 % from B1 at
  800 (858.9 against 1,043.7 ms): no pair.
- **YOLO11s against YOLO26s, within the session:** FP32 at 800 1,043.7 against 1,007.3 ms (−3.5 %; −2.9 % without B1's
  off-screen images); INT8 at 1120 987.0 against 1,045.1 ms (+5.9 %); FP32 at 960 1,397.6 against 1,411.9 ms (+1.0 %);
  pruned at 960 858.9 against 862.3 ms (+0.4 %). All are inside this session's drift (5.94 %; the INT8 gap is 5.89 %),
  so the two models' latency cannot be told apart on the phone. The Pi 5 put YOLO26s +5.8 % above YOLO11s at 1120 in
  INT8 within one session.
- **INT8 against FP32 at 960, across sessions:** B1 1,397.6 (session 3) against 733.5 ms (session 1), 1.91×; B2
  1,411.9 (session 3) against 763.0 ms (session 2), 1.85×. They come from different sessions, but the speed-up is far
  larger than any session's drift.

### Where the drift is

- Over the reference's first 50 images: 596.4 → 672.6 ms. Over images 51–200: 615.7 → 643.1 ms, **+4.5 %**.
- The first reference run's four blocks of 50 images had median net times of 587, 566, 599 and 628 ms. The last
  reference run's: 657, 640, 628 and 629 ms.
- The medians of 10 equal parts of each cell (about 161 images each), in ms:
  - B1 800 FP32: 963.7, 967.7, 1,027.1, 1,058.1, 1,043.4, 1,033.4, 3,194.9, 1,266.8, 1,051.7, 1,043.3.
  - B3 1120 INT8: 984.9, 996.3, 1,005.3, 995.3, 1,001.3, 985.1, 905.7, 896.3, 882.6, 893.6.
  - B2 800 FP32: 1,002.7, 1,006.4, 1,011.9, 1,002.8, 1,007.8, 1,004.4, 1,004.2, 1,005.6, 1,007.7, 1,021.0.
  - B4 1120 INT8: 980.5, 1,055.1, 1,051.6, 1,040.7, 1,044.5, 1,046.5, 1,050.0, 1,043.8, 1,042.2, 1,044.0.
  - B1 960 FP32: 1,371.6, 1,415.7, 1,420.8, 1,441.1, 1,349.8, 1,329.7, 1,355.9, 1,421.4, 1,342.7, 1,339.0.
  - B2 960 FP32: 1,398.6, 1,412.2, 1,414.2, 1,445.7, 1,414.5, 1,412.5, 1,408.6, 1,407.0, 1,410.6, 1,419.7.
  - P2 960 FP32: 830.8, 882.0, 871.5, 866.3, 867.3, 901.8, 813.0, 820.6, 804.4, 811.0.
  - P4 960 FP32: 852.2, 865.6, 858.4, 856.7, 869.0, 858.5, 862.7, 861.5, 859.2, 882.1.
- The step after the gate that sessions 1 and 2 showed is weaker here. Most cells start a little fast (their first 50
  images run 0–13 % below their medians) and hold one level after that. B1 at 800 steps up about 8 % after about 7.5
  minutes. Two
  cells run *faster* in their last four parts: B3 at 1120 by about 10 % and P2 at 960 by about 7 %. The phone shows no
  CPU temperature or clock, so none of this is explained (#33, #34).
- So the reference's 200 images, about 2.5 minutes, again sit in the phase in which the phone settles, and the drift
  check, as #8 defines it, measures it. This session ran for 4 h 55 min, against 2 h 19 min (session 2, +3.4 %) and
  1 h 52 min (session 1, +4.9 %).

### The slow stretches (#34)

Blocks of 10 images, slow when their median net time is at least 1.3× the run's median net time.

| Run | Blocks | Slow blocks |
|---|---|---|
| reference run, first | 20 | 0 |
| B1 800 FP32 | 161 | 16: images 591–600 (1,340 ms), and 1041–1190, 15 in a row (3,226–4,402 ms), against 1,025 ms |
| B3 1120 INT8 | 161 | 0 |
| B2 800 FP32 | 161 | 1: images 751–760 (1,308 ms against 989 ms) |
| B4 1120 INT8 | 161 | 0 |
| B1 960 FP32 | 161 | 0 |
| B2 960 FP32 | 161 | 1: images 511–520 (1,975 ms against 1,388 ms) |
| P2 960 FP32 | 161 | 0 |
| P4 960 FP32 | 161 | 0 |
| reference run, last | 20 | 0 |

- B1 800's 15 slow blocks in a row are the 10 minutes off screen: the runner on CPUs 0–2 ran at about 3.3–4.4 s per
  image, 3.1–4.3× its median. The cpuset explains this stretch. As in sessions 1 and 2, it does not explain the few
  single slow blocks.

## Events

- **The laptop lost the phone at about 16:34.** The phone's Wi-Fi left the laptop's hotspot, and the wizard's progress
  view lost its SSH connection; the runner on the phone kept running. The user reconnected the phone's Wi-Fi. While the
  phone's settings were on screen, from 16:34:31 to about 16:44:20, the runner's cpuset was `/foreground` and then
  `/moderate`, CPUs 0–2. This fell on B1 at 800, images about 1041–1190 of 1,610. One more `/foreground` sample, at
  19:01:39, fell on B1 at 960 and slowed no block.
- The wizard's progress view polled the phone over SSH every 30 s, except during the outage.
- The wizard copied the session home and scored the eight cells.

## The decision

Recorded in session 1's record, at about 13:20 IST on 2026-11-01: the user chose to keep the phone's sessions on
session 1's terms if they failed the drift check. The reasons are session 1's:

- A re-run is 5 hours, and a session that long fails ±3 % again.
- The accuracy does not depend on the drift or on the cpuset.
- No conclusion drawn from this session moves with a 5.9 % shift. The two pairs are 5.7 % and 3.8 % apart, inside 10 %
  with or without B1's off-screen images, and each ran back to back. Pruning is about 1.6× faster, and INT8 about 1.9× faster, than FP32 at 960.
  The YOLO11s and YOLO26s latencies are reported as not separable.

**How the cells are used:** they are reported with the failed drift check (`cells.csv` notes it), and the phone's
latency stays provisional (#34). A phone latency difference smaller than this session's drift (5.9 %) is not reported
as a difference. B1 at 800 is reported at its median over all 1,610 images, with its off-screen stretch noted; the
off-screen images move that median by 0.6 %.
