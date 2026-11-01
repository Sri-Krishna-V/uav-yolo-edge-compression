# Galaxy M34 matrix, session 1: session record

Written on 2026-11-01 at about 13:30 IST, after the session ended. No record was written before it ran: the rules are
#8's and the plan's, and the analysis below is the one used for the 25 October full cells
([m34_fullcell_record.md](m34_fullcell_record.md), [m34_fullcell_int8_record.md](m34_fullcell_int8_record.md)).

## Question

- What are the latency and accuracy, on the Galaxy M34 with ONNX Runtime, of: YOLO11s at 640 in FP32 and in
  per-channel INT8 (A1), YOLO11s at 960 in per-channel INT8 (B1), and YOLO26s at 640 in FP32 and in per-channel INT8
  (A2), on all 1,610 test-dev images?
- Is A1 at 640 in FP32 against B1 at 960 in per-channel INT8 an equal-latency pair on the phone (#8: medians within
  10 % of the smaller), measured back to back in one session?
- How much did the phone's new software change the baseline's latency against 25 October?

## Plan and code

- Plan: `plans/m34-matrix-1.json` (committed in 979ded4), `plan_sha256` in `session.json`
  `fa08111439cc6416f99a213ecdd59af2090e0d54211173a9b75d1e1355158229`. Five cells, seed 5. The order, shuffled once:
  A1 640 FP32, B1 960 INT8, A2 640 FP32, A1 640 INT8, A2 640 INT8. The pair runs first, back to back.
- The reference runs, first and last: YOLO11s, ONNX Runtime FP32, 640, the first 200 test-dev images (#8).
- The code is unchanged since 31 October (`studybench.py` and `studysession.py`, last changed in 16221bb).
- **New software on the phone.** Termux and Termux:API were reinstalled on 1 November. Python 3.14.8 and numpy 2.5.4
  as before; OpenCV 5.0.0 and ONNX Runtime 1.31.1, against 4.14.0 and 1.31.0 on 25 October. Each cell's `meta.json`
  carries its versions.
- The cpuset log: `bench/m34/cpuset-m34-matrix-1.log`, a sample every 10 s, copied home after the session.
- Run from the laptop with the wizard (`ENV_FILE=pi/.env.m34 bash pi/wizard.sh plans/m34-matrix-1.json`), the phone
  plugged in, Termux kept on screen.

## Results

**The drift check failed: +4.9 %, past the ±3 % limit, so the runner flagged every cell for a re-run.** The cpuset
held `/top-app` for the whole session and no cell failed. On 1 November at about 13:20 IST the user decided to keep the
session and not re-run it (see [The decision](#the-decision)). The phone's latency stays provisional (#34).

| Run (in order) | Run starts (gate opens), IST | Battery at its gate | Images | Median latency, ms |
|---|---|---|---|---|
| reference run, first | 11:09:46 | 33.9 °C | 200 | 648.3 |
| A1, YOLO11s 640, FP32 | 11:15:35 | 33.5 °C | 1,610 | 675.2 |
| B1, YOLO11s 960, INT8 per-channel | 11:39:30 | 34.3 °C | 1,610 | 733.5 |
| A2, YOLO26s 640, FP32 | 12:04:37 | 35.0 °C | 1,610 | 667.4 |
| A1, YOLO11s 640, INT8 per-channel | 12:27:51 | 34.2 °C | 1,610 | 335.6 |
| A2, YOLO26s 640, INT8 per-channel | 12:42:07 | 34.2 °C | 1,610 | 346.0 |
| reference run, last | 12:56:43 | 34.7 °C | 200 | 680.1 |

- **Drift:** 648.3 → 680.1 ms, +4.9 %: outside ±3 %.
- **Cpuset:** all 666 samples, from 11:06:45 to 12:59:25, read `/top-app` with CPUs 0–7.
- **No cell failed.** Each run took one attempt. The session ran from 11:06:45 to 12:59:31 IST (1 h 52 min).
- **Battery:** plugged in throughout. It was charged to about 50 % with fast charging before the start, then fast
  charging was turned off. It read 93 %, charging, at about 13:15.

### Latency and accuracy

All five cells use the one-to-many head. Latency is the median `e2e_deploy`, at the deployment threshold 0.25.

| Cell | Median, ms | p10, ms | p90, ms | Time of the cell | AP | AP50 | AP-small | AP-medium | AP-large | AR500 |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 640 FP32 | 675.2 | 649.2 | 769.9 | 20 min 54 s | 0.1875 | 0.3265 | 0.0904 | 0.2837 | 0.4203 | 0.3563 |
| B1 960 INT8 | 733.5 | 667.3 | 855.4 | 22 min 05 s | 0.2239 | 0.3892 | 0.1259 | 0.3297 | 0.4613 | 0.4042 |
| A2 640 FP32 | 667.4 | 649.4 | 726.4 | 20 min 12 s | 0.1943 | 0.3394 | 0.0995 | 0.2877 | 0.3972 | 0.3694 |
| A1 640 INT8 | 335.6 | 304.7 | 392.1 | 11 min 14 s | 0.1696 | 0.3068 | 0.0781 | 0.2593 | 0.4044 | 0.3367 |
| A2 640 INT8 | 346.0 | 310.3 | 422.1 | 11 min 34 s | 0.1795 | 0.3227 | 0.0886 | 0.2704 | 0.3817 | 0.3525 |

- **The accuracy matches the Pi 5's.** The three INT8 cells give exactly the AP and AP-small of the Pi 5's session 7
  cells. The two FP32 cells differ from the Pi 5's sessions 1 and 2 by less than 3 × 10⁻⁷. The phone's new software
  does not change accuracy.
- **The equal-latency pair holds:** B1 960 INT8 733.5 ms against A1 640 FP32 675.2 ms, 8.6 % apart (within 10 %). B1 ran
  second. If the phone slowed through the session, B1 is measured slow, and the true gap is at most 8.6 %. At equal
  latency, B1 has +0.0364 AP and +0.0355 AP-small over A1.
- **INT8 against FP32, within the session:** A1 335.6 / 675.2 = 0.497 (2.01× faster); A2 346.0 / 667.4 = 0.518
  (1.93×).
- **YOLO11s against YOLO26s at 640:** FP32 675.2 against 667.4 ms (−1.2 %); INT8 335.6 against 346.0 ms (+3.1 %). Both
  are inside this session's drift, so the two models' latency cannot be told apart on the phone.
- **Against 25 October:** A1 640 FP32 was 636.6 ms then and 675.2 ms now (+6.1 %). Its reference ran at 613.7 ms then,
  648.3 ms now (+5.6 %). The new software, or the phone's state, makes the baseline about 6 % slower. Nothing here
  separates the two. The 25 October INT8 cell (304.8 ms) used the stock per-tensor INT8 recipe, not per-channel, so it
  is not compared with these.

### Where the drift is

- Over the reference's first 50 images: 628.7 → 696.8 ms. Over images 51–200: 663.5 → 674.6 ms, **+1.7 %**.
- Each run starts fast after its 3-minute gate and settles within a few minutes. The first reference run's four blocks
  of 50 images had median net times of 615, 647, 628 and 658 ms. The last reference run did not start fast: 682, 654,
  664 and 661 ms.
- A1 640 FP32's first 200 images (the reference's images, run right after it) have a median of 649.2 ms, against the
  first reference run's 648.3 ms.
- The medians of 10 equal parts of each cell (about 161 images each), in ms:
  - A1 640 FP32: 648.9, 660.1, 675.8, 674.4, 674.3, 682.1, 679.2, 739.9, 677.0, 674.6.
  - B1 960 INT8: 683.9, 744.0, 738.9, 742.5, 735.3, 739.6, 753.6, 679.4, 672.6, 676.8.
  - A2 640 FP32: 614.0, 668.3, 670.9, 671.5, 669.9, 669.9, 663.1, 668.2, 672.1, 668.9.
  - A1 640 INT8: 308.2, 304.7, 336.2, 344.6, 338.5, 339.1, 337.7, 336.8, 341.3, 339.2.
  - A2 640 INT8: 311.4, 310.1, 354.4, 345.5, 350.4, 350.4, 354.3, 348.2, 345.8, 358.9.
- So the reference's 200 images, about 2.7 minutes, sit in the phase where the phone settles. That phase depends on how
  the phone ran before the gate, and the drift check, as #8 defines it, measures it. A full cell runs for 11–22 minutes and
  spends most of them at the settled level. The phone shows no CPU temperature or clock, so this is not confirmed (#33,
  #34).
- The 25 October sessions ran for 25–45 minutes and drifted +2.8 % and +2.95 %. This one ran for 1 h 52 min.

### The slow stretches (#34)

Blocks of 10 images, slow when their median net time is at least 1.3× the run's median net time.

| Run | Blocks | Slow blocks |
|---|---|---|
| reference run, first | 20 | 1: images 51–60 (1,103 ms against 635 ms) |
| A1 640 FP32 | 161 | 6: images 831–840, 1111–1120, 1181–1190, 1211–1220, 1241–1250, 1601–1610 (873–1,069 ms against 663 ms) |
| B1 960 INT8 | 161 | 1: images 131–140 (977 ms against 710 ms) |
| A2 640 FP32 | 161 | 2: images 431–440, 681–690 (948 and 863 ms against 655 ms) |
| A1 640 INT8 | 161 | 2: images 71–80, 1291–1300 (595 and 505 ms against 322 ms) |
| A2 640 INT8 | 161 | 9, spread through the run (433–643 ms against 333 ms) |
| reference run, last | 20 | 0 |

### Events

- **An aborted first start, at 10:53 IST.** The first run of this plan started while Termux was off screen. Its cpuset
  was `/moderate`, CPUs 0–2, and the battery read 39.2 °C after fast charging. The reference run's first images ran at
  1.0–2.5 s until Termux came back to `/top-app` at about 10:58. The user stopped that runner and deleted its folder on the
  phone. Its cpuset log is kept as `bench/m34/cpuset-m34-matrix-1-aborted-1053.log`. The session above is a clean
  restart at 11:06:45, after the battery cooled to 34.9 °C.
- The wizard's progress view polled the phone over SSH every 30 s for the whole session, as on 25 October. The agent
  checked the cpuset log over SSH a few times in the first minutes.
- The wizard copied the session home and scored the five cells, then printed the drift warning. The user stopped the
  wizard with Ctrl-C at that point. Nothing was lost: only the closing summary was skipped.

## The decision

Recorded on 2026-11-01 at about 13:20 IST. The user chose to **keep the session, not re-run it**, and to run session 2
next on the same terms. The reasons:

- A re-run is 2 hours. A 2-hour phone session would likely fail ±3 % again, as this one did. The 25 October sessions
  were 25–45 minutes long and only just passed.
- The accuracy does not depend on the drift.
- No conclusion drawn from the phone moves with a 5 % shift. The pair is 8.6 % apart, and drift can only narrow that
  gap. INT8 is about 2× faster than FP32. The YOLO11s and YOLO26s latencies are reported as not separable.

**How the cells are used:** they are reported with the failed drift check (`cells.csv` notes it), and the phone's
latency stays provisional (#34). A phone latency difference smaller than this session's drift (4.9 %) is not reported
as a difference.
