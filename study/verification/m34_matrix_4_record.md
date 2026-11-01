# Galaxy M34 matrix, session 4: session record

Written on 2026-11-02 at about 00:30 IST, after the session ended. No record was written before it ran: the rules
are #8's and the plan's, and the analysis below is session 1's ([m34_matrix_1_record.md](m34_matrix_1_record.md)),
run on this session's files with the same code.

## Question

- What are the latency and accuracy, on the Galaxy M34 with ONNX Runtime, of: B3 in per-channel INT8 at 1280, B1 and
  B2 in per-channel INT8 at 800, and B1 in FP32 at 960 again, on all 1,610 test-dev images?
- Is B3 in INT8 at 1280 against B1 in FP32 at 960 an equal-latency pair on the phone (#8: medians within 10 % of the
  smaller), measured back to back in one session? Session 3 measured B1 in FP32 at 960 at 1,397.6 ms, and B3 in INT8
  at 1280 scaled to about 1,290 ms.

## Plan and code

- Plan: `plans/m34-matrix-4.json` (committed in 221321b, which split the earlier 9-cell plan of 71c7970 into sessions 4
  and 5), `plan_sha256` in `session.json`. Four cells, seed 5. The order, shuffled once: B1 960 FP32, B3 1280 INT8,
  B2 800 INT8, B1 800 INT8. The pair runs first, back to back, the FP32 cell first.
- The reference runs, first and last: YOLO11s, ONNX Runtime FP32, 640, the first 200 test-dev images (#8).
- The code is unchanged since 31 October (`studybench.py` and `studysession.py`, last changed in 16221bb).
- The software on the phone is sessions 1–3's: Python 3.14.8, numpy 2.5.4, OpenCV 5.0.0, ONNX Runtime 1.31.1.
- The cpuset log: `bench/m34/cpuset-m34-matrix-4.log`, a sample every 10 s, copied home after the session.
- B1 and B2 run at 800 from their 960 checkpoints, at an input resolution they were not trained at, as on the Pi 5. B3
  runs at 1280 from its 1280 checkpoint.
- Run from the laptop with the wizard (`ENV_FILE=pi/.env.m34 bash pi/wizard.sh plans/m34-matrix-4.json`), the phone
  plugged in, Termux kept on screen. It started about 50 minutes after session 3 ended.

## Results

**The drift check failed: +8.4 %, past the ±3 % limit, so the runner flagged every cell for a re-run.** The cpuset
held `/top-app` for the whole session and no cell failed. The user had decided to keep the phone's sessions on session
1's terms if they failed the drift check (see [The decision](#the-decision)). The phone's latency stays provisional
(#34).

| Run (in order) | Run starts (gate opens), IST | Battery at its gate | Images | Median latency, ms |
|---|---|---|---|---|
| reference run, first | 21:58:06 | 31.9 °C | 200 | 594.2 |
| B1, YOLO11s 960, FP32 | 22:03:33 | 31.9 °C | 1,610 | 1,440.5 |
| B3, YOLO11s 1280, INT8 per-channel | 22:48:48 | 34.5 °C | 1,610 | 1,296.1 |
| B2, YOLO26s 800, INT8 per-channel | 23:30:05 | 33.6 °C | 1,610 | 531.2 |
| B1, YOLO11s 800, INT8 per-channel | 23:49:40 | 33.3 °C | 1,610 | 514.1 |
| reference run, last | 00:08:43 | 33.4 °C | 200 | 644.2 |

- **Drift:** 594.2 → 644.2 ms, +8.4 %: outside ±3 %.
- **Cpuset:** all 803 samples, from 21:55:05 to 00:11:14, read `/top-app` with CPUs 0–7, with no gap over 15 s.
- **No cell failed.** Each run took one attempt. The session ran from 21:55:05 to 00:11:22 IST (2 h 16 min).
- **Battery:** 31.9–34.5 °C at the gates. It read 100 %, full, plugged in (AC), 33.2 °C, at about 00:15.

### Latency and accuracy

All four cells use the one-to-many head. Latency is the median `e2e_deploy`, at the deployment threshold 0.25.

| Cell | Median, ms | p10, ms | p90, ms | Time of the cell | AP | AP50 | AP-small | AP-medium | AP-large | AR500 |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 960 FP32 | 1,440.5 | 1,401.6 | 1,580.1 | 42 min 13 s | 0.2392 | 0.4067 | 0.1365 | 0.3496 | 0.4635 | 0.4211 |
| B3 1280 INT8 | 1,296.1 | 1,281.1 | 1,446.6 | 38 min 16 s | 0.2593 | 0.4404 | 0.1579 | 0.3695 | 0.4835 | 0.4525 |
| B2 800 INT8 | 531.2 | 488.5 | 621.0 | 16 min 33 s | 0.2074 | 0.3630 | 0.1083 | 0.3119 | 0.4656 | 0.3888 |
| B1 800 INT8 | 514.1 | 463.8 | 589.0 | 16 min 01 s | 0.2053 | 0.3607 | 0.1049 | 0.3088 | 0.4793 | 0.3790 |

- **The accuracy matches the Pi 5's.** The three INT8 cells give exactly the AP, AP50, AP75, AP-small, AP-medium,
  AP-large and AR500 of the Pi 5's session 7 cells. B1 at 960 in FP32 differs from the Pi 5's session 1 cell by less
  than 3 × 10⁻⁶, and equals session 3's phone cell.
- **B3 at 1280 does not form an equal-latency pair with B1 at 960:** 1,296.1 against 1,440.5 ms, **11.1 % apart**, just
  outside 10 %. B3 ran second and is the faster: if the phone slowed through the session, B3 is measured slow and the
  true gap is wider. Against session 3's B1 at 960 (1,397.6 ms) the gap would be 7.8 %, but that crosses sessions,
  which #8's pair rule does not allow (as for B2 against A2, session 2). B3 has +0.0214 AP-small and +0.0201 AP over
  B1, at 11.1 % less latency.
- **B1 at 960 in FP32, again:** 1,440.5 ms against session 3's 1,397.6 ms, +3.1 %, inside either session's drift
  (5.9 %, 8.4 %). By the latest-run rule, `cells.csv` and the tables now use this session's cell.
- **INT8 at 800 against FP32 at 800, across sessions:** B1 1,043.7 (session 3) / 514.1 ms = 2.03×; B2 1,007.3 (session 3)
  / 531.2 ms = 1.90×. The speed-up is far larger than either session's drift.
- **INT8 at 1280 against INT8 at 1120, across sessions:** B3 1,296.1 against 987.0 ms (session 3), 1.31×: the pixel
  ratio, (1280 / 1120)² = 1.31.
- **YOLO11s against YOLO26s, INT8 at 800, within the session:** 514.1 against 531.2 ms (YOLO26s +3.3 %), inside this
  session's drift: not separable.

### Where the drift is

- Over the reference's first 50 images: 584.9 → 653.1 ms. Over images 51–200: 603.7 → 643.9 ms, **+6.7 %**. Unlike in
  sessions 1–3, most of this session's drift is not in the first 50 images.
- The first reference run's four blocks of 50 images had median net times of 573, 571, 587 and 614 ms. The last
  reference run's: 640, 646, 639 and 619 ms. The first reference run (594.2 ms) is the fastest of the phone's four
  sessions (648.3, 630.5, 609.3, 594.2 ms). This session started about 50 minutes after session 3 ended, with the
  phone idle in between.
- The medians of 10 equal parts of each cell (about 161 images each), in ms:
  - B1 960 FP32: 1,373.5, 1,451.0, 1,432.7, 1,435.2, 1,492.6, 1,471.8, 1,438.0, 1,436.4, 1,435.0, 1,437.8.
  - B3 1280 INT8: 1,296.4, 1,292.5, 1,289.9, 1,299.1, 1,294.5, 1,296.3, 1,296.6, 1,299.4, 1,298.7, 1,303.1.
  - B2 800 INT8: 473.9, 524.7, 534.8, 535.8, 529.7, 531.5, 529.0, 540.5, 531.9, 531.3.
  - B1 800 INT8: 466.7, 465.2, 510.6, 513.5, 517.3, 514.3, 525.1, 536.2, 518.2, 515.8.
- So B1 at 960 and both cells at 800 start fast and step up, by about 5 % (FP32) and 10–12 % (INT8), after 1.5–4
  minutes, as in session 2. B3 at 1280 holds one level from start to end. The first reference run sits in the fast
  phase again. The phone shows no CPU temperature or clock, so this is not explained (#33, #34).

### The slow stretches (#34)

Blocks of 10 images, slow when their median net time is at least 1.3× the run's median net time.

| Run | Blocks | Slow blocks |
|---|---|---|
| reference run, first | 20 | 0 |
| B1 960 FP32 | 161 | 0 |
| B3 1280 INT8 | 161 | 2: images 211–220, 1341–1350 (1,656 and 1,935 ms against 1,257 ms) |
| B2 800 INT8 | 161 | 2: images 261–270, 521–530 (741 and 891 ms against 513 ms) |
| B1 800 INT8 | 161 | 2, a pair: images 1071–1090 (724 and 693 ms against 496 ms) |
| reference run, last | 20 | 0 |

- All 803 cpuset samples read `/top-app`, so, as in sessions 1 and 2, the cpuset does not explain these.

## Events

- None. The session started once, at 21:55:05, with Termux on screen; the cpuset logger was waiting for the runner and
  caught it from its first sample. The laptop kept its SSH connection to the phone for the whole session.
- The wizard copied the session home and scored the four cells.

## The decision

Recorded in session 1's record, at about 13:20 IST on 2026-11-01: the user chose to keep the phone's sessions on
session 1's terms if they failed the drift check. The reasons are session 1's:

- A re-run would likely fail ±3 % again: every phone session since the reinstall has, at +4.9, +3.4, +5.9 and +8.4 %.
- The accuracy does not depend on the drift.
- No conclusion drawn from this session moves with an 8.4 % shift. The one pair candidate is outside 10 % as measured,
  and drift can only widen its gap. INT8 is about 2× faster than FP32 at 800. The YOLO11s and YOLO26s latencies are
  reported as not separable.

**How the cells are used:** they are reported with the failed drift check (`cells.csv` notes it), and the phone's
latency stays provisional (#34). A phone latency difference smaller than this session's drift (8.4 %) is not reported
as a difference.
