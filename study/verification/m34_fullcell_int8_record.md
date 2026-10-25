# Galaxy M34 INT8 full cell: session record

Written on 2026-10-25 at about 18:45 IST, before the session started. The results are added after the run, and nothing
above them is changed.

## Question

- What are the quantized configuration's latency and accuracy on the Galaxy M34: YOLO11s, checkpoint A1
  (`exports/yolo11s-640-s0`), ONNX Runtime INT8, at the matrix resolution (640), on all 1,610 test-dev images?
- How do they compare with the baseline's full cell ([m34_fullcell_record.md](m34_fullcell_record.md))?

## Plan and code

- Plan: `plans/m34-fullcell-int8.json`, sha256 `b6978abb48562147b1168cfdd874e63d4928ab6abc00ce59efc8ce49a0d71c56`. One
  cell: `onnxruntime-int8`, all 1,610 test-dev images. The reference runs, first and last, are as before (FP32, the
  first 200 images).
- The rules of #8, the code (same sha256 values) and the wizard are the same as in
  [m34_speedtest_record.md](m34_speedtest_record.md).
- The cpuset log: `bench/m34/cpuset-m34-fullcell-int8.log`, a sample every 10 s.
- The conditions are those confirmed for the speed tests. The phone and the laptop are on the extension box.

## Analysis plan (fixed before the run)

- **Latency:** the cell's median in `summary.csv` (`e2e_deploy`, at the deployment threshold 0.25).
- **Time of the cell:** from its "attempt 1 starts" to its "attempt 1 complete" in `session.log`.
- **Accuracy:** AP, AP50 and AP-small on all 1,610 images, scored on the laptop.
- **Valid** under the same three conditions: drift within ±3 %, every cpuset sample `/top-app`, no failed cell.
- **Comparison with the baseline**, reported in two ways, because the two full cells ran in different sessions:
  - across sessions: INT8 cell median / FP32 cell median (636.6 ms);
  - within each session: the cell's median / the mean of its own session's two reference runs. For the FP32 full
    cell that was 636.6 / 622.4 = 1.023.
  - The speed test's within-session ratio (0.433, [m34_speedtest_record.md](m34_speedtest_record.md)) remains the
    controlled number. These ratios check it at full length.
- **Accuracy change:** ΔAP and ΔAP-small, INT8 minus FP32, on the same 1,610 images.
- **The slow stretches (#34):** blocks of 10 images, slow when their median net time is at least 1.3× the run's median.
  Report the count and their positions. Report the rise through the cell as the medians of 10 equal parts, as for the
  FP32 full cell.
- **The latency is provisional** until #34 is answered.

## Results

**The session is valid by the analysis plan.** The drift is +2.95 %, just inside the ±3 % limit. The latency stays
provisional until #34 is answered.

| Run (in order) | Run starts (gate opens), IST | Battery at its gate | Images | Median latency, ms |
|---|---|---|---|---|
| reference run, first | 18:46:39 | 35.0 °C | 200 | 618.1 |
| INT8 full cell | 18:52:14 | 34.3 °C | 1,610 | 304.8 |
| reference run, last | 19:05:50 | 35.6 °C | 200 | 636.3 |

- **Drift:** 618.1 → 636.3 ms, +2.95 %: within ±3 %.
- **Cpuset:** all 146 samples, from 18:43:38 to 19:08:19, read `/top-app` with CPUs 0–7.
- **No cell failed.** Each run took one attempt.
- **Latency of the quantized configuration on the M34:** 304.8 ms median (`e2e_deploy`); p10 283.6 ms, p90 372.6 ms.
  At the evaluation threshold, pre + net + post is 337.0 ms (post 35.0 ms).
- **Time of the cell:** 10 min 35 s, with the 20 warm-up images. The whole session took 24 min 49 s (18:43:38 to
  19:08:27).
- **Battery:** at 100 % and "FULL" (not charging) for the whole session. The FP32 full cell ran while charging, from
  31.7 °C. This session started at 35.0 °C.

### Comparison with the FP32 full cell

| | FP32 full cell | INT8 full cell | INT8 − FP32 | Relative |
|---|---|---|---|---|
| AP | 0.1875 | 0.1682 | −0.0193 | −10.3 % |
| AP50 | 0.3265 | 0.3056 | −0.0209 | −6.4 % |
| AP-small | 0.0904 | 0.0769 | −0.0135 | −14.9 % |
| AP-medium | 0.2837 | 0.2557 | −0.0280 | −9.9 % |
| AP-large | 0.4203 | 0.4013 | −0.0190 | −4.5 % |
| AR500 | 0.3563 | 0.3330 | −0.0233 | −6.5 % |

- **Latency, across sessions:** 304.8 / 636.6 = **0.479** (INT8 2.09× faster).
- **Latency, each cell against its own session's reference runs:** INT8 304.8 / 627.2 = 0.486; FP32 636.6 / 622.4 =
  1.023. Their ratio is **0.475** (2.10×).
- **The speed test gave 0.433 (2.31×)**, within one session, on 200 images. The full cells give a smaller speed-up. The
  next section shows why.
- AP-small loses the most in relative terms (−14.9 %). This is one checkpoint with no confidence interval, so it is
  not a test of the small-object effect (#9).

### The rise through the cell, and the slow stretches (#34)

The medians of 10 equal parts of the INT8 cell (about 161 images each), in ms: 267.3, 291.7, 302.6, 302.7, 304.1,
300.5, 307.2, 306.7, 313.0, 315.5. That is **+18 %** from the first part to the last. The FP32 full cell rose +5 %.

- The INT8 cell's first part (267.3 ms) is within 2 % of the speed test's INT8 median (271.6 ms, 200 images). A
  200-image INT8 run measures the phone before this rise. A full cell measures it after.
- So on this phone, the INT8 speed-up depends on how long the phone has run: 2.31× over 200 images, 2.09× over 1,610.
  Heat is a likely cause: INT8 keeps the fast cores busier per second than FP32 does. The phone shows no CPU
  temperature or clock of the fast cores, so this is not confirmed (#33, #34).

| Run | Blocks of 10 images | Slow blocks (median net ≥ 1.3× the run's) |
|---|---|---|
| reference run, first | 20 | 0 |
| INT8 full cell | 161 | 2: images 621–630 (837 ms against 293 ms, 2.9×) and 631–640 (385 ms) |
| reference run, last | 20 | 0 |

One short slow stretch, about 20 images long, in the middle of the cell.

### Events during the run

- The agent's watcher began to check the session at about 19:05, during the last reference run (19:05:50 to 19:08:27),
  because the session ran about 3 minutes longer than estimated. It made about 6 SSH logins, one every 30 s: each runs
  `cat` and `kill -0`. This is the same kind of load as the wizard's progress view. The drift check passed.
