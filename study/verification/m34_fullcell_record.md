# Galaxy M34 full cell: session record

The dev started the session at about 17:55 IST on 2026-10-25. This record was written at about 17:57 IST, after the
start and before any result was seen: the agent did not contact the phone until the session was expected to end. The
results are added after the run, and nothing above them is changed.

## Question

- How long does a full cell take on the Galaxy M34?
- What are the baseline's latency and accuracy there: YOLO11s, checkpoint A1 (`exports/yolo11s-640-s0`), ONNX Runtime
  FP32, at the matrix resolution (640), on all 1,610 test-dev images?
- Do the slow stretches of the speed tests (#34) come again in a run this long?

## Plan and code

- Plan: `plans/m34-fullcell.json`, sha256 `d71c1a0e2fbf2aa4d41203602a799920e6c4a018e766ab251234f35ddce470c6`. One cell:
  `onnxruntime-fp32`, all 1,610 test-dev images. The reference runs, first and last, are as in the speed tests.
- The rules of #8, the code and the wizard are the same as in [m34_speedtest_record.md](m34_speedtest_record.md). The
  sha256 values of the scripts are unchanged.
- The cpuset log was armed at about 17:00 IST: `bench/m34/cpuset-m34-fullcell.log`, a sample every 10 s.
- The conditions are those confirmed for the speed tests. The phone and the laptop are on the extension box.

## Analysis plan (fixed before any result was seen)

- **Latency:** the cell's median in `summary.csv` (`e2e_deploy`, at the deployment threshold 0.25).
- **Time of a full cell:** from the cell's "attempt 1 starts" to its "attempt 1 complete" in `session.log`; and the
  whole session, from its start to "session complete".
- **Accuracy:** AP, AP50 and AP-small on all 1,610 images, scored on the laptop. No laptop score of the same export is
  on file to compare it with, so it is reported alone.
- **Valid** under the same three conditions as the speed tests: drift within ±3 %, every cpuset sample `/top-app`, and
  no failed cell.
- **The slow stretches (#34):** split the cell into blocks of 10 images. A block is slow when its median net time is at
  least 1.3× the cell's median net time. Report how many blocks are slow and where they are in the run. The reference
  runs get the same analysis.
- **The latency is provisional** until #34 is answered, whatever the session gives.

## Results

**The session is valid by the analysis plan.** The latency stays provisional until #34 is answered.

| Run (in order) | Run starts (gate opens), IST | Battery at its gate | Images | Median latency, ms |
|---|---|---|---|---|
| reference run, first | 17:58:39 | 31.7 °C | 200 | 613.7 |
| FP32 full cell | 18:04:15 | 31.7 °C | 1,610 | 636.6 |
| reference run, last | 18:26:42 | 35.8 °C | 200 | 631.1 |

- **Drift:** 613.7 → 631.1 ms, +2.8 %: within ±3 %.
- **Cpuset:** all 199 samples, from 17:55:37 to 18:29:18, read `/top-app` with CPUs 0–7.
- **No cell failed.** Each run took one attempt.
- **Latency of the baseline on the M34:** 636.6 ms median (`e2e_deploy`); p10 610.1 ms, p90 699.3 ms. At the
  evaluation threshold, pre + net + post is 664.9 ms (post 32.5 ms against 2.3 ms at the deployment threshold).
- **Accuracy on all 1,610 test-dev images:** AP 0.1875, AP50 0.3265, AP-small 0.0904 (AP-medium 0.2837, AP-large
  0.4203, AR500 0.3563).
- **Time of a full cell:** 19 min 26 s, from "attempt 1 starts" to "attempt 1 complete", with the 20 warm-up images.
  The whole session took 33 min 42 s (17:55:37 to 18:29:19).

### The slow stretches (#34)

| Run | Blocks of 10 images | Slow blocks (median net ≥ 1.3× the run's) |
|---|---|---|
| reference run, first | 20 | 1: images 11–20 (962 ms against 603 ms) |
| FP32 full cell | 161 | 1: images 811–820 (932 ms against 625 ms) |
| reference run, last | 20 | 0 |

The long stretches of the speed tests (20–70 images at 1.4–2.5×) did not come back. Each slow block here is a single
block of 10 images.

### Not in the analysis plan: a slow rise through the cell

The cell's latency rises slowly from its start to its end. The medians of 10 equal parts (about 161 images each) are 618,
623, 631, 622, 623, 648, 639, 645, 648 and 651 ms: +5 % from the first part to the last. The battery warmed from 31.7 °C
to 35.8 °C between the cell's gate and the last reference run's gate. Heat is a likely cause, but the phone shows no
CPU temperature or throttle signal, so this is not confirmed. A full cell's median includes this rise. A 200-image cell's
median includes less of it.
