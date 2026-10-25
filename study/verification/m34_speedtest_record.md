# Galaxy M34 speed test: session record

Written before the run, on 2026-10-25 at about 15:50 IST. Everything above "Results" was fixed before the session
started. The results are added after the run, and nothing above them is changed.

## Question

On the Galaxy M34 (platform), with ONNX Runtime (runtime) under Termux, is the quantized configuration (INT8) faster
than the baseline (FP32)? Model: YOLO11s, checkpoint A1 (`exports/yolo11s-640-s0`), at the matrix resolution, 640.

## Plan and code

- Plan: `plans/m34-speedtest.json`, sha256 `c694c08726067cf33b3261bea67c75176f32e6d2844a764805b62f928f7632f7`.
  - Two cells: `onnxruntime-fp32` and `onnxruntime-int8`, on the first 200 test-dev images each.
  - The reference runs, first and last: the reference baseline, ONNX Runtime FP32, 640, on the first 200 test-dev images.
- The rules of #8 (PLATFORMS.md, "The rules it applies"): 4 threads, 20 untimed warm-up images, a fixed 3-minute
  start wait on the phone, cell order shuffled with seed 0, the drift check at ±3 %, one re-run of a crashed cell.
- Code at git `b3f9a9a`, with these files as run:
  - `scripts/studybench.py` sha256 `4b2a19a13404637dc6e0921dc5f5264d69b0c4405fac0553bece68b0209ea7f6`
  - `scripts/studysession.py` sha256 `2096cf64ab2b68e94b48301ab41f2299db180ddd73aea402ecbff5968edbdea3`
  - `pi/wizard.sh` sha256 `6c6c12e10094e716ed3676896d1c986521473c6837f5779fb04f57cdaedfc577`
- Driven by the Pi 5 wizard (`pi/wizard.sh`, #25), from the laptop over SSH. The phone variant of #27 is not built.

## Analysis plan (fixed before the run)

- **Metric:** per cell, the median latency in `summary.csv` (`latency_ms`, with `latency_source` = `e2e_deploy`):
  pre + net + post at the deployment threshold, 0.25, batch 1.
- **Result:** the ratio INT8 median / FP32 median.
- **Accuracy:** a 200-image cell is scored only on those images. It is a plumbing check and is not reported.
- **The session is valid only if all of these hold:**
  - the drift check passes (the reference runs, first and last, within 3 %);
  - every sample of the cpuset log during the session reads `/top-app` (see "The background test");
  - no cell failed.
- **If it is not valid,** its numbers are not used. The cells run again in a new session, under a new plan name
  (`plans/m34-speedtest-2.json`).
- **Expectation, not a prediction:** the research for #27 scaled about 600–1,000 ms per image for FP32 and 350–750 ms
  for INT8 from Pi 5 numbers. No published latency exists for this SoC.

## Platform

Read over SSH at 15:36 IST ([m34_facts.txt](m34_facts.txt)):

- SM-M346B, Android 16 (SDK 36), One UI 8.5, build `BP4A.251205.006.M346BXXSEFZH9`.
- Exynos 1280 (`s5e8825`):
  - cpu0–5 are Cortex-A55 (`0xd05`), 2.0 GHz, cpufreq policy0;
  - cpu6–7 are Cortex-A78 (`0xd41`), 2.4 GHz, policy6;
  - governor `energy_aware`; `asimddp` present; MemTotal 5,517,824 kB.
- Termux 0.118.3 (F-Droid), Python 3.14.8, numpy 2.5.4, OpenCV 4.14.0, ONNX Runtime 1.31.0. The bench asks for
  `CPUExecutionProvider` only (the build also offers NNAPI and XNNPACK).
- Install: `libavif` and `dbus` were installed by hand. Termux's opencv 4.14.0-4 and qt6-qtbase 6.12.0 need them but do not
  list them.
- Thermal zones: not readable (`Permission denied`).
- Battery at 15:36: 67 %, AC charger, charging, 33.2 °C.

## Conditions, confirmed by the dev before the run (15:40 IST)

- Developer options: Stay awake on; Disable child process restrictions on.
- Termux and Termux:API: battery Unrestricted, in Never sleeping apps; Put unused apps to sleep off.
- Power saving and Adaptive power saving off. Fast charging off. Auto optimization's Restart when needed off.
  Performance profile Standard.
- Do Not Disturb on. Brightness low and fixed. Other apps closed. On the wall charger.
- Room: 28 °C (the dev's reading).
- Network: the laptop's Windows mobile hotspot (5 GHz); the phone at 192.168.137.218.

The runner reads none of these on the phone (#27, #33). They are the dev's answers.

## The background test (#33), 15:41–15:46 IST

**Method.** A shell on the phone recorded its own cpuset and `Cpus_allowed_list` once a second. The dev kept Termux on
the screen, pushed Home for about 30 s, and opened Termux again. Log: [m34_bgtest-2026-10-25.log](m34_bgtest-2026-10-25.log).

**Result.**

| From | cpuset | CPUs allowed |
|---|---|---|
| 15:41:35 | `/top-app` | 0–7 |
| 15:43:56 (Home pushed) | `/foreground-boost` | 0–7 |
| 15:44:11 (15 s later) | `/moderate` | **0–2** |
| 15:44:31 (Termux on the screen again) | `/top-app` | 0–7 |

On this phone, 15 s after Termux leaves the screen, every Termux process can run only on cpu0–2: three A55 cores, and no
A78. The bench's 4 threads would then share 3 slow cores. With Termux on the screen again, `/top-app` is back within
about 1 s.

**Consequence.** A run is valid only while Termux stays on the screen. This is the hazard of termux-app #5035 and #5086,
now confirmed on the M34, and it is worse here (3 cores instead of 4).

## Instruments outside the protocol

- **The cpuset log** (`~/cpuset_logger.sh` on the phone): every 10 s, the runner's cpuset and allowed CPUs, to
  `bench/m34/cpuset-m34-speedtest.log`. Each sample costs one `cat` and one `awk`.
- **The wizard's progress view:** one SSH login every 30 s during the session, the same as on a Pi.

Both run while images are timed. Their cost is small and not measured.

## Known limitations

- The phone's conditions are the dev's answers, not readings by the runner (#27, #33).
- The runner does not record the room temperature on the phone; it is only in this record.
- 4 threads on 2 A78 + 6 A55 cores: the cross-platform rule of #8 is kept, and where the threads run is not controlled.
  Only the cpuset is logged.
- The bench's `cpu_mhz` column reads cpu0, which is an A55, so it shows the slow cluster's clock (#33).
- The phone gives no throttle signal: the runner records the throttle state as "unknown".
- The battery charges during the session and gets warmer. The runner logs its temperature before each run.

## Results

**The session is not valid.** The drift check failed, so by the analysis plan its numbers are not used, and the cells
run again in `plans/m34-speedtest-2.json`.

| Run (in order) | Run starts (gate opens), IST | Battery at its gate | Median latency, ms |
|---|---|---|---|
| reference run, first | 15:53:38 | 34.4 °C | 703.7 |
| FP32 cell | 15:59:53 | 34.6 °C | 684.6 |
| INT8 cell | 16:05:46 | 36.0 °C | 298.6 |
| reference run, last | 16:10:10 | 36.2 °C | 638.9 |

- **Drift:** 703.7 → 638.9 ms, −9.2 %. The limit is ±3 %. Every cell of the session is flagged for a re-run.
- **Cpuset:** all 131 samples, from 15:50:36 to 16:12:40, read `/top-app` with CPUs 0–7. Termux kept the fast cores.
- **No cell failed.** The session completed at 16:12:45.
- **Where the drift comes from.** Medians of the four quarters (50 images each) of each run, in ms:

  | Run | Q1 | Q2 | Q3 | Q4 | p10 | p90 |
  |---|---|---|---|---|---|---|
  | reference run, first | 672.6 | 654.2 | 707.3 | 739.1 | 630.8 | 980.0 |
  | FP32 cell | 683.2 | 674.7 | 685.5 | 691.0 | 640.6 | 768.0 |
  | INT8 cell | 269.3 | 300.5 | 300.4 | 310.8 | 267.1 | 366.5 |
  | reference run, last | 616.2 | 639.5 | 657.8 | 639.2 | 607.1 | 689.1 |

  The first reference run slowed down in its second half and has a long tail (p90 980 ms, against 689–768 ms in the
  other FP32 runs). The cpuset log rules out the move off the fast cores. The data cannot show the cause: the session
  does not record where each thread ran (#33).
- **Not reported:** by the analysis plan, the INT8/FP32 ratio of this session is not a result. The re-run gives it.

### Events during the run

- The session started at 15:50:35 IST. The session log says "room not given": the wizard asks for the room's
  temperature only on a Pi. The dev's reading (28 °C) is above, under "Conditions".
- At about 16:06:15 IST, during the INT8 cell's timed images (started 16:05:46), the agent made one SSH status check on
  the phone: `ls`, `tail`, `termux-battery-status` (an Android broadcast to Termux:API) and `cat`. That is extra load
  for about 1 s, during a cell of about 3 minutes. No other SSH use was made during the session, except the
  wizard's progress view.

### Notes

- The sha256 values above are of the files with LF line endings, as git stores them. A Windows checkout with CRLF
  endings gives other values. The runner's own `plan_sha256` hashes the parsed plan, so it does not depend on line endings.
- The plumbing check (the 200 images only, not reported): FP32 AP 0.1981, AP-small 0.1143; INT8 AP 0.1805, AP-small
  0.0996.

## The re-run: `plans/m34-speedtest-2.json`

The analysis plan above applies unchanged. The plan has the same two cells; only its note differs. sha256
`b9948d49020a3ade7efb64c3611dfb94d34a2da7244e2d824b735db011d10739`. The scripts have the same sha256 as above.

### Changes from the first session

- Between the sessions, the dev moved the phone's charger and the laptop's charger to an extension box. Then the phone
  was off the hotspot until the dev connected it again (Auto reconnect to the college Wi-Fi is now off).
- The session started at 16:31:15 IST, about 2 minutes after the phone was back on the hotspot. The 5-minute settling
  wait that the agent suggested was not kept.
- Battery: 78 % at 16:20, 89 % at the end, charging all the time.
- The agent did not contact the phone during the session. Its watcher first connected at 16:54:10, after the session
  completed (16:53:16). The dev's own use of the phone during the session is not known.

### Results of the re-run

**The session is valid by the analysis plan.**

| Run (in order) | Run starts (gate opens), IST | Battery at its gate | Median latency, ms |
|---|---|---|---|
| reference run, first | 16:34:17 | 34.2 °C | 658.3 |
| FP32 cell | 16:40:39 | 34.8 °C | 627.5 |
| INT8 cell | 16:46:14 | 35.0 °C | 271.6 |
| reference run, last | 16:50:30 | 34.9 °C | 643.6 |

- **Drift:** 658.3 → 643.6 ms, −2.2 %: within ±3 %.
- **Cpuset:** all 130 samples, from 16:31:19 to 16:53:11, read `/top-app` with CPUs 0–7.
- **No cell failed.** Each run took one attempt.
- **Result: INT8 / FP32 = 271.6 / 627.5 = 0.433.** On the Galaxy M34, with ONNX Runtime 1.31 and 4 threads, the
  quantized configuration of A1 at 640 is 2.31× faster than the baseline. With the two reference runs as the FP32
  number instead of the cell, it is 2.37× to 2.42×.
- The invalid first session gave 298.6 / 684.6 = 0.436. That number is not used. It only shows that the direction did
  not change.
- The plumbing check gives the same AP as in the first session, to 4 digits, for both cells.

### Slow episodes in the first reference runs

Both sessions' first reference runs have a stretch of images where the latency is far higher than in the rest of the
run. Medians of 10 images, net (the runtime call) in ms:

| Images | First session | Re-run |
|---|---|---|
| 1–130 | 621–932 (the first 10 at 932, then 621–720) | 587–710 |
| 131–150 | 690–706 | **925–1,000** |
| 151–170 | **869–1,275** | **1,265–1,393** |
| 171–200 | 689–710 | **912–1,600** |

- Pre-processing also gets slower in these stretches (8–9 ms to 10–13 ms). The whole process slows down, not only the
  runtime call.
- The clock of cpu0 (an A55) does not change.
- The cpuset is `/top-app` all the time, so the move off the fast cores does not explain it.
- In the re-run, the episode lasts about 70 images, approximately 16:36:20 to 16:37:37 IST.
- The cause is not known. The session records no per-image clock time, no thread placement and no other processes' load.

This reaches the median: the FP32 cell (627.5 ms) is 4.7 % faster than the first reference run (658.3 ms), with the same
model on the same images.

### Decision: no further sessions without the dev

Before this re-run ended, the agent fixed four conditions for running the next sessions on its own (the full cell, then
an INT8 full cell): the drift check within ±3 %; the FP32 cell within 3 % of both reference runs; every cpuset sample
`/top-app`; no failed cell. The second condition failed (−4.7 % against the first reference run), and the slow episodes
are not explained. So the agent ran no further session. The full cell (1,610 images, about 20 minutes of timed images)
waits for the dev, and the slow episodes are #34.
