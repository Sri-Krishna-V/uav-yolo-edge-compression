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

(Added after the run.)
