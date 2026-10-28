# Raspberry Pi 5 fan pilot: session record

Written on 2026-10-28, after the runs. Vidhyarth ran the sessions with `pi/wizard.sh` and sent the results. No
analysis plan was written before the run: the pass rule is #29's, as `pi/README.md` (part 3) gave it to the dev.

## Question

With the fan on the Pi 5's FAN socket (#29), does a full cell run without throttling? If it throttles, at which of
2000, 1800 and 1500 MHz does it hold?

## Plan and code

- Plan: `plans/pi5-fullcell.json` (the runner's `plan_sha256` `e954723d…0b65`). One cell: YOLO11s, checkpoint A1
  (`exports/yolo11s-640-s0`), ONNX Runtime FP32 at 640, on all 1,610 test-dev images, plus the drift check.
- The re-run at the 2000 MHz lock: `plans/pi5-fullcell-2000.json`, a copy of the same plan (same `plan_sha256`).
- The rules of #8: 4 threads, 20 untimed warm-up images, the 50 °C start gate with its 3-minute timeout, the drift
  check at ±3 %, one re-run of a throttled or crashed cell.
- `scripts/studybench.py` sha256 `4b2a19a1…0ea7f6` and `scripts/studysession.py` sha256 `2096cf64…edbdea3`, the
  same files as the M34 sessions. The dev's `pi/wizard.sh` had one local fix, pushed later in `7d8d9c8`: the Pi's `awk`
  printed the SD card's free bytes as `2.6e+10`, which bash's arithmetic in stage 3 cannot read.

**Pass (#29):** no throttle bits for the whole run, the clock at 2400 MHz (or at the locked value) throughout, and a
peak below about 80 °C. `pi/README.md` reduced this to the session's own verdict: `summary.csv` says `throttled` or not.

## Platform

From [pi5_facts.txt](pi5_facts.txt), read by `pi/provision.sh` on 2026-10-26 at 22:25 IST, and from the session logs:

- Raspberry Pi 5 Model B Rev 1.0, 4 GB (4,049 MB), firmware and bootloader of 2025-05-08.
- Raspberry Pi OS Lite (Legacy, 64-bit), Debian 12 Bookworm, image "Raspberry Pi reference 2026-10-06", kernel
  `6.12.109+rpt-rpi-2712`, Python 3.11.2. SD card 29 GB (26 GB free).
- Packages: ONNX Runtime 1.31.0, OpenVINO 2026.4.1, ncnn 1.0.20260526, numpy 2.4.6, opencv-python-headless 5.0.0.93.
- The governor `performance` on all 4 cores, no desktop, the apt timers off. `get_throttled` `0x0` after the fresh boot.
- Cooling: the fan on the FAN socket, 1,770–1,875 rpm at each session's start; `temp_soft_limit` 0 in the firmware's
  config. Idle CPU 50.1–50.7 °C.
- Room: 26 °C, as the dev entered it in the wizard. How it was measured is not recorded.
- Network: the dev's hotspot. The wizard's progress view logs in over SSH every 30 s during the session.

## Results

### At the stock clock, 2400 MHz: throttled

Session `bench/pi5/pi5-fullcell`, 26 October, 22:38–23:12 IST.

| Run (in order) | Starts, IST | Images | Median latency, ms | Peak CPU | Clock below 2400 MHz | Verdict |
|---|---|---|---|---|---|---|
| reference run, first | 22:38:39 | 200 | 355.5 | 80.4 °C | 0 images | throttled (the bit set at its last sample) |
| full cell, attempt 1 | 22:43:06 | 1,610 | 355.7 | 82.6 °C | 67 images, down to 2311 | throttled |
| full cell, attempt 2 (the re-run) | 22:56:51 | 1,610 | 356.1 | 83.2 °C | 182 images, down to 2256 | throttled |
| reference run, last | 23:10:36 | 200 | 355.3 | 82.0 °C | 3 images, down to 2366 | none |

- **The throttle is the soft temperature limit.** `get_throttled` reads `0x80000` (it has occurred) through the cell
  and `0x80008` (active) in some samples. The under-voltage and frequency-cap bits never set.
- **The CPU settles at 81 °C,** the median of each tenth of the cell from the fourth tenth on. Above 80 °C the firmware
  takes the clock down to 2256–2366 MHz, on and off.
- **The median latency does not move.** The tenths of attempt 1 have medians from 353.6 to 356.5 ms, and the
  unthrottled last reference run gives 355.3 ms. At this load the soft throttle costs less than 1 %.
- Drift: 355.5 → 355.3 ms, −0.1 %.
- By the rule, the pilot failed, and the dev went on to the 2000 MHz lock (`pi/README.md`, part 3b).

### At the 2000 MHz lock: passed

`arm_freq=2000` in `/boot/firmware/config.txt`, then a reboot. Session `bench/pi5/pi5-fullcell-2000`, 26 October,
23:34–23:56 IST.

| Run (in order) | Starts, IST | Images | Median latency, ms | p10–p90, ms | Peak CPU | Clock |
|---|---|---|---|---|---|---|
| reference run, first | 23:35:18 | 200 | 388.2 | 386.8–391.1 | 77.7 °C | 2000 throughout |
| full cell | 23:39:55 | 1,610 | 388.5 | 386.9–391.0 | 79.3 °C | 2000 throughout |
| reference run, last | 23:54:43 | 200 | 392.4 | 389.6–395.6 | 78.2 °C | 2000 throughout |

- **No throttle bit** in any sample of any run (`0x0` throughout). **The pilot passed at 2000 MHz.** 1800 and 1500
  were not needed.
- **The temperature levels off.** The tenths of the cell have medians of 75.5–77.1 °C, flat from the second tenth on,
  so a longer cell is not expected to climb much higher. The peak, 79.3 °C, is just under the 80 °C line.
- Drift: 388.2 → 392.4 ms, +1.1 %.
- **Time of a full cell:** 11 min 48 s from "attempt 1 starts" to "complete", with the 20 warm-up images. The whole
  session took 21 min 31 s.
- **Accuracy on all 1,610 test-dev images:** AP 0.1875, AP50 0.3265, AP-small 0.0904 (AP-medium 0.2837, AP-large
  0.4203, AR500 0.3563). The same, to 4 digits, as the same export on the Galaxy M34
  ([m34_fullcell_record.md](m34_fullcell_record.md)), and as both throttled attempts above.

### What the lock costs

The same cell takes 388.5 ms at 2000 MHz against 355.7 ms at 2400 MHz: **+9.2 %** latency. Every Pi 5 number from
here on carries the lock, and the paper reports the Pi 5 at 2000 MHz. The lock stays in the Pi's `config.txt` for every
later Pi 5 session (#29, #8).

### The start gate

Of the 7 gates in these two sessions, 2 opened on the rule (CPU below 50 °C) and 5 at the 3-minute timeout, with the
CPU at 50.7–52.9 °C. With the fan, the Pi idles at about 50 °C, right on the gate's line. The speed test's gates did the
same (4 of 8 on the rule). See #29's open item on the gate.

## Known limitations

- **The pilot cell is not the heaviest one.** #29 asked for the heaviest Pi 5 cell (YOLO26s, ONNX Runtime FP32) for
  15 minutes. No YOLO26s export from A2 exists yet (#37), so the pilot ran A1. The 960 and 1280 cells run about 2–4×
  longer; the flat temperature at 2000 MHz suggests they hold, and the runner re-runs any cell that throttles.
- The room temperature is the dev's entry; the instrument is not recorded.
