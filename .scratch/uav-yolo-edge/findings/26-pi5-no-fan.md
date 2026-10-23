# Findings: the Pi 5 has no fan. What that does to the measurements, and how to run the protocol honestly

Ticket: #26 (energy runs on the Pi 5); the protocol is `study/PLATFORMS.md` § "Running a session".
Researched: 2026-10-23. Paper due 2026-10-31.
Source snapshot: `raspberrypi/documentation` @ `ddea98e` (2026-10-13); `raspberrypi/linux` branch `rpi-6.12.y`.

**Question.** The protocol assumed the official Active Cooler. We actually have a bare Pi 5 (no fan, possibly no
heatsink) on the official 27 W supply, running 4-thread CPU inference (ONNX Runtime / OpenVINO / NCNN) headless. What
are the throttling rules? How fast does a bare board throttle, and by how much? What does that do to latency and to
J/frame? Which way of running without a fan keeps the paper valid?

**Verdict in one line:** a bare Pi 5 under 4-core load **reaches the 80 °C soft limit in under a minute** and spends
the rest of a 3-minute or 14-minute run throttled, at 1.5–2.3 GHz instead of 2.4 GHz. Throttled runs measure the cooling,
not the model. Re-running a throttled cell at the end of the session fixes nothing, because the re-run throttles too. The
protocol only survives if the board is kept below 80 °C. The cheapest certain fix is an Active Cooler (about US$6). Without
one: stand the board on its edge in front of a desk fan held at a fixed setting, and prove with a 15-minute pilot on the
heaviest cell that it never throttles. If it does, lock the CPU clock with `arm_freq` and report it.

---

## 1. Throttling thresholds on the Pi 5 (primary: official docs, firmware engineer, kernel DT)

**The temperatures.**
- Official docs, *Frequency management and thermal control*: *"temperatures do not exceed a limit which we define as
  85°C on all models. It is possible to set this to a lower value, but not to a higher one"* and *"When the core
  temperature is between 80°C and 85°C, the Arm cores will be progressively throttled back. If the temperature reaches
  85°C, both the Arm cores and the GPU will be throttled back."*
  ([frequency-management.adoc L3, L5](https://github.com/raspberrypi/documentation/blob/ddea98e98dafe9419feb4fd82710c2fbdef7dc44/documentation/asciidoc/computers/raspberry-pi/frequency-management.adoc#L3-L5))
- `config.txt` reference: `temp_limit` = *"Overheat protection … Values over 85 are clamped to 85"*, default 85. **`temp_soft_limit`
  is "*3A+/3B+ only*"**, so the Pi 3B+'s 60 °C soft limit does **not** apply to the Pi 5
  ([overclocking.adoc L98–102, L293](https://github.com/raspberrypi/documentation/blob/ddea98e98dafe9419feb4fd82710c2fbdef7dc44/documentation/asciidoc/computers/config_txt/overclocking.adoc#L98-L102)).
- Raspberry Pi white paper RP-010139-WP-1 (*Use-case-specific thermal performance of Raspberry Pi SBCs*, release 1,
  1 Jun 2026): soft limit 80 °C, *"The firmware gradually reduces the clock speed"*; hard limit 85 °C, *"The firmware
  aggressively forces the clock speed and voltage down, which can result in a 30-50% reduction in performance."*
  ([PDF p. 6](https://pip-assets.raspberrypi.com/categories/685-app-notes-guides-whitepapers/documents/RP-010139-WP-1-Use-case-specific%20thermal%20performance%20of%20Raspberry%20Pi%20SBCs.pdf)).
  Its wording says the 80 °C soft throttle applies "on Raspberry Pi 4 Model B". The forum answer below and the paper's
  own Pi 5 run (§2) show that it applies to the Pi 5 as well.

**What each threshold does to the clock (staff answer).**
- **jdb (Raspberry Pi Engineer & Forum Moderator), 27 Mar 2024:** *"There is a soft-throttle that starts at 80°C and a
  hard throttle at 85°C. The soft-throttle linearly decreases the maximum allowed ARM frequency by 100MHz per degree until
  85°C, then the idle frequency is forced."* ([forum t=368073](https://forums.raspberrypi.com/viewtopic.php?t=368073))
- The Pi 5's idle frequency is **1500 MHz** (`arm_freq_min` default for "Pi 5/500/500+"; `arm_freq` default 2400)
  ([overclocking.adoc table, L112–125, L199–209](https://github.com/raspberrypi/documentation/blob/ddea98e98dafe9419feb4fd82710c2fbdef7dc44/documentation/asciidoc/computers/config_txt/overclocking.adoc#L110-L125)).
  So the curve is **2.4 GHz below 80 °C, falling about 100 MHz per °C from 80 °C to 85 °C, then 1.5 GHz at 85 °C.**
- Alasdair Allan (Raspberry Pi staff, author of the Pi 5 cooling post), in the comments: a bare board *"will throttle
  the CPU down from 2.4GHz to 1.5GHz"* ([raspberrypi.com, 4 Oct 2023](https://www.raspberrypi.com/news/heating-and-cooling-raspberry-pi-5/)).
- Measured steps match this. Tom's Hardware saw *"2311 MHz, 2256 MHz, 2201 MHz and 2146 MHz"* before *"2.4 GHz down to
  1.5 GHz"* ([Tom's Hardware review, 23 Oct 2023](https://www.tomshardware.com/reviews/raspberry-pi-5)). These are steps of about 55 MHz.
- ⚠️ **Unresolved:** Roy Longbottom (forum member, self-described alpha tester, *not* staff) logged a no-fan Pi 5 at
  **1000 MHz for much of a run, at 89–91.7 °C**. That is below the 1500 MHz floor and above the 85 °C limit
  ([forum t=363656](https://forums.raspberrypi.com/viewtopic.php?t=363656); posts from Jan 2024, on early firmware). The
  forum also lists the Pi 5's clock range as 1000–2400 MHz. Treat 1.5 GHz as the floor on current firmware, but expect
  lower if the board keeps heating past 85 °C.

**Who does the throttling.** The firmware does it, not Linux. dom (Raspberry Pi Engineer): *"temperature throttling is
implemented on the gpu side, so isn't directly attached to kernel side trip points"*. The kernel trips *"just implement a
reboot when temperature is excessively high"* ([forum t=388927, Jun 2025](https://forums.raspberrypi.com/viewtopic.php?t=388927)).
The kernel device tree agrees:
- the only non-fan trip is `cpu-crit` at **110 000 m°C, type "critical"** ([bcm2712-ds.dtsi L48–61](https://github.com/raspberrypi/linux/blob/rpi-6.12.y/arch/arm64/boot/dts/broadcom/bcm2712-ds.dtsi#L48-L61));
- the Pi 5's other trips are fan steps only: 50 / 60 / 67.5 / 75 °C, each with 5 °C hysteresis
  ([bcm2712-rpi-5-b.dts L498–549](https://github.com/raspberrypi/linux/blob/rpi-6.12.y/arch/arm64/boot/dts/broadcom/bcm2712-rpi-5-b.dts#L498-L549)),
  matching the docs' fan curve ([frequency-management.adoc L85–93](https://github.com/raspberrypi/documentation/blob/ddea98e98dafe9419feb4fd82710c2fbdef7dc44/documentation/asciidoc/computers/raspberry-pi/frequency-management.adoc#L85-L93)).

**Consequence for the bench (load-bearing):** Linux's `scaling_cur_freq` **does not show firmware throttling.** Docs: *"this
frequency is the kernel _requested_ frequency, and it is possible that any throttling … might mean the CPU is actually
running more slowly than reported. An instantaneous measurement of the actual Arm CPU frequency can be retrieved using …
`vcgencmd measure_clock arm`"* ([overclocking.adoc L351](https://github.com/raspberrypi/documentation/blob/ddea98e98dafe9419feb4fd82710c2fbdef7dc44/documentation/asciidoc/computers/config_txt/overclocking.adoc#L351)).
A forum user confirmed the mismatch on a heatsink-less Pi 5 ([t=388927](https://forums.raspberrypi.com/viewtopic.php?t=388927)).
`studybench.py`'s `cpu_mhz()` reads `scaling_cur_freq` (L509–511). With the `performance` governor it will log 2400 for
every image, throttled or not. The docs also say that *"Linux-based temperature measurements can be inaccurate"* and that
`vcgencmd measure_temp` is accurate ([frequency-management.adoc L49](https://github.com/raspberrypi/documentation/blob/ddea98e98dafe9419feb4fd82710c2fbdef7dc44/documentation/asciidoc/computers/raspberry-pi/frequency-management.adoc#L47-L54)).
The bench's `temp_c` and the start gate read `thermal_zone0`.

**`vcgencmd get_throttled` bits** ([graphics-utilities.adoc L45–84](https://github.com/raspberrypi/documentation/blob/ddea98e98dafe9419feb4fd82710c2fbdef7dc44/documentation/asciidoc/computers/os/graphics-utilities.adoc#L45-L84)):

| Bit | Hex | Now / since boot | Meaning |
|---|---|---|---|
| 0 / 16 | `0x1` / `0x10000` | now / occurred | under-voltage |
| 1 / 17 | `0x2` / `0x20000` | now / occurred | Arm frequency capped |
| 2 / 18 | `0x4` / `0x40000` | now / occurred | currently throttled |
| 3 / 19 | `0x8` / `0x80000` | now / occurred | soft temperature limit active |

How they map to the thresholds:
- **Bit 3 (`0x8`) is the 80–85 °C soft throttle.** A forum user tested this after jdb's answer: *"soft-throttling is
  reported correctly by get_throttled"* ([t=368073](https://forums.raspberrypi.com/viewtopic.php?t=368073)).
- Raspberry Pi's own bare-board run caught it within 30 s: at 83.4 °C and 2256 MHz the flags were *"soft-temp-limit
  freq-cap-occurred throttle-occurred"* (white paper Fig. 7, §2).
- Which of bits 1 and 2 fires at the 85 °C hard limit is not stated in any official source I found (**unverified**).
  Treat any of bits 1–3 as "throttled".

---

## 2. A bare Pi 5 under sustained all-core load: measured numbers

| Source (who, when) | Cooling, setup | Load | Result |
|---|---|---|---|
| Raspberry Pi Ltd white paper RP-010139, Fig. 7 (runs dated 13–14 Apr 2026) | **Bare board in a Bumper, no heatsink** | `stress-ng --cpu 4 --cpu-method all`, 30 s | 58.7 → **83.4 °C in 30 s**, clock **2256 MHz**, soft-temp-limit active, throttle occurred. Avg 76.3 °C ([PDF p. 21](https://pip-assets.raspberrypi.com/categories/685-app-notes-guides-whitepapers/documents/RP-010139-WP-1-Use-case-specific%20thermal%20performance%20of%20Raspberry%20Pi%20SBCs.pdf)) |
| Same paper, Fig. 6 | Active Cooler + M.2 HAT+ + SSD, case lid off | same, 120 s | max 71.4 °C, 2400 MHz, no throttling |
| Raspberry Pi blog, Alasdair Allan (staff), 4 Oct 2023 | Bare, flat, open air on lab bench, air-conditioned office (ambient not given) | `stress --cpu 4`, "extended testing" | **idle ≈ 65 °C**; load *"climbs to and then remains stable just above the 85°C thermal limit"* → *"sustained thermal throttling"*; clock 2.4 → 1.5 GHz ([post](https://www.raspberrypi.com/news/heating-and-cooling-raspberry-pi-5/)) |
| Same post | Active Cooler, fan **unplugged** (heatsink only) | same | throttles at **≈ T₀ + 200 s** |
| Same post | Active Cooler, fan on | same | idle ≈ 45 °C, load 60 °C (max 62–63 °C); back to 45 °C *"within a further 300 seconds"* |
| Tom's Hardware, Piltch & Pounder, 23 Oct 2023 | Bare (ambient, orientation not given) | stress test | **idle 50.5 °C, 2.7 W**; load **86.7 °C, 7 W, 2.4 → 1.5 GHz** ([review](https://www.tomshardware.com/reviews/raspberry-pi-5)) |
| Same review | Generic stick-on heatsinks (SoC, PMIC, Wi-Fi) | same | idle 41.1 °C; load **85.6 °C, still throttled**. *"Buying cheap heat sinks is a waste of money"* |
| Same review | Active Cooler / Pimoroni Fan Shim | same | 59.3 °C / 60.4 °C |
| Core Electronics (Jaryd), updated 1 Nov 2023 | Bare, ≈ 25 °C air-conditioned office | Stressberry, 30 min | throttled **in under a minute**; ≈ 60 °C just opening a browser ([guide](https://core-electronics.com.au/guides/answering-your-pi-5-questions-cooling-overclocking-power-and-more/)) |
| Same guide | Active Cooler without fan / generic low-profile heatsink | Stressberry | throttling after **≈ 10 min**; ≈ 10 °C hotter than with the fan |
| R. Longbottom, forum (not staff), Jan 2024 | **"no fan"**, before the Active Cooler was available (heatsink presence not stated), 4 A then 5 A PSU | own stress suite, 15–30 min | 1500 MHz by ≈ 60 s at ≈ 86 °C; **1000 MHz by ≈ 7 min**; avg 1174–1239 MHz, avg 88.7 °C, max 91.7 °C; **−43 to −49 %** integer/FP throughput, −40 % CPU+GPU *"without the fan"* ([t=363656](https://forums.raspberrypi.com/viewtopic.php?t=363656)) |
| Marasini & Zhou, arXiv:2609.04705 (Sep 2026) | Active Cooler with **fan disconnected**, 23 ± 2 °C | YOLOv8n, OpenVINO FP32, 30 min | plateau **84.8 °C, 1823 throttle events**, 12.31 FPS vs 14.53 FPS actively cooled (**−15 %**) ([arXiv](https://arxiv.org/html/2609.04705)) |

What the official sources give as "typical":
- RP-010139 Table 2 (layout garbled in extraction; this is my reading): no cooling idle 50–60 °C, load 80–85 °C;
  heatsink 45–55 / 70–75 °C; fan 35–45 / 55–65 °C.
- *"Fan-assisted cooling can reduce the temperature by 10-20°C"*.
- Its conclusion recommends *"a fan for heavy workloads or when using a Raspberry Pi 5"* ([PDF pp. 13–16](https://pip-assets.raspberrypi.com/categories/685-app-notes-guides-whitepapers/documents/RP-010139-WP-1-Use-case-specific%20thermal%20performance%20of%20Raspberry%20Pi%20SBCs.pdf)).

**Summary:**
- **Time to throttle** on a bare board under 4-core load: **30–60 s** (Raspberry Pi Ltd, Core Electronics).
- **Steady state:** 85–87 °C (Raspberry Pi blog, Tom's), up to ≈ 90 °C (Longbottom).
- **Clock:** settles near **1.5 GHz**, a 37.5 % clock loss. The official white paper puts the hard-limit loss at 30–50 %.
  Longbottom measured 40–49 % at 1.0–1.24 GHz average.
- **Bare idle in open air:** 50.5 °C (Tom's) to ≈ 65 °C (Raspberry Pi). Raspberry Pi's typical range is 50–60 °C.
- **A heatsink alone only delays throttling:** about 200 s (Raspberry Pi), about 10 min (Core Electronics), or not at all
  with stick-on sinks (Tom's).
- **Board variation (from Jeff Geerling's measurements, a secondary source).** The 2 GB board's D0 stepping has a 32.5 %
  smaller die. It draws about 0.9 W less at idle and under `stress-ng` than the 4 GB and 8 GB C1 boards, and runs 4–5 °C
  cooler under load ([Geerling, 29 Aug 2024](https://www.jeffgeerling.com/blog/2024/new-2gb-pi-5-has-33-smaller-die-30-idle-power-savings/)).
  So results differ by board. Record the board revision.

**Desk fan / external airflow on a bare Pi 5: no measured data found.**
- Raspberry Pi's white paper RP-003608-WP-3 (*Cooling a Raspberry Pi device*, release 3, 13 Feb 2026) recommends it but
  gives no numbers: *"If you are prototyping on a desk using bare boards, a quick and easy way to keep the device cool is to
  use a desktop fan! Any extra airflow around the board will greatly increase cooling."*
  ([PDF p. 7](https://pip-assets.raspberrypi.com/categories/685-app-notes-guides-whitepapers/documents/RP-003608-WP-3-Cooling%20a%20Raspberry%20Pi%20device.pdf)).
- The only quantitative analogue is the Active Cooler's own blower: **1.09 CFM** ([product page](https://www.raspberrypi.com/products/active-cooler/))
  over a heatsink. It holds a stressed Pi 5 at about 60 °C.
- Whether a desk fan blowing on a **bare** die keeps it below 80 °C at 4-core load is **unverified**. It has to be measured
  (§5, pilot).

**Orientation.**
- Docs: *"Mount the Raspberry Pi vertically for the best airflow and thus slightly improved heat dissipation"*
  ([frequency-management.adoc L58](https://github.com/raspberrypi/documentation/blob/ddea98e98dafe9419feb4fd82710c2fbdef7dc44/documentation/asciidoc/computers/raspberry-pi/frequency-management.adoc#L56-L58)).
- RP-003608: mounting a bare board on its edge lets hot air rise from both sides, *"standard convection can drop the
  temperature considerably"*.
- **No Pi 5 measurement exists.** Alasdair Allan, asked about vertical mounting, replied *"haven't tested in that
  configuration"* ([comments](https://www.raspberrypi.com/news/heating-and-cooling-raspberry-pi-5/)).
- The only data is a **Pi 4** test (MagPi / Gareth Halfacree, 28 Nov 2019). There, vertical mounting idled about 2 °C
  lower, *"heats a lot more slowly"* and kept a *"dramatically improved average clock speed"*, with no figures in the text.
  The flat bare Pi 4 throttled at 177 s ([raspberrypi.com](https://www.raspberrypi.com/news/thermal-testing-raspberry-pi-4/)).

---

## 3. What this does to the latency benchmarks

**Will a 3-min or a 15-min run throttle at 4 threads of CNN inference? Yes, both, on a bare board (inferred from the
above, with strong support).**
- CNN inference draws about as much power as a stress test:
  - YOLOv8n under OpenVINO drew **7.2 W** at the USB-C input on a passively cooled Pi 5 ([arXiv:2609.04705](https://arxiv.org/html/2609.04705), Table "static states" S0);
  - Tom's stress test drew 7 W at 86.7 °C;
  - YOLO11s/26s at 640 is about 2.5× the GFLOPs of v8n, so the cores stay saturated.
- The bare board crosses 80 °C in 30–60 s under a load like that (§2). A 20-image warm-up (about 10 s at about 0.5 s per
  image) ends long before then.
- So in **every** cell:
  - the first ~30–60 s run at 2.4 GHz, the rest at roughly 1.5–2.2 GHz;
  - the median latency is a mix whose proportions depend on the cell's length, on its starting temperature and on how hot
    the cell itself runs.
- A 3-minute energy run is mostly throttled. A 14-minute full cell is almost entirely throttled.

**Published Pi 5 inference without cooling:**
- **Marasini & Zhou, arXiv:2609.04705** (Sep 2026, preprint):
  - Setup: heatsink only, fan unplugged; YOLOv8n, OpenVINO FP32; 23 ± 2 °C ambient; at least 15 min cool-down between
    runs; official 27 W PSU; inline USB-C meter.
  - At stock clocks: plateau 84.8 °C, **1823 throttle events in 30 min**, 12.31 FPS, against 14.53 FPS with the Active
    Cooler (−15 %).
  - **Capping the clock at 1800 MHz gave 12.13 FPS with zero throttle events at an 80.3 °C plateau.** So a "2.4 GHz"
    run that throttles performs like a run locked at 1.8 GHz. At 1500 MHz: 10.76 FPS, 73.4 °C.
  - Above 27 °C ambient, their linear thermal model broke down. They attribute this to *"nonlinear leakage"*: at 31 °C
    the 1500 MHz plateau was 88–89 °C against 81 °C predicted (single exploratory runs).
- **Ranjan et al., arXiv:2607.09835** (YOLO26 vs predecessors, aquaculture; ONNX Runtime on a Pi 5):
  - *"Initial deployment trials on the Raspberry Pi 5 without active cooling resulted in CPU throttling within minutes of
    sustained inference, with peak temperatures above 80 °C."*
  - With an active cooler: idle 44.4 °C, peak < 70 °C, `0x0` throttle state, *"unthrottled operation at 2.4 GHz across
    all reported benchmark runs"*.
  - They give no uncooled latency numbers; they discarded those runs ([arXiv PDF §2.5, §3.5](https://arxiv.org/pdf/2607.09835)).
- **Ghoniya et al., IJERT 15(3), Mar 2026** (low-tier venue; cooling not stated):
  - YOLOv8n TFLite on a Pi 5, all 4 cores: *"After approximately 120 seconds, the BCM2712 die temperature reached 84.0°C,
    triggering DVFS throttling"*. Clock fell 2.4 → about 2.18 GHz and FPS 2.6 → 2.1 (−19.2 %).
  - They prescribe *"a 60-second thermal warm-up"*.
  - ⚠️ The 9 % clock drop against a 19 % FPS drop is internally odd. Cite with care, if at all ([IJERT](https://www.ijert.org/comparative-benchmarking-of-cpu-gpu-and-npu-architectures-for-real-time-yolov8n-inference-on-embedded-edge-platforms-ijertv15is030232)).
- **Pi 4 precedent:** Benoit-Cattin, Velasco-Montero & Fernández-Berni, arXiv:2010.06291 (2020):
  - 5 CNNs × 4 frameworks × 2 OSes, with and without a fan;
  - the fan improved throughput **by up to ≈ 90 %** and prevented throttling in all 80 cases;
  - uncooled, ambient between 22 and 36 °C moved throughput **by up to 27.7 %** ([arXiv](https://arxiv.org/abs/2010.06291)).

**How Pi 5 benchmark papers state and control cooling (the reviewer's expectation):**
- **arXiv:2609.04705:** names the cooler; logs ambient with a DHT11 at the start and end of each run; at least 15 min
  cool-down; logs throttle events.
- **arXiv:2607.09835:** active cooler; *"device temperature and thermal throttling status were verified prior to each model
  evaluation"*; reports idle and peak temperatures and the `0x0` state.
- **Vdoviak et al., *Sensors* 26(16):5188 (2026)**, YOLO-pose, PyTorch / ONNX Runtime / NCNN, **4 threads**:
  - fan cooling, ≈ 21 °C ambient;
  - logged `measure_temp` and `get_throttled` at the latency cadence;
  - max 77.4 °C, *"The firmware thermal-throttling flag was not raised in any of the 135 configurations"*;
  - they concede it *"does not rule out throttling under longer, back-to-back continuous operation"* ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13517955/)).
  - Their phrase *"integrated on-board fan"* is ambiguous: the Pi 5 has no fan on the board, so it is presumably an Active Cooler.

The norm is: **name the cooling, report ambient, show the throttle flag was 0x0**. A paper whose Pi 5 cells throttle,
without saying so, would be out of line with every recent Pi 5 benchmark found.

**Why throttling is worse than "slower numbers" for *this* paper (my analysis):** the paper compares compression
configurations, so throttling is not noise but a **confound correlated with the treatment**:
- A heavier cell (FP32, YOLO26s's extra layers) heats more and is throttled harder.
- A lighter cell (INT8, pruned) runs cooler and keeps more of its 2.4 GHz time.
- So the measured speed-up of compression on the Pi = true speed-up × a thermal bonus. That inflates exactly the quantity
  the paper reports.
- The cell order is shuffled and the start gate is not reached (§5f), so the starting temperature also varies by position.
  The drift check (reference cell first and last, ±3 %) will probably trip, because the first reference run starts from a
  cold board.

---

## 4. What it does to energy (J/frame)

**General physics (primary, well established).** CPU power at a fixed frequency and voltage rises with die temperature,
because sub-threshold leakage grows exponentially with temperature.
- De Vogeleer, Memmi, Jouvelot & Coelho, arXiv:1404.3381 (2014), on real application processors:
  - They measured the Exynos 5 big.LITTLE on an ODROID-XU+E (28 nm, Cortex-A7/A15) and a Galaxy S2 (45 nm A9),
    sweeping 25–85 °C at fixed frequency and core count.
  - Result: *"the power/temperature relationship is indeed very likely exponential over a 20°C to 85°C temperature range"*.
    An exponential fit beats a quadratic, and a quadratic beats a linear one (sign test p < 0.01) ([arXiv](https://arxiv.org/abs/1404.3381)).
  - Their fitted A15 model (their Fig. 5, my evaluation of it), 4 cores at fixed frequency:
    - **+11 % power going from 60 °C to 85 °C** at 1.2 GHz, +12 % at 1.6 GHz;
    - **+15–17 % from 45 °C to 85 °C**;
    - this matches their measured trace (Fig. 3b: about 2.4 → 2.8 W over 30–80 °C at 1.2 GHz).
- The same authors, arXiv:1401.4655: *"our CPU at 1.3 GHz shows an inflated power consumption of around 5% between a CPU
  temperature of 36°C and 46°C"*. They cite You et al.: +3 % for 30 → 40 °C on a 0.1 µm part ([arXiv PDF](https://arxiv.org/pdf/1401.4655)).
- Freescale / UC Irvine, ASP-DAC 2007 slides (65 nm industrial core): *"Leakage power more than doubles with a 20°C
  increase in the temperature of the die"* ([slides](https://www.aspdac.com/aspdac2007/pdf/archive/3B-1.pdf)).
  Total power rises much less than that, because leakage is only part of the total.
- Raspberry Pi's white paper RP-010139 confirms the mechanism for its own chips only qualitatively (*"fast" silicon "leak[s]"
  more current, "meaning they can run hotter and consume more power"*). It gives no coefficient.

**On the Pi 5 / BCM2712 specifically: no clean leakage-vs-temperature measurement exists (searched forums, papers,
reviews).**
- BCM2712 is a 16 nm part (Geerling, secondary). FinFET processes have a smaller leakage share than 28 nm planar ones, so
  the Exynos numbers above are probably an **upper bound** for the Pi 5 (**inferred, unverified**).
- The closest Pi 5 data, arXiv:2609.04705 (USB-C wall power, YOLOv8n):

  | State | Plateau | FPS | W | J/frame |
  |---|---|---|---|---|
  | 2.4 GHz, throttling (heatsink only) | 84.8 °C | 12.31 | 7.22 | **0.586** |
  | locked 1.8 GHz, no throttling | 80.3 °C | 12.13 | 6.36 | **0.524** |
  | locked 1.5 GHz | 73.4 °C | 10.76 | 5.64 | 0.524 |
  | 2.4 GHz with Active Cooler (fan power included) | 63.4 °C | 14.53 | 8.18 | 0.563 |

  - The **hot, throttling run cost about 4 % more J/frame than the cool, unthrottled one even though the cool one paid for
    its fan.**
  - Subtracting about 0.3 W for the fan (the docs give 60 mA at 5 V for the official case fan
    [frequency-management.adoc L81](https://github.com/raspberrypi/documentation/blob/ddea98e98dafe9419feb4fd82710c2fbdef7dc44/documentation/asciidoc/computers/raspberry-pi/frequency-management.adoc#L79-L81);
    the Active Cooler's draw is not published) gives about 0.54 J/frame for the cool board. So a **throttled hot run is
    ≈ 8 % worse** (my arithmetic, **inferred**).
  - The **locked-1.8 GHz run is 11 % better** than the throttling one at almost the same FPS. Throttling spends time at
    2.4 GHz and high voltage, then falls back.
  - These effects mix DVFS voltage and leakage; the data cannot separate them.

**What this means for our rail-sum J/frame:**
1. **Direction:** on a bare board, J/frame comes out **higher** than with the Active Cooler, by roughly 5–10 % (Pi 5 wall
   data above; a rail sum excludes PSU losses, so the relative effect on the SoC rails could be larger; **unverified**).
2. **The confound again:** cells that run hotter pay more leakage per frame, so J/frame differences between compression
   configurations pick up a temperature term. A 60 → 85 °C difference is worth up to about 10 % on the A15 model, less on
   16 nm.
3. **Idle baseline:** the 1-minute idle reading is taken at whatever temperature the gate left the board, 50–65 °C bare.
   The `performance` governor keeps the idle clock at 2.4 GHz, so the idle rail sum is itself temperature-dependent.
   Net J/frame (run − idle) is the cleaner of the two figures, but only if idle and run temperatures are logged.
4. **The fan's power stays out of the measurement in every option.** It is outside the PMIC rail sum (as noted in the
   brief), and a desk fan has its own mains supply. So removing the Active Cooler does not "save" energy in our metric. It
   only changes the temperature.

---

## 5. How to run the protocol without a fan: options with evidence

Current rules (`study/PLATFORMS.md` L85–105):
- 4 threads, 20 warm-up images;
- start when the CPU is below 50 °C, or after 3 minutes;
- a throttled cell re-runs once at the end of the session;
- a full cell is 1,610 images (about 14 min).

Throttle detection: `studybench.py` samples `get_throttled` at start, every 50 images (about 25 s) and at the end.
`studysession.py` (L256–268) marks a run throttled if a current bit was ever seen, or if a sticky bit 16–19 newly appeared
between the start and the end. That logic is sound. Two caveats:
- **after the first throttle since boot, the sticky bits stay set** (I found no documented way to clear them without a
  reboot; **unverified**), so later runs rely on the 25-s spot checks of the current bits alone;
- the logged MHz column is blind to throttling (§1).

| Option | Evidence | Cost to validity | Verdict |
|---|---|---|---|
| **(0) Buy the Active Cooler** (≈ US$6 per [Tom's](https://www.tomshardware.com/reviews/raspberry-pi-5); push-pin mount, pre-applied pads, [product page](https://www.raspberrypi.com/products/active-cooler/)) | Load 59–63 °C, no throttling (Raspberry Pi, Tom's, arXiv:2607.09835, *Sensors* 2026) | None. The protocol, PLATFORMS.md and the comparisons to other papers all stand as written | **Best, if it can arrive by about 27 Oct** |
| **(a) Cheap passive heatsink** | Stick-on sinks: still throttled at 85.6 °C (Tom's). Active Cooler without fan: throttles after about 200 s (Raspberry Pi) to about 10 min (Core Electronics); 1823 events in 30 min with YOLO (arXiv:2609.04705). A full metal heatsink case (Argon NEO, FLIRC style, [RP-003608](https://pip-assets.raspberrypi.com/categories/685-app-notes-guides-whitepapers/documents/RP-003608-WP-3-Cooling%20a%20Raspberry%20Pi%20device.pdf) p. 8) is better but has no Pi 5 sustained-load numbers found | Delays throttling, does not prevent it. A 14-min cell still throttles. 3-min runs may or may not, depending on the start temperature, which adds variance | **Not enough alone**; worth adding under (b) |
| **(b) Desk fan / external airflow** | Officially recommended for bare boards ([RP-003608 p. 7](https://pip-assets.raspberrypi.com/categories/685-app-notes-guides-whitepapers/documents/RP-003608-WP-3-Cooling%20a%20Raspberry%20Pi%20device.pdf)). **No Pi 5 numbers anywhere**: effect unverified | Fine if it holds the board below 80 °C on the heaviest cell. To keep it the same for every run: one fan, one speed setting, fixed distance and angle (mark the desk), oscillation off, same room, window and AC state. Log ambient at every gate (a USB or DHT-type thermometer). Photograph the setup for the paper. Residual risk: ambient drift across a long session | **Recommended, pending a pilot** |
| **(c) Vertical board, case off** | Docs: vertical gives *"slightly improved heat dissipation"*; RP-003608: edge mounting lets convection *"drop the temperature considerably"*. Pi 4: slower heating, higher average clock. No Pi 5 numbers | Free, no validity cost; must be the same for all runs | **Do it, together with (b)** |
| **(d) Accept throttling and report it** | Clock 1.5–2.3 GHz, −15 % (YOLO) to −49 % (stress) throughput (§2–3) | Large. Latency depends on the cooling, the cell's own heat, the cell's length and its position. The compression speed-up is inflated by a thermal bonus (§3), and J/frame gets a temperature term (§4). Not comparable to Ultralytics' Pi 5 table or other papers (all actively cooled). The "re-run once" rule is moot: the re-run throttles too | **Last resort only.** If used, reframe the Pi column as "sustained, fanless" and report the per-run time-at-clock |
| **(e) Fewer threads** (e.g. 2) | No Pi 5 measurement of 2-thread inference temperatures found. Lower power is expected (fewer active cores; the De Vogeleer model scales with core count) but **unverified** that it stays below 80 °C | Changes a fixed protocol parameter (4 threads on every platform, PLATFORMS.md L85). Cross-platform rows stop being like for like. Speed-ups at 2 threads differ from those at 4 | **Not recommended** |
| **(f) Different start gate / longer cool-down** | Bare idle in open air: 50.5 °C (Tom's), 50–60 °C (Raspberry Pi typical), ≈ 65 °C (Raspberry Pi blog). Our `performance` governor probably idles hotter still (**inferred**). Cool-down time of a bare board: no data (Active Cooler: 300 s to 45 °C) | The 50 °C gate will rarely open on a bare board, so every run starts "after 3 minutes" at an uncontrolled temperature. A different gate only standardises the *start*: it cannot stop a bare board throttling about 60 s into a 14-min run | **Needed, but not sufficient.** Use a gate relative to the session's measured idle plateau (below), with a longer timeout |
| **(g) Lock the clock (`arm_freq`) below the throttle point** (extra option) | arXiv:2609.04705: heatsink only, 1800 MHz → 80.3 °C, 0 throttle events; 1500 MHz → 73.4 °C. Lower J/frame than throttling. `arm_freq` is an official knob ([overclocking.adoc](https://github.com/raspberrypi/documentation/blob/ddea98e98dafe9419feb4fd82710c2fbdef7dc44/documentation/asciidoc/computers/config_txt/overclocking.adoc); RP-003608 p. 10 shows `[pi5] arm_freq=1500`). Bare-board (no heatsink) plateaus at these clocks: **unverified** | Every Pi number becomes "Pi 5 at X GHz", not stock, so absolute latencies are not comparable to published Pi 5 numbers. But they are **deterministic and stationary**, and relative comparisons between configurations are clean. (Speed-up ratios may differ slightly from those at 2.4 GHz if memory-bound fractions differ: **inferred**) | **Fallback if (b)+(c) fails the pilot** |

### Recommended protocol (one option)

**Run (b) + (c), proven by a pilot, with (g) as the pre-agreed fallback.** Buy (0) anyway if it can arrive in time. If it
does, skip all of this.

1. **Physical setup, fixed for the whole study:**
   - board on its edge, case off, in open air;
   - a desk fan at a marked distance and angle, fixed speed, no oscillation;
   - a cheap passive heatsink on the SoC if one is to hand (it only helps);
   - log the ambient temperature at every gate.
2. **Pilot (about 20 min, before any reported session):**
   - run the heaviest Pi cell: YOLO26s, ONNX Runtime FP32, 640 (the slowest per Ultralytics' table, PLATFORMS.md L43);
   - run it for a full 1,610 images from a cold board;
   - log `vcgencmd measure_clock arm`, `vcgencmd measure_temp` and `get_throttled` at 1 Hz in a side loop.
3. **Pass:** zero bits 1–3 for the whole run, `measure_clock arm` at 2400 MHz throughout, and peak temperature below about
   78 °C, to leave margin for warmer ambient. Then run the protocol at stock clocks and report "bare board, vertical, desk
   fan, ambient X–Y °C, no throttling (flags 0x0)".
4. **Fail:** set `arm_freq` to the highest of 2000 / 1800 / 1500 MHz that passes the same pilot. Apply it to **every** Pi
   cell, reference and energy run, and report "clock locked at X MHz for thermal stationarity". Do not mix locked and
   unlocked runs.
5. **Gate:**
   - replace "below 50 °C" with "within +3 °C of the idle plateau measured at the session's start, or after 10 min";
   - log the gate temperature, the ambient and the wait.
   - With a fan the idle plateau may well sit below 50 °C, and the original gate then works unchanged (**unverified until
     the pilot**).
6. **Instrument the truth, not the request** (bench changes for the owner to decide; not made here):
   - log `vcgencmd measure_clock arm` instead of `scaling_cur_freq`;
   - log `vcgencmd measure_temp` for temperature;
   - check the current throttle bits more often than every 50 images, or keep a 1 Hz side logger.
7. **Throttle rule:**
   - keep "re-run once" only for the pass case, where a throttle is an anomaly (ambient spike);
   - under (g) any throttle means the cap is wrong: stop and lower it;
   - never report a throttled attempt's latency or J/frame as the cell's value.
8. **Energy runs:** same cooling; report the mean run temperature next to each J/frame. Compare J/frame only between cells
   whose mean temperatures are within a few °C, or say how far apart they were (§4).

---

## Verdict for the protocol

- **The current protocol is not valid on a bare Pi 5.**
  - A bare board throttles **within 30–60 s** of 4-core load. Raspberry Pi Ltd's own bare run reached 83.4 °C and 2256 MHz
    in 30 s.
  - It then settles at **85–87 °C and about 1.5 GHz** (2.4 GHz is stock). That costs **15 % (YOLOv8n) to 30–50 % (stress)
    of throughput**.
  - Every 3-min and 14-min cell would throttle. The re-run would throttle again.
  - The 50 °C gate cannot open, because bare idle is 50–65 °C.
  - Throttling flatters the lighter, compressed cells, so it biases the paper's main comparison.
- **J/frame gets worse when hot.** Leakage makes power at a fixed clock grow roughly exponentially with temperature: about
  +11–17 % on a 28 nm A15 between 45–60 °C and 85 °C. On the Pi 5, a hot throttling YOLO run cost about 4–8 % more J/frame
  than an actively cooled one, and 11 % more than a cool run locked at 1.8 GHz.
- **Do this:**
  1. Get an Active Cooler (≈ US$6) if it can arrive by about 27 Oct.
  2. Otherwise: board on its edge, case off, a desk fan held at a fixed setting, ambient logged.
  3. Prove it with a 15-minute pilot on YOLO26s ONNX FP32 that logs `vcgencmd measure_clock arm` (not `scaling_cur_freq`,
     which cannot see throttling).
  4. If the pilot throttles, lock `arm_freq` (1.8 GHz held 80 °C with only a heatsink) for every Pi run and report it.
  5. Do not run and report throttled cells.

---

## Sources

Primary, Raspberry Pi documentation and firmware/kernel (fetched 2026-10-23):
- Frequency management and thermal control — [frequency-management.adoc @ ddea98e](https://github.com/raspberrypi/documentation/blob/ddea98e98dafe9419feb4fd82710c2fbdef7dc44/documentation/asciidoc/computers/raspberry-pi/frequency-management.adoc)
- `config.txt` overclocking options (temp_limit, temp_soft_limit, arm_freq/arm_freq_min, scaling_cur_freq caveat) — [overclocking.adoc @ ddea98e](https://github.com/raspberrypi/documentation/blob/ddea98e98dafe9419feb4fd82710c2fbdef7dc44/documentation/asciidoc/computers/config_txt/overclocking.adoc)
- `vcgencmd get_throttled` bits — [graphics-utilities.adoc @ ddea98e](https://github.com/raspberrypi/documentation/blob/ddea98e98dafe9419feb4fd82710c2fbdef7dc44/documentation/asciidoc/computers/os/graphics-utilities.adoc)
- Kernel DT: [bcm2712-ds.dtsi](https://github.com/raspberrypi/linux/blob/rpi-6.12.y/arch/arm64/boot/dts/broadcom/bcm2712-ds.dtsi) (cpu-crit 110 °C), [bcm2712-rpi-5-b.dts](https://github.com/raspberrypi/linux/blob/rpi-6.12.y/arch/arm64/boot/dts/broadcom/bcm2712-rpi-5-b.dts) (fan trips)
- White paper RP-010139-WP-1, *Use-case-specific thermal performance of Raspberry Pi SBCs* (1 Jun 2026) — [PDF](https://pip-assets.raspberrypi.com/categories/685-app-notes-guides-whitepapers/documents/RP-010139-WP-1-Use-case-specific%20thermal%20performance%20of%20Raspberry%20Pi%20SBCs.pdf)
- White paper RP-003608-WP-3, *Cooling a Raspberry Pi device* (13 Feb 2026) — [PDF](https://pip-assets.raspberrypi.com/categories/685-app-notes-guides-whitepapers/documents/RP-003608-WP-3-Cooling%20a%20Raspberry%20Pi%20device.pdf)
- Raspberry Pi 5 product brief (Oct 2026; operating temperature 0–70 °C) — [PDF](https://datasheets.raspberrypi.com/rpi5/raspberry-pi-5-product-brief.pdf)
- Active Cooler product page (1.09 CFM) — <https://www.raspberrypi.com/products/active-cooler/>

Primary, Raspberry Pi staff:
- A. Allan, *Heating and cooling Raspberry Pi 5* (4 Oct 2023) and comments — <https://www.raspberrypi.com/news/heating-and-cooling-raspberry-pi-5/>
- jdb (Raspberry Pi Engineer), soft throttle 80 °C at 100 MHz/°C, then idle frequency at 85 °C — <https://forums.raspberrypi.com/viewtopic.php?t=368073>
- dom (Raspberry Pi Engineer), throttling is firmware-side; kernel trips only reboot — <https://forums.raspberrypi.com/viewtopic.php?t=388927>
- MagPi / G. Halfacree, *Thermal testing Raspberry Pi 4* (28 Nov 2019; Pi 4 orientation data) — <https://www.raspberrypi.com/news/thermal-testing-raspberry-pi-4/>

Measured, named tests (secondary to the chip vendor, but reproducible):
- Tom's Hardware, Piltch & Pounder, *Raspberry Pi 5 Review* (23 Oct 2023) — <https://www.tomshardware.com/reviews/raspberry-pi-5>
- Core Electronics, *Answering your Pi 5 questions* (updated 1 Nov 2023) — <https://core-electronics.com.au/guides/answering-your-pi-5-questions-cooling-overclocking-power-and-more/>
- R. Longbottom, *Raspberry Pi 5 Benchmarks and Stress Tests* (forum, Jan 2024; not staff; cooling config of the no-fan runs not stated) — <https://forums.raspberrypi.com/viewtopic.php?t=363656>
- J. Geerling, *New 2GB Pi 5 has 33% smaller die, 30% idle power savings* (29 Aug 2024) — <https://www.jeffgeerling.com/blog/2024/new-2gb-pi-5-has-33-smaller-die-30-idle-power-savings/>

Papers:
- Marasini & Zhou, *Sustainable Edge Vision via Empirically Calibrated DVFS…*, arXiv:2609.04705v1 (4 Sep 2026) — <https://arxiv.org/html/2609.04705>
- Ranjan et al., *Does YOLO26 Truly Offer Advantages…? Aquaculture*, arXiv:2607.09835 — <https://arxiv.org/pdf/2607.09835>
- Vdoviak, Sledevič & Matuzevičius, *Accelerator-Free Thermal Pose Estimation… on the Raspberry Pi 5*, Sensors 26(16):5188 (Aug 2026) — <https://pmc.ncbi.nlm.nih.gov/articles/PMC13517955/>
- Ghoniya et al., IJERT 15(3) (Mar 2026; low-tier, internally inconsistent) — <https://www.ijert.org/comparative-benchmarking-of-cpu-gpu-and-npu-architectures-for-real-time-yolov8n-inference-on-embedded-edge-platforms-ijertv15is030232>
- Benoit-Cattin, Velasco-Montero & Fernández-Berni, *Impact of Thermal Throttling on Long-Term Visual Inference in a CPU-based Edge Device*, arXiv:2010.06291 (Pi 4) — <https://arxiv.org/abs/2010.06291>
- De Vogeleer, Memmi, Jouvelot & Coelho, *Modeling the Temperature Bias of Power Consumption for Nanometer-Scale CPUs in Application Processors*, arXiv:1404.3381 — <https://arxiv.org/abs/1404.3381>
- De Vogeleer et al., *The Energy/Frequency Convexity Rule*, arXiv:1401.4655 — <https://arxiv.org/abs/1401.4655>
- Gupta, Dutt, Kurdahi, Khouri & Abadir (UC Irvine / Freescale), *LeAF: A Leakage-Aware Floorplanner*, ASP-DAC 2007 slides — <https://www.aspdac.com/aspdac2007/pdf/archive/3B-1.pdf>

Searched and found nothing:
- a measured desk-fan or vertical-orientation test on a bare Pi 5;
- a clean leakage-vs-temperature measurement on BCM2712 or any Cortex-A76 SoC;
- 2-thread inference temperatures on a Pi 5;
- Jeff Geerling bare-Pi 5 sustained-load numbers (his posts cover the Active Cooler, overclocking, and a 10-s bare-die sysbench run only);
- Gough Lui Pi 5 thermal tests;
- an official statement of which `get_throttled` bit fires at the 85 °C hard limit;
- a way to clear the sticky bits without rebooting.
