# Findings: Raspberry Pi 5 power telemetry (PMIC) vs an inline USB-C meter

Ticket: `issues/06-power-measurement-capability.md`
Researched: 2026-10-19
Method: primary sources first. These are the Raspberry Pi docs repo, the `raspberrypi/utils` man page, Raspberry Pi white papers, the `rpi-eeprom` release notes, Raspberry Pi blog posts with staff comments, Pi forum threads, meter manuals and maker pages, and source code on GitHub. Forum posts are labelled with the poster's standing. Other secondary sources are labelled where they are used.

**The ticket's premise is wrong.** It says that "vcgencmd reports voltage and temperature but not draw". On a Pi 5, `vcgencmd pmic_read_adc` reports a **current and a voltage for 12 PMIC output rails**, which gives on-board power as the sum of V×I. What it does *not* report is the current on the 5V input. So it measures the PMIC's output rails, not the power drawn at the wall or the USB-C input. That is still a real measurement, not a "TDP guess".

---

## 1. What `pmic_read_adc` reports

**Per-rail current: yes.** The official man page says: *"On Raspberry Pi 5, displays the readings of the ADCs in the PMIC: the current drawn from and the voltage of each PMIC supply rail, plus the external 5V supply voltage (EXT5V_V)."* It goes on: *"The readings do not include USB current or anything else powered directly from the 5V supply, because that power does not pass through the PMIC."* ([raspberrypi/utils `vcgencmd/vcgencmd.1`](https://github.com/raspberrypi/utils/blob/master/vcgencmd/vcgencmd.1). An outside contributor added this text in [PR #220](https://github.com/raspberrypi/utils/pull/220), and Phil Elwell of Raspberry Pi committed it on 2026-10-14.)

There are 26 channels: 12 currents and 14 voltages. The PMIC is the Renesas/Dialog DA9091.

| Rail | Current (`_A`) | Voltage (`_V`) |
|---|---|---|
| `3V7_WL_SW` (Wi-Fi) | yes | yes |
| `3V3_SYS` | yes | yes |
| `1V8_SYS` | yes | yes |
| `DDR_VDD2` | yes | yes |
| `DDR_VDDQ` | yes | yes |
| `1V1_SYS` | yes | yes |
| `0V8_SW` | yes | yes |
| `VDD_CORE` (SoC core, the dominant load) | yes | yes |
| `3V3_DAC` | yes | yes |
| `3V3_ADC` | yes | yes |
| `0V8_AON` | yes | yes |
| `HDMI` | yes | yes (≈5.07 V) |
| **`EXT5V`** (5V input) | **NO** | yes |
| `BATT` (RTC cell) | no | yes |

Sources for the channel list: real output in the [jfikar/RPi5-power README](https://github.com/jfikar/RPi5-power), and the counts captured on a Pi 5 running firmware 2025/03/19 in [raspberrypi/documentation PR #4330](https://github.com/raspberrypi/documentation/pull/4330). That PR was closed without being merged, so the main docs site still has no `pmic_read_adc` entry in its vcgencmd section. A single channel can be read with `vcgencmd pmic_read_adc EXT5V_V`. An unrecognised channel name prints all 26 and still exits 0.

**EXT5V gives voltage only.** The official docs say: *"The PMIC has built-in ADCs that, among other things, can measure the supply voltage `EXT5V_V` … You can't see USB current or anything else connected directly to 5V, because this bypasses the PMIC. You should not expect this to add up to the wattage of the source power supply."* ([power-supplies.adoc, "Power supplies and Raspberry Pi OS"](https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/raspberry-pi/power-supplies.adoc)). Forum regulars say the same: rpdom wrote *"The one it doesn't have is the main 5V supply current as it has no way of measuring that"* ([t=359249](https://forums.raspberrypi.com/viewtopic.php?t=359249)), and bensimmo wrote *"No you don't get current for the 5V USB-C input"* ([t=367244](https://forums.raspberrypi.com/viewtopic.php?t=367244)). Neither of them is Raspberry Pi staff.

**What the readings do not cover:**
- **USB port VBUS.** Up to 1.6 A in total. Official docs.
- **Anything else fed directly from 5V.** That includes the GPIO 5V pins, HATs, and M.2/NVMe boards. The docs give the general rule, and jfikar names HATs and NVMe.
- **The fan header.** The docs rate the case fan at *"60 mA at 5 V (300 mW)"*, so it is a 5V load and therefore outside the readings. *This is an inference from the 5V rule. No source states it about the fan directly.*
- **PMIC conversion losses.** This covers the efficiency of the 5V-to-rail converters. Forum regular trejan: *"pmic_read_adc … doesn't include any conversion losses"* ([t=359249](https://forums.raspberrypi.com/viewtopic.php?t=359249)).
- **Losses before the PMIC.** These are in the USB-C connector, the input protection and the cable. *This is an inference.*
- **3V3 is covered**, contrary to the ticket's worry. `3V3_SYS`, `3V3_DAC` and `3V3_ADC` are all metered. Which components sit on which rail is **undocumented**. A forum request asking for exactly that mapping got no answer ([t=365540](https://forums.raspberrypi.com/viewtopic.php?t=365540)).

**A common mistake: each current is measured at its own rail's voltage.** `VDD_CORE_A` = 6 A at about 0.72 V is about 4.3 W, not 30 W. Forum users who multiplied these currents by 5 V got "5–6× too high" values ([t=390109](https://forums.raspberrypi.com/viewtopic.php?t=390109), [t=367074](https://forums.raspberrypi.com/viewtopic.php?t=367074)). Always compute Σ V_i·I_i.

## 2. PMIC sum compared with total board power

**There is one published load sweep, from a hobbyist rather than Raspberry Pi.** jfikar measured about 30 operating points between roughly 1.5 and 15 W of PMIC sum, against a USB-C inline meter whose model is not named. The fitted line is:

> `real_consumption [W] = pmic_output [W] × 1.1451 + 0.5879`

That is a **+0.59 W offset with a 14.5 % scale error**. ([README, script and plot.png](https://github.com/jfikar/RPi5-power), 2024-04/05). The script samples once a second for 50 s, averages the samples, and applies this fit.

- Reading his plot by eye (approximate): the scatter around the line is about ±0.2 W below 6 W of PMIC sum, and grows to about ±0.7–0.8 W around 8–15 W.
- What the fit implies: the PMIC sum reads about **31 % low at 2 W**, **21 % low at 5 W** and **18 % low at 8 W**. A shortfall that shrinks with load and has a fixed offset fits the pattern of roughly 87 % conversion efficiency plus about 0.6 W of 5V-side load. *That interpretation is my inference.*

**Other independent checks (forum, idle only):**

| Who | Meter | Meter reading | PMIC sum | jfikar fit predicts |
|---|---|---|---|---|
| hqcamuser, 2024-04 ([t=359249](https://forums.raspberrypi.com/viewtopic.php?t=359249)) | POWER-Z KM003C | 2.9–3.1 W | 2.05–2.3 W | 2.94–3.22 W (agrees) |
| cleverca22, 2023-12 ([t=360658](https://forums.raspberrypi.com/viewtopic.php?t=360658)) | not named | 3.4 W | 1.94 W | 2.81 W (0.6 W low; peripherals unknown) |

So the error is **affine (an offset plus a scale), and stable on one board.** The offset depends on the setup, because it moves with whatever hangs off 5V. One board's fit therefore works as a calibration for *that* board and peripheral set. jfikar's own coefficients cannot be transplanted to our board without checking them.

**Is the PMIC current measurement calibrated? Nobody says.**
- The DA9091 datasheet is not public. Raspberry Pi engineer jamesh: *"There are no plans to release this datasheet AFAIK"* ([t=359073](https://forums.raspberrypi.com/viewtopic.php?t=359073)).
- The only firmware-side evidence is the 2023-09-28 bootloader note *"vcgencmd pmic_read_adcs fixes — Fix the LDO names and current scaling codes"* ([rpi-eeprom firmware-2712 release notes](https://github.com/raspberrypi/rpi-eeprom/blob/master/firmware-2712/release-notes.md)). That shows scaling constants exist. It does not show that each board is calibrated.
- One 2026 arXiv paper states "±10 % per the PMIC datasheet" and "~1 Hz bounded by vcgencmd" ([arXiv 2609.11932](https://arxiv.org/html/2609.11932v1), §6.2). **Neither claim is cited, and the datasheet is not public. Do not repeat them.**
- The same white paper that documents the Pi 5's PMIC extras warns *"no guarantee that this functionality will be maintained in future versions of the PMIC"* ([RP-004340-WP](https://pip.raspberrypi.com/documents/RP-004340-WP)). That paper covers `measure_temp pmic` and the CC ADCs, not `pmic_read_adc`.

**Why the offset matters for energy-per-frame rankings** (worked example using jfikar's fit):
- Model A draws 6 W of PMIC sum at 10 FPS, which is 0.600 J/frame on the PMIC sum.
- Model B draws 2.9 W at 4.9 FPS, which is 0.592 J/frame. **B wins on the PMIC sum.**
- Corrected to board power, A is 0.746 J/frame and B is 0.798 J/frame. **A wins.**

Close rankings can flip, because the offset penalises slower models. Clear differences survive.

## 3. Practicalities

- **Firmware and OS:**
  - The command has existed since before the Pi 5 launched (the release note above predates retail units).
  - On a Pi 5 the "start.elf" firmware lives inside the bootloader EEPROM (*"Effectively, the bootloader has an embedded version of `start.elf`"*, [bootflow-eeprom.adoc](https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/raspberry-pi/bootflow-eeprom.adoc)). So any retail Pi 5 has it, and `rpi-eeprom-update` keeps it current.
  - No later release note changes the ADC readings.
  - The kernel `raspberrypi-hwmon` driver exposes **voltages only** (`in0..in3`), with no currents ([kernel docs v7.2](https://kernel.org/doc/html/v7.2/_sources/hwmon/raspberrypi-hwmon.rst.txt)). `vcgencmd` or the mailbox is the only way to read currents.
- **How it works:** `vcgencmd` sends a single mailbox property request (`GET_GENCMD_RESULT` 0x00030080) through `/dev/vcio` ([utils commit](https://github.com/raspberrypi/utils/commit/b9fda0c4122e480995ca23c064a342752c227dd7): *"cut down to only support the mailbox interface"*). The firmware owns the PMIC over I2C (PhilE, Raspberry Pi engineer: *"the firmware needs full control of the DA9091"*, [t=359073](https://forums.raspberrypi.com/viewtopic.php?t=359073)).
- **Polling rate: UNVERIFIED.** No primary source gives the time per call or a maximum rate. Every script found polls at 1 Hz or slower. Measure it on our Pi with `time (for i in $(seq 200); do vcgencmd pmic_read_adc >/dev/null; done)`. Reading only the channels you need may be cheaper, but that is untested.
- **Readings are noisy and unaveraged.** jfikar: *"readings tend to fluctuate a lot"*, hence his 50 s averages. cleverca22, a forum user: `3V7_WL_SW` *"can spike at up to 800mW"*, and they suspect *"the PMIC isnt doing any averaging"* ([t=360658](https://forums.raspberrypi.com/viewtopic.php?t=360658)). Average over the whole timed run. Keep Wi-Fi off, or in the same state for every run.
- **Does polling slow the CPU? UNVERIFIED for `pmic_read_adc` itself.**
  - The ARM-side cost is a fork/exec plus an ioctl. The ADC work is done by the VideoCore firmware. *That second point is an inference.*
  - The closest published number is a Pi 5 running YOLOv8n under OpenVINO with a 2 Hz telemetry process (vcgencmd plus psutil). It cost **1.90 % FPS (1.33 ms/frame)**, paired n=3 ([arXiv 2609.04705](https://arxiv.org/pdf/2609.04705), Table I).
  - To keep the cost down, use one long-lived sampler that issues the ioctl directly instead of spawning `vcgencmd`, then A/B the FPS with and without the sampler.
- **Existing tools that total the rails:**
  - [jfikar/RPi5-power](https://github.com/jfikar/RPi5-power): shell script, 1 Hz × 50 s, applies his fit.
  - [bcrisp4/pi5_exporter](https://pkg.go.dev/github.com/bcrisp4/pi5_exporter): Go, uses `/dev/vcio` directly. `pi5_pmic_measured_power_watts` is documented as the *"Sum of measured rails (not total board input power)"*. Default interval is 15 s.
  - [glossyio/traffic-monitor `rpi5-nodered-telemetry.py`](https://github.com/glossyio/traffic-monitor/blob/main/utils/rpi5-nodered-telemetry.py): a Python example of the ioctl approach that avoids spawning a process.
  - [librerpi/rpi-tools `pi5_voltage.py`](https://github.com/librerpi/rpi-tools/blob/master/pi5_voltage.py): Prometheus exporter, per rail, no total.
  - hallmanan's per-rail Python script, posted inline in [t=359249](https://forums.raspberrypi.com/viewtopic.php?t=359249).

## 4. Inline USB-C meters on the Pi 5

**What the PSU offers and what the Pi asks for.** The official 27 W PSU offers PDOs *"5V 3A, 5V 5A, 9V 3A, 12V 2.25A, and 15V 1.8A"*. The Pi requests only 5V PDOs and uses 5V/5A if it is offered. Otherwise it *"will revert to 3A"* ([RP-009856-WP, "USB Power Delivery on Raspberry Pi 5", 2026-03-05](https://pip-assets.raspberrypi.com/categories/685-app-notes-guides-whitepapers/documents/RP-009856-WP-1-USB%20Power%20delivery%20on%20Raspberry%20Pi%205.pdf)).

**What changes in 3A mode** (same white paper, plus [power-supplies.adoc](https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/raspberry-pi/power-supplies.adoc)):
- The USB ports are limited to **600 mA in total**, instead of 1.6 A.
- **USB boot is disabled.**
- The desktop shows a "power is limited" warning.
- The white paper says the Pi *"will function perfectly well with a 3A supply, provided that only lower-power peripherals are attached."*

**No source documents any CPU clock or voltage limit tied to the PSU mode.** The launch post explains the USB cap as protecting the *CPU's* margin: *"we must limit downstream USB current to 600mA to ensure that we have sufficient margin to support these workloads"*. Peak is *"around 12W"* for "power virus" loads ([Introducing Raspberry Pi 5](https://www.raspberrypi.com/news/introducing-raspberry-pi-5/)). For a headless CPU-inference benchmark with no USB load, 3A mode should not affect results. *This is inferred; it has not been A/B tested.*

The overrides are `usb_max_current_enable=1` in `config.txt`, or `PSU_MAX_CURRENT=5000` in the EEPROM config, which skips PD negotiation ([eeprom-bootloader.adoc](https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/raspberry-pi/eeprom-bootloader.adoc)).

**Checking whether 5A was negotiated through a meter:** run `vcgencmd get_config usb_max_current_enable`, which is non-zero after a 5 A negotiation (docs). Or read `/proc/device-tree/chosen/power/max_current` (mA, a big-endian u32), for example `printf '%d\n' 0x$(xxd -p /proc/device-tree/chosen/power/max_current)`. *I wrote that command; it has not been tested.*

**Does each meter pass PD through so the Pi still sees 5 A? Not verified for any meter.** No source tests the Pi 5's 5 A negotiation through a specific meter.

| Meter | PD path to the Pi | Range | PC logging | Fit for this study |
|---|---|---|---|---|
| **UM25C** (Joy-IT manual) | USB-C in and out, *"only VBUS, GND, CC1, CC2"*, so CC (and therefore PD) should pass. Not PD-aware. Through USB-A ports there is no PD, so the Pi falls back to 3A. | 4–24 V, **0–5 A** | Bluetooth to "UM25C PC software V1.3", **2 Hz** update rate, Excel export ([manual](https://www.joy-it.net/files/files/Produkte/JT-UM25C/JT-UM25C_Manual_2025-07-08.pdf)) | Poor: 2 Hz, Bluetooth, no PD decoding |
| **FNIRSI FNB48 / 48S / 48P** | Has a PD trigger. Pass-through not documented. | FNB48S: 4–24 V, 6.5 A (Robu listing) | FNB48 and FNB48S work with the open-source logger below. FNB48P is not listed there. | Acceptable; check the exact variant |
| **FNIRSI FNB58** | Has a "PD Communication" switch. The manual says to keep it **OFF** except when using the PD trigger, listener or E-Marker functions, and warns that leaving it ON can trigger high voltage ([community transcription of the manual](https://github.com/Shadester/fnb58manual/blob/master/FNIRSI_FNB58_Complete_Manual.md)). Gough Lui describes it as a switch that isolates PD communication (secondary). | 4–28 V, 0–7 A ([fnirsi.com](https://www.fnirsi.com/products/fnb58)) | Maker page: *"2 sps–100 sps"*. The open-source [baryluk/fnirsi-usb-power-data-logger](https://github.com/baryluk/fnirsi-usb-power-data-logger) streams **100 samples/s** over USB, timestamped in UNIX epoch with 1 ms resolution (Linux). The vendor app saves a binary `.cfn` file (Gough Lui, secondary). | **Good budget pick** |
| **ChargerLAB POWER-Z KM003C** | Built as a PD analyzer. Was used inline between the official 27 W PSU and a Pi 5 in a published YOLO/OpenVINO energy study, but **that paper does not report the PD outcome** ([arXiv 2609.04705](https://arxiv.org/pdf/2609.04705) §IV-A/D). | 0–50 V, about 6 A (retailers; maker spec page unreachable) | Vendor PC app at **1 / 10 / 50 / 1000 SPS**, CSV or SQLite export ([ChargerLAB tutorial](https://www.chargerlab.com/how-to-use-chargerlab-power-z-km003c-tester-to-record-charging-data-offline/)). Whether the export has timestamps is unverified. | **Best pick** |

**Secondary figures for the FNB58** (Gough Lui, [review](https://goughlui.com/2023/04/29/review-tested-fnirsi-fnb58-usb-fast-charge-tester/)), compared against a Keithley 2450 and an R&S NGM202:
- Voltage reads about 0.06 % high.
- Current reads about 0.07 % low, within ±1.1 mA.
- Burden is about 20 mΩ, which drops about 61 mV at 3 A.
- The meter itself draws 54–64 mA.

At our roughly 1–2 A this is negligible next to the PMIC's 15–30 % shortfall.

**Practical note:** log the meter on the laptop, not on the Pi, so the Pi carries no extra USB load or CPU work. Then align the two clocks. The arXiv 2609.04705 authors reported *"~30–60 s of manual drift in Power-Z start/stop timing"* and kept only samples above 4 W of board power to cut out idle time.

## 5. Indian prices (approximate, from search snapshots on 2026-10-19)

- **FNIRSI FNB58:** about **₹4,900–5,800**.
  - Toolworld: ₹4,900 including GST without Bluetooth, ₹5,800 with Bluetooth.
  - Flipkart FNB58-WBT: about ₹5,040.
  - IndiaMART: ₹5,200–5,990 excluding tax.
  - Robu.in shows no FNB58 listing. It lists only the FNB48S, and no price could be read.
- **POWER-Z KM003C:** about **₹8,400–12,000** on Amazon.in ([listing](https://www.amazon.in/ChargerLAB-KM003C-Portable-Charging-Voltmeter/dp/B0BJ24PVNJ), price varies by seller). IndiaMART lists it at ₹9,800 excluding tax. ChargerLAB warns about counterfeits from third-party sellers.
- **The ticket's ₹1,500–3,000 budget is too low** for a logging meter that understands PD. A UM25C fits that price but logs at only 2 Hz.

---

## Verdict for the ticket

**A hardware meter is OPTIONAL, used for cross-calibration. It is not required for every run.** The ticket's premise and its fallback ("drop Pi energy metrics") should both go.

1. **Primary instrument: the PMIC rail sum (free, on-board).** Log Σ V_i·I_i from `pmic_read_adc` throughout each timed run, averaged over the whole run. Report it under its real name: *"PMIC-measured on-board rail energy per frame (excludes 5V-direct loads and PMIC conversion losses)."* It is a measurement. It is fine for comparing models and backends on the same board when the differences are clear.
2. **For absolute or wall-referenced J/frame, and for close rankings,** borrow a meter (KM003C, or an FNB58 at about ₹5k) **for one session**. Do a load sweep covering idle, 1–4 core stress, and each model/backend. Fit a board-specific `P_board = a·P_pmic + b`, then report both numbers. jfikar's (a=1.145, b=0.59 W) shows what to expect, but it is not our calibration.
3. If no meter can be found, report PMIC-domain energy only, with that definition stated as a limitation. Cite jfikar's fit only as an outside indication of the gap (+15 % and +0.6 W).
4. Whatever is chosen, measure and report the sampler's own cost: FPS with and without the sampler.

**Not verified:**
- Whether any named meter lets the Pi 5 negotiate 5 V/5 A. Check `usb_max_current_enable` with the meter in line.
- Polling latency and maximum rate of `pmic_read_adc`.
- The CPU cost of `pmic_read_adc` specifically.
- Whether the DA9091 current ADC is calibrated per board, and its accuracy (no public datasheet).
- Whether 3A mode has any effect at all on CPU performance (no source says it does).
- The KM003C's full maker spec, and whether its CSV export has timestamps.
- Whether the community FNB58 manual matches FNIRSI's official one.
- All prices.
