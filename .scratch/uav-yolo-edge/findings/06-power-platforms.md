# Findings: software power telemetry on the Galaxy M35, the Jetsons and the Ryzen laptop

Ticket: `issues/06-power-measurement-capability.md`
Researched: 2026-10-19. The laptop facts in §C come from the study laptop itself (HP Victus 15-fb2xxx, Ryzen 7 8845HS,
Windows 11 build 26300). They were read from a non-elevated shell on AC power.

**Verdict in one line:** all three platforms can produce energy per frame from software alone, but each one measures a
different boundary:

- the phone measures the whole handset, from the battery;
- the Jetson measures the module input only;
- the laptop measures the SoC package only, and that figure is modelled rather than measured.

So report each number together with its boundary. Do not rank the platforms against each other on raw joules per frame.

---

## A. Samsung Galaxy M35 (Exynos 1380)

> **Correction, 2026-10-19: the study phone is a Galaxy M34 5G, not an M35.** adb reports `SM-M346B`, product `m34xnsxx`,
> SoC `s5e8825` (Exynos 1280), Android 16. The dev confirmed that "M35" on the map was a mistake. The kernel-source
> analysis below is for the M35 and does **not** carry over. What was observed on the actual M34:
>
> - `adb shell cmd battery get -f current_now` works for the shell user. The `-f` flag forces a fresh read, so no app is needed for spot checks.
>   `/sys/class/power_supply/` is permission-denied to shell, and `dumpsys battery` shows voltage but no current.
> - **The unit is mA, positive while charging.** Over USB the readings were 276–946 (with one transient −27). In µA these would be sub-mA, which is implausible.
>   `current_average` = 738 alongside `current_now` = 725.
> - **The value changed on every read** when polled about every 0.37 s for 15 s (40 of 40 reads). The gauge therefore refreshes at least
>   that often; its true period is still unmeasured.
> - **Discharge under load was NOT verified.** Two attempts failed:
>   - an `adb shell` script started with `setsid nohup` was killed the moment the USB cable came out;
>   - Samsung reset `adb tcpip` mode on unplug, and Android Wireless-debugging ports were unreachable over the campus Wi-Fi ("Amrita"), although ping worked.
>
>   So on this phone the harness must be an **app** that holds a `PARTIAL_WAKE_LOCK`, logs to a file on the device, and is
>   pulled over USB after the run. Shell-driven measurement across an unplug does not work here.

### A1. `CURRENT_NOW` / `CURRENT_AVERAGE`: units, sign and update rate

**The API contract.** Source: AOSP `frameworks/base/core/java/android/os/BatteryManager.java`.

- `CURRENT_NOW`: *"Instantaneous battery current in microamperes, as an integer. Positive values indicate net current
  entering the battery from a charge source, negative values indicate net current discharging from the battery."*
- `CURRENT_AVERAGE` uses the same units and sign. *"The time period over which the average is computed may depend on the
  fuel gauge hardware and its configuration."*
- If a property is unsupported, `getIntProperty` returns `Integer.MIN_VALUE` (for targetSdk ≥ 28) or `0` (for older targets).
- The kernel ABI (`Documentation/ABI/testing/sysfs-class-power`) agrees. Both `current_now` and `current_avg` are
  *"Represented in microamps. Negative values are used for discharging batteries"*.

**The platform does no rescaling.** AOSP `system/core/healthd/BatteryMonitor.cpp` `getProperty()` reads the sysfs
`current_now` node on every call and passes the integer straight through. The unit is therefore whatever Samsung's kernel
driver writes.

**The unit depends on the Samsung firmware.** Both kernels below use the same common driver,
`drivers/battery/common/sec_battery.c`, with an SM5714 fuel gauge. The M35 defconfig sets `CONFIG_FUELGAUGE_SM5714=m`.

| Kernel source | `POWER_SUPPLY_PROP_CURRENT_NOW` asks the gauge for | Unit Android sees |
|---|---|---|
| Galaxy A54 (same SoC), Samsung OSS for firmware `A546BXXU6BXA2`, kernel 5.15.104 | `SEC_BATTERY_CURRENT_MA`, passed through | **mA**, despite the API saying µA. The same function explicitly converts voltage to µV, so this is a real mismatch. |
| Galaxy M35 `m35x` tree, kernel 5.15.153 (third-party mirror of Samsung OSS, exact firmware build unknown) | `SEC_BATTERY_CURRENT_UA` | **µA**, but with **1 mA resolution**. The SM5714 driver computes `((raw&0x7fff)*1000)/2044` as an integer in mA (one step is about 0.49 mA), then multiplies by 1000. |

- A secondary, anecdotal source agrees: a user on Samsung's developer forum reported mA readings from a Galaxy Z Fold3 on
  Android 12.
- **The sign** follows the contract in source: the SM5714 driver negates the value when the gauge's discharge bit is set.
- **`CURRENT_AVERAGE` on the M35 is stale.** The driver returns the gauge's own averaging register as it was cached at
  the last `sm5714_get_curr()` call. That function runs on each `CURRENT_NOW` read and in the periodic battery monitor. The
  averaging window is undocumented. **Poll `CURRENT_NOW` and ignore `CURRENT_AVERAGE`.**
- **The update period was not verified.** Each `CURRENT_NOW` read triggers a fresh I²C read of the gauge, so Android does
  not cap the polling rate. The SM5714's internal conversion period is undocumented, though, and its datasheet is not
  public. For scale, AOSP lists other gauges at 175.8 ms (MAX17050, Nexus 6/9) up to 3.5 s (Nexus 10) for `CURRENT_NOW`.
  To measure it on the device, poll every 50 ms for 30 s and count how often the value actually changes.
- **`CHARGE_COUNTER` on the M35** is `Capacity × raw_SoC` with SoC in steps of 0.1 %. That makes it far too coarse for a
  one-minute run, which is about 6 mAh per step on a 6000 mAh pack (capacity not checked).
- **Where voltage comes from.** For P = V·I, take `EXTRA_VOLTAGE` (mV) from the sticky `ACTION_BATTERY_CHANGED` broadcast.
  See the trap with `dumpsys battery unplug` in §A3.

### A2. Perfetto / ODPM power rails: Pixel-class hardware only, and none on the M35

What the first-party docs say:

- **Perfetto** (`docs/data-sources/battery-counters.md`): battery counters are *"available on most Google Pixel
  smartphones"*. ODPM *"requires the dedicated hardware on the device. This hardware is not yet available on most
  production phones."*
- **AOSP "Power stats HAL"**: the rail APIs are an *"optional feature"*. Without power-monitoring hardware,
  `getRailInfo()`, `getEnergyData()` and `streamEnergyData()` *"return NOT_SUPPORTED"*.

What the M35 kernel source shows:

- The PMICs are S2MPU15 and S2MPU16. The only metering driver built for them is a general-purpose 12-bit ADC
  (`CONFIG_S2MPU15_ADC_8835=m`), which exposes IIO voltage channels only.
- The tree does contain PMIC power-meter drivers, but only for the flagship S2MPS25/26 (`s2mps2x_powermeter_9935.c`).
- `s5e8835-m35xxx_defconfig` has **zero** matches for `POWERMETER`, `PMETER` or `ODPM`.

**Conclusion:** expect no rails on the M35. Confidence is high, but this has not been checked on the device. To confirm,
run `adb shell dumpsys powerstats`: the *"available Channels"* section will be empty if the phone has no energy meters. The
Perfetto `android.power` battery counters still work, but they are the same fuel-gauge data as §A1, with the same caveats.

### A3. Method for an unrooted phone

1. **Physically unplug USB.**
   - `dumpsys battery unplug` only fakes the framework's plug state (AOSP `BatteryService.java`: *"Force battery
     unplugged, freezing battery state"*). Charge current keeps flowing.
   - Worse, it **freezes the `ACTION_BATTERY_CHANGED` values**, including voltage and level, until `dumpsys battery
     reset`. A logger that takes V from that broadcast would record a constant.
   - The command that really stops charging is `suspend_input`, and it is only offered when `Build.IS_DEBUGGABLE`. A retail
     M35 is not debuggable.
   - Perfetto's docs say the same: while plugged in you see charging current, and the workaround is a hub that cuts VBUS.
2. **Use adb over Wi-Fi.** On Android 11+, wireless debugging pairs without any USB cable (developer.android.com
   `tools/adb`). Better still, have the app log to a file, and pull the file after the run.
3. **Do not let Doze interfere.**
   - Option 1: keep the screen on at a fixed brightness and subtract an idle baseline taken at the same brightness.
     Almeida et al. (IMC 2021) kept screens on for exactly this reason: to stop Doze killing jobs.
   - Option 2: run screen-off with `adb shell dumpsys deviceidle disable` (an AOSP shell command), plus a foreground
     service.
   - Either way, fix the screen state across all models.
4. **Choose the run length.** These are first-party numbers for comparison, not M35 values:
   - AOSP recommends measuring above 20 mA, and says gauge offset dominates small currents.
   - For the MAX17050, ±100 µA needs about a 1-minute window, and a <100 µA 3-σ result needs about 6 minutes.
   - MLPerf Power requires *"a minimum of 60 seconds of power data"*, and loops shorter workloads.
   - YOLO on the M35 should draw hundreds of mA, so the 1 mA quantisation is below 1 %. The real risks are the unknown
     gauge period and aliasing against the load.
   - **Rule: at least 60 s of steady-state window after warm-up, at least 3 repeats, and the same starting state-of-charge
     band and temperature.** AOSP: *"Ensure the starting battery charge is the same for each device."*
5. **Compute** E/frame = Σ V·|I|·Δt / N_frames. Report it both gross and net of idle.

**Published methodology (mobile DNN energy):**

- **MLPerf Power** (Tschand et al., arXiv 2410.12032, HPCA 2025): *"we currently do not support the measurement of power of
  mobile systems in MLPerf Power due to the complexity and variability of smartphones."* So MLPerf Mobile is
  performance-only.
- **AI Benchmark** (Ignatov et al., arXiv 1810.01109 and 1910.06663): a full-text check found **no power measurement**.
  Power appears only as motivation and as vendor TOPS/W figures.
- **Almeida et al., "Smart at what cost?"** (IMC 2021, arXiv 2109.13963): a Monsoon AAA10F on open-deck Qualcomm HDK
  boards, not on retail phones. That remains the gold standard, which this study does not have.
- **AOSP "Measure device power"**: the first-party fuel-gauge method. It assumes same-device A/B comparisons and warns
  *"measurements are not guaranteed to be of high absolute accuracy"*.

---

## B. NVIDIA Jetson

### B1. Original Jetson Nano (JetPack 4.6, L4T R32.7)

- **What NVIDIA documents** (L4T R32.7.5 "Power Management for Jetson Nano and Jetson TX1"):
  - a three-channel INA3221 at I²C 0x40;
  - sysfs at `/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/`;
  - nodes `rail_name_<N>`, `in_current<N>_input` (mA), `in_voltage<N>_input` (mV) and `in_power<N>_input` (mW);
  - "Jetson Nano 2GB does not have an INA3221 power monitor."
- **The channel order differs between doc and device tree.** The doc's table gives channel 0 = VDD_IN *"Main module power
  input"*, 1 = VDD_GPU, 2 = VDD_CPU. The shipped DT source says otherwise:
  - `tegra210-porg-powermon-p3448-0000-a00.dtsi`, included by both the a02 and b00 devkit DTs, defines `channel@0` =
    `POM_5V_GPU`, `channel@1` = `POM_5V_IN` and `channel@2` = `POM_5V_CPU`, all with 5 mΩ shunts. tegrastats prints these
    DT names.
  - So **`in_power0_input` is probably the GPU rail, not module input.** Always map channels through `rail_name_N`.
- **Root is needed.**
  - In the L4T R32 kernel (OE4T mirror `linux-tegra-4.9`, branch r32.7.4), the IIO core creates channel attributes with
    `S_IRUSR` (0400).
  - The INA driver's own attributes (`rail_name_*`, the limits) are `S_IRUSR|S_IWUSR`.
  - So a non-root reader gets EACCES unless a udev rule relaxes the permissions. Whether a non-root `tegrastats` still
    prints POM fields was **not verified**. Use `sudo tegrastats`, or run the logger as root.
- **How often a fresh value appears.** The DT sets `ti,enable-forced-continuous` and `ti,continuous-config = <0x7607>`:
  3 channels, a 64-sample average, 140 µs bus and 140 µs shunt conversions, continuous mode. By the INA3221 formula
  (C × S × (BC + SC), from the kernel `hwmon/ina3221.rst`), that is **one fresh value about every 54 ms**.
  - tegrastats `--interval` is in ms with a default of 1000. **No minimum is documented** in either the R32.7 or the r36.4
    docs. Sampling faster than about 54 ms only returns repeats.
- **Quantisation:** the shunt LSB is 40 µV, giving 8 mA on a 5 mΩ shunt, which is about **40 mW per step at 5 V**.
- **What `POM_5V_IN` leaves out.** It is the module's input. Carrier-board consumers are outside it: the USB ports and a
  USB camera, Ethernet, the fan, and M.2. This is inferred from the "main module power input" wording. Separate
  carrier-board rails are documented only for the TX1 carrier.
- **The board-level gap.** Shalavi et al. (below) fitted external whole-board power against the internal sensor on a Nano
  as P_ext = 1.11·P_int + 232.6 mW.

### B2. Jetson Orin Nano (JetPack 6, r36.4.4)

- **What NVIDIA documents** (r36.4.4 "Platform Power and Performance", Orin NX/Nano section):
  - one INA3221 at 0x40, under the upstream **hwmon** driver at `/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon<x>/`;
  - nodes `in<i>_label`, `in<i>_input` (mV) and `curr<j>_input` (mA), with `curr4` the summation channel.
- **The rails:**

  | Channel | Rail | NVIDIA's description |
  |---|---|---|
  | 1 | **VDD_IN** | *"Total Module Power"* |
  | 2 | **VDD_CPU_GPU_CV** | *"Total power consumed by CPU, CPU [sic] and CV cores i.e. DLA and PVA"* |
  | 3 | **VDD_SOC** | *"memory subsystem and various engines like nvdec, nvenc, vi, vic, isp etc."* |

- **Caveats:**
  - **There is no power node.** Power = `in_input × curr_input`, which is what tegrastats computes.
  - **CPU and GPU share one rail**, so GPU-only power cannot be measured.
  - **VDD_IN also drives over-current throttling.** For example, on the Orin Nano 8GB it is limited to 15 W average and
    20 W instantaneous, or 25 W in Super mode. Crossing the limit causes hardware clock throttling of CPU and GPU. Check
    `oc*_event_cnt` after each run.
  - **Permissions:** the upstream `hwmon/ina3221.c` `is_visible` returns 0444 for inputs, so non-root reads should work.
    This was read from upstream source; NVIDIA's 5.15 build was not checked, so confirm with `ls -l`.
  - **No averaging by default.** The upstream DT `tegra234-p3767.dtsi` uses 5 mΩ shunts with no averaging override. The
    driver resets to `0x7127`: 1 sample, 1.1 ms conversions, continuous mode. That is **a new value every 6.6 ms, each a
    single ≈1.1 ms snapshot with no averaging**, which invites aliasing against bursty per-frame load. NVIDIA's shipped DTB
    may differ, so read `samples` and `update_interval` on the board. As root you can raise `samples`, for example to
    64 or 256.

### B3. INA3221 accuracy caveats (both boards)

- **TI chip specification:** offset ±80 µV max and gain error 0.25 % max. On 5 mΩ that offset means ±16 mA, or about
  **±80 mW at 5 V per channel**, worst case (derived).
- **NVIDIA** (Nano/TX1 page): *"assume a 5% guard band for INA measurements greater than 200 milliwatts. Below that,
  accuracy may deviate by as much as 15%."* The r36 Orin page does not repeat this note.
- **Shalavi et al.**, "Accurate Calibration of Power Measurements from Internal Power Sensors on NVIDIA Jetson Devices"
  (IEEE EDGE 2023, arXiv 2306.13107). The reference was a PicoScope on the board input.
  - Internal sensors *"underestimate the actual power by up to 50% in most cases"*.
  - The gap is nearly constant per device, so a linear calibration brings the error within ±3% (Nano: 1.11x + 232.60 mW,
    ±0.8%).
- **PolyThrottle** (arXiv 2310.19991, AGX Orin and TX2; secondary for the Orin Nano):
  - an external USB meter read *"consistently around 10% higher"*;
  - polling tegrastats at 50 ms itself raised measured power by 5–10%, an observer effect.

---

## C. Windows laptop: Ryzen 7 8845HS (Hawk Point, Family 25 Model 117), Windows 11

### C1. What the "Energy Meter" counters are

- **The interface is Windows EMI.** Microsoft Learn, "Energy Meter Interface": meters *"periodically measure voltage and
  current on a rail, calculate a power product, and integrate the total energy consumed over time."* The EMI unit is
  **picowatt-hours**, and `AbsoluteTime` is in 100 ns intervals.
- **The units, from the counter help text on this machine:**

  | Counter | Type | Help text |
  |---|---|---|
  | Energy | `NumberOfItems64`, cumulative | *"total energy consumed on the metered channel in picowatt-hours"* |
  | Power | `AverageCount64` (PDH divides between two collections) | *"average power over each sampling interval in milliwatts"* |
  | Time | | *"duration of meter operation in milliseconds"* |

- **Two separate drivers provide the channels.** This was established by querying `IOCTL_EMI_GET_METADATA` on each EMI
  device interface `{45BD8344-…}`, as a non-admin user:

  | EMI device | Driver | OEM / Model | Channels |
  |---|---|---|---|
  | `ACPI\AMDI0102` "AMD PMF" | `amdpmf`, AMD 26.10.15.0 | AMD / – | Socket Power, Apu Power, System Power, GPU Power, NPU Power, CPU Power |
  | 8 × CPU device | `AmdPPM`, Microsoft inbox `cpu.inf` 10.0.26100.9549 | Microsoft / PPM | RAPL_Package0_PKG (+Core0), RAPL_Package0_Core1…7_CORE |

- **Observed values,** computed as ΔEnergy/ΔTime, on AC under the Balanced scheme:

  | Condition | RAPL PKG | Socket | Apu | Σ cores | CPU / GPU / NPU / System Power |
  |---|---|---|---|---|---|
  | Idle, 6 s | 10.1 W | 11.2 W | 11.2 W | 9.9 W | **0, never increments** |
  | 16-process CPU burn | 15.9 W | 16.4 W | 16.4 W | 13.1 W | **0** |

  - The four zero channels are not populated on this laptop, at least on AC with the engines idle.
  - `_Total` sums every channel: the package, its own cores, Socket and Apu. **It double-counts, so never use it.**
- **What each channel measures:**
  - **`RAPL_Package0_*`:** these come from Microsoft's PPM driver, and Microsoft does not document the source register. On
    Linux, this CPU's package energy is MSR `0xC001029B` and its per-core energy is `0xC001029A` (`intel_rapl_msr.c`,
    `msr-index.h`). The names and the per-core structure make it **very likely the same counters, but this is unverified.**
  - **Socket / Apu / CPU / GPU / NPU / System Power:** these are AMD PMF channels, which AMD does not document for
    Windows. The AMD-authored Linux `amd-pmf` driver reads an SMU metrics table with fields `current_socketpower`,
    `apu_power` and `dgpu_power` (mW). Newer tables add `npu_power` and `system_power`. **Inferred:** the Windows channels
    are the same SMU telemetry, and "GPU Power" may mean the dGPU rather than the 780M.
  - **The dGPU is invisible to these counters.** The laptop also has an **RTX 4050 Laptop GPU** and an NPU, and the
    RTX 4050 does not appear in any of the channels above. If inference ever moves to the RTX 4050, use `nvidia-smi
    power.draw`. Its help text: *"On Ampere or newer devices, returns average power draw over 1 sec … accurate to within
    +/- 5 watts."*
- **Does `RAPL_Package0_PKG` equal Linux `amd_energy`?** No such comparison is possible on this chip:
  - `amd_energy` only matched Family 17h Model 30h and Family 19h Models 01h and 30h, which are EPYC server parts;
  - it was deleted in Linux 5.13 by the commit *"hwmon: Remove amd_energy driver"*.
  - The Linux path for this CPU is powercap `intel-rapl`, which reads the same `0xC001029B`. That path is root-only since
    CVE-2020-12912. On Windows the counters are readable without admin rights (observed).

### C2. Update rate

- **The underlying hardware updates every 1 ms.** AMD's RAPL MSRs *"are updated every 1ms"* (AMD-authored
  `amd_energy.rst`, Linux 5.12), and Schöne et al. measured 1 ms on Zen 2.
- **On Windows, both providers read on demand.** In a tight loop, every read returned a new Energy and Time value. Reads
  were 2–7 ms apart because of PDH read cost, and the smallest Time step was 1–2 ms.
- **Wrap-around is not a concern.** The counter is a 64-bit accumulation: about 1.4 × 10¹⁴ pWh, or 143 Wh, since boot was
  observed.
- **So skip the Power counter for energy per frame.** Read the cumulative Energy counter at run start and run end, and
  divide by the frame count.

**Measured under a real load, 2026-10-19.** The run used YOLO11s FP32 ONNX (`study/runs/prototype-small-object/onnx-fp32/best.onnx`) on ONNX Runtime
1.31 CPU with 8 intra-op threads and a random 640×640 input. The sequence was 15 s idle, then 40 s of inference, then 15 s idle,
with `typeperf` sampling at 1 Hz. The probe script is not in the repo; the method is described here.

| Power | FPS | PKG idle → load → idle | Energy/Time over the load window | J/frame (gross PKG) |
|---|---|---|---|---|
| Battery, Balanced | 9.47 | 7.5 → 16.4 → 13.6 W | 16.4 W | 1.72 |
| AC, Balanced | 8.78 | 8.5 → 16.2 → 7.1 W | 16.2 W | 1.80 |

- The cumulative Energy counter matches the mean of the Power samples to within 0.1 W. This confirms that Energy is in pWh, Time in ms and Power in mW.
- `Socket Power` and `Apu Power` read 0.2–0.5 W above `RAPL_Package0_PKG` and moved in step with it.
- **Idle drifts by several watts between neighbouring windows** (7.1–13.6 W). Run-to-run FPS moved about 7 % with nothing changed.
  Net-of-idle energy is therefore only as good as the idle window next to it. That is a question for the measurement protocol.

### C3. Accuracy caveats

- **AMD RAPL is modelled.** Schöne et al., "Energy Efficiency Aspects of the AMD Zen 2 Architecture" (IEEE Cluster 2021,
  arXiv 2108.00808), tested Zen 2 EPYC:
  - *"The results indicate that the energy data is modeled, not measured"*;
  - data-dependent power is not reflected;
  - memory energy is *"not (fully) included in the package domain"*;
  - *"Energy measurements of AMDs RAPL implementation should be considered inaccurate."*
  - **This has not been re-validated on Zen 4 or Hawk Point; applying it here is an extrapolation.**
- **The boundary is the package, not the laptop.** It excludes DRAM, the display, the SSD, the dGPU and VRM losses. That
  is acceptable for the ticket's CPU-only inference as a *relative* comparison between models. As an absolute
  energy-per-frame number it is weak.

### C4. AC versus battery

- **Microsoft:** the default power-mode slider is set separately for AC and DC. Each mode engages different PPM settings
  (for example EPP), and Battery Saver exists on DC only (Learn, "Customize the Windows performance power slider").
- **AMD PMF** (Linux `amd/pmf/sps.c`): the SPL, SPPT and FPPT package power limits are looked up by **power source
  (AC/DC)** and slider position. Unplugging therefore changes the power limits, and with them throughput and joules per
  frame.
- **This laptop on 2026-10-19:** the Balanced scheme, with both AC and DC overlays at the default (null GUID). HP's own
  performance-mode utility may also rewrite the limits; **not verified.**
- **Rule:** run on AC, keep the power mode fixed, and log `powercfg /getactivescheme` plus the overlay registry values
  with each run. EMI channels measure the SoC, so battery charging does not contaminate them.

---

## Verdict for the ticket

| Platform | Mechanism | What it measures | Sample rate | Trust | The one discipline rule |
|---|---|---|---|---|---|
| Galaxy M35 (**study phone is actually the M34**; see the correction in §A) | `BatteryManager.CURRENT_NOW` × `EXTRA_VOLTAGE` (SM5714 fuel gauge) | **Whole phone** at the battery: SoC, screen, radios | Read on demand; gauge period **unknown**; 1 mA resolution | Low absolute, OK relative (same device, same state) | **USB physically unplugged** (Wi-Fi adb or log-to-file), ≥60 s steady window × ≥3, idle baseline subtracted, units checked on the device |
| Jetson Nano (JP 4.6) | INA3221 via `sudo tegrastats` or IIO sysfs, rail `POM_5V_IN` | **Module input** (no carrier, USB or fan) | New 64-sample average about every 54 ms | Medium: ±5 % above 200 mW; sensor reads below board power (fit 1.11x + 233 mW) | **Map the rail by `rail_name_N`** (doc and DT disagree), run as root, integrate VDD_IN only |
| Jetson Orin Nano (JP 6) | INA3221 hwmon; VDD_IN = mV × mA | **Module input**; CPU, GPU and CV share one rail | 6.6 ms, single snapshot, no averaging (upstream default; check `samples`) | Medium; up to 10–50 % under board power in the literature | **Use VDD_IN only, set and record `samples`, and require `oc*_event_cnt` unchanged** |
| Ryzen 8845HS laptop | "Energy Meter" EMI counters: `RAPL_Package0_PKG` (MS PPM) or `Socket Power` (AMD PMF) | **SoC package only** (no DRAM, screen, SSD or dGPU) | On demand; hardware 1 ms | Low–medium: modelled; relative CPU-only comparisons only | **Difference the cumulative Energy (pWh) counter, on AC with a fixed power mode.** Never use `_Total` or the always-zero CPU/GPU/NPU/System channels. |

**Not verified (do these on the devices before relying on them):**

- **M35:**
  - the gauge's update period;
  - whether retail M35 firmware matches the µA kernel source;
  - that `dumpsys powerstats` lists no channels;
  - the charge-counter step size.
- **Nano:**
  - whether non-root `tegrastats` shows POM fields;
  - the minimum `--interval`;
  - what tegrastats "average" averages over;
  - which physical rail `in_power0_input` is on the actual board.
- **Orin Nano:**
  - the averaging set in NVIDIA's shipped DTB;
  - non-root permissions on NVIDIA's 5.15 kernel;
  - that VDD_IN excludes the carrier (inferred).
- **Laptop:**
  - the MSR behind `RAPL_Package0_PKG`;
  - what the AMD PMF channels actually mean;
  - whether CPU/GPU/NPU/System Power populate on battery or under NPU or GPU load;
  - RAPL accuracy on Zen 4.

---

## Sources

Primary:

- AOSP `BatteryManager.java`: https://android.googlesource.com/platform/frameworks/base/+/refs/heads/main/core/java/android/os/BatteryManager.java
- AOSP `HealthInfo.aidl`: https://android.googlesource.com/platform/hardware/interfaces/+/refs/heads/main/health/aidl/android/hardware/health/HealthInfo.aidl
- AOSP healthd `BatteryMonitor.cpp`: https://android.googlesource.com/platform/system/core/+/refs/heads/main/healthd/BatteryMonitor.cpp
- AOSP `BatteryService.java` (unplug, suspend_input): https://android.googlesource.com/platform/frameworks/base/+/refs/heads/main/services/core/java/com/android/server/BatteryService.java
- AOSP `DeviceIdleController.java` (deviceidle disable): https://android.googlesource.com/platform/frameworks/base/+/refs/heads/main/apex/jobscheduler/service/java/com/android/server/DeviceIdleController.java
- AOSP `IPowerStats.aidl` and `PowerStatsService.java` (dumpsys powerstats): https://android.googlesource.com/platform/hardware/interfaces/+/refs/heads/main/power/stats/aidl/
- AOSP "Measure device power": https://source.android.com/docs/core/power/device
- AOSP "Power stats HAL": https://source.android.com/docs/core/power/power-stats-hal
- Perfetto battery counters and ODPM: https://perfetto.dev/docs/data-sources/battery-counters
- Android adb (wireless debugging): https://developer.android.com/tools/adb
- Linux ABI `sysfs-class-power`: https://github.com/torvalds/linux/blob/master/Documentation/ABI/testing/sysfs-class-power
- Samsung kernel (A54, firmware A546BXXU6BXA2), mirror of Samsung OSS: https://github.com/Mesa-Labs-Archive/android_kernel_samsung_s5e8835 (`drivers/battery/common/sec_battery.c`, `drivers/battery/fuelgauge/sm5714_fuelgauge/`)
- Samsung kernel (M35 `m35x`), mirror of Samsung OSS: https://github.com/SavedByLight/android_kernel_samsung_m35x (`sec_battery.c`, `sm5714_fuelgauge.c`, `arch/arm64/configs/s5e8835-m35xxx_defconfig`, `drivers/iio/adc/samsung/pmic/s2mpu15_adc_8835.c`)
- NVIDIA L4T R32.7.5 Nano power management: https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-3275/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_nano.html
- NVIDIA L4T R32.7.5 tegrastats: https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-3275/Tegra%20Linux%20Driver%20Package%20Development%20Guide/AppendixTegraStats.html
- NVIDIA Jetson Linux r36.4.4 Orin power: https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html
- NVIDIA Jetson Linux r36.4.4 tegrastats: https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html
- L4T R32 kernel (INA3221 IIO driver, Nano DT, IIO core): https://github.com/OE4T/linux-tegra-4.9 (branch `oe4t-patches-l4t-r32.7.4`; `nvidia/drivers/staging/iio/meter/ina3221.c`, `nvidia/platform/t210/porg/kernel-dts/porg-platforms/tegra210-porg-powermon-p3448-0000-a00.dtsi`, `drivers/iio/industrialio-core.c`)
- Linux upstream: `Documentation/hwmon/ina3221.rst`, `drivers/hwmon/ina3221.c`, `arch/arm64/boot/dts/nvidia/tegra234-p3767.dtsi`, `drivers/powercap/intel_rapl_msr.c`, `drivers/platform/x86/amd/pmf/{pmf.h,sps.c,Kconfig}`, and `Documentation/hwmon/amd_energy.rst` at tag v5.12 (removed in v5.13) at https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git
- TI INA3221 product page: https://www.ti.com/product/INA3221
- Microsoft Learn, Energy Meter Interface: https://learn.microsoft.com/en-us/windows-hardware/drivers/powermeter/energy-meter-interface
- Microsoft Learn, `EMI_CHANNEL_MEASUREMENT_DATA`: https://learn.microsoft.com/en-us/windows/win32/api/emi/ns-emi-emi_channel_measurement_data
- Microsoft Learn, power slider: https://learn.microsoft.com/en-us/windows-hardware/customize/desktop/customize-power-slider
- On-machine evidence (2026-10-19): `Energy Meter` counter help, EMI metadata IOCTLs, driver enumeration, `nvidia-smi --help-query-gpu`.

Secondary (peer-reviewed or preprint papers, plus forum posts):

- Tschand et al., MLPerf Power, arXiv 2410.12032 (HPCA 2025)
- Ignatov et al., AI Benchmark, arXiv 1810.01109 and 1910.06663
- Almeida et al., "Smart at what cost?", IMC 2021, arXiv 2109.13963
- Shalavi et al., IEEE EDGE 2023, arXiv 2306.13107
- Yan et al., PolyThrottle, arXiv 2310.19991
- Schöne et al., IEEE Cluster 2021, arXiv 2108.00808
- Samsung Developer Forum thread on `CURRENT_NOW` units (anecdotal): https://forum.developer.samsung.com/t/charging-current-battery-property-current-now/13517
