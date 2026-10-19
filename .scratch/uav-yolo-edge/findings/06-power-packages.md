# Findings: the four "software power meter" packages (py-power-profile, perun, pyJoules, pyRAPL)

Ticket: `issues/06-power-measurement-capability.md`
Researched: 2026-10-19
Method: primary sources only. I downloaded each package's sdist from PyPI and read the source; repo state and history came from the GitHub API, and kernel and Microsoft docs were checked for the interfaces underneath. The Windows Energy Meter counters were **read live on the team laptop** (this machine reports `AMD Ryzen 7 8845HS w/ Radeon 780M Graphics`, Windows 11, non-admin shell). Nothing was run on the Pi, Jetson or phone. Anything not verified is marked **[unverified]**.

**Bottom line:** none of the four gives a real reading on any of our four platforms. Three of them only read Linux `powercap` RAPL, plus NVML or AMD SMI for discrete GPUs. The fourth, py-power-profile, ends up estimating TDP x CPU% on every one of our platforms, and that estimate is broken. Usable sources exist for the laptop and the Jetson, but they come from outside this list (§6).

---

## 0. Does "py-power-profile" exist? Yes, under that exact name.

[`py-power-profile` on PyPI](https://pypi.org/project/py-power-profile/) has one release, **0.1.0, dated 2025-07-21**. It requires Python >=3.9. The repo is [Sherin-SEF-AI/py-power-profile](https://github.com/Sherin-SEF-AI/py-power-profile), which has **3 commits, all made on 2025-07-21**, 0 stars, and nothing since. It is a single-day project. The similarly named `powerprofile` on PyPI (gisce) is an unrelated library for electricity tariff profiles.

---

## 1. py-power-profile 0.1.0: estimates, and the estimator is broken

**What it reads.** It is a CLI only: `py-power profile script.py`. It has no context manager and no Python API (`pyproject.toml` `[project.scripts]`; `cli.py`). It profiles with `sys.settrace` and calls `backend.start()` and `backend.stop()` on every Python function call and return (`tracer.py`). `utils.get_auto_backend()` tries three backends in order:

| Backend | Source | What it really does |
|---|---|---|
| `rapl` | `backends/rapl.py` | `import pyRAPL; pyRAPL.setup()` is the availability check, and `pyRAPL.RAPLMonitor.sample()` is the reading. **pyRAPL has no `RAPLMonitor`.** I grepped the whole pyRAPL 0.2.3.1 source; its `__init__.py` exports only `setup`, `Measurement`, `measureit`, `Sensor` and a few others. So even on Intel Linux every reading throws, and the tracer only prints "Energy measurement failed". This backend **never produces a number**. |
| `hwmon` | `backends/hwmon.py` | Takes the first `/sys/class/hwmon/hwmon*/power*_input` and reads it **once at start and once at stop**, treating the value as mW. The kernel ABI says `powerY_input` is **microWatt** ([sysfs-class-hwmon](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/Documentation/ABI/testing/sysfs-class-hwmon)). The code then multiplies mW x s by 1000 again, so a reported "mJ" would be about **10^6 x too large**. None of our boards exposes such a file anyway (§5). |
| `psutil_est` | `backends/psutil_est.py` | `power = tdp_watts x avg(cpu%) / 100`. The default is `tdp_watts = 15.0` (`config.py`), which can be overridden with `PY_POWER_TDP_WATTS`. This is a **model, not a measurement**. It is also mis-sampled: `start()` and `stop()` each call `psutil.cpu_percent(interval=0.1)`. That blocks for 100 ms and measures the CPU **in the 100 ms before the block starts and the 100 ms after it ends, never during it**, and it adds about 200 ms to every traced function call. |

A second structural bug: the backend holds a single `_start_time`/`_start_energy`, and every nested Python call resets it. So any function that calls other functions gets the wrong value (`tracer.py` `_trace_callback`).

**Platforms.** On Pi 5, Windows AMD and Jetson Orin, `rapl` and `hwmon` are not available, so it **silently falls back to `psutil_est`**, the 15 W x CPU% guess. Jetson Nano on JetPack 4.6 / Python 3.6 is ruled out by the >=3.9 floor. On Android it is unsupported **[unverified: psutil under Termux not tested]**.

**Footprint and permissions.** It pulls in typer, rich, psutil and tomli; the optional extra `[rapl]` pulls in pyRAPL. It needs no root, because none of its backends ever reads a protected file.

**Sampling.** It works per function through `settrace` and cannot attribute energy to a block of N inferences.

**Verdict: reject.** On all of our platforms it reports a TDP guess labelled as a profile.

---

## 2. perun 1.0.0: a real tool, but its sensors are not on our hardware

Release **1.0.0, 2026-08-31** ([PyPI](https://pypi.org/project/perun/), [CHANGELOG](https://github.com/Helmholtz-AI-Energy/perun/blob/main/CHANGELOG.md)). It is actively maintained: the last push was 2026-10-15 at [Helmholtz-AI-Energy/perun](https://github.com/Helmholtz-AI-Energy/perun). It requires **Python >=3.10**, and no perun release has ever supported Python 3.6 (the floor was 3.8 or 3.9 from 2022 to 2024, then 3.10).

**What it reads.** These are the backends listed in `perun/backend/__init__.py`:
- **`powercap_rapl.py`** reads `/sys/class/powercap/intel-rapl:N/energy_uj`, plus the `dram` and `package` subzones. The docs say: "*supports recent Intel and AMD CPUs*" and "*cpu energy readings from perun only support linux environments with read access to the powercap-rapl interface, which can only be read by **root** on Linux 5.10 and later*" (`docs/install.rst`). If `energy_uj` is unreadable, it **silently skips** the device; the message only appears at debug log level (lines 78-83).
- **`nvml.py`** (NVIDIA dGPU) and **`rocmsmi.py`** (AMD dGPU, via amdsmi).
- **`hwmon_grace.py`** reads `power*_average` under `/sys/class/hwmon/*/device/`. It is written for NVIDIA Grace servers.
- **`psutil.py`** gives **utilisation, memory, disk and network only. It produces no power figure.** perun has no TDP model. The only estimate is a user-entered constant, `power_overhead`, which defaults to 0 W (`configuration.py`, `api/cli.py`).

So perun only **measures**; it never estimates. On a machine without RAPL, NVML, AMD SMI or Grace hwmon, it reports no energy at all.

**Platforms.**
- **Pi 5**: unsupported. There is no powercap RAPL, and the Pi kernel's hwmon drivers expose no `power*_average` (§5).
- **Windows AMD**: unsupported. The docs say CPU energy is Linux-only.
- **Android**: unsupported.
- **Jetson**: unsupported. Nano on JetPack 4.6 fails the Python floor. On Orin, the INA3221 hwmon exposes `in*_input`/`curr*_input` but no `power*_average` (§5), and NVML does not work on Jetson (§6).

**Footprint.** h5py, numpy, pandas, psutil and py-cpuinfo, plus optional mpi4py, nvidia-ml-py and amdsmi. This is the heaviest of the four.

**Sampling.** A separate process samples every sensor every `sampling_period`, which defaults to **1 s** with a minimum of 0.1 s (`configuration.py`). The `@monitor("region")` decorator "*only stores timestamps at the start and the end*" (`docs/usage.rst`). Region energy is then computed by interpolating the sampled power series and integrating it with `np.trapezoid` (`processing.py`, `processEnergyData`). It handles counter wraparound. This works for a block of N inferences much longer than 1 s; it does not work per frame.

**Verdict: the best-engineered of the four, but it gives zero energy numbers on our matrix.** It would only matter if the laptop booted Linux.

---

## 3. pyJoules 0.5.1: the right API shape, but only Intel RAPL on Linux plus NVML

The PyPI release is **0.5.1, dated 2020-11-16** ([PyPI](https://pypi.org/project/pyJoules/)). A GitHub tag v0.5.2 (2021-10-05) was **never published to PyPI**. Repo [powerapi-ng/pyJoules](https://github.com/powerapi-ng/pyJoules) received a batch of bug-fix merges on 2025-12-29 (overflow handling #38, package-domain detection #42). These are **unreleased**, so `pip install` still gives the 2020 code. It requires Python >=3.7.

**What it reads.**
- `device/rapl_device.py` is hard-coded to `RAPL_API_DIR = '/sys/class/powercap/intel-rapl'` and reads the `energy_uj` files of the package, dram, core and uncore zones.
- `device/nvidia_device.py` calls `pynvml.nvmlDeviceGetTotalEnergyConsumption`. The README says that only works on "*Volta architecture*" or newer dGPUs.
- The README says RAPL "*estimates power consumption of the CPU*", and the counter it reads is the hardware or firmware one.
- There is no fallback model. Open issue [#54](https://github.com/powerapi-ng/pyJoules/issues/54) (2026-08-04) says that on AMD-without-powercap, ARM and Apple machines it "*silently returns nothing*".

**AMD on Linux.** It works if the kernel registers AMD Fam17h/19h RAPL under the same `intel-rapl` powercap path ([patch series, Oct 2020](https://lkml.iu.edu/hypermail/linux/kernel/2010.3/02739.html)). Zen 4, which is our 8845HS (Family 19h), is in that family. The package and core zones work; there is no DRAM zone.

**Platforms.**
- **Windows**: no. It only uses `/sys` paths.
- **Pi 5**: no.
- **Android**: no.
- **Jetson**: no. There is no RAPL, and NVML is not supported on Jetson (§6).

**Footprint and permissions.** It is pure Python with zero required dependencies. It needs **root** for `energy_uj` on kernels 5.10 and later (same restriction documented by perun and PowerJoular).

**Sampling.** It takes the difference of the counter between `start`/`record`/`stop` (`energy_meter.py`), and is available as the decorator `@measure_energy` or the `EnergyContext` context manager. **This attributes energy to a block of N inferences exactly.** Resolution is the RAPL update period; the README warns that periods that are too short return null values.

**Verdict: unusable here.** It would be the right tool on an Intel or AMD Linux laptop, and our laptop runs Windows.

---

## 4. pyRAPL 0.2.3.1: Intel RAPL on Linux only (AMD works via the same path), and stale

The release is **0.2.3.1, dated 2019-12-19** ([PyPI](https://pypi.org/project/pyRAPL/)). Repo [powerapi-ng/pyRAPL](https://github.com/powerapi-ng/pyRAPL) has had no release since then. Its last activity was a batch of PR merges on 2025-03-07. It requires Python >=3.7, because it uses `time.time_ns` (`measurement.py`).

**What it reads.** `device_api.py` reads `/sys/devices/system/cpu/present`, then the `name` and `energy_uj` files under `/sys/class/powercap/intel-rapl/intel-rapl:N`. It covers PKG and DRAM only. The README says "*uses the Intel RAPL technology that estimates power consumption of a CPU*". It has **no counter-wraparound handling**: `SubstractableList` simply subtracts, and open issue [#19](https://github.com/powerapi-ng/pyRAPL/issues/19) reports negative energy during NN inference **on AMD sockets**. That issue also confirms the package runs on AMD Linux.

**Platforms.**
- **Windows**: no. `setup()` opens `/sys/...`.
- **Pi 5**: no.
- **Android**: no.
- **Jetson**: no.
- On **AMD Linux** it gets the package zone only; DRAM is skipped.

**Footprint and permissions.** It is pure Python with no dependencies, and it needs **root** on kernels 5.10 and later.

**Sampling.** `Measurement` `begin()`/`end()`, the `with` form, and the `@measureit(number=N)` decorator all take a counter delta, and `measureit` divides by N. **It can attribute energy to a block of N inferences.** The README's examples use `@pyRAPL.measure`, which the code does not export; the real name is `measureit`.

**Verdict: unusable here.** It is the older, narrower predecessor of pyJoules.

---

## 5. Why the Linux-sysfs packages find nothing on our boards

- **Pi 5.** The Pi kernel's own hwmon drivers expose no power attributes:
  - `raspberrypi-hwmon.c` registers `rpi_volt` with only `in0_lcrit_alarm`, the under-voltage flag.
  - `rp1-adc.c` exposes only `in1..4_input` and `temp1_input` ([raspberrypi/linux rpi-6.12.y](https://github.com/raspberrypi/linux/tree/rpi-6.12.y/drivers/hwmon)).
  - The PMIC ADC is reached through firmware (`vcgencmd pmic_read_adc`), not through hwmon.
  - **None of the four packages calls vcgencmd or reads the PMIC.** There is no RAPL on Cortex-A76.
  - The remaining hwmon nodes on a stock Pi 5 are thermal and fan **[unverified on the board]**.
- **Jetson.**
  - On JetPack 4.x (Nano), INA3221 is an **IIO** device: `/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power{0,1,2}_input`, in mW, with rails POM_5V_IN, GPU and CPU. It is not hwmon.
  - On JetPack 5 and later (Orin), it is the upstream `ina3221` **hwmon** driver, which exposes only `in*_input` (mV) and `curr*_input` (mA). Power has to be computed as V x I.
  - Sources: jetson-stats [`jtop/core/power.py`](https://github.com/rbonghi/jetson_stats/blob/master/jtop/core/power.py) comments "*hwmon are JP5 compatible, iio:device from JP4*"; upstream [`drivers/hwmon/ina3221.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/drivers/hwmon/ina3221.c) channel info has in and curr only.
  - So none of the four reads the Jetson rails.
- **AMD RAPL is itself modelled.** On Zen 2, TU Dresden found "*the energy data is modeled, not measured*", that it reads lower than external measurement, and that it misses memory energy ([Schöne et al., arXiv:2108.00808](https://arxiv.org/abs/2108.00808), §VII). **[unverified for Zen 4]** The laptop figure should therefore be reported as "CPU package power as reported by the SoC", not as wall power.

---

## 6. Alternatives that actually target our hardware (brief)

- **Laptop (Windows 11, AMD): read the Energy Meter counters directly. Verified on this laptop.**
  - `Get-Counter -ListSet "Energy Meter"` lists `RAPL_Package0_PKG`, `RAPL_Package0_Core0..7_CORE`, `Apu Power`, `Socket Power`, `CPU/GPU/NPU Power` and `System Power`, each with `Power`, `Energy` and `Time` counters.
  - A **non-admin** read returned PKG ≈ 5.9-7.9 W at light load.
  - Units, inferred from the deltas: Power in mW, Energy in picowatt-hours, Time in ms. Over a 3.04 s interval, ΔE/Δt gave about 6.0 W, matching the Power counter. **[units inferred; not documented by Microsoft]**
  - The source of `Apu Power`/`Socket Power` is **[unverified]**.
  - [Microsoft's EMI page](https://learn.microsoft.com/en-us/windows-hardware/drivers/powermeter/energy-meter-interface) documents the IOCTL interface and its picowatt-hour unit, but does not mention RAPL.
  - Ready-made readers:
    - **CodeCarbon >= 3.3.0 (2026-08-04)** added Windows EMI (`codecarbon/core/windows_emi.py`: "*for both Intel and AMD CPUs. No third-party driver, no admin rights*"). It selects `*_PKG` channels and needs Python >= 3.10.
    - **PowerJoular 2.0** reads EMI, but it is still `2.0.0-alpha-3`.
    - A ~20-line ctypes or `win32pdh` helper that takes Energy deltas around a block of N inferences gives exactly the pyJoules-style context manager we want.
- **Jetson.**
  - **jetson-stats / jtop** reads the INA3221 rails on both JetPack generations (§5). Its client polls every 1 s by default (`jtop(interval=1.0)`), and it needs a root-installed service.
  - Version choice: **7.2.2** (2026-10-01) requires Python >=3.7. **4.3.2** (2025-03-19) declares >=2.7, so pin it on Nano with Python 3.6 **[declared, not tested on 3.6]**.
  - `tegrastats --interval <ms>` is the zero-install alternative.
  - NVML-based tools (pyJoules, perun nvml) are out: NVIDIA staff say "*NVML library is not supported on Jetson*" ([forum](https://forums.developer.nvidia.com/t/does-nvml-fully-support-jetson-orin-gpu-devices/332969); a forum post, not formal documentation).
- **Pi 5, model route.**
  - **PowerJoular** lists Pi "5B" support. The model in [Joular Core `src/linux/joular_core-rpi.adb`](https://github.com/joular/joularcore/blob/develop/src/linux/joular_core-rpi.adb) is a **degree-9 polynomial of whole-system CPU usage** (only for 64-bit OS). It evaluates to **3.48 W at 0%, 6.20 W at 50% and 9.52 W at 100%**.
  - The README says only that the models are "*research-based regression power models*" and were "*trained on one particular revision*". Pi 5 support was added on 2024-01-08 (commit `d2ed059`), after the IE2022 paper. **The training data and meter for the Pi 5 model are not published [unverified]**.
  - It is an Ada binary. A Python ctypes example exists, but it requires building the shared library with `gprbuild`.
- **Pi 5, CodeCarbon.** With no RAPL it falls back to "cpu_load" mode. If the CPU model is not in its TDP table, it uses `4 W x threads`, which is 16 W on the Pi 5, shaped by a load curve (`core/cpu.py` `DEFAULT_POWER_PER_CORE = 4`; `external/hardware.py`). This is a TDP guess **[the Cortex-A76 table miss is inferred from code, not run]**.
- **Pi 5, PMIC.** **There is no packaged PyPI library.** Ad-hoc scripts parse `vcgencmd pmic_read_adc`, for example [librerpi/rpi-tools `pi5_voltage.py`](https://github.com/librerpi/rpi-tools/blob/master/pi5_voltage.py), a Prometheus exporter that parses `*_V volt(...)` and `*_A current(...)` lines. The mechanism itself is covered by the separate PMIC finding.
- **Android.** None of the above applies. The ticket's BatteryManager / batterystats route stands.

---

## 7. Verdict for the ticket

| Package \ Platform | Pi 5 (ARM Linux) | Laptop (Win 11, AMD) | Galaxy M35 (Android) | Jetson (Nano JP4.6 / Orin) |
|---|---|---|---|---|
| **py-power-profile 0.1.0** | estimate (15 W x CPU%, mis-sampled) | estimate (same) | unsupported [unverified] | Nano: unsupported (Py 3.6); Orin: estimate |
| **perun 1.0.0** | unsupported (utilisation only, no energy) | unsupported | unsupported | unsupported (Py floor; no NVML; no power hwmon) |
| **pyJoules 0.5.1** | unsupported | unsupported | unsupported | unsupported |
| **pyRAPL 0.2.3.1** | unsupported | unsupported | unsupported | unsupported |
| *for reference:* Energy Meter (EMI) | n/a | **real reading: CPU package, SoC-reported (verified, no admin)** | n/a | n/a |
| *for reference:* jtop / tegrastats | n/a | n/a | n/a | **real reading: INA3221 rails** |

"Real measurement" for pyJoules or pyRAPL would only appear on an Intel or AMD **Linux** laptop with root. That is not our configuration.

**Recommendation: adopt none of the four.** For the dev's request ("pick the one that best fits"), the honest answer is that each of them, examined on its own, is either Linux-RAPL/NVML-only (perun, pyJoules, pyRAPL) or a broken TDP estimator (py-power-profile). Use these instead:
- **Laptop**: read `Energy Meter(RAPL_Package0_PKG)\Energy` deltas around each block of N inferences, using a tiny in-repo helper or CodeCarbon >= 3.3.0 `windows_emi`. Report it as **CPU-package power**.
- **Jetson**: jtop or tegrastats, using the total input rail (Nano `POM_5V_IN`; Orin `VDD_IN`) for board power.
- **Android**: BatteryManager, as already planned.

**What a software-only Pi 5 number would honestly be.** It is either:
- **(a) a measurement of the PMIC's own output rails.** Each rail's V and I come from `vcgencmd pmic_read_adc` and are summed. This misses PMIC conversion loss and anything fed from the 5 V input before the PMIC (USB and peripherals). **It is not input power.** The exact rail coverage is deferred to the PMIC finding; **[unverified here]**.
- **(b) a model.** That means PowerJoular's CPU-utilisation polynomial (3.5-9.5 W, provenance unpublished), or a TDP x load guess (CodeCarbon, py-power-profile).

Neither is a measurement of what the Pi draws from the wall. Gap 02's "power correlation" on the primary platform therefore still needs the inline USB-C meter. Without it, the paper should say (a) is a PMIC-rail measurement and treat it as a stated limitation. Separately, the ticket's line "*vcgencmd reports voltage and temperature but not draw*" is out of date for the Pi 5, since `pmic_read_adc` reports per-rail current. That line should be corrected once the PMIC finding lands.

---

## Not verified

- py-power-profile under Termux on Android, and psutil access to `/proc/stat` there.
- That a stock Pi 5 has no other hwmon `power*` node (checked from driver source, not on the board).
- Energy Meter counter units (inferred from deltas); the provenance of the `Apu Power` and `Socket Power` channels.
- Whether AMD RAPL is modelled on Zen 4 (shown only for Zen 2).
- How the PowerJoular Pi 5 model was trained.
- jetson-stats 4.3.2 actually running on Python 3.6.
- NVML on Jetson (forum-staff statement only).

## Sources

- PyPI JSON metadata: [py-power-profile](https://pypi.org/project/py-power-profile/), [perun](https://pypi.org/project/perun/), [pyJoules](https://pypi.org/project/pyJoules/), [pyRAPL](https://pypi.org/project/pyRAPL/), [codecarbon](https://pypi.org/project/codecarbon/), [jetson-stats](https://pypi.org/project/jetson-stats/).
- Package source read from the sdists:
  - py-power-profile 0.1.0: `backends/{rapl,hwmon,psutil_est}.py`, `tracer.py`, `utils.py`, `config.py`, `README.md`
  - perun 1.0.0: `backend/*.py`, `processing.py`, `configuration.py`, `api/decorator.py`, `docs/install.rst`, `docs/usage.rst`
  - pyJoules 0.5.1: `device/rapl_device.py`, `device/nvidia_device.py`, `energy_meter.py`, `README.md`
  - pyRAPL 0.2.3.1: `__init__.py`, `device_api.py`, `sensor.py`, `measurement.py`, `README.md`
  - codecarbon 3.3.1: `core/windows_emi.py`, `core/resource_tracker.py`, `core/cpu.py`, `external/hardware.py`
- GitHub repos and issues: [pyJoules #54](https://github.com/powerapi-ng/pyJoules/issues/54), [#34](https://github.com/powerapi-ng/pyJoules/issues/34), [#38](https://github.com/powerapi-ng/pyJoules/pull/38), [#42](https://github.com/powerapi-ng/pyJoules/pull/42); [pyRAPL #19](https://github.com/powerapi-ng/pyRAPL/issues/19); [PowerJoular README](https://github.com/joular/powerjoular); [Joular Core](https://github.com/joular/joularcore); [jetson_stats](https://github.com/rbonghi/jetson_stats); [CodeCarbon 3.3.0 release](https://github.com/mlco2/codecarbon/releases/tag/3.3.0).
- Kernel:
  - [sysfs-class-hwmon ABI](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/Documentation/ABI/testing/sysfs-class-hwmon)
  - [ina3221.c](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/drivers/hwmon/ina3221.c)
  - [AMD Fam17h/19h powercap RAPL patches](https://lkml.iu.edu/hypermail/linux/kernel/2010.3/02739.html)
  - [raspberrypi/linux hwmon drivers](https://github.com/raspberrypi/linux/tree/rpi-6.12.y/drivers/hwmon)
- [Microsoft: Energy Meter Interface](https://learn.microsoft.com/en-us/windows-hardware/drivers/powermeter/energy-meter-interface)
- [Schöne et al., Energy Efficiency Aspects of the AMD Zen 2 Architecture](https://arxiv.org/abs/2108.00808)
- [NVIDIA forum: NVML on Jetson](https://forums.developer.nvidia.com/t/does-nvml-fully-support-jetson-orin-gpu-devices/332969); [NVIDIA forum: reading Nano power](https://forums.developer.nvidia.com/t/read-power-consumption-of-nano/167779)
- Live check on the team laptop, 2026-10-19: `Get-Counter -ListSet "Energy Meter"`, and `Get-Counter` samples of `RAPL_Package0_PKG` Power, Energy and Time.
