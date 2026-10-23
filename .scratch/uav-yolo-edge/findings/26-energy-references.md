# Findings: citable references for the Pi 5 energy method, and is the method sound?

Ticket: #26 (energy runs on the Pi 5).
Researched: 2026-10-23. Paper due 2026-10-31.

**Question.** We measure energy on the Raspberry Pi 5 by polling `vcgencmd pmic_read_adc` every 0.5 s and summing
V×I over the PMIC output rails. Each cell is a 1-min idle window followed by a 3-min inference run (pre + model + post,
conf 0.25, batch 1, 4 threads). We report gross J/frame = P̄_run·t/N and net J/frame = (P̄_run − P̄_idle)·t/N. J/frame is
compared only within the Pi 5. The sampler's FPS cost is measured once. There is no external meter. What can we cite
for this method, and what would a reviewer attack?

**Verdict in one line:** the method is defensible as a **within-device, relative** measurement. Three things carry it:
the Raspberry Pi docs, which say the rail sum is not supply power; a public meter calibration showing the rail sum is
linear in input power; and peer-reviewed evidence that on-board sensors under-read by a near-constant mapping, so
same-device comparisons survive while absolute and cross-device ones do not. The weak points a reviewer will hit are
**(1) one run per cell**, where the literature norm is n=3, **(2) possible aliasing** between the 0.5 s sampler and
the ~0.2 s frame period, and **(3) no record of the starting temperature or cooling state** for each cell. All three can
be fixed cheaply before the deadline (§6).

Legend: ✅ = verified against the primary source itself (paper HTML/PDF, official doc, or source code).
⚠️ = verified only from an abstract, metadata or a secondary page, or partly unverifiable. ❌ = claim did not hold up.

---

## 0. Re-verification of the earlier internal note

| Earlier claim | Status | What the source actually says |
|---|---|---|
| jfikar calibration `P_in ≈ 1.1451·P_pmic + 0.5879` | ✅ | README: *"real_consumption [W] = pmic_output [W] \* 1.1451 + 0.5879"*. Script sums V×I over 12 current/voltage pairs and averages 50 one-second samples. <https://github.com/jfikar/RPi5-power>, [script](https://raw.githubusercontent.com/jfikar/RPi5-power/main/rpi5_power.sh) |
| "rail sum reads 18–31 % low" | ✅, with a caveat | Follows from the fit, but **the size depends on load**: 1 − P/(1.1451P + 0.5879) is 30.5 % at 2 W, 22.6 % at 4 W, 17.9 % at 8 W and ~15.5 % at 15 W. The fit's plot covers roughly 1.5–15 W PMIC output, about 30 points. The meter is an unnamed "USB-C power-meter", and this is a hobbyist repo (2024), not peer-reviewed. |
| vcgencmd man page documents `pmic_read_adc` | ✅ | *"On Raspberry Pi 5, displays the readings of the ADCs in the PMIC: the current drawn from and the voltage of each PMIC supply rail, plus the external 5V supply voltage (EXT5V_V)"* … *"The readings do not include USB current or anything else powered directly from the 5V supply, because that power does not pass through the PMIC."* No accuracy figure is given. [vcgencmd.1](https://github.com/raspberrypi/utils/blob/master/vcgencmd/vcgencmd.1) |
| Raspberry Pi docs, power-supply section | ✅ | *"You can't see USB current or anything else connected directly to 5V, because this bypasses the PMIC."* *"You should not expect this to add up to the wattage of the source power supply. However, it can be useful to monitor things like the core voltage."* [raspberry-pi.html → Power supply](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html) |
| arXiv 2609.04705: Pi 5 YOLOv8n, POWER-Z KM003C, 1.9 % FPS cost at 2 Hz | ✅, with a correction | The meter is a POWER-Z KM003C inline at 1 kS/s. The overhead is *"1.90% relative FPS reduction (1.33 ms/frame absolute) from paired n=3, 5-minute benchmarks"*: 14.579 → 14.302 FPS (Table I). **Correction:** that telemetry samples temperature, frequency, CPU and memory use, and `get_throttled`. **It does not call `pmic_read_adc`**, so the 1.9 % does not transfer directly to our sampler. [arXiv:2609.04705](https://arxiv.org/html/2609.04705) |
| arXiv 2609.11932: ±10 % PMIC accuracy | ⚠️ unsupported | Quoted text: *"whose ±10% datasheet accuracy precludes confirmatory claims"*. It names no datasheet. I found **no public DA9091 datasheet**, and searches for Renesas DA9091 ADC specs returned nothing. Do not cite the ±10 %. The paper is still usable as an example of a Pi 5 PMIC energy method (§1). [arXiv:2609.11932](https://arxiv.org/html/2609.11932) |

Independent cross-check (forum, not citable as evidence): a Raspberry Pi forum user saw a POWER-Z KM003C read
**2.9–3.1 W at idle while a rail-sum script read 2.05–2.3 W**. Another user says total draw always exceeds the rail
sum, because the PMIC does not see the 5 V rail and the readings exclude conversion losses
([forum t=359249](https://forums.raspberrypi.com/viewtopic.php?t=359249)). The jfikar fit predicts 2.94–3.22 W for
2.05–2.3 W, so the two independent observations agree.

---

## 1. Pi 5 energy via the PMIC, and Pi 5 papers that used an external meter

### 1a. Papers that read the PMIC with `vcgencmd pmic_read_adc`

All are arXiv preprints. **I found no peer-reviewed paper that uses `pmic_read_adc`**, and none validates it against a
meter. The jfikar repo remains the only public calibration.

| Paper | What they measured | Rail sum | Rate | Validated? |
|---|---|---|---|---|
| Oliveira, *Throughput per Megabyte…on a Raspberry Pi 5*, arXiv:2609.11932 (2026) ✅ | Mean power and energy per HTTP request | *"∑ᵢIᵢ⋅Vᵢ over all PMIC rails reported by vcgencmd pmic_read_adc at 1 Hz"* (Table 6). Energy = P×t_w/N. Labels energy *"Exploratory only"* and calls the metric *"partially tautological (P×t/N)"*. | 1 Hz | No. Cites an unnamed "datasheet" for ±10 %. |
| Setyawan et al., *CoDAT…Efficient Edge Action Recognition*, arXiv:2608.06691 (2026) ✅ | mJ/frame for action recognition on a Pi 5 CPU, 4 threads | *"summing P̄=V×I across all matched voltage-current rail pairs"*. E_frame = P̄·T_total/(N_frames·N_rep). Protocol: 3 runs × 100 reps, 10 s warm-up, 30 s cooldown. | not stated | No. ❌ The paper names the PMIC as *"MXL7704"*, which is wrong for the Pi 5 (the docs and other papers say DA9091). Cite it for the method only. [HTML](https://arxiv.org/html/2608.06691v1) |
| Tummalapalli et al., *LLM Inference at the Edge…*, arXiv:2603.23640 (2026) ⚠️ | Pi 5 + Hailo-10H power | *"estimated from the RPi 5 PMIC's VDD_CORE_A current and VDD_CORE and EXT5V_V voltage rails"* *"sampled via the INA219 sensor at 1 kHz"*. The description is internally inconsistent: an INA219 cannot read PMIC ADCs. | "1 kHz" | No. **Avoid citing.** [HTML](https://arxiv.org/html/2603.23640v1) |
| macclab-stevens/power-measurement-pi (GitHub, no paper) ⚠️ | Per-op YOLO power, VDD_CORE only | — | ~190 Hz for one rail | No. Useful number: one rail takes **~4–5 ms** per call and the **full 26-line dump ~24 ms**. [repo](https://github.com/macclab-stevens/power-measurement-pi) |

### 1b. Pi 5 papers with an external meter: the stronger alternative to cite

| Paper | Instrument and boundary | Energy definition and idle | Repeats |
|---|---|---|---|
| Marasini & Zhou, arXiv:2609.04705 (2026) ✅ | POWER-Z KM003C **inline on the USB-C supply, 1 kS/s**, so fan draw on the same bus is included | J/frame = P̄/FPS. No idle subtraction; an *"active-power filter"* keeps samples above 4 W ("Pi 5 idles at ≈2.5 W") and *"compensates for ±30–60 s of manual drift in Power-Z start/stop timing"* | *"three times per main condition"*, 30-min runs, ≥15-min cooldown, start gated on the idle-temperature band, ambient 23 ± 2 °C |
| Liao, Kondle, Liu, Gu, *Seeing at the Edge…*, arXiv:2610.20136 (2026) ✅ | **INA228 in series with the Pi 5's 5 V input**, a hardware energy accumulator, plus a 50 Hz software trace for cross-checking | Reports **gross and active**: *E_active = E_gross − P_idle·t*, with idle *"measured once per daemon state"*. A trial is valid only if hardware and software agree within 10 % | 1 warm-up + 3 measured trials per prompt; Holm correction |
| Alqahtani, Cheema, Rodriguez, Toosi, arXiv:2409.16808 (v3 2026) ✅ | UM25C USB meter, read over Bluetooth from a separate device. Rate not stated. Pi 3/4/5 ± Coral, Pi 5 + AI HAT+, Jetsons | **(TE − BE)/NR**: base energy from *"idle for a fixed duration of 5 minutes"*, then total energy over the same duration under load | *"Each experiment is repeated three times"*, 5-min runs |
| Suchý & Turčaník, *Review of large YOLOv8 and RT-DETR energy efficiency on edge devices for real-time detection*, Sci. Rep. 16:10908 (2026) ✅ (via PMC) | Pi 5 fed by an OWON SPE8205 bench supply, with an **FNIRSI FNB48P USB meter inline**. Values *"represent the complete electrical load of the platform"* | E = ΣP(tᵢ)Δt. Reports FPS/W. No idle handling stated | Not stated. [PMC13039985](https://pmc.ncbi.nlm.nih.gov/articles/PMC13039985/) |

**Takeaways.**
- Peer-reviewed Pi 5 energy work uses **external meters on the 5 V input**. We should say plainly that ours is a lower
  boundary (the PMIC outputs) and cite 2609.04705 and 2610.20136 as the stronger method we could not use.
- **2610.20136 is the closest methodological twin to ours.** It reports gross and idle-subtracted energy and argues for
  within-platform ratios. Its own boundary admission is useful: *"Absolute Joules are uncalibrated."*

---

## 2. Energy-per-inference methodology in edge ML

**MLPerf Power.** Tschand et al., HPCA 2025, arXiv:2410.12032 ✅
([HTML v2](https://arxiv.org/html/2410.12032v2); DOI 10.1109/HPCA61900.2025.00092 via Crossref)
- Boundary: *"a fundamental tenet of MLPerf Power is the emphasis on measuring full system power consumption"* (§IV-A).
  Myth #1 rejects the idea that measuring only *"specific ML components, such as accelerators or GPUs, is adequate"* (§III-C).
- Energy per inference: *"The host then integrates the power over the indicated interval and divides by the number of
  inferences to calculate the energy per inference"* (§IV-B). This is exactly our gross J/frame.
- Window: *"If a workload finishes executing before the 60-second mark, it is run in a loop until this threshold is
  reached"*; *"we require a minimum of 60 seconds of power data collection for most scenarios"* (§IV-A, §IV-D). **Our
  180 s run and 60 s idle window both meet this.**
- Inference power is measured *"during the actual inference computations"*. Offline pre-processing and data setup are
  excluded. Our window includes per-frame pre- and post-processing on purpose (end-to-end), so state that difference.
- Edge systems: *"must employ SPEC-approved devices"* (§IV-D); the edge methodology *"requires the use of a
  SPEC-certified power meter such as the Yokogawa WT310"* (§III-A). Telemetry: *"we require documentation of the
  telemetry systems used, including their accuracy specifications and calibration procedures"* (§IV-D). Out-of-band:
  *"We recommend out-of-band measurement techniques to minimize interference with the ML job itself"* (§IV-C).
- **Implication:** our method is MLPerf-*inspired* (energy = ∫P/N, ≥60 s window), **not MLPerf-compliant** (no SPEC
  analyzer, not at the wall). Say exactly that.

**MLCommons power rules** ✅
([inference_policies/power_measurement.adoc](https://github.com/mlcommons/inference_policies/blob/master/power_measurement.adoc))
- *"Power is measured at the system level ("at the wall")"*.
- *"Power is evaluated only in the performance phase"*.
- Against cherry-picking: *"It is not permitted to run the same benchmark and scenario 3 times and report the highest
  performance"*.
- The document sets no idle-subtraction rule and no numeric sampling rate. The SPEC PTDaemon sampling rate is not
  stated there, and I did not verify it elsewhere.

**Idle baseline / net (dynamic) energy.**
- Alqahtani et al. (above) ✅ define net energy per request as (TE − BE)/NR with a 5-min idle baseline. Their *base
  energy* is our idle window.
- Liao et al. 2610.20136 ✅ report both, and show the choice matters: an NPU-vs-CPU reduction goes from 65.5 % (gross)
  to 80.4 % (active). **Report both, as we already plan.**
- Hadidi et al., IISWC 2019 ✅ (PDF read) report idle and average power per device separately (Table "Measured idle and
  average power while executing DNNs"). Power on USB-powered boards came from a *"USB digital multimeter [UM25C] that
  records voltage and current every second"*. This is precedent for **1 Hz sampling in a peer-reviewed edge-DNN energy
  study**, and our 2 Hz is faster.

**Energy = P̄·latency at high rate on phones.** Tu et al., SEC 2023 ✅
([HTML](https://arxiv.org/html/2310.18329v2)) use a Monsoon monitor at 5 kHz, *"Energy consumption is calculated by
multiplying the measured power consumption by the model/kernel inference latency"*, and *"Each power and latency value
is the average of at least 100 inference runs."*

**Repeats and statistics.** Hoefler & Belli, SC15 ⚠️ (bibliographic data verified, full text not re-read) is the
standard reference for reporting variability and confidence intervals rather than single numbers. Among Pi 5 papers,
**n = 3 is the norm**: 2609.04705, 2409.16808, 2610.20136 and 2608.06691 all use three runs.

---

## 3. Software power telemetry vs meters: biased, partial boundaries, compare within one platform only

| Source | Finding we cite | Status |
|---|---|---|
| Raspberry Pi docs and vcgencmd man page (§0) | The PMIC sum excludes the 5 V loads and *"You should not expect this to add up to the wattage of the source power supply"*. This is the **vendor's own boundary statement**. | ✅ |
| Shalavi, Khoshsirat, Stellini, Zanella, Rossi, *Accurate Calibration of Power Measurements from Internal Power Sensors on NVIDIA Jetson Devices*, IEEE EDGE 2023, pp. 166–170 | INA3221 sensors *"underestimate the actual power by up to 50% in most cases"*. The gap is attributed to draw between plug and sensor and is near-constant per device. A **per-device linear** fit `a·x + b` (e.g. Nano 1.11x + 232.6 mW, Xavier NX 1.10x + 3130 mW) brings the error to ≤3 %. The fit differs per device. This is the **direct Jetson analogue of jfikar's Pi 5 fit**, and it is peer-reviewed. [arXiv:2306.13107](https://arxiv.org/html/2306.13107v1) | ✅ |
| Yang, Adámek, Armour, *Part-time Power Measurements: nvidia-smi's Lack of Attention*, arXiv:2312.02741 (v3 2024) | Error is *"proportional, rather than the flat ±5W NVIDIA claimed"*. *"the relationship remains almost perfectly linear for each GPU itself"*, so *"the direction of power efficiency improvement will be accurate, albeit with a minor magnitude error"*. But *"comparing results from different GPUs would be erroneous"*. A100/H100 sample only *25 % of runtime*. **This is the cleanest quote for our "within-platform only" rule.** It is a preprint: no venue found. | ✅ |
| Holly, Wendt, Lechner, *Profiling Energy Consumption of DNNs on NVIDIA Jetson Nano*, IGSC 2020, DOI 10.1109/IGSC51522.2020.9290876 | On-board vs shunt and oscilloscope: up to 1.0 W difference (per search snippet). | ⚠️ not read |
| Tu et al., SEC 2023 | Phone built-in current sensors update every ~100 ms–1 s and *"cannot fully capture the fine-grained, kernel-level power variations"*. On a Huawei P40 Lite they gave **10.3 % (CPU) and 30.64 % (GPU) energy error** against a Monsoon monitor. | ✅ |
| Khan, Hirki, Niemi, Nurminen, Ou, *RAPL in Action*, ACM TOMPECS 3(2), 2018, DOI 10.1145/3177754 | RAPL is highly correlated with plug power and has *negligible* overhead, but is a package/DRAM boundary, not the whole system. | ⚠️ abstract only (Crossref) |
| Liao et al. 2610.20136 | Shared proportional errors (shunt, gain, oscillator) **cancel within a platform and in ratios, but not in absolute Joules**. They also *"compare deployment stacks rather than silicon"*. | ✅ |

**Why our rule holds, made explicit.** Assume jfikar's linear model, P_in = a·P_pmic + b with a = 1.1451 and b = 0.5879 W.
- **Net J/frame:** the offset b cancels in P_run − P_idle, so net J/frame from the rail sum is a **constant 1/a = 87.3 %**
  of the true input-side net. **Ratios of net J/frame between Pi 5 cells are exact** under this model.
- **Gross J/frame ratios** are compressed slightly toward 1 by the missing offset. For example, 4 W vs 6 W on the rail
  sum gives a ratio of 1.50, but 1.44 at the input.
- **Across platforms**, a and b differ per device (Shalavi Table, Yang §4.4), so no ratio survives.

This is my derivation from cited fits, not a published result. It is worth one sentence in the paper.

---

## 4. Sampler overhead

- **2609.04705** ✅ is the only Pi 5 measurement: **1.90 % FPS loss at 2 Hz**, from paired n=3 × 5-min runs. Their
  sampler does not read the PMIC, so it is a lower bound for ours.
- **Cost of a `pmic_read_adc` call:** ~24 ms for the full dump (macclab repo ⚠️). At 2 Hz that is ~5 % of one core's
  time, plus `fork`/`exec` of `vcgencmd` each time. Our once-off with/without FPS measurement is exactly the right
  control. Report it the way 2609.04705 does: relative %, absolute ms/frame, n, run length.
- **The sampler also adds power.** Because the sampler runs in both the idle and run windows, its power largely
  cancels in **net** J/frame. It does not cancel in **gross** J/frame. Our idle window therefore must include the
  sampler; check that it does.
- MLPerf Power recommends out-of-band measurement *"to minimize interference with the ML job itself"* (§IV-C) ✅. Khan
  et al. report RAPL overhead as negligible ⚠️.
- Mytkowicz et al., *Producing wrong data without doing anything obviously wrong!*, ASPLOS 2009, DOI
  10.1145/1508284.1508275 ⚠️ (bibliographic only) is the general measurement-bias/observer-effect reference if one is
  wanted.

---

## 5. Temperature and energy

- **Leakage rises with temperature, and power and temperature form a feedback loop.** Bhat, Gumussoy, Ogras, *Analysis
  and Control of Power-Temperature Dynamics in Heterogeneous Multiprocessors*, IEEE TCST (2020/21), arXiv:2003.11081 ✅:
  *"As a result of the increase in temperature, there is an exponential rise in the leakage power."* *"Power
  consumption and temperature form a well-known positive feedback system."* They model leakage as V·κ₁·T²·e^(κ₂/T),
  measured on an Odroid-XU3 (ARM big.LITTLE). [ar5iv](https://ar5iv.labs.arxiv.org/html/2003.11081)
- **Classic device reference:** Kim et al., *Leakage current: Moore's law meets static power*, IEEE Computer 36(12):68–75,
  2003, DOI 10.1109/MC.2003.1250885 ⚠️ (Crossref metadata only; the temperature statement is not re-read).
- **On the Pi 5 itself**, 2609.04705 ✅:
  - leakage breaks their linear thermal model: *"observed 88–89 °C → nonlinear leakage at elevated junction temp"*;
  - the passive envelope *"closes at ambient temperatures (≥27 °C) where nonlinear leakage defeats DVFS-based control"*;
  - Static-S0 throttled 1,823 times and records **0.586 J/frame vs 0.524** for the unthrottled S1/S2 states. Frequency
    differs between those states, so this is not a pure temperature effect.
- **Throttling changes throughput.** Benoit-Cattin, Velasco-Montero, Fernández-Berni, *Impact of Thermal Throttling on
  Long-Term Visual Inference in a CPU-based Edge Device*, Electronics 9:2106 (2020), arXiv:2010.06291 ⚠️:
  - active cooling improved throughput by up to ~90 %;
  - ambient 22–36 °C moved throughput by up to 27.7 %;
  - this was measured on a Pi 4B.
- **Pi 5 fan.** The fan pin is *"+5V"* and the official case fan *"draws approximately 60 mA at 5 V (300 mW)"*
  ([docs](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html)) ✅.
  - My inference: fan power is on the 5 V path, so the **rail sum never sees it**.
  - The fan still changes leakage *indirectly* by cooling the chip.
  - Fan steps: 0 % below 50 °C, 30 % at 50 °C, 100 % at 75 °C.

---

## 6. What a reviewer could reject, and the literature's answer

| Objection | How the literature handles it | Our fix (cheap first) |
|---|---|---|
| **No wall or USB meter: "your watts are wrong."** | The vendor states the boundary (§0). On-board sensors are linear per device (Shalavi; Yang). Liao et al. publish with *"uncalibrated"* absolute Joules but valid within-platform ratios. | Call it **"PMIC rail power (a lower bound on board input power)"**. Report jfikar-corrected input power as a secondary, clearly labelled estimate. Make claims **only as within-Pi ratios**, net J/frame preferred (offset cancels, §3). Never quote ±10 %. |
| **One run per cell.** | Pi 5 papers use n = 3 (2609.04705, 2409.16808, 2610.20136, 2608.06691). MLCommons forbids best-of-N. Hoefler & Belli ask for variability. | (a) Within-run uncertainty from the 360 samples by **batch means**: e.g. 6 × 30 s batches gives the s.e. of P̄ without assuming independence. (b) **Repeat 2–3 cells three times** (say the fastest, slowest and one INT8) to estimate run-to-run CV, and state it. If CV < the gaps between cells, single runs are defensible. |
| **0.5 s sampling vs a ~5 FPS (~0.2 s) workload.** | Sampling rate does not bias the **mean** if sample phases are spread over the frame cycle. Hadidi et al. used 1 Hz in a peer-reviewed study. Yang et al. show a sensor that samples only part of the runtime can bias energy, and recommend **phase-shifting / randomised delays**. | Real risk: 0.5 s / 0.2 s = 2.5 frame periods, so a strictly periodic sampler hits **only two frame phases**. Jitter from `vcgencmd` (~24 ms) and the OS probably breaks this, but nothing documents the PMIC ADC's integration window. Add **uniform random jitter** (e.g. 0.5 s ± 0.1 s) to the sampler, or show on one cell that P̄ matches a randomised schedule within the batch-means s.e. |
| **Idle measured first, cold; run is hot.** | 2609.04705 gates each run on the SoC returning to the idle-temperature band, with ≥15 min cooldown and logged ambient. | Log SoC temperature at idle start and at run end, plus `get_throttled`, for every cell. Gate the start on a temperature band. Report cooling (fan on/off) and ambient. Note that net J/frame includes temperature-driven leakage. |
| **Pre/post-processing inside the window, unlike MLPerf.** | MLPerf measures the inference computation only. End-to-end studies (Sci. Rep. 2026, 2610.20136) include the pipeline. | State the boundary: "end-to-end per-frame (pre + model + post at conf 0.25)". It is the deployment-relevant number. |
| **Sampler perturbs the workload.** | 2609.04705 reports 1.9 % at 2 Hz. MLPerf prefers out-of-band measurement. | Already measured. Report % and ms/frame, and confirm the idle window also ran the sampler. |
| **Rail list or pairing errors.** | jfikar pairs currents with the first 12 voltages by line order. 2608.06691 pairs by name. | Make sure our parser pairs **by rail name** (`X_A` ↔ `X_V`) and excludes `EXT5V_V` (it has no current). Say so in the paper. |

---

## Reference list for the paper

[1] Raspberry Pi Ltd. *vcgencmd(1) manual page*, `raspberrypi/utils`. <https://github.com/raspberrypi/utils/blob/master/vcgencmd/vcgencmd.1> (accessed 2026-10-23).
For: what `pmic_read_adc` reports, and that 5 V loads are excluded.

[2] Raspberry Pi Ltd. *Raspberry Pi hardware documentation: Power supply.* <https://www.raspberrypi.com/documentation/computers/raspberry-pi.html> (accessed 2026-10-23).
For: the rail sum *"should not … add up to the wattage of the source power supply"*; the 5 V fan pin; fan thresholds; throttling at 80–85 °C.

[3] J. Fikar. *RPi5-power* (software and calibration), GitHub, 2024. <https://github.com/jfikar/RPi5-power>.
For: the linear rail-sum → USB-C-meter calibration, P_in = 1.1451·P + 0.5879 W. Not peer-reviewed.

[4] A. Tschand et al. "MLPerf Power: Benchmarking the Energy Efficiency of Machine Learning Systems from μWatts to MWatts for Sustainable AI." *IEEE HPCA 2025*, pp. 1201–1216. DOI 10.1109/HPCA61900.2025.00092; arXiv:2410.12032.
For: energy/inference = ∫P/N; ≥60 s window; full-system boundary; documenting telemetry accuracy.

[5] MLCommons. *MLPerf Inference Power Measurement* rules. <https://github.com/mlcommons/inference_policies/blob/master/power_measurement.adoc>.
For: wall-power boundary; no best-of-N reporting.

[6] A. Marasini, Z. Zhou. "Sustainable Edge Vision via Empirically Calibrated DVFS: Eliminating Thermal Throttling on Passively Cooled Hardware." arXiv:2609.04705, 2026.
For: Pi 5 YOLO J/frame with an inline USB-C meter; 1.9 % FPS cost of 2 Hz telemetry; n = 3; leakage on the Pi 5.

[7] J. Liao, H. Kondle, D. Liu, Z. Gu. "Seeing at the Edge: Energy Characterization of Vision-Language Model Inference on Edge Platforms." arXiv:2610.20136, 2026.
For: gross vs idle-subtracted energy on the Pi 5; proportional errors cancel in within-platform ratios.

[8] D. K. Alqahtani, M. A. Cheema, M. A. Rodriguez, A. N. Toosi. "A Comprehensive Evaluation of Deep Learning Object Detection Models on Heterogeneous Edge Devices." arXiv:2409.16808 (v3), 2026.
For: (TE − BE)/N net energy with a measured idle baseline on the Pi 5; 3 repeats.

[9] W. Oliveira. "Throughput per Megabyte: A Pilot Benchmark of Language-Stack Efficiency for Self-Hosted HTTP Services on a Raspberry Pi 5." arXiv:2609.11932, 2026.
For: prior use of ∑V·I over `pmic_read_adc` rails at 1 Hz. Not for its ±10 % figure.

[10] N. Setyawan et al. "CoDAT: Collaborative Dual-Attention Transformer with Low-Cost Temporal Modeling for Efficient Edge Action Recognition." arXiv:2608.06691, 2026.
For: prior per-frame energy from summed PMIC rails on a Pi 5 CPU.

[11] I. Suchý, M. Turčaník. "Review of large YOLOv8 and RT-DETR energy efficiency on edge devices for real-time detection." *Scientific Reports* 16:10908, 2026. DOI 10.1038/s41598-026-46453-6.
For: peer-reviewed Pi 5 YOLO energy with an inline USB meter (stronger alternative).

[12] R. Hadidi, J. Cao, Y. Xie, B. Asgari, T. Krishna, H. Kim. "Characterizing the Deployment of Deep Neural Networks on Commercial Edge Devices." *IEEE IISWC 2019*, pp. 35–48. DOI 10.1109/IISWC47752.2019.9041955.
For: edge-DNN energy per inference with a 1 Hz USB meter; idle vs active power.

[13] N. Shalavi, A. Khoshsirat, M. Stellini, A. Zanella, M. Rossi. "Accurate Calibration of Power Measurements from Internal Power Sensors on NVIDIA Jetson Devices." *IEEE EDGE 2023*, pp. 166–170. DOI 10.1109/EDGE60047.2023.00034; arXiv:2306.13107.
For: on-board sensors under-read by up to 50 %, with a per-device linear correction.

[14] Z. Yang, K. Adámek, W. Armour. "Part-time Power Measurements: nvidia-smi's Lack of Attention." arXiv:2312.02741, 2023 (v3 2024).
For: the on-board sensor error is linear per device, so within-device comparisons hold and cross-device ones do not; partial sampling biases energy.

[15] X. Tu, A. Mallik, D. Chen, K. Han, O. Altintas, H. Wang, J. Xie. "Unveiling Energy Efficiency in Deep Learning: Measurement, Prediction, and Scoring across Edge Devices." *ACM/IEEE SEC 2023*, pp. 80–93. DOI 10.1145/3583740.3628442.
For: phone built-in sensors err 10–31 % against a Monsoon monitor.

[16] K. N. Khan, M. Hirki, T. Niemi, J. K. Nurminen, Z. Ou. "RAPL in Action: Experiences in Using RAPL for Power Measurements." *ACM TOMPECS* 3(2):1–26, 2018. DOI 10.1145/3177754.
For: software power counters correlate with plug power but are a partial boundary; telemetry overhead.

[17] G. Bhat, S. Gumussoy, U. Y. Ogras. "Analysis and Control of Power-Temperature Dynamics in Heterogeneous Multiprocessors." *IEEE Trans. Control Systems Technology*, DOI 10.1109/TCST.2020.2974421; arXiv:2003.11081.
For: leakage rises exponentially with temperature; power–temperature feedback.

[18] T. Benoit-Cattin, D. Velasco-Montero, J. Fernández-Berni. "Impact of Thermal Throttling on Long-Term Visual Inference in a CPU-based Edge Device." *Electronics* 9:2106, 2020. arXiv:2010.06291.
For: throttling and ambient temperature change sustained inference throughput on a Pi.

[19] T. Hoefler, R. Belli. "Scientific Benchmarking of Parallel Computing Systems: Twelve Ways to Tell the Masses when Reporting Performance Results." *SC15*, 73:1–73:12. DOI 10.1145/2807591.2807644.
For: reporting variability and repeats.

Optional: [20] N. S. Kim et al. "Leakage current: Moore's law meets static power." *IEEE Computer* 36(12):68–75, 2003. DOI 10.1109/MC.2003.1250885. Classic leakage reference; full text not re-read.

Not recommended: arXiv:2603.23640, whose PMIC/INA219 description is inconsistent.

---

## Suggested methods paragraph

> **Energy.** On the Raspberry Pi 5 we read the PMIC's on-board ADCs with `vcgencmd pmic_read_adc` [1] every 0.5 s
> and sum V×I over all output rails, pairing each rail's current with its voltage by name. This *rail power* excludes
> loads fed directly from 5 V (USB, fan) and the PMIC's conversion losses [1, 2], so it is a lower bound on board input
> power. A public calibration against a USB-C meter finds it linear in input power [3], as on-board sensors on other
> edge boards are [13, 14]. Each cell has a 60 s idle window followed by a 180 s run, which meets MLPerf Power's 60 s
> minimum [4]. The run is end-to-end: pre-processing, inference and post-processing at confidence 0.25, batch 1,
> 4 threads. We report J/frame = P̄_run·t/N [4] and net J/frame = (P̄_run − P̄_idle)·t/N [7, 8]. Because sensor bias is
> device-specific [13, 14], we compare J/frame only between Pi 5 configurations, never across platforms. Polling cost
> X % FPS (measured with and without the sampler; cf. [6]). Run-to-run variation was Y % (three repeats of Z cells) [19].

(≈150 words. Fill in X, Y, Z. Drop the last sentence if the repeats are not run.)

---

## Sources

Primary, docs and code: [1] [2] [3] [5] above; [forum t=359249](https://forums.raspberrypi.com/viewtopic.php?t=359249)
(corroboration only); [macclab-stevens/power-measurement-pi](https://github.com/macclab-stevens/power-measurement-pi)
(call latency only).

Papers read in full or in the relevant sections (✅): arXiv 2609.04705, 2609.11932, 2608.06691, 2610.20136, 2409.16808
(v3), 2410.12032 (v2), 2306.13107, 2312.02741 (v3), 2310.18329 (v2), 2003.11081 (ar5iv), 2603.23640; Sci. Rep. 16:10908
(PMC); Hadidi et al. IISWC 2019 (NSF PAR PDF). Bibliographic data for DOIs from Crossref.

Metadata or abstract only (⚠️): Khan et al. 2018; Kim et al. 2003; Holly et al. 2020; Hoefler & Belli 2015;
Mytkowicz et al. 2009; Benoit-Cattin et al. 2020 (abstract).

Searched and found nothing:
- a peer-reviewed paper using `pmic_read_adc`;
- any paper validating `pmic_read_adc` against a meter (only jfikar's repo);
- a public DA9091 datasheet or ADC accuracy spec;
- SPEC PTDaemon's required sampling rate (not in the MLCommons doc).
