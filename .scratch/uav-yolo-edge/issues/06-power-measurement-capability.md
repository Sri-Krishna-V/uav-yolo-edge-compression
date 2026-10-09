# Power measurement capability

Type: task
Status: open
Blocked by: none

## Question

Acquire the ability to measure power and energy-per-frame on every platform in the matrix, or decide explicitly and on the record to drop the metric where we cannot.

**The constraint that forces this ticket:** the Raspberry Pi 5 has no software power telemetry. The Cortex-A76 has no RAPL equivalent, psutil does not expose power, and vcgencmd reports voltage and temperature but not draw. Software-only power figures on the Pi would be a TDP guess dressed as a measurement, and the Pi is the primary platform of the study. Gap 02 in the proposal explicitly promises power correlation, so this is load-bearing for the stated contribution.

Per-platform status:

| Platform | Mechanism | Status |
|---|---|---|
| Raspberry Pi 5 | inline USB-C PD power meter (hardware) | NOT OWNED - action needed |
| Jetson | tegrastats / INA3221 rails | free, on-board |
| Galaxy M35 | Android BatteryManager / dumpsys batterystats | free, noisy, needs long runs and screen-off discipline |
| Laptop CPU | Intel RAPL or AMD equivalent | free, package power only, acceptable since inference is CPU-only there |

Work to do:

1. Buy or borrow an inline USB-C power meter for the Pi 5 (roughly 1,500-3,000 INR; a UM25C or equivalent PD meter). Check with the EEE dept first - they likely have one, and they already have the Jetson.
2. Verify each mechanism actually reports plausible numbers under load, before any benchmark depends on it.
3. If the Pi meter cannot be obtained, record that decision here and cut energy metrics to the three self-reporting platforms, to be stated as an explicit limitation in the paper.

Resolution records what was acquired, where the readings come from on each platform, and the sampling method.
