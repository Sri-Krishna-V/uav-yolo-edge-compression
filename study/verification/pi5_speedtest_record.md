# Raspberry Pi 5 speed test: session record

Written on 2026-10-28, after the run. Vidhyarth ran the session with `pi/wizard.sh` and sent the results. No analysis
plan was written before the run; the question and the cells are #25's.

## Question

On the Raspberry Pi 5 (platform), is the quantized configuration (INT8) slower than the baseline (FP32), for ONNX
Runtime and for OpenVINO? NCNN runs beside them. Model: YOLO11s, checkpoint A1 (`exports/yolo11s-640-s0`), at 640.

## Plan and code

- Plan: `plans/pi5-speedtest.json` (the runner's `plan_sha256` `6c767f87…7050a`). Five cells on the first 200
  test-dev images each: ONNX Runtime FP32 and INT8, OpenVINO FP32 and INT8, and NCNN. The reference runs, first and
  last: ONNX Runtime FP32 at 640 on the same 200 images.
- The rules of #8: 4 threads, 20 untimed warm-up images, the 50 °C start gate with its 3-minute timeout, cell order
  shuffled with seed 0, the drift check at ±3 %, one re-run of a crashed or throttled cell.
- The same scripts, Pi and setup as the fan pilot ([pi5_pilot_record.md](pi5_pilot_record.md)), and **the CPU locked
  at 2000 MHz**, the lock that the pilot passed at. Room 26 °C, as the dev entered it.

## Results

Session `bench/pi5/pi5-speedtest`, 27 October, 01:03–01:29 IST. **Drift: 389.9 → 388.1 ms, −0.5 %.** No run
throttled (`0x0` throughout), and the clock read 2000 MHz on every image.

| Cell (in run order) | Starts, IST | Precision recorded | Median latency, ms | p10–p90, ms | vs ORT FP32 |
|---|---|---|---|---|---|
| reference run, first | 01:04:20 | FP32 | 389.9 | 388.3–398.6 | |
| OpenVINO FP32 | 01:08:27 | FP32 | 294.5 | 293.5–296.2 | 1.33× faster |
| ONNX Runtime INT8 | 01:12:02 | INT8 | 271.6 | 269.5–278.3 | 1.44× faster |
| ONNX Runtime FP32 | 01:16:13 | FP32 | 390.2 | 387.6–393.8 | 1.00 |
| NCNN | 01:20:49 | FP16 | 171.4 | 170.5–172.9 | 2.28× faster |
| OpenVINO INT8 | 01:24:37, 01:27:39 | — | **failed** | | |
| reference run, last | 01:27:51 | FP32 | 388.1 | 386.6–392.3 | |

Latency is the median of `e2e_deploy`: pre + net + post at the deployment threshold, 0.25, batch 1.

### The answer for ONNX Runtime

**INT8 is not slower on the Pi 5. INT8 / FP32 = 271.6 / 390.2 = 0.696: INT8 is 1.44× faster.** With the two reference
runs as the FP32 number instead of the cell, it is 1.43× to 1.44×. The Galaxy M34 gave 2.31× for the same two exports
([m34_speedtest_record.md](m34_speedtest_record.md)).

### OpenVINO INT8: no cell on the Pi 5

Both attempts stopped within 2 s, before any image, on the precision guard of INVARIANTS.md §8:

> OpenVINO 2026.4.1 compiled `model_int8_openvino_model/model.xml` in FP16 despite `INFERENCE_PRECISION_HINT=f32`:
> not a cell. Layers: {'Convolution f16': 45, 'Convolution f32': 43, 'MatMul f32': 2}

- **No layer runs on u8 or i8.** On the Pi 5's ARM CPU, OpenVINO 2026.4.1 executes the quantized model in f16 and f32,
  so even without the guard this would not be an INT8 cell.
- The OpenVINO FP32 cell of the same session compiled in f32, as §8 requires.
- §8's sentence "A quantized OpenVINO model keeps its INT8 layers under these hints" holds on the laptop, not on the
  Pi 5. What this means for the matrix is a decision for the dev: see the ticket raised from this record.

### NCNN

The cell is named `ncnn-fp32` after its export, and the runner records **FP16**: ncnn's defaults compute in FP16 on the
Pi 5's Cortex-A76, as INVARIANTS.md §8 expects. By the precision rule it is a baseline.

### Plumbing check (200 images, not reported)

| Cell | AP | AP-small |
|---|---|---|
| ONNX Runtime FP32 | 0.1981 | 0.1143 |
| ONNX Runtime INT8 | 0.1805 | 0.0996 |
| OpenVINO FP32 | 0.1981 | 0.1143 |
| NCNN (FP16) | 0.1976 | 0.1140 |

The ONNX Runtime cells give the same AP as on the M34, to 4 digits.

### The start gate

4 of the 8 gates opened on the rule (CPU below 50 °C), after 10–150 s; the other 4 at the 3-minute timeout, with the
CPU at 50.7–51.8 °C. See the pilot record.

## Known limitations

- 200 images per cell: enough for the latency question, not for accuracy.
- One session, one seed's order. The cells ran once each.
- The latencies are at 2000 MHz, not the Pi 5's stock 2400 MHz. At 2400 MHz the same ONNX Runtime FP32 cell is about
  8 % faster (the pilot record).
