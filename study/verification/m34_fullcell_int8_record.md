# Galaxy M34 INT8 full cell: session record

Written on 2026-10-25 at about 18:45 IST, before the session started. The results are added after the run, and nothing
above them is changed.

## Question

- What are the quantized configuration's latency and accuracy on the Galaxy M34: YOLO11s, checkpoint A1
  (`exports/yolo11s-640-s0`), ONNX Runtime INT8, at the matrix resolution (640), on all 1,610 test-dev images?
- How do they compare with the baseline's full cell ([m34_fullcell_record.md](m34_fullcell_record.md))?

## Plan and code

- Plan: `plans/m34-fullcell-int8.json`, sha256 `b6978abb48562147b1168cfdd874e63d4928ab6abc00ce59efc8ce49a0d71c56`. One
  cell: `onnxruntime-int8`, all 1,610 test-dev images. The reference runs, first and last, are as before (FP32, the
  first 200 images).
- The rules of #8, the code (same sha256 values) and the wizard are the same as in
  [m34_speedtest_record.md](m34_speedtest_record.md).
- The cpuset log: `bench/m34/cpuset-m34-fullcell-int8.log`, a sample every 10 s.
- The conditions are those confirmed for the speed tests. The phone and the laptop are on the extension box.

## Analysis plan (fixed before the run)

- **Latency:** the cell's median in `summary.csv` (`e2e_deploy`, at the deployment threshold 0.25).
- **Time of the cell:** from its "attempt 1 starts" to its "attempt 1 complete" in `session.log`.
- **Accuracy:** AP, AP50 and AP-small on all 1,610 images, scored on the laptop.
- **Valid** under the same three conditions: drift within ±3 %, every cpuset sample `/top-app`, no failed cell.
- **Comparison with the baseline**, reported in two ways, because the two full cells ran in different sessions:
  - across sessions: INT8 cell median / FP32 cell median (636.6 ms);
  - within each session: the cell's median / the mean of its own session's two reference runs. For the FP32 full
    cell that was 636.6 / 622.4 = 1.023.
  - The speed test's within-session ratio (0.433, [m34_speedtest_record.md](m34_speedtest_record.md)) remains the
    controlled number. These ratios check it at full length.
- **Accuracy change:** ΔAP and ΔAP-small, INT8 minus FP32, on the same 1,610 images.
- **The slow stretches (#34):** blocks of 10 images, slow when their median net time is at least 1.3× the run's median.
  Report the count and their positions. Report the rise through the cell as the medians of 10 equal parts, as for the
  FP32 full cell.
- **The latency is provisional** until #34 is answered.

## Results

(Added after the run.)
