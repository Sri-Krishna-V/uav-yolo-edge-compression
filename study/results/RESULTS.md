# Results

## Result B: NMS time against objects per image on the Pi 5

Built by `study/scripts/studynms.py` (#45) from the per-image `timings.csv`, `meta.json` and `score.json` of cells already
benchmarked; nothing was run. Every post-processing time below is labelled with its threshold: **0.25 is the deployment
threshold** (the one latency is measured at), **0.001 is the evaluation threshold** (the one accuracy is scored at).

Files: `figures/result_b_nms.pdf` / `.png` (A2, ONNX Runtime INT8: post-processing against objects per image and against the
candidates entering NMS), `figures/result_b_nms_pooled.pdf` / `.png` (the one-to-many head over every Pi 5 cell at 640),
`tables/result_b_heads.csv` / `.tex` (the table), `tables/result_b_bins.csv` (the figure's bins), `tables/result_b_slopes.csv`.

### The head cells

A2 (YOLO26s, 640), one-to-many head against NMS-free head, same runtime and precision, four pairs: ONNX Runtime FP32, ONNX
Runtime INT8 (stock) and OpenVINO FP32 from session 2, and ONNX Runtime INT8 per-channel from session 7. Medians, in ms:

| Pair | Post-processing at 0.25, one-to-many / NMS-free | Post-processing at 0.001, one-to-many / NMS-free | Latency, one-to-many / NMS-free | AP | AP-small |
|---|---|---|---|---|---|
| ONNX Runtime FP32 (session 2) | 1.11 / 0.17 | 15.35 / 0.20 | 383.1 t / 386.9 t | 0.1943 / 0.1891 | 0.0995 / 0.0991 |
| ONNX Runtime INT8 (session 2) | 1.10 / 0.18 | 15.27 / 0.20 | 275.9 / 279.1 | 0.1549 / 0.1540 | 0.0698 / 0.0770 |
| OpenVINO FP32 (session 2) | 1.12 / 0.15 | 15.59 / 0.16 | 302.2 t / 310.5 t | 0.1943 / 0.1891 | 0.0995 / 0.0991 |
| ONNX Runtime INT8 per-channel (session 7) | 1.07 / 0.18 | 14.82 / 0.20 | 277.4 / 279.5 | 0.1795 / 0.1739 | 0.0886 / 0.0881 |

t: the cell throttled (summary.csv). The full table, with pre-processing, network, each head's share of the latency and the
difference between the heads in every column, is `tables/result_b_heads.csv`.

- **At the deployment threshold, NMS costs about 1 ms on the Pi 5**: the one-to-many head's post-processing at 0.25 is
  0.89-0.97 ms longer than the NMS-free head's, and is 0.29-0.40 % of the latency (the NMS-free head's, 0.04-0.07 %).
- **At the evaluation threshold it costs about 15 ms**: post-processing at 0.001 is 14.8-15.6 ms for the one-to-many head
  against 0.16-0.20 ms for the NMS-free head. That is the scoring pass, not the deployed one.
- The NMS-free head still has the longer latency (+3.2 ms on the clean INT8 pair): its network takes 4.0 ms more than
  the 0.9 ms it saves in post-processing.

### How the cost grows with the objects in an image

The figure bins the 1,610 test-dev images by ground-truth objects per image (1 to 461, median 36) into ten bins of about
160, and plots each bin's median and interquartile range. ONNX Runtime INT8, one-to-many head:

- **Post-processing at 0.25 grows with the objects, from 0.58 ms (median of the 148 images with 1-11 objects) to 2.29 ms (the 163
  images with 88-461)**: 1.15 ms per 100 objects, intercept 0.72 ms, r = 0.72 over the images. At the busiest, NMS at the deployment threshold
  is still 0.8 % of the latency.
- **Post-processing at 0.001 grows from 7.8 ms to 19.5 ms**, steeply up to about 25 objects and slowly after (5.95 ms per 100
  objects over the images, intercept 12.0 ms, r = 0.61: a line fits it less well).
- **The candidates entering NMS explain it better than the objects do, at 0.001**: 3.80 ms per 1,000 candidates, r = 0.82
  (0.53 ms per 1,000 at 0.25, r = 0.69; the count at 0.25 is not logged, so that is against the count at 0.001). `n_cand` is
  studybench's count of rows entering NMS in the evaluation pass: boxes with a class score above 0.001, one row per (box,
  class), before the 30,000-row cap. It runs from 128 to 7,469 an image (median 1,414).
- **The NMS-free head is flat**: its bin medians run 0.09-0.22 ms at 0.25 and 0.12-0.25 ms at 0.001 with no trend in the
  objects (all four pairs: slopes of at most 0.002 ms per 100 objects at 0.25 and 0.011 at 0.001, r below 0.14).
- The other three pairs agree: 1.15-1.23 ms per 100 objects at 0.25 and 5.95-7.37 at 0.001 (`tables/result_b_slopes.csv`).
- **Pooled over the 26 one-to-many Pi 5 cells at 640** (sessions 1, 2, 3 and 7: both study models, compressed cells included), the
  curve is the same shape. Post-processing at 0.25 goes from 0.55 ms (lowest bin) to 2.18 ms (highest); each cell's own curve runs from
  0.52-0.59 ms to 1.91-2.43 ms. At 0.001 it goes from 6.1 ms to 19.8 ms (cells: 4.6-9.3 to 18.6-21.0 ms). The figure shows each cell's curve.

Reading the figure: panel (a) is against ground-truth objects per image; panel (b) against the candidates entering NMS at
0.001. The NMS-free head's own `n_cand` is its (at most 500) boxes over the threshold and says nothing about NMS, so its times
in panel (b) are plotted against the one-to-many head's count for the same image (both ran the same 1,610 images). The right-hand
bin of each curve spans many images with a long tail (objects up to 461), and is plotted at its median (123 objects).
Within an image, the two post-processing passes run in alternating order (the deployment pass first on odd images), so
neither always gets the warm caches.

### How the table was checked

`study/scripts/tests/test_studynms.py` (test 4) builds the table from the tracked session 2 data and compares it with
`verification/pi5_matrix_2_record.md`, *The NMS-free head (result B)*: for all three pairs and both heads, AP, AP-small and
AP-large (4 decimals), latency and network (1 decimal) and post-processing at 0.25 and at 0.001 (2 decimals) equal the
record's. Medians are `meta.json`'s `timing_ms`, AP and precision `score.json`'s. The record's last column (clock median,
images below 2000 MHz) is not in this table; only whether the cell throttled is. Session 7's pair is found by the same code
(test 4 also checks its AP 0.1739, AP-small 0.0881, latency 279.5 against `pi5_matrix_7_record.md`). Objects per image were
checked against the YOLO label files under `datasets/VisDrone/labels/test`: for all 1,610 images, the ground-truth json's count
for image id `i` equals the line count of the label file of `files[i]` (75,102 objects in all), so the id-to-file mapping holds.
