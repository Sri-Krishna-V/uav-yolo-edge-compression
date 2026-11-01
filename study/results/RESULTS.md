# Results

## Result A: the equal-latency comparisons on the Pi 5

Built by `study/scripts/studypairs.py` (#46) from `cells.csv`; nothing was run. Re-run it after `studyresults.py` when a
session lands. An **equal-latency comparison** here is a compressed configuration at a larger input resolution against
a baseline at a smaller one: same platform, model, runtime and head, and median latencies 10 % apart or less, the
difference divided by the smaller of the two (#8). Distilled cells save no latency, so they are never the compressed side. INT8 is ONNX Runtime's,
so the INT8 cells pair with ONNX Runtime FP32 baselines. Differences are the compressed cell minus the baseline.

Files: `tables/equal_latency.csv` / `.tex` (the pairs; the `.tex` holds the Pi 5's), `tables/equal_latency_all.csv`
(every pair of cells within 10 % on a platform, across runtimes too: for the writer, not a paper table) and
`figures/result_a_pi5.pdf` / `.png` (AP-small against latency, one panel per runtime). The phone has 2 cells, so it has
no figure yet. It gets one once it has three.

**The answer.** On the Pi 5, a compression budget spent on input resolution pays off for INT8 and not for pruning.
In every ONNX Runtime pair, for both study models, INT8 at the larger input resolution beats the FP32 baseline in
AP-small, by +0.0044 to +0.0145. Pruned at 960, against the baseline at 800, loses 0.0027-0.0042 AP-small on every
runtime. The INT8 cells are the per-channel ones (INVARIANTS.md section 8).

| Model | Compressed cell | ms | Baseline | ms | Apart | AP-small diff (stock) | AP diff | AP-large diff |
|---|---|---|---|---|---|---|---|---|
| YOLO11s | B1 INT8 at 800 † | 416.2 | A1 FP32 at 640 | 389.8 t | 6.8 % | **+0.0145** (+0.0137) | +0.0178 | +0.0590 |
| YOLO11s | B1 INT8 at 960 | 617.2 | B1 FP32 at 800 † | 629.9 t | 2.1 % | **+0.0096** (+0.0095) | +0.0049 | −0.0010 |
| YOLO11s | B3 INT8 at 1120 † | 832.0 | B1 FP32 at 960 | 909.7 t | 9.3 % | **+0.0049** (+0.0025 §) | +0.0027 | +0.0241 |
| YOLO26s | B2 INT8 at 960 | 646.0 | B2 FP32 at 800 † | 611.9 t | 5.6 % | **+0.0125** (+0.0008) | +0.0077 | −0.0190 |
| YOLO26s | B4 INT8 at 1120 † | 880.6 | B2 FP32 at 960 | 926.5 t | 5.2 % | **+0.0044** (−0.0140 §) | +0.0063 | −0.0048 |
| YOLO11s | P2 FP32 at 960 | 574.6 | B1 FP32 at 800 † | 629.9 t | 9.6 % | −0.0042 | −0.0135 | −0.0695 |
| YOLO26s | P4 FP32 at 960 | 571.7 t | B2 FP32 at 800 † | 611.9 t | 7.0 % | −0.0027 | −0.0119 | −0.0513 |

ONNX Runtime. †: a checkpoint run at an input resolution it was not trained at. t: flagged throttled. (stock): the same
pair with the stock INT8 cell; §: that stock cell is session 6's. AP-medium and each cell's clock are in the CSV.

- **The INT8 gain is +3.2 to +16.1 % of the baseline's AP-small**, and AP rises too (+0.0027 to +0.0178).
- **The gain shrinks as the input resolution grows.** One step up from 640 gives +0.0145 (YOLO11s). From 800 it gives
  +0.0096 and +0.0125, and from 960 +0.0049 and +0.0044.
- **The INT8 gain is in the small objects.** AP-small rises in all five INT8 pairs. AP-large falls in three of them (B1
  and B2 at 960, B4 at 1120: −0.0010 to −0.0190).
- **Pruning loses on every runtime.** P2 and P4 at 960 against the baseline at 800 give AP-small −0.0042 and −0.0027 on
  ONNX Runtime and OpenVINO, and −0.0035 and −0.0032 on NCNN (FP16). On OpenVINO they are 0.0 % and 4.0 % apart, on NCNN
  2.1 % and 1.1 %. AP falls by 0.0117-0.0135.
- **The stock INT8 recipe (sessions 1-6) gave YOLO26s a tie and a loss**: B2 INT8 at 960 +0.0008, B4 INT8 at 1120
  −0.0140. The per-channel recipe turns them into gains. YOLO11s's pairs barely move. The table keeps each stock row
  after its per-channel one, as a row of its own.
- **No pair.** B2 INT8 at 800 is 12.3 % above A2 FP32 at 640 (430.3 against 383.1 ms). The combined cells run only at
  640, the smallest input resolution. OpenVINO has no INT8 cell on the Pi 5 (#38), and NCNN has none, so their only pairs
  are the pruned ones.

Caveats:

- **The FP32 baselines throttled; the INT8 cells did not.** Every ONNX Runtime baseline in a pair was flagged throttled.
  Their images below 2000 MHz: A1 at 640, 381; B1 at 800, 351; B1 at 960, 680; B2 at 800, 388; B2 at 960, 1,377 (median
  clock 1911 MHz). No INT8 cell has an image below 2000 MHz. P4 at 960 throttled too (1,381 images, median 1911 MHz).
  - A throttled latency is if anything high. So an unthrottled baseline would sit further from the two INT8 cells that
    are slower than their baseline (B1 at 800, B2 at 960), and nearer the others.
  - Those two pairs stay pairs. A1 at 640 (389.8 ms) is 0.3 % above the pilot's unthrottled 388.5 ms (session 1's
    record), which puts B1 INT8 at 800 7.1 % above it.
  - B2 at 800 would have to be 4.0 % faster for its pair to pass 10 %. The throttle re-runs moved medians by −3.6 to
    +1.9 % (INVARIANTS.md section 4.7).
- **Session 6.** No headline pair has a session 6 cell. The INT8 cells at 1120 are session 7's, and session 6's baselines
  at 1120 pair with nothing as measured. The two stock rows at 1120 are session 6's (§).
  - B3 INT8 at 1280 (1153.2 ms) and B3 FP32 at 1120 (1315.4, session 6) are 14.1 % apart as measured. With session 6's
    latency taken as 6 % high, they are 7.6 % apart.
  - Session 6's latencies stay as measured (decided on #46), so this is not a pair. It is a footnoted row of
    `equal_latency_all.csv` (AP-small +0.0085).
  - That CSV has 243 rows, and 10 of them are within 10 % only with the offset taken out.
- **Untrained input resolutions.** Every pair has a cell at 800 or 1120: a checkpoint from the 960 or 1280 models, run
  at an input resolution it was not trained at (†; hollow in the figure). The paper says its accuracy there is a little
  lower (#8).
  - Where that cell is the compressed one (B1 INT8 at 800, B3 and B4 INT8 at 1120), it understates the gain.
  - Where it is the baseline (B1 and B2 FP32 at 800), it overstates the INT8 gain and understates the pruned loss.
- **No error bars yet.** Each number is one seed (s0) and one pass over test-dev; the paired bootstrap is #48's.

Reading the figure:

- Each line joins a model's baselines from 640 to 1280, and its numbers are their input resolutions.
- A compressed cell is labelled with its input resolution unless it is at 640.
- An arrow runs from a baseline to the compressed cell it pairs with.
- The marker's shape is the configuration. Hollow markers are untrained input resolutions.
- Only the one-to-many head is drawn: the NMS-free head is result B's.
- Each stock INT8 cell is left out, since its per-channel cell is drawn.

### How it was checked

`study/scripts/tests/test_studypairs.py`:

- **The rule, on literal cells.**
  - Same platform, model, runtime and head; the compressed side at the larger input resolution.
  - 10.0 % is in, and a pair 10.3 % apart over the smaller (9.3 % over the larger) is out.
  - A distilled cell is never the compressed side.
- **The INT8 headline, on literal cells.** The per-channel pair takes over from the stock one once it exists, with no
  code change.
- **The real `cells.csv`, without session 7.** The ONNX Runtime pairs are exactly #46's check list of seven, each the
  headline.
- **The real `cells.csv`, with session 7.** The five pairs of `verification/pi5_matrix_7_record.md`'s table, parsed from
  the record, equal it on:
  - both latencies and the baseline's throttle flag;
  - how far apart they are;
  - the AP-small difference, per-channel and stock.
- **The figure's data.**
  - One panel per runtime, both models in each.
  - Each line runs through 640, 800, 960, 1120 and 1280 in that order.
  - One arrow per headline pair, and no stock or NMS-free cell drawn.
  - The phone gets the figure only once it has three cells; a cell run again in a later session counts once.

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

## The cells table: `cells.csv`

Built by `study/scripts/studyresults.py` (#44): one row per cell, from every Pi 5 matrix session and every Galaxy M34
full-cell or matrix session, by the attempt that `summary.csv` names in `run_dir`. Every table and figure here reads
it; re-run the script when a session lands (`python study/scripts/studyresults.py`). The script's docstring defines
every column; the names do not change.

- **76 cells now**: the Pi 5's 74 (sessions 1-7, all scored) and the phone's 2 (25 October). Twenty of them run a
  checkpoint at an input resolution it was not trained at (`untrained_imgsz` = 1: 800 from the 960 models, 1120 from the
  1280 ones).
- **Throttle columns** (INVARIANTS.md §4.7) are `studyclock.py`'s (#43), from `timings.csv`: `throttled` as recorded,
  `below_2000_mhz`, `clock_min_mhz`, `clock_median_mhz`, `peak_temp_c`, and the bit-3 reads. They are empty for the
  phone, which has no clock lock and no throttle signal.
- **Software versions** come from each cell's `meta.json` (`opencv_version`, `runtime_version`, ...), so the phone's
  sessions of 25 October (OpenCV 4.14.0, ONNX Runtime 1.31.0) and 1 November (5.0.0, 1.31.1) stay apart.
- **Notes** a table must footnote: session 6's cells, "latency about 6 % high against the other sessions' (measured by
  session 7)"; session 7's, "drift flagged across a restart; kept (#37)"; session 7's B2 at 800, whose first attempt
  was interrupted.
- **Checked** by `study/scripts/tests/test_studyresults.py`. Every cell's precision, AP, AP50, AP-small, AR@500 and
  latency equal its row in the session records' results tables (`verification/pi5_matrix_{1..7}_record.md`), and the
  two M34 cells equal `m34_fullcell_record.md` and `m34_fullcell_int8_record.md`.
