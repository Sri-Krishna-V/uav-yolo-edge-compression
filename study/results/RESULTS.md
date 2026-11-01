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

## Result C: YOLO11s against YOLO26s on the Pi 5

Built by `study/scripts/studytables.py` (#47) from `cells.csv`; nothing was run. Re-run it after `studyresults.py` when a
session lands. Each pair is the two study models in the same configuration, on the same runtime, at the same precision and
input resolution, with the one-to-many head. The latency ratio is YOLO26s's median latency over YOLO11s's; differences are
YOLO26s minus YOLO11s.

Files: `tables/result_c.csv` / `.tex` (the `.tex` holds the Pi 5's baselines and quantized cells; the CSV also has the
pruned, distilled and combined pairs) and `figures/result_c_pi5.pdf` / `.png` (the latency ratio against input
resolution, one line per runtime and precision).

**The answer.** YOLO26s is the more accurate study model at every input resolution, in FP32 and in per-channel INT8. It
is also the slower one, and its latency gap grows with the input resolution. On the two clean runtimes, where no image
ran below 2000 MHz:

| | 640 | 800 | 960 | 1120 | 1280 |
|---|---|---|---|---|---|
| YOLO26s's latency, NCNN FP16 | +1.3 % | +3.3 % | +4.0 % | +6.1 % | +9.0 % |
| YOLO26s's latency, ONNX Runtime INT8 | +1.9 % | +3.4 % | +4.7 % | +5.8 % | — |
| AP difference, FP32 | +0.0069 | +0.0028 | +0.0016 | +0.0068 | +0.0038 |
| AP-small difference, FP32 | +0.0091 | +0.0021 | +0.0026 | +0.0042 | +0.0039 |
| AP difference, INT8 | +0.0099 | +0.0021 | +0.0056 | +0.0052 | — |
| AP-small difference, INT8 | +0.0105 | +0.0034 | +0.0050 | +0.0021 | — |
| AP difference, INT8 with the stock recipe | −0.0133 | −0.0183 | −0.0157 | −0.0166 | — |

INT8 is per-channel unless the row says stock. FP32 is ONNX Runtime's; OpenVINO gives the same AP to 4 digits, and NCNN
FP16 gives +0.0017 to +0.0066. B4 has no INT8 cell at 1280 (no stock export, INVARIANTS.md section 8).

- **FP32 and INT8 apart.** In FP32, YOLO26s leads by +0.0016 to +0.0069 AP and +0.0021 to +0.0091 AP-small. With the
  per-channel INT8 recipe it leads at every size too. With the stock recipe it trails at every size, by 0.0133-0.0183
  AP, because stock INT8 costs YOLO26s 2.0-2.7× as many AP points as it costs YOLO11s (#41; result D).
- **The latency gap grows with the input resolution**: on NCNN from +1.3 % at 640 to +9.0 % at 1280; on INT8 from
  +1.9 % to +5.8 % (per-channel) and +1.2 % to +6.7 % (stock) at 640-1120. NCNN and per-channel INT8 agree within 0.7
  points at every size they share. YOLO26s has 5.0 % more GFLOPs at 640 and 5.7 % more at 960 (session 2's record), so the
  gap stays below the FLOP gap up to 960, then reaches 9.0 % at 1280 (the records have no FLOP count at 1280). Nothing in
  the records explains the growth.
- **The compressed configurations at 640** (CSV): YOLO26s's pruned, distilled and combined cells are within −2.8 to
  +1.6 % of YOLO11s's latency on ONNX Runtime and NCNN, and +0.0004 to +0.0029 AP ahead, per-channel. The stock combined
  pair is −0.0135 AP, A2's INT8 deficit carried through. P3 against P1 and P4 against P2 run within 10 % of each other's
  latency on every runtime, which is #10's condition for comparing the pruned models.

Caveats:

- **The FP32 runtimes carry the throttle.** Every ONNX Runtime FP32 and OpenVINO pair has a cell that ran images below
  2000 MHz (hollow in the figure). Their ratios swing: ONNX Runtime FP32 from −2.9 % (800) to +6.3 % (1120), OpenVINO
  +2.8 % to +7.5 %. A2's ONNX Runtime attempt held a median 1911 MHz against A1's 2000 and was still 1.7 % faster (session
  2's record). Read the gap from NCNN and INT8.
- **Session 6.** At 1120 both cells of each pair ran in one session (6, or 7 for per-channel INT8), so session 6's
  offset cancels in the ratio.
- **Untrained input resolutions.** At 800 and 1120 both checkpoints run at an input resolution they were not trained at;
  the latency is unaffected and both models are off their trained size alike.
- **No error bars.** One seed (s0), one pass over test-dev (#48). Two trained models differ by seed noise too: YOLO11s's
  three seeds at 640 spread 0.0007 AP and 0.0027 AP-small on test-dev (#9); YOLO26s's spread is not measured. A
  three-seed range is a yardstick, not a test. The AP leads in the table (+0.0016 to +0.0099) are larger than it; the
  AP-small leads of +0.0021 (800, FP32), +0.0026 (960, FP32) and +0.0021 (1120, INT8) are within it, and so is the
  combined pair's +0.0004 AP.

### How result C was checked

`study/scripts/tests/test_studytables.py`:

- **The pair rule, on literal cells.** Same configuration, runtime, precision, INT8 recipe, input resolution and head; the
  ratio is YOLO26s over YOLO11s and the differences YOLO26s minus YOLO11s; a pair is clean only when neither cell ran an
  image below 2000 MHz.
- **Against the records.** All 23 baseline and quantized pairs on the Pi 5 equal the records' YOLO11s-against-YOLO26s
  tables (sessions 2, 4, 5, 6 and 7) in both APs and YOLO26s's latency gap. Five of them: NCNN +1.3 % at 640 (session 2)
  and +9.0 % at 1280 (session 4), per-channel INT8 +5.8 % at 1120 (session 7), OpenVINO +7.5 % at 960 (session 2), ONNX
  Runtime FP32 +6.3 % at 1120 (session 6).
- **The figure's data.** One line per runtime and precision through its input resolutions (NCNN +1.3, +3.3, +4.0, +6.1,
  +9.0 %); the NCNN and INT8 points clean, the ONNX Runtime FP32 ones not; the stock INT8 line behind the per-channel one.
- **Session 7 swaps in with no code change.** Without session 7's cells the INT8 headline pairs are the stock ones; with
  them, the per-channel ones.

## Result D: the runtime and INT8 on the Pi 5

Built by `study/scripts/studytables.py` (#47). Each INT8 cell is set against the FP32 cell of the same checkpoint, input
resolution and runtime (ONNX Runtime, the Pi 5's only INT8 runtime: #38). The speed-up is the FP32 latency over the INT8
latency; the change is INT8 minus FP32, in % of FP32.

Files: `tables/result_d_int8.csv` / `.tex` (INT8 against FP32, per-channel and stock side by side, the Pi 5 and the Galaxy
M34) and `tables/result_d_runtimes.csv` / `.tex` (the three Pi 5 runtimes side by side for each checkpoint). These tables
have no clock columns, so each latency carries the images it ran below 2000 MHz in square brackets (INVARIANTS.md section
4.7); the median clocks are in the latency table.

**The answer.** On the Pi 5, ONNX Runtime INT8 is 1.38-1.51× faster than ONNX Runtime FP32 with either recipe, where
neither cell is session 6's. The per-channel recipe costs 2-10 % of AP; YOLO11s's loss falls as the input grows, and
YOLO26s's levels off above 960. NCNN FP16 beats INT8 on both counts.

| Checkpoint | Input resolution | Speed-up, per-channel (stock) | AP change, per-channel (stock) | AP-small change, per-channel (stock) |
|---|---|---|---|---|
| A1, YOLO11s | 640 | 1.43× (1.43×) | −9.5 % (−10.3 %) | −13.6 % (−15.0 %) |
| B1 | 800 † | 1.51× (1.51×) | −6.3 % (−6.8 %) | −9.8 % (−10.5 %) |
| B1 | 960 | 1.47× (1.48×) | −6.4 % (−6.9 %) | −7.8 % (−7.8 %) |
| B3 | 1120 † | (1.58×) § (1.49×) | −4.8 % (−5.5 %) | −5.4 % (−7.0 %) |
| B3 | 1280 | 1.47× (1.47×) | −2.2 % (−3.9 %) | −2.5 % (−3.7 %) |
| A2, YOLO26s | 640 | 1.38× (1.39×) | −7.6 % (−20.3 %) | −11.0 % (−29.9 %) |
| B2 | 800 † | 1.42× (1.42×) | −6.5 % (−16.3 %) | −8.5 % (−19.6 %) |
| B2 | 960 | 1.43× (1.44×) | −4.7 % (−14.1 %) | −5.9 % (−14.3 %) |
| B4 | 1120 † | (1.59×) § (1.48×) | −5.3 % (−14.3 %) | −6.5 % (−18.6 %) |

†: a checkpoint run at an input resolution it was not trained at. §: session 7's per-channel cell against session 6's
FP32 cell, so about 6 % high and not to be used. At 1120 the speed-up is the stock cells' from within session 6, 1.49×
and 1.48× (#47's comment of 1 November); `result_d_int8.csv`'s `speedup` column gives it, with `speedup_recipe`.
A2's NMS-free head gives 1.38× (1.39×) and −8.0 % (−18.5 %) AP. C1 and C2 have INT8 cells only: no FP32 export of
them was run.

- **The speed-up does not depend on the recipe**: 1.38-1.51× per-channel and 1.39-1.51× stock, where both cells ran
  outside session 6. The two recipes have the same nodes and activation scales and differ only in their weights' scales
  (INVARIANTS.md section 8).
- **The per-channel recipe's loss falls as the input grows, for YOLO11s**: −9.5 % at 640 to −2.2 % at 1280. YOLO26s's
  falls from −7.6 % at 640 to −4.7 % at 960, then −5.3 % at 1120. AP-small loses more than AP in every per-channel row,
  and in every stock row but B3 at 1280 (see *Small-object degradation*).
- **The stock recipe cost YOLO26s 14.1-20.3 % of its AP**, 2.0-2.7× as many AP points as YOLO11s lost (2.0-2.6× in % of
  each model's AP). Per-channel brings it to 4.7-7.6 %, YOLO11s's level, and leaves YOLO11s's loss almost where it was
  (#41).
- **The Pi 5 against the Galaxy M34.** A1 at 640 with the stock recipe is 1.43× faster in INT8 on the Pi 5 and 2.09× on
  the phone (304.8 against 636.6 ms), for the same AP (0.1682 on both). The phone's two cells ran in different sessions
  (25 October); its speed test gave 2.31× within one session, over 200 images (`m34_fullcell_int8_record.md`). The phone
  has no per-channel cell in `cells.csv` yet.
- **The three Pi 5 runtimes** (`result_d_runtimes`): over ONNX Runtime FP32, NCNN FP16 is 1.98-2.28× faster and OpenVINO
  FP32 1.12-1.33×, for every checkpoint. NCNN's FP16 costs 0.0003-0.0019 AP against FP32. **NCNN FP16 beats ONNX Runtime
  INT8 at every size**: 33-37 % less latency and 0.0045-0.0169 more AP. At 1120 the gap reads 28 %, but that sets session
  6's NCNN cells (about 6 % high) against session 7's INT8 cells, so it understates NCNN's lead there.

Caveats:

- **The FP32 side throttled.** Every ONNX Runtime FP32 cell in the INT8 table was flagged throttled (351-1,408 images
  below 2000 MHz), except B4 at 1120 (19 images, not flagged). A throttled latency is if anything high, so the speed-ups
  over ONNX Runtime FP32 (INT8's, NCNN's and OpenVINO's) are if anything high too. The throttle re-runs moved medians by
  −3.6 to +1.9 % (INVARIANTS.md section 4.7). No NCNN or INT8 cell on the Pi 5 ran an image below 2000 MHz, so NCNN
  against INT8 is clean.
- **The phone has no clock lock or throttle signal**, and its INT8 cell slowed 18 % from its first tenth to its last
  (`m34_fullcell_int8_record.md`).
- One seed and one pass; no error bars (#48).

### How result D was checked

`study/scripts/tests/test_studytables.py`:

- **The rules, on literal cells.** The speed-up is the same runtime's FP32 latency over INT8's, the change INT8 minus FP32
  over FP32. A speed-up across session 6 is flagged, and then the stock one from within session 6 is the headline. A
  checkpoint with no FP32 cell keeps its row, with no speed-up.
- **Against the records.** All 10 rows of session 7's per-channel table: FP32 AP, both INT8 APs and their change, the
  AP-small chain, both latencies, and which pair spans session 6. All 9 rows of session 6's stock table: the AP and
  AP-small change and the speed-up, e.g. A1 at 640 1.43× and B4 at 1120 −0.0374 (−14.3 %). The phone's 2.09×
  (`m34_fullcell_int8_record.md`).
- **The runtime table against the records.** All 10 rows of session 6's two resolution ladders: ONNX Runtime FP32,
  OpenVINO and NCNN latency, NCNN's AP, and the stock INT8 cell's AP and latency, e.g. A1 at 640 on NCNN 171.5 ms (AP
  0.1866), B2 at 800 on OpenVINO 489.6 ms, B3 at 1280 on ONNX Runtime FP32 1695.1 ms, B4 at 1120 stock INT8 0.2235 AP at
  943.1 ms; B4 at 1280 has no INT8 cell.

## Small-object degradation (the supporting section)

Built by `study/scripts/studytables.py` (#47) from the accuracy table's 640 cells. Each quantized, pruned and combined cell
is set against its model's reference baseline (ONNX Runtime FP32 at 640). The relative drop is (reference − compressed) /
reference, as #9 pre-registered it, and the ratio is the AP-small drop over the AP drop. #9's bar: below 1.3× absent,
1.3-2× weak, and 2× or more, with a paired-bootstrap CI of (small drop − all drop) that excludes 0, confirmed.

Files: `tables/small_object.csv` / `.tex` (the Pi 5 rows, with the AP-medium and AP-large drops, beside #9's laptop rows).

| Compression | YOLO11s: AP drop, AP-small drop, ratio | YOLO26s: AP drop, AP-small drop, ratio |
|---|---|---|
| Quantized, per-channel | 9.5 %, 13.6 %, **1.43×** | 7.6 %, 11.0 %, **1.44×** |
| Quantized, stock | 10.3 %, 15.0 %, 1.45× | 20.3 %, 29.9 %, 1.47× |
| Pruned | 16.1 %, 18.3 %, **1.13×** | 18.5 %, 24.2 %, **1.31×** |
| Combined, per-channel | 23.1 %, 30.2 %, **1.31×** | 25.6 %, 35.2 %, **1.37×** |
| Combined, stock | 24.3 %, 32.9 %, 1.35× | 33.9 %, 45.9 %, 1.36× |
| #9, laptop CPU (YOLO11s), 95 % CI | ONNX Runtime INT8 1.40× [1.27, 1.64], weak; OpenVINO INT8 1.51× [1.05, 1.97], weak ‡; pruned 1.17× [0.99, 1.32], absent | |

‡: #9's OpenVINO FP32 reference ran in bf16 (INVARIANTS.md section 8). #23, which scores it again at true FP32, has not
run, so the row is the bf16 one, footnoted.

- **No compression reaches #9's 2× bar on the Pi 5.** The AP-small drop is 1.13-1.47× the AP drop. Every row but P1 is
  weak by the ratio; P1 is absent, and P3 and C1 per-channel (both 1.31×) are just over the line.
- **The Pi 5 gives #9's laptop numbers.** A1's stock INT8 drops 10.3 % AP and 15.0 % AP-small, against the laptop's 10.2 %
  and 14.3 % (1.45× against 1.40×). P1 drops 16.1 % and 18.3 % (1.13×), against the laptop's 17.1 % and 20.0 % from the
  `.pt` checkpoints (1.17×, absent).
- **The INT8 recipe changes the size of the loss, not its small-object share.** Per-channel cuts YOLO26s's AP drop from
  20.3 % to 7.6 %, and its ratio stays at 1.44-1.47×. Both models' quantized ratios sit at 1.43-1.47× in both recipes.
- **Pruning costs YOLO26s's small objects more than YOLO11s's**: 1.31× against 1.13×. The combined cells give
  1.31-1.37×.

Caveats:

- **No CI on the Pi 5 rows** (#48), so none of them can be "confirmed" by #9's rule, only placed by its ratio.
- One seed (s0). AP-large is the noisy bucket (its seed spread on test-dev is 2.6 points, #9), so the CSV's AP-large
  drops are context only.
- A combined cell is pruned, distilled and quantized, so its drop against the reference baseline mixes three steps.
- The OpenVINO laptop row is the bf16 one until #23 runs.

### How the small-object numbers were checked

`study/scripts/tests/test_studytables.py`:

- **The rule, on literal cells.** Quantized (both recipes), pruned and combined against the reference baseline, no
  distilled row; the drop is (reference − compressed) / reference; #9's bands (1.17× absent, 1.43× weak).
- **Against the records.** A1's stock drops of 10.3 % and 15.0 % and A2's of 20.3 % and 29.9 % (session 6); P1's
  −0.0302 and P3's −0.0360 AP against their baselines (session 2); and the ratios 1.43×, 1.44×, 1.45× and 1.47× of #47's
  comment of 1 November.

## The accuracy and latency tables

Built by `study/scripts/studytables.py` (#47).

- **`tables/accuracy.csv` / `.tex`**: configuration × study model at the matrix resolution (640) on the Pi 5, one-to-many
  head. AP, AP50, AP-small, AP-medium, AP-large. Baseline, pruned and distilled come from the ONNX Runtime FP32 cells,
  quantized and combined from ONNX Runtime INT8: per-channel as the headline, the stock row after it. 14 rows. At 640 the
  distilled cells are the only compressed configuration above the reference baseline (D1 +0.0075 AP, D2 +0.0036); pruning
  costs 0.0302 and 0.0360 AP; combined per-channel gives 0.1442 and 0.1446.
- **`tables/latency.csv` / `.tex`**: every cell, platform × runtime × model × configuration × input resolution (76 rows:
  74 on the Pi 5, 2 on the phone). Median latency, its pre-processing, network and post-processing at 0.25 (the CSV also has
  post-processing at 0.001), precision, and the throttle columns of INVARIANTS.md section 4.7: images below 2000 MHz and
  the median clock. Session 6's cells carry a footnote (latencies about 6 % high). The `.tex` is a `longtable`.
- Both `.tex` files, and the result C and D ones, need `booktabs` (and `longtable` for the latency table). Their notes
  are single-line `\multicolumn` rows, as in Result A's table, so a long one may need wrapping in the paper.

### How the accuracy and latency tables were checked

`study/scripts/tests/test_studytables.py`:

- **The rules, on literal cells.** The accuracy table takes ONNX Runtime FP32 for baseline, pruned and distilled, and
  ONNX Runtime INT8 for quantized and combined, per-channel first and stock after it; no other runtime, input resolution,
  head or platform. The latency table holds every cell, and session 6's carry its footnote.
- **The accuracy table against the records**: every AP and AP-small in it. A1 and A2 and their INT8 cells from session
  7's per-channel table; the pruned, distilled and stock combined cells from session 3's table; the per-channel combined
  cells from session 7's results table (C1 0.1442, C2 0.1446 AP); and A1's AP50, 0.3265 (session 1).
- **The latency table against the records**, five cells:
  - A1 at 640, ONNX Runtime FP32: 389.8 ms, 381 images below 2000 MHz, median 2000 MHz (session 1).
  - A2 at 640, ONNX Runtime FP32: 383.1 ms, network 378.4 ms, post-processing at 0.25 1.11 ms, 1,373 images below, median
    1911 MHz (session 2).
  - B3 at 1120, ONNX Runtime FP32: 1315.4 ms, 513 images below, footnoted (session 6).
  - A1 at 640, NCNN FP16: 171.5 ms, none below (session 1).
  - The phone's A1 INT8: 304.8 ms (`m34_fullcell_int8_record.md`).
- **Session 7 swaps in with no code change.** Without session 7's cells, the INT8 headline rows of the accuracy table,
  result C and the small-object section are all stock; with them, all per-channel.

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
