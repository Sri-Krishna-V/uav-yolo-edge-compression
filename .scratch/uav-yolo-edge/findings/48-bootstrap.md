# Findings: a cheap paired bootstrap for the equal-latency pairs' AP and AP-small differences

Ticket: #48 (parent #1; related #37, #40).
Measured: 2026-11-01, laptop CPU, faster-coco-eval 1.8.0, Python 3.12 venv `study/.venv`, VisDrone test-dev (1,610 images),
every heavy step under `studycap` (Job Object, 4 GB cap). Nothing under `study/` was changed.

**Question.** Can the paired bootstrap over the 1,610 test-dev images be done from each cell's saved `predictions.json`
cheaply enough to finish today, and by what method?

**Verdict in one line:** yes. Evaluate each cell once with `separate_eval=True`, which keeps the per-image results, then
per resample redo only the C++ accumulate on a list built from the kept per-image results. A resample drawing every image
once reproduces `score.json`'s `coco/AP_all` and `coco/AP_small` exactly, and a resample with repeats is bit-identical to
remapping the duplicated images to new ids and running a full evaluate. One pair costs about 0.7 s per resample (after a
~12 s evaluate per cell), so 8 pairs x 200 resamples take about 20 min and 8 x 1,000 about 1.5 h, at under 3 GB.

---

## 1. API facts (faster-coco-eval 1.8.0, `study/.venv/Lib/site-packages/faster_coco_eval/core/`)

1. **The default path discards the per-image results.** `COCOeval_faster.evaluate()` with `separate_eval=False` (the
   default, `cocoeval.py:34`, `:78`, which is how `studyscore.coco_evaluate` calls it) runs
   `_C.COCOevalEvaluateAccumulate(...)` and stores only the accumulated `self.eval` (`faster_eval_api.py:144`). Nothing
   per-image survives. The Python `self.evalImgs` (`cocoeval.py:68`) stays an empty `defaultdict`; the docstring at
   `faster_eval_api.py:~50` says it is deliberately not populated.
2. **`separate_eval=True` keeps them.** `evaluate()` then runs `_C.COCOevalEvaluateImages(...)` and stores the result in
   `self._evalImgs_cpp` (`faster_eval_api.py:130-142`); `accumulate()` is a separate call,
   `_C.COCOevalAccumulate(self._paramsEval, self._evalImgs_cpp)` (`faster_eval_api.py:168-170`). `_paramsEval` is a deep
   copy of `params` taken at `faster_eval_api.py:128`.
3. **What `_evalImgs_cpp` is** (measured, the extension is a compiled `.pyd` with no source): a plain Python `list` of
   `faster_eval_api_cpp.ImageEvaluation` objects (no public attributes), length K x A x I = 10 x 4 x 1,610 = 64,400. Its
   layout is `[category][area range][image position]`, i.e. index `(k*A + a)*I + i`. This was found empirically: with the
   blocks gathered in that order a resample matched the remapped-id reference exactly (synthetic probe and the real cell);
   an image-major order gave wrong numbers (AP-small 0.219 against 0.088).
4. **`accumulate` can be re-run on a different list, repeats included.** `_C.COCOevalAccumulate(params, list)` takes the
   list as given: pass it a list rebuilt from the kept objects, with an image position repeated, and it counts the image
   once per occurrence. The `imgIds` in `params` are not used to select images (passing a duplicated `imgIds` with the
   unmodified list ran but changed nothing in the output shape; tiny_probe.py). So the resample is made by reindexing the
   list, not by editing `params`.
5. **Pycocotools contrast.** The pure-Python `accumulate` (pycocotools `cocoeval.py`) does `setI = set(p.imgIds)` and
   `i_list = [n for n, i in enumerate(p.imgIds) if i in setI]`, so a repeated image id collapses to one and plain
   pycocotools cannot weight an image twice. That is why the ticket's "repeat their per-image results" needs the list-level
   reindex above (or an id remap), not a longer `imgIds`.
6. **Trimming is allowed and exact.** `accumulate` only needs the blocks its params name. Giving it
   `areaRng=[all, small]` (only the first two blocks of each category) and `maxDets=[500]` returns `precision[..., a, -1]`
   bit-identical to the full accumulate (3 of 3 draws, plus the identity draw), and 2.3x faster. The per-image results were
   computed at `maxDet = p.maxDets[-1] = 500` (`faster_eval_api.py:134`), so nothing is lost. `AP_all` and `AP_small` are
   the mean over `precision[:, :, :, a, -1]` where > -1, as `_summarize` does (`cocoeval.py:~440-460`).
7. `math_matches` / `extra_calc` are off (`cocoeval.py:75`), so `accumulate()` adds no per-annotation bookkeeping.

## 2. Methods compared

| Method | Cost per resample | Verdict |
|---|---|---|
| **A. Evaluate once (`separate_eval=True`), reindex the kept list, `COCOevalAccumulate`** | 0.4 s per cell (trimmed), 0.9 s full | **Recommended.** Exact, cheap, no extra memory. |
| B. Remap the duplicated images to new ids in GT and detections, full evaluate per resample | 7.0 s per cell (measured), plus building 700k dets | Exact (the check below), 8-17x dearer, rebuilds JSON dicts every time. Use only as the correctness reference. |
| C. Plain pycocotools-style `accumulate` with a longer `imgIds` | n/a | Wrong: repeats collapse (fact 5). |

## 3. Measurements (real cell: `pi5-matrix-1-yolo11s/.../yolo11s-640-s0/640/onnxruntime-fp32/attempt-2`, 709,951 detections)

Probe `probe_real.py` (one cell) and `paired_probe.py` (one pair), under a 4 GB studycap job; free RAM at start 7.5 GB.

- **Load** `predictions.json` and remap/sort: 2.3 s, 0.48 GB resident.
- **`COCO` + `loadRes`**: 3.1 s. **`evaluate()` (separate_eval=True)**: 5.3 s. One cell, all in: about 11-13 s.
  (The ticket's "minutes" for a full evaluation does not hold for this build on this machine.)
- **Production `accumulate()+summarize()`** on the kept results: 1.2 s.
- **Peak memory**: one cell 1.76 GB commit after evaluate, 2.3 GB at the end of its probe; a pair (two cells' per-image
  lists held at once, predictions freed between) 2.72 GB peak commit, 0.9-1.2 GB resident. Fits the 4 GB cap with margin.
- **One resample** (reindex + C++ accumulate), full A/maxDets: 0.9-1.2 s per cell (reindex 0.02 s, accumulate 1.0-1.2 s).
  Trimmed to all+small and maxDets 500: 0.36-0.44 s per cell. **Paired, both cells: 1.68 s full, 0.67 s trimmed**
  (200 resamples: 335 s full, 134 s trimmed; both gave the same intervals, since the seed is the same).
- **Identity** (a draw of every image exactly once):
  - Production path, 15 `coco/*` stats against `score.json`: all exactly equal.
  - Re-accumulate path, `AP_all` 0.1874793899021204 and `AP_small` 0.09039407025619958: exactly equal to `score.json`'s
    `coco/AP_all` and `coco/AP_small`. Same for the paired probe's int8 cell (`AP_all` 0.16818970420581286, `AP_small`
    0.07686940014139579): both cells exact.
- **Resample against remapped-id full evaluate** (one random draw of 1,610 with repeats): `AP_all` 0.18888817593947618,
  `AP_small` 0.09221624463351376 from both methods, exactly equal. The remapped full evaluate took 7.0 s, peak commit 2.39 GB.
- **Paired example** (YOLO11s 640, ORT FP32 minus ORT INT8 per-tensor, 200 resamples, seed 48; a pair chosen only to time
  the method, not an equal-latency pair of the table): AP difference +0.0193, 95 % interval [0.0180, 0.0215]; AP-small
  difference +0.0135, interval [0.0124, 0.0163]; bootstrap SD 0.0009 and 0.0010. Percentile intervals; both exclude 0.

## 4. Time estimate (this laptop, one process, trimmed accumulate)

Per pair: 2 cells x ~12 s evaluate + resamples x 0.67 s. Cells shared by several pairs would only need evaluating once
if the script caches them, but it is not needed; counting every pair as fresh:

| Resamples | Per pair | 8 pairs |
|---|---|---|
| 200 | 24 s + 134 s = about 2.6 min | **about 21 min** |
| 1,000 | 24 s + 670 s = about 11.6 min | **about 93 min** |
| (untrimmed, 200) | 24 s + 335 s | about 48 min |

Run one pair per capped process (cap 4 GB, timeout 20 min for 200, 30 min for 1,000), one at a time. The loop is
single-threaded C++ plus a Python list build, so a second process would double the memory (2.7 GB each) for the 15 GB
laptop with 3-5 GB often free; not recommended. Spare time buys resamples, not parallelism: 1,000 resamples fit today
if the laptop stays otherwise idle; 200 is the floor the ticket asks for and is safely done in under half an hour.

## 5. Recommendation

1. Use method A with the trimmed accumulate (all + small, maxDets [500]) in a new script that takes the pair list from
   `equal_latency.csv` (the table was not in the working tree when this was run, so the pairs were not enumerated), loads
   the two cells in turn under `studycap`, and writes the resample draws and differences. Reuse `paired_probe.py`'s core
   (`evaluate`, `accumulate`).
2. Draw the image positions once per resample and apply them to both cells (the probe does this: one `idx` per resample).
   Take the 95 % interval as the 2.5 and 97.5 percentiles of the differences, and report the number of resamples in the table.
3. Keep the acceptance check inside the script: before the loop, assert that the every-image-once accumulate equals each
   cell's `score.json` `coco/AP_all` and `coco/AP_small` exactly (the probe does, for both cells).
4. Start with 200 resamples for all pairs (about 21 min) to have a usable table, then re-run at 1,000 (about 1.5 h) if time
   allows. At 200 the interval end-points still move in the 4th decimal (0.0124 vs 0.0126 on repeat seeds is plausible), which
   does not change whether a difference excludes 0 for pairs this far from 0; only borderline pairs need the larger count.
5. Caveat: the per-image list layout (fact 3) and the independence of `imgIds` (fact 4) were established by experiment on a
   closed-source extension. Pin faster-coco-eval 1.8.0, and keep the identity assertion so a version change fails loudly.
6. This is a case-resampling bootstrap on images, so it treats each image's detections as fixed per cell; it measures
   test-set sampling error, not seed or training variance (the #9 three-seed spread is a different yardstick).

## 6. Scripts

In [`48-bootstrap/`](48-bootstrap/), beside this file (copied from the session scratchpad, where the probes ran and
their JSON outputs stay):

- `paired_probe.py`: the paired bootstrap for one pair, with identity checks (the method to port). Output `paired200_trim.json`
  (untrimmed run: `paired200.json`).
- `probe_real.py`: one cell; timings, memory, identity, optional `--verify-remap` against the remapped-id full evaluate
  (outputs `real1.json`, `real_verify.json`).
- `trim_probe.py`: the trimmed against the full accumulate, three draws.
- `tiny_probe.py`: synthetic 30-image probe that established the API facts and the list layout.
- `run_capped.py`: runs any of the above under `studycap.run` (cap GB, timeout min, script, args).
