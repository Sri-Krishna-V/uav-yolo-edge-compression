# Findings: why YOLO26s loses 2.4x YOLO11s's AP to ONNX Runtime INT8, and a fix

Ticket: #41 (found in Pi 5 matrix session 5; related #13, #37).
Measured: 2026-10-31, laptop CPU (ONNX Runtime 1.31.0, 4 threads, imgsz 800, VisDrone test-dev).
Scripts and raw runs: the session scratchpad `int8diag/` folder (section 8). The fix was adopted on 31 October and
ported to the repo as `study/scripts/studyqdq.py` and `studyexport.py`'s `onnxruntime-int8pc` variant (section 5).

**Question.** On the Pi 5 at 800, B2 (YOLO26s, `yolo26s-960-s0`) loses 16.3 % AP from FP32 to INT8, and B1 (YOLO11s,
`yolo11s-960-s0`) loses 6.8 %. Is that YOLO26s, or the recipe? What recovers it?

**Verdict in one line:** it is the recipe meeting YOLO26s's narrow Detect-head box branch. `quantize_static` stores
every Conv weight per tensor. The box branch's inner convs (`/model.23/cv2.x.0`, `cv2.x.1`) have 32 channels in
YOLO26s and 64 in YOLO11s, and their per-channel weight ranges differ so much that 12–31 % of their output channels
carry more than 20 % error against their own RMS. Quantizing the weights per channel, with the stock activation
scales, takes B2 from −16.6 % to −6.6 % AP on the full test-dev. That is the same loss as B1's stock INT8 (−6.6 %). It
needs no recalibration and no graph change. The −6.6 % that remains in both models is not explained, and the fix's
Pi 5 latency is not measured.

---

## 1. Method

1. **Reproduce on the laptop first.** The laptop's stock INT8 reproduces the Pi:
   - B2: 0.2221 → 0.1852 (−16.6 %), against 0.2218 → 0.1857 (−16.3 %) on the Pi.
   - B1: 0.2190 → 0.2046 (−6.6 %), against 0.2190 → 0.2040 (−6.8 %) on the Pi.

   Each INT8 variant is compared with the FP32 export of the same model, scored the same way (INVARIANTS.md §4.4).
2. **Graph surgery on the stock QDQ export** (`qdq_surgery.py`). The stock graph is `quantize_static` with its defaults
   (QDQ, QInt8/QInt8, per-tensor weights, MinMax), with every node that isn't Conv, Gemm or MatMul excluded. Each
   variant is built from it and the FP32 export, reusing the stock activation scales, so nothing is calibrated. The
   operations:
   - `fp32(group)`: the group's Conv/MatMul nodes run in float (FP32 weights and bias, no input or output Q/DQ).
   - `per_channel(group)`: the weights are re-quantized per output channel (symmetric INT8, absmax/127,
     DequantizeLinear axis 0), and the int32 bias at input scale × weight scale[c], as
     `quantize_static(per_channel=True)` stores it.
   - `drop_out_q`: a Conv's output is left unquantized. `float_skip`: Add and Concat read the float tensor.

   Checks: every operation changes the output (no variant is identical to stock), and everything in float plus
   `float_skip` gives exactly the FP32 output.
3. **Screening** on the first 200 test-dev images (`runvariants.py` → `runcell.py`: `studybench.py`, then
   `studyscore.py` under `studycap`, 4 threads). A `--limit` run ranks variants against each other and is never final.
   The noise is about 0.003 AP.
4. **Full test-dev** (1,610 images) for FP32, stock INT8 and per-channel INT8, for both models (`runfull.py`).
5. **Weight statistics** (`weight_stats.py`): per Conv, the INT8 weight error per tensor against per channel, from the
   FP32 exports.

## 2. Full test-dev

1,610 images, laptop, 4 threads. The last four columns are the loss against the model's FP32 export.

| Model, variant | AP | AP50 | AP75 | AP-small | AP-large | AP lost | AP50 lost | AP-small lost | AP-large lost |
|---|---|---|---|---|---|---|---|---|---|
| B1 YOLO11s, FP32 | 0.2190 | 0.3755 | 0.2225 | 0.1164 | 0.4598 | | | | |
| B1, stock INT8 | 0.2046 | 0.3593 | 0.2061 | 0.1049 | 0.4697 | −6.6 % | −4.3 % | −9.9 % | +2.2 % |
| B1, per-channel INT8 | 0.2049 | 0.3602 | 0.2057 | 0.1049 | 0.4551 | −6.4 % | −4.1 % | −9.9 % | −1.0 % |
| B2 YOLO26s, FP32 | 0.2221 | 0.3814 | 0.2237 | 0.1184 | 0.4871 | | | | |
| B2, stock INT8 | 0.1852 | 0.3409 | 0.1858 | 0.0944 | 0.4000 | **−16.6 %** | −10.6 % | −20.3 % | −17.9 % |
| B2, per-channel INT8 | **0.2074** | 0.3628 | 0.2073 | 0.1091 | 0.4768 | **−6.6 %** | −4.9 % | −7.9 % | −2.1 % |

The laptop's median `e2e_deploy` latency, in ms. It is given for completeness only: the laptop's INT8 is slower than
its FP32, so it says nothing about the Pi.

| Model | FP32 | Stock INT8 | Per-channel INT8 |
|---|---|---|---|
| B1 | 254.2 | 379.9 | 369.7 |
| B2 | 227.1 | 392.5 | 360.6 |

Each run was made once. The first full B2 INT8 run was lost when the laptop went to sleep (it hit its 60-minute
timeout) and was run again.

## 3. Screening, first 200 images

**The groups.**
- `head` = all of `/model.23/` (the Detect head); `box` = `cv2.*`; `cls` = `cv3.*`.
- `box_inner` = `cv2.x.0` and `cv2.x.1`; `box_final` = `cv2.x.2`.
- `box_p3`, `box_p4`, `box_p5` = the box branch of the stride 8, 16 and 32 level.
- `attn` = the attention blocks' convs and MatMuls; `matmul` = the 4 MatMuls only.
- `backbone` = `/model.0`–`/model.9`; `neck` = `/model.10`–`/model.22`.
- `fp32_X` = the group in float; `pc_X` = the group per channel (`pc` = every Conv); `fs` = `float_skip`.
- `dropq_box_final` = the box output convs keep INT8 inputs and weights, but their output is not quantized.

**Caveats.**
- No latency is given for the 200-image runs: 2–3 jobs ran at once.
- Every `pc*` row here was built with the first version of `qdq_surgery.py`, which left each bias at the stock
  per-tensor scale. It was fixed afterwards, and the full-test-dev `pc` rows (section 2) use the fixed models. The
  `fp32_*`, `fs` and stock rows are unaffected. The difference is small: `fusion_check.py` gives stock and the fixed
  per-channel model the same fused-against-float-QDQ difference (boxes about 0.28 px, top score about 0.036). The `pc*`
  screening rows were not run again.

The last four columns are the loss against FP32 for AP, AP50, AP-small and AP-large.

B2, YOLO26s, 200 images:

| Variant | AP | AP50 | AP75 | AP-small | AP-large | AP | AP50 | AP-small | AP-large |
|---|---|---|---|---|---|---|---|---|---|
| FP32 | 0.2355 | 0.4200 | 0.2294 | 0.1479 | 0.4558 | | | | |
| stock INT8 | 0.2104 | 0.3966 | 0.2047 | 0.1259 | 0.3949 | −10.6 % | −5.6 % | −14.9 % | −13.4 % |
| fp32_attn | 0.2106 | 0.3968 | 0.2014 | 0.1239 | 0.3914 | −10.5 % | −5.5 % | −16.2 % | −14.1 % |
| fp32_matmul | 0.2112 | 0.3969 | 0.2061 | 0.1248 | 0.3921 | −10.3 % | −5.5 % | −15.6 % | −14.0 % |
| fp32_backbone | 0.2162 | 0.4066 | 0.2106 | 0.1322 | 0.4214 | −8.2 % | −3.2 % | −10.6 % | −7.6 % |
| fp32_neck | 0.2121 | 0.3983 | 0.2062 | 0.1299 | 0.3882 | −9.9 % | −5.2 % | −12.1 % | −14.8 % |
| fp32_head | 0.2312 | 0.4125 | 0.2244 | 0.1402 | 0.4355 | −1.8 % | −1.8 % | −5.2 % | −4.5 % |
| fp32_head_box | 0.2273 | 0.4078 | 0.2224 | 0.1387 | 0.4260 | −3.4 % | −2.9 % | −6.2 % | −6.6 % |
| fp32_head_cls | 0.2136 | 0.3999 | 0.2063 | 0.1292 | 0.3943 | −9.3 % | −4.8 % | −12.7 % | −13.5 % |
| fp32_head_final | 0.2144 | 0.3999 | 0.2073 | 0.1297 | 0.4043 | −9.0 % | −4.8 % | −12.3 % | −11.3 % |
| fp32_box_final | 0.2122 | 0.3967 | 0.2042 | 0.1270 | 0.4002 | −9.9 % | −5.5 % | −14.1 % | −12.2 % |
| fp32_box_inner | 0.2253 | 0.4065 | 0.2216 | 0.1362 | 0.4197 | −4.3 % | −3.2 % | −7.9 % | −7.9 % |
| fp32_box_p3 | 0.2158 | 0.4039 | 0.2118 | 0.1330 | 0.3958 | −8.3 % | −3.8 % | −10.0 % | −13.2 % |
| fp32_box_p4 | 0.2205 | 0.3996 | 0.2143 | 0.1305 | 0.4217 | −6.4 % | −4.9 % | −11.7 % | −7.5 % |
| fp32_box_p5 | 0.2112 | 0.3975 | 0.2050 | 0.1268 | 0.3976 | −10.3 % | −5.4 % | −14.2 % | −12.8 % |
| dropq_box_final | 0.2115 | 0.3967 | 0.2048 | 0.1271 | 0.3982 | −10.2 % | −5.6 % | −14.0 % | −12.6 % |
| fs | 0.2107 | 0.3976 | 0.2063 | 0.1276 | 0.3898 | −10.5 % | −5.3 % | −13.7 % | −14.5 % |
| pc | 0.2289 | 0.4119 | 0.2175 | 0.1372 | 0.4641 | −2.8 % | −1.9 % | −7.2 % | +1.8 % |
| pc_fs | 0.2249 | 0.4083 | 0.2140 | 0.1342 | 0.4410 | −4.5 % | −2.8 % | −9.3 % | −3.3 % |
| pc_head | 0.2268 | 0.4093 | 0.2187 | 0.1377 | 0.4287 | −3.7 % | −2.5 % | −6.9 % | −6.0 % |
| pc_box | 0.2231 | 0.4046 | 0.2198 | 0.1352 | 0.4193 | −5.3 % | −3.7 % | −8.6 % | −8.0 % |
| pc_cls | 0.2133 | 0.4011 | 0.2072 | 0.1272 | 0.3924 | −9.4 % | −4.5 % | −14.0 % | −13.9 % |
| pc_nohead | 0.2126 | 0.4016 | 0.2046 | 0.1269 | 0.4187 | −9.7 % | −4.4 % | −14.2 % | −8.2 % |

B1, YOLO11s, 200 images:

| Variant | AP | AP50 | AP75 | AP-small | AP-large | AP | AP50 | AP-small | AP-large |
|---|---|---|---|---|---|---|---|---|---|
| FP32 | 0.2435 | 0.4325 | 0.2378 | 0.1509 | 0.4124 | | | | |
| stock INT8 | 0.2266 | 0.4155 | 0.2194 | 0.1408 | 0.4058 | −6.9 % | −3.9 % | −6.7 % | −1.6 % |
| fp32_attn | 0.2256 | 0.4126 | 0.2181 | 0.1427 | 0.4221 | −7.3 % | −4.6 % | −5.4 % | +2.4 % |
| fp32_backbone | 0.2332 | 0.4217 | 0.2259 | 0.1454 | 0.4055 | −4.2 % | −2.5 % | −3.6 % | −1.7 % |
| fp32_head | 0.2343 | 0.4236 | 0.2281 | 0.1439 | 0.4042 | −3.8 % | −2.1 % | −4.6 % | −2.0 % |
| fp32_head_box | 0.2305 | 0.4177 | 0.2250 | 0.1468 | 0.4155 | −5.3 % | −3.4 % | −2.7 % | +0.7 % |
| fp32_head_cls | 0.2300 | 0.4193 | 0.2201 | 0.1375 | 0.4040 | −5.5 % | −3.1 % | −8.9 % | −2.0 % |
| pc | 0.2286 | 0.4165 | 0.2225 | 0.1403 | 0.4395 | −6.1 % | −3.7 % | −7.0 % | +6.6 % |
| pc_head | 0.2260 | 0.4113 | 0.2162 | 0.1408 | 0.4245 | −7.2 % | −4.9 % | −6.7 % | +2.9 % |

**Not scored.** The screening chains were cut by the laptop's sleep and not restarted: B2 `pc_fp32box` and
`pc_fp32boxinner`, B1 `pc_box` and `pc_fp32box`. `e4_levels.py` (the output error per stride level) was written but
not run.

## 4. The cause, as far as measured

1. **B2's extra loss sits in the box branch.** The 200-image gap is 0.0251 AP. The whole head in float closes 0.0208,
   the box branch 0.0169, `box_inner` 0.0149 (59 %). The class branch alone closes 0.0032, `box_final` 0.0018 and
   `dropq_box_final` 0.0011: all three are within the noise.
2. **Attention does not matter.** `fp32_attn` gives 0.2106 and `fp32_matmul` 0.2112, against 0.2104 for stock.
   `float_skip` and `fp32_neck` change nothing. The backbone closes 0.0058.
3. **The stride-16 box level matters most.** `box_p4` closes 0.0101, `box_p3` 0.0054 and `box_p5` 0.0008 (noise). Not
   explained.
4. **The mechanism the data support: per-tensor weight rounding in thin convs.**
   - YOLO26s's box convs have 32 output channels (YOLO11s: 64).
   - The spread of the absolute maximum across output channels is 6.6x, 9.8x and 5.6x in the three `cv2.x.1` convs
     (YOLO11s: 2.6–4.3x), and 2.8–4.2x in `cv2.x.0`.
   - Under per-tensor INT8, 31 % of the output channels of `cv2.0.1` and `cv2.1.1`, 28 % of `cv2.2.0`, 19 % of
     `cv2.0.0` and 12 % of `cv2.1.0` carry more than 20 % error against their own RMS. YOLO11s's box branch has 0–3 %,
     and YOLO11s over all its convs 0.1 %.
   - The whole-tensor relative weight error is 4–13 % per tensor, against 2–6 % per channel.
   - Per channel closes 0.0185 of the 200-image gap (74 %): `pc_box` 0.0127, `pc_head` 0.0164, `pc_nohead` 0.0022,
     `pc_cls` 0.0029.
5. **Two independent ways of removing the error agree.** Float `box_inner` (+0.0149) and per-channel box (+0.0127)
   recover about the same.
6. **B1 does not have this problem.** Per channel does nothing for it on the full test-dev (0.2046 → 0.2049). Its loss
   is spread over the head, the backbone and the box branch, and no group alone removes it.
7. **B2's AP75 and AP-large drops look like box-localisation noise.** Stock: AP75 −16.9 %, AP-large −17.9 %. Per
   channel: AP75 −7.3 %, AP-large −2.1 %.

**Suggested, not shown:** why the box branch is more sensitive in YOLO26s. Two candidates: it is half as wide, and it
regresses box edges directly, with no DFL softmax (its graph has 2 Softmax nodes, both in attention). Neither was tested.

## 5. The fix and its cost

**Recommended:** per-channel weights on every Conv, with the stock activation scales (the `pc` variant).

| | Stock INT8 | Per-channel INT8 | FP32 |
|---|---|---|---|
| B2 AP | 0.1852 (−16.6 %) | 0.2074 (−6.6 %) | 0.2221 |
| B1 AP | 0.2046 (−6.6 %) | 0.2049 (−6.4 %) | 0.2190 |

- **Accuracy.** B2 recovers 0.0222 AP, 60 % of its loss, and stays 0.0147 AP below its FP32. B2's INT8 goes from behind
  B1's (0.1852 against 0.2046) to ahead of it (0.2074 against 0.2049). B1 is unchanged, so the recipe is fair to both
  models.
- **Latency.** No cost on the laptop. The Pi 5 is not measured. The graph is unchanged apart from per-channel weight
  and bias scales; whether ONNX Runtime's per-channel kernels run as fast on ARM was not checked.
- **Building it.** `python scripts/studyexport.py --weights <best.pt> --imgsz <imgsz> --only onnxruntime-int8pc`, from
  the cell's FP32 export and its stock INT8 export (ported from `build_fixed_int8.py` on 31 October). Its B1 and B2
  models at 800 are byte-identical to the ones scored here (B2 SHA-256
  `f6008380f9bd691be2fbbb3a09a706001ce24f00ad082f679dddd7aab2ddf57a`, B1
  `2a4f2d5cbc9eef326c669f8a0e977d3e3c50400baaacb77a31b5353afdeaba21`).
- **Limits.** A cell with no stock INT8 export (YOLO26s at 1280) needs a fresh `quantize_static(per_channel=True)`,
  which was not run. A float box branch on top of per channel (`pc_fp32box`) was not scored, so it is not recommended.

## 6. For the paper

B2's INT8 loss on the Pi (−16.3 %) belongs mostly to the recipe (per-tensor weights) meeting YOLO26s's thin box branch,
not to its attention or its class branch. With per-channel weights, the laptop gives −6.6 % for both models. State it
as a laptop-measured recovery until the Pi has run the fixed export. The laptop matched the Pi within 0.0006 AP in all
four stock cells compared, but that says nothing about the fixed export.

## 7. Unknown

- The Pi 5 latency and AP of the per-channel INT8. Laptop latencies are not evidence.
- The `pc` export reuses the stock activation scales; a fresh `quantize_static(per_channel=True)` calibration was not
  tried.
- The −6.6 % that remains in both models. For B2 after per channel, no group was tried.
- Why YOLO26s's box branch is the sensitive one, and why stride 16 matters most.
- Other image sizes (session 2's 640 and 960 pairs).
- The 200-image screening understates the full-set effect (B2 stock −10.6 % on 200 images, −16.6 % on 1,610), so the
  ranking of the groups could move.
- 4 threads only (ONNX Runtime's INT8 output changes with the thread count, INVARIANTS.md §4.5); one checkpoint per model.

## 8. Scripts

All in the session scratchpad's `int8diag/` folder
(`C:\Users\srikr\AppData\Local\Temp\claude\c--Users-srikr-Desktop-Studies-Minor-Term-5\503b9376-030b-487d-84e4-bea752973fd6\scratchpad\int8diag\`).
The fix (`qdq_surgery.py`'s per-channel path and `build_fixed_int8.py`) is in the repo as `study/scripts/studyqdq.py`
and `studyexport.py`'s `onnxruntime-int8pc` variant. The diagnosis variants (`variants.py`, the `fp32`, `drop_out_q` and
`float_skip` paths) are not.

- `qdq_surgery.py`, `variants.py`, `test_surgery.py`, `fusion_check.py`
- `build_fixed_int8.py` (the fix)
- `runcell.py`, `runvariants.py`, `runfull.py`, `full5.sh`, `table.py`
- `weight_stats.py` → `weight_stats.txt`
- `e1_outputs.py`, `head_scales.py`, `graph.py`, `show.py`
- `runs/<model>_<variant>_{n200,full}/` (`score.json`, `meta.json`, `predictions.json`)
- `models/` (the variant ONNX files; `*_pc_oldbias.onnx` are the per-channel models before the bias fix)
