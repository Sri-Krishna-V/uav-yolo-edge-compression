# Findings: the results talk's story, its charts, and its references

Ticket: #49 (parent #1). Written 2026-11-01 for the faculty review on 2 November. Sources:
- a read of all 49 issues with their comments;
- `novelty-check.md`;
- 26 references, each checked against a primary page (arXiv abstract, Crossref DOI metadata, publisher page, or the
  Ultralytics docs).

The ten charts are `study/results/figures/story/s01–s10`, made by `study/scripts/studystory.py`. The deck is the Slides
artifact https://claude.ai/artifact/Q7QQGnXeaS8rCniyFjmeDH, with a local copy in `study/results/deck/`.

## 1. The story arc and the evidence for each beat

| Beat | Claim | Evidence | Chart |
|---|---|---|---|
| Problem | Two thirds of what a drone sees is small: 50,837 of the 75,102 test-dev objects (67.7 %) are COCO-small | `runs/gt/visdrone-test-640.json`, area < 32² | s01 |
| Problem | Input resolution 640 to 1280 raises AP-small most (+79 % and +67 %, against +22 to +34 % for medium and large), at 4.35× the latency | `cells.csv`, ONNX Runtime FP32 baselines | s02 |
| Gap | Compression is judged by FLOPs and mAP; the INT8-for-resolution idea exists [10, 11], but not at equal measured latency, not with AP-small, and not with pruning | the references below; `novelty-check.md` | slide |
| Question | Is a compression budget better spent on input resolution? The small-object bar (2×) was set in advance on #9, missed on 15–17 Oct, and on 20 Oct the study changed course (#10) | #9, #10 | timeline slide |
| Result A | Every INT8 pair gains AP-small (+3 to +16 % of the plain model's); every pruned pair loses (−2 to −4 %) | `equal_latency.csv`, headline pairs | s03 |
| Why | At 960, INT8 buys 1.4–1.5× speed for 6–8 % of AP-small; a smaller input buys the same speed for 15 %; pruning costs 17–18 % | `cells.csv` | s04 |
| Phone | INT8 is 1.9–2.0× faster on the phone, against 1.4× on the Pi 5 (provisional) | `result_d_int8.csv` | s05 |
| Result B | Post-processing is 0.4 % of a 277 ms image; the NMS-free head ends up 3 ms slower | `result_b_heads.csv` | s06 |
| Result C | YOLO26s's extra latency grows from +1.3 % at 640 to +9.0 % at 1280 (NCNN FP16) | `result_c.csv` | s07 |
| Result D | NCNN FP16 runs at 172 ms against 272 ms for ONNX Runtime INT8, with AP-small 0.089 against 0.078 | `cells.csv` | s08 |
| Result D | Per-channel INT8 costs YOLO26s 7.6 % of AP; the stock recipe costs it 20.3 % | `result_d_int8.csv` | s09 |
| Small objects | AP-small drops 1.13–1.44× as much as AP (per-channel and pruned; stock INT8 up to 1.47×), under the 2× bar | `small_object.csv` | s10 |

**Design rule for the ten charts.**
- One highlight colour, the deck's blue `#2563eb`, marks what the slide's sentence is about; everything else is grey (Wong,
  "Salience to relevance", Nat Methods 8:889, 2011).
- Values are written on the marks, and there is no legend where a label will do (Rougier et al. 2014, rule 8).
- Every chart is 160 mm wide with 11 pt text, which is about 36 px on a 1,920 px slide.

## 2. References (as on the deck's last two slides)

Every entry was verified against a primary page on 1 Nov 2026, except where a caveat says otherwise.

1. P. Zhu et al., "Detection and Tracking Meet Drones Challenge," IEEE TPAMI 44(11), 2022. doi:10.1109/TPAMI.2021.3119563
2. T.-Y. Lin et al., "Microsoft COCO," ECCV 2014. doi:10.1007/978-3-319-10602-1_48
3. G. Cheng et al., "Towards Large-Scale Small Object Detection: Survey and Benchmarks," IEEE TPAMI 2023.
   doi:10.1109/TPAMI.2023.3290594. Volume and issue still to confirm.
4. F. C. Akyon et al., "Slicing Aided Hyper Inference …," ICIP 2022. doi:10.1109/ICIP46576.2022.9897990
5. M. Tan, R. Pang, Q. V. Le, "EfficientDet," CVPR 2020. arXiv:1911.09070
6. D. Blalock et al., "What is the State of Neural Network Pruning?" MLSys 2020. arXiv:2003.03033
7. M. Dehghani et al., "The Efficiency Misnomer," ICLR 2022. arXiv:2110.12894
8. G. Zhou, D. Zhang, "Deployment-oriented pruning for real-time object detection: a survey," Applied Intelligence 56,
   2026. doi:10.1007/s10489-026-07416-6
9. H. Habash et al., aerial real-time detection survey, Sensors 25(24), 2025. doi:10.3390/s25247563
10. G. Plastiras et al., "… Joint Quantization and Tiling," AICAS 2020. doi:10.1109/AICAS48895.2020.9073885. This is the
    closest prior work.
11. M. Vandersteegen et al., "Super accurate low latency object detection on a surveillance UAV," MVA 2019.
    arXiv:1904.02024
12. G. Jocher, J. Qiu, "Ultralytics YOLO11," software, 2024. There is no paper; cite the software.
13. G. Jocher et al., "Ultralytics YOLO26," arXiv:2606.03748, 2026
14. B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," CVPR
    2018. doi:10.1109/CVPR.2018.00286
15. R. Krishnamoorthi, "Quantizing deep convolutional networks for efficient inference," arXiv:1806.08342, 2018
16. H. Li et al., "Pruning Filters for Efficient ConvNets," ICLR 2017. arXiv:1608.08710
17. G. Fang et al., "DepGraph," CVPR 2023. doi:10.1109/CVPR52729.2023.01544
18. G. Hinton, O. Vinyals, J. Dean, "Distilling the Knowledge in a Neural Network," arXiv:1503.02531, 2015
19. V. J. Reddi et al., "MLPerf Inference Benchmark," ISCA 2020. doi:10.1109/ISCA45697.2020.00045
20. A. Wang et al., "YOLOv10," NeurIPS 2024. arXiv:2405.14458
21. T. C. Le et al., "Enabling Fully Integer-Only Inference for Lightweight Detection Transformers," arXiv:2607.24981,
    2026. Its small-object ratios come from our full-text read, not from the abstract page.
22. J. I. Choi, Q. Tian, "Visual Saliency-Guided Channel Pruning …," arXiv:2303.02512, 2023. The venue is not confirmed.
23. Y. Ma et al., "I-YOLOv11n," Sensors 25(15), 2025. doi:10.3390/s25154857
24. T. Zhang et al., "MD-YOLO," Algorithms 19(9), 2026. doi:10.3390/a19090737
25. L. Rey et al., YOLO on edge devices for drones, Electronics 14(3), 2025. doi:10.3390/electronics14030638
26. A. Marasini, Z. Zhou, "Sustainable Edge Vision via Empirically Calibrated DVFS …," arXiv:2609.04705, 2026

Checked but not cited on the slides:
- Du et al., VisDrone-DET2019, ICCVW 2019
- Nagel et al., arXiv:2106.08295
- He et al., channel pruning, ICCV 2017
- Tan & Le, EfficientNet
- RT-DETR, arXiv:2304.08069
- Q-YOLO, arXiv:2307.04816

## 3. Corrections found while doing this

- #49's body says session 6 latencies were "3–5 % high". Session 7 measured about 6 %, so correct the body.
- `papers/README.md` says the Applied Intelligence PDF is not in the repo, but a copy sits at the repo root.
- The reason for the 10 % pair rule, and for choosing s-scale models, is not recorded on any ticket. Faculty may ask.
