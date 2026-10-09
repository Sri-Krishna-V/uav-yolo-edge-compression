# Official-toolkit A/B: in or out

Type: grilling
Status: open
Blocked by: none

## Question

Does the study score a baseline checkpoint with the official VisDrone toolkit (`evalDET.m`) alongside `studyeval`? If so, with which tool, on what scope, and where in the paper does the result go?

*Freeze the experimental invariants* froze a COCO-style protocol (faster-coco-eval, maxDets 500, native-pixel area ranges) plus a written caveat. Three deltas from `evalDET.m` remain, all derived from reading the code, all biasing our numbers low:

- **Ignored regions.** The official toolkit deletes detections ≥50% inside an ignored region; ours counts them as false positives.
- **Class averaging.** The official AP is weighted by image frequency; ours is a macro average.
- **PR integration.** Official uses VOC all-point; ours uses COCO 101-point.

The **direction** is solid, but the **magnitude has never been measured**: nobody has published the comparison, and the closest prior work (arXiv 2605.24831) reports Ultralytics-pipeline numbers only. It costs **zero GPU-hours**, because `studyeval` already writes `predictions.json` per evaluation. Converting to `<x>,<y>,<w>,<h>,<score>,<category+1>,-1,-1` and scoring is CPU-only.

Decide:

1. **In or out.** In means a methods paragraph or small table quantifying the gap on baseline val and test-dev. Out means the caveat stands alone.
2. **Tool.** MATLAB (does the college licence cover it?), GNU Octave (does `evalDET.m` run unmodified?), or a third-party Python port. No port has been verified against `dropObjectsInIgr.m`'s 50%-area rule, so a port needs a diff against MATLAB or Octave output on at least one split before it is trusted.
3. **Scope: baseline only, or also one compressed configuration.** This is the part that matters for contribution (c). Ignored regions sit over dense crowds of tiny objects, exactly where AP-small lives. If the protocol gap *moves* under compression, part of a measured ΔAP-small could be a protocol artefact, and the caveat alone would not cover it.
