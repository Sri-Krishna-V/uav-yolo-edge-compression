# Final matrix and schedule

Type: grilling
Status: open
Blocked by: 05, 06, 08, 11, 12, 14

## Question

Lock the experiment matrix that will actually be run, and allocate it across the available GPU-hours and the remaining calendar.

The nominal matrix is 2 models x 5 configurations (baseline, quantized, pruned, distilled, combined) x 4 platforms = 40 deployment cells, on top of roughly 16 training and recovery runs. Estimated training cost is 80-150 GPU-hours against a supply of Kaggle 30 h/wk, Colab free, and one RTX 4050.

Decisions to reach:

1. **Which cells are required** for the claims of the paper versus merely nice to have. Not every model x configuration x platform combination earns its GPU-hours.
2. **Cut order.** If the schedule slips, what gets dropped first - decided NOW rather than in a panic in week 7. The standing preference on the map is to cut cells, never rigor.
3. **Schedule.** Week-by-week allocation of training runs to machines, with the Jetson explicitly last, since it is dept access and its stack is the least certain.
4. **Critical path.** Which results must exist before the paper can be written at all, versus which merely strengthen it.
5. **Checkpoint policy** so that a disconnected free-tier session costs minutes, not a day.
6. **A stop rule** - the date after which no new experiments start and writing begins regardless.

**Constraints changed on 2026-10-09, and they are tighter than this ticket was written for:**

- **One month, not two** (deadline ~2026-11-09), but **two executors** instead of one. The teammate runs the YOLO26s training line and owns Jetson Nano lab access; Pi, phone and Jetson benchmarking is covered by the pair.
- **GPU supply is ~60 h/week**: Kaggle T4 ×2 only, two accounts, one per executor. No laptop training, no P100.
- **The baseline set alone is roughly 1–1.5 weeks of that supply** (10 runs, ~96 GPU-h at current estimates). Its internal cut order is already fixed in `study/RUNS.md`: 1280 first, then 960, then the YOLO26s noise runs — never the YOLO11s 640 runs, which carry the critical path and the noise floor.
- **Training now runs in parallel with the remaining decision tickets** (map Notes, plan-don't-do override). So this ticket schedules what is *left* around a GPU that is already busy, rather than scheduling from zero.

This ticket is deliberately last. It cannot be answered honestly until YOLO26 viability, the real cost of the training recipe, the power capability, and the small-object verdict are all known.
