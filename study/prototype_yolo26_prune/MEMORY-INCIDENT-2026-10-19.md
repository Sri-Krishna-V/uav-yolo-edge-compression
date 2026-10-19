# Incident: feasibility.py exhausted RAM and made the laptop unusable (2026-10-19)

Written by a Claude session outside this project, which diagnosed the laptop. **Facts** are measured. **Hypotheses** are labelled and not yet proven. Please confirm them before you build a fix on them.

## What happened

The command `.venv\Scripts\python.exe prototype_yolo26_prune/feasibility.py` ran from 18:21 to 22:05 IST. The user then had it killed (`Stop-Process -Force`), because the machine had stopped responding. It was most likely started from a Claude Code session in VS Code: that session's process started in the same minute.

| Time | Stage | Result |
|---|---|---|
| 18:22 | `0_unpruned_export` | OK in 26 s |
| 18:22 → 19:50 | `1_prune_yolo26s-prune5` | **`MemoryError` after 5218 s (87 min)**. Recorded in `results/feasibility.json`. |
| 19:50 → 22:05 | `1_prune_yolo26s-prune30` | Running when killed. Nothing recorded, because `save()` only runs after a stage finishes. |

The machine state at 22:00, while the 30% stage was running:

- The Python process (PID 7520) held **25.8 GB of private memory**. The laptop has 15.2 GB of RAM.
  - The memory was no longer growing.
  - It had used only 2414 s of CPU in 3 h 40 min, so it was almost entirely stalled on paging.
- Free RAM was 1.0 GB. Commit was 51.7 / 61.2 GB. 13.6 GB of the page file was in use (peak 19.8 GB).
- The disk showed 100% active from paging.
- C: free space fell from 70.7 to 48.3 GB, because Windows grew the system-managed page file.

After the kill, commit dropped to 27.2 GB and free RAM rose to 6.4 GB.

The prune stages left no files behind. `runs/prototype-yolo26-prune/` holds only the stage-0 exports, and no `yolo26s-prune5.pt` was written.

**Ruled out:** the laptop's new background tools (Disk Guard, WSL Backup, Battery Guard) used under 100 MB together, and WSL was not running. They had nothing to do with this.

Environment: Windows 11, 15.2 GB RAM, RTX 4050 6 GB, Python 3.12.6, torch 2.14.0+cu130, ultralytics 8.4.149, torch-pruning 1.6.1. The DepGraph trace runs on CPU, because `studyprune.prune` builds its example tensor on CPU.

## Problem 1: why pruning YOLO26s blows up (the feasibility finding itself)

**Fact.** The traceback in `results/feasibility.json` is:

```
studyprune.py:107 prune -> tp.pruner.MagnitudePruner(...)
torch_pruning/pruner/algorithms/base_pruner.py:253  DG.get_all_groups(...)
torch_pruning/dependency/graph.py:314 get_pruning_group -> :251 _fix_dependency_graph_non_recursive
torch_pruning/dependency/index_mapping.py:41 __call__  new_idxs.extend(...)  -> MemoryError
```

The memory goes into torch-pruning's index-mapping lists while it builds pruning groups. Neither the model nor the data is the problem. This happens at only 5% pruning, where building the groups should take seconds or minutes.

**Hypothesis (leading).** `studyprune.prune` ignores only `C2PSA` and `Detect`. In YOLO11, the attention lives in `C2PSA`. In YOLO26s, there is also an `Attention` (via `PSABlock`) inside the last `C3k2` (layer 22, `attn=True`). Nothing ignores it, so DepGraph traces through its `qkv` view/split and its matmuls. Index mappings through reshape, split or matmul paths can multiply on every pass, and that would match unbounded `new_idxs.extend` growth. `feasibility.py` already plans this exact test (`1_prune_yolo26s-prune30-attnfixed`, with `extra_ignored=(Attention,)`), but it runs late, after the stages that blow up.

Other suspects, in case that hypothesis fails:

- YOLO26's dual heads (one-to-many and one-to-one) both read the same P3/P4/P5 tensors.
- `unchunk_c2f` may not handle the attention variant of `C3k2` cleanly.

**How to confirm without risking the machine.** Write a small standalone diagnostic with a hard memory cap (see Problem 2):

1. Load `yolo26s.pt`, apply `studyprune.unchunk_c2f`, and build `tp.DependencyGraph().build_dependency(model, example)`.
2. Iterate `DG.get_all_groups(ignored_layers=..., root_module_types=...)` one group at a time.
3. Log each group's root module name, the elapsed time, and the process RSS.

The group where RSS takes off is the culprit. Run this twice: once with the stock ignore list, once with `Attention` added.

## Problem 2: why a failure turned into hours of thrashing (harness bug)

1. **`stage()` catches `MemoryError` and keeps going.** The docstring says "a failure IS a finding, so it never aborts the run". For a resource-exhaustion failure that is harmful:
   - Python does not reliably give the memory back to the OS after a `MemoryError`, so the next stage starts with an already huge process.
   - The next stage, `yolo26s-prune30`, runs the same DepGraph path and fails the same way, only slower.
2. **On Windows, `MemoryError` comes very late.** It is raised only when the commit limit is hit, and the system-managed page file grows first. So the process doesn't fail fast; it pages the whole machine for hours. A cap inside Python can't catch this.
3. **No stage has a time limit**, although these stages should take seconds to minutes.

## What a proper fix looks like

The prototype's own rule is that `scripts/studyprune.py` stays **unchanged**. So keep these fixes in `feasibility.py`, and use the existing `extra_ignored` hook for ignore-list experiments.

1. **Run each prune and export stage in a child process with a hard memory cap and a timeout.**
   - Cap memory at about 6 GB. The laptop runs at about 9–10 GB with VS Code, Claude and Chrome open, so a larger cap starts paging.
   - Use about 15 minutes per prune stage, or about 10× the time the YOLO11s control takes.
   - For the cap, use a Windows Job Object (`JOB_OBJECT_LIMIT_PROCESS_MEMORY`, e.g. via `pywin32`'s `win32job`), or a parent-side watchdog that polls the child's RSS with `psutil` and kills it. Either way, record "exceeded N GB" or "timed out after N min" as the stage result.
   - Keep the "failure is a finding" behaviour, but now it applies to a child process that dies cleanly.
2. **Don't repeat a known blow-up.** If a family's 5% prune fails on memory or time, mark its 30% prune as "skipped: same failure as prune5" instead of running it.
3. **Put the cheap, informative checks first:**
   1. The YOLO11s control (it is known to work, see P1 in `RUNS.md`), to prove that the environment and the harness are fine.
   2. The YOLO26s attention-fixed variant.
   3. The stock YOLO26s ignore list.
4. **Save partial results before a stage starts**, e.g. `{"ok": null, "started": ...}`. Then a killed run still shows which stage it was in.
5. **Record the outcome as a finding.** If the hypothesis holds, the result is something like: "YOLO26s is not prunable by the unchanged studyprune.py; DepGraph group construction exceeds 25 GB at 5% because of the un-ignored Attention in C3k2 layer 22. With Attention ignored: <result>."

## Done means

- A full `feasibility.py` run finishes on this laptop without free RAM dropping below about 3 GB, and without the page file growing.
  - Check in Task Manager, or poll `Get-CimInstance Win32_OperatingSystem`.
- Every stage ends with `ok: true`, or with `ok: false` and a clear reason (error, memory cap or timeout). No stage is missing.
- Problem 1's cause is either confirmed by the per-group diagnostic, or replaced by the cause that diagnostic actually finds.

---

## Resolution (2026-10-19, 22:35 IST)

### Problem 1: cause confirmed, with one correction

`diagnose_groups.py` builds the dependency graph as `studyprune.prune` does, then walks `get_all_groups` one root at a time under a 3 GB job-object cap. Reports are in `results/diagnose/`.

| Run | Result |
|---|---|
| YOLO11s, stock ignore list | 53 groups in 0.6 s, 1.9 GB |
| YOLO11s, **C2PSA not ignored** | **runs away**: root `model.10.m.0.attn.pe.conv` |
| YOLO26s, stock ignore list | **runs away**: root `model.22.m.0.1.attn.pe.conv` |
| YOLO26s, one-to-one head off (`end2end=False`) | **runs away**: same root |
| YOLO26s, + ignore `Attention` | 61 groups in 0.6 s, 2.0 GB |

**Confirmed:** the cause is the un-ignored `Attention` inside layer 22's `C3k2` (`attn=True`). The group that never terminates is the one rooted at that attention's depthwise `pe` conv.

**Mechanism** (`yolo26s-trace-pe.json`):

- Torch-Pruning maps the attention's `view`s as stride-2 flattens. These are the views between `(B, C, H, W)` and `(B, heads, head_dim, N)` around the `bmm`.
- On the loop proj → residual add → reshape → bmm → reshape, the expanding mappings outnumber the collapsing ones, so the index list doubles every lap: 256 → 512 → … → 131,072, all far past the 256 real channels.
- `has_pruning_op` never sees a repeat, so the loop never ends.

**Ruled out:** the dual head. Disabling the one-to-one branch changes nothing.

**YOLO11 has the same bug.** It was hidden only because `studyprune` ignores C2PSA, which holds YOLO11's one attention.

**Correction to "What a proper fix looks like":** adding `Attention` to the ignore list stops the runaway, but the prune then fails in `studyprune.rechunk_c2f` with "unequal C2f halves 176/256":

- Torch-Pruning's `ignored_layers` never reach `build_dependency`. They only stop a module from being a group root and reject finished groups.
- So the `cv1` half of layer 22, which feeds the attention's residual stream, stays pinned at 256 channels while the `cv0` half is pruned. The rechunk guard correctly refuses to stitch them back together.

**The fix that works** is to ignore the whole attention-bearing `C3k2`, the way `studyprune` already ignores the whole C2PSA. `feasibility.py` does this through `extra_ignored=("AttnC3k2",)`, so `studyprune.py` stays unchanged. The predicate matches YOLO26s layers 10, 22 and 23, and on YOLO11s it adds nothing (layers 10 and 23 only), so P1 would be unaffected.

### Problem 2: harness fixed

- **Every stage runs in its own child process** (`memcap.py`):
  - a Windows Job Object cap on committed memory (prune 3 GB, export and latency 3.5 GB, training 6 GB)
  - a timeout (10, 10, 15 and 30 min)
  - a parent watchdog that kills the process tree if system available RAM falls below 1 GB
- **The child records a `MemoryError` without allocating inside the `except` block.** Earlier, formatting the traceback while its frames still held the runaway lists raised a second `MemoryError`. The parent also reads a result-less death at the cap as a memory-cap failure.
- **Known blow-ups aren't repeated.** A memory or timeout failure at 5% marks the matching 30% stage as skipped.
- **The run order is control first, known failure last.** A `started` marker is saved before every stage.
- **The incident run's record is kept** as `results/feasibility-incident-run-2026-10-19.json`.

### Done means: checked

- The full run finishes in about 5 minutes. Every stage ends with `ok: true`, or with `ok: false` and a reason. The stock YOLO26s prune now dies at its 3 GB cap in 18 s, and its 30% sibling is skipped.
- The page file did **not** grow: allocated size stayed at 42,732 MB. Its in-use part rose by 0.6 GB.
- Free RAM **did** drop below 3 GB:
  - The lowest point was 1.65 GB, during the training smoke (a legitimate 4.7 GB peak), and 2.0 GB during the capped runaway stage.
  - The machine started the run with only 2.8–4.2 GB available, with VS Code, Claude and Chrome open, so the 3 GB target can't be met with those apps running. The 1 GB floor is the guarantee the harness actually gives.

### Feasibility result (for the ticket, not the incident)

- **YOLO26s is prunable** with the attention C3k2 ignored, at 5% and 30%, `round_to=8`:
  - 30%: params −32.3% (YOLO11s: −40.1%), GFLOPs 23.1 → 13.8.
  - The per-layer channel pattern is identical to YOLO11s's.
  - Attention widths are consistent, and both heads agree on input width.
  - Eval one-to-many, eval one-to-one and train forwards all run.
- **Pruned exports match PyTorch**, one-to-many head:
  - ONNX: box ≤ 0.0006 px, score 0.
  - NCNN: box ≤ 0.001 px.
- **Unpruned YOLO26s → NCNN works on 8.4.149.** Ultralytics #23365 was fixed in 8.4.7.
- **Laptop CPU latency, 30% pruned** (indicative only): ORT 65 → 37 ms, NCNN 85 → 58 ms.
- **The recovery-training smoke passes.** `PrunedTrainer` ran one epoch with SGD, losses were finite, the reloaded checkpoint is still pruned, and its ONNX export matches.
- **Open:** the one-to-one ONNX export agrees on confidences (max diff 0), but boxes differ by up to 640 px and 2–12 of the top 100 classes differ. It looks like tie-ordering in top-k, but that is **unverified**, and stage 0 never tested the unpruned one-to-one export for comparison.
