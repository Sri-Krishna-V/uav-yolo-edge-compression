# Training on Kaggle: the team guide

This guide takes you from "I have never used Kaggle" to "my training runs are finishing on their own".
Read it once, top to bottom. After that, the wizard does the remembering for you.

**Who this is for:** the two people training the study's baseline models.

| Person | Trains | Run list, in order |
|---|---|---|
| srikr | YOLO11s | A1, A3, A4, B1, B3 |
| teammate | YOLO26s | A2, A5, A6, B2, B4 |

What each run is, and why, is in [`../RUNS.md`](../RUNS.md). You don't need it to follow this guide.

---

## The short version

1. Get ready on your computer: Git, Python, and a copy of this repo ([section 1](#1-get-ready-on-your-computer-15-min)).
2. Run `bash study/kaggle/wizard.sh` and do what it says ([section 2](#2-the-first-time-run-the-wizard)).
3. Each time a training session finishes, run `bash study/kaggle/wizard.sh next` ([section 3](#3-after-every-session-run-the-wizard-again)).

That's the whole job. The rest of this page explains what is happening, the rules that keep the
results valid, and what to do when something goes wrong.

---

## 1. Get ready on your computer (15 min)

You need three things on your own computer. Training does **not** happen there; your computer only runs
the wizard and keeps a copy of the results.

1. **Git.** Windows: install [Git for Windows](https://git-scm.com/download/win). It includes **Git Bash**,
   the terminal you'll use for everything below. macOS: run `git --version` in Terminal and accept the
   install prompt. Linux: install it with your package manager.
2. **Python 3.9 or newer.** From [python.org](https://www.python.org/downloads/). On Windows, tick
   *"Add python.exe to PATH"* in the installer. You only need plain Python: no packages.
3. **This repo.** Open Git Bash (Windows) or Terminal (macOS/Linux) and run:

   ```bash
   git clone https://github.com/Sri-Krishna-V/uav-yolo-edge-compression.git
   cd uav-yolo-edge-compression
   ```

   If you already cloned it, run `git pull` instead to get the latest version.

You also need a **Kaggle account with a verified phone number**. The wizard walks you through that.

---

## 2. The first time: run the wizard

From inside the repo folder, run:

```bash
bash study/kaggle/wizard.sh
```

The wizard opens each web page for you, says exactly what to click, copies the notebook code into your
clipboard, and remembers your answers. You can stop it with **Ctrl-C** at any time and run it again later.

It has 9 stages:

| Stage | What you do | Time |
|---|---|---|
| 1. About you | Your name, your run list, a folder for checkpoint copies | 1 min |
| 2. Kaggle account | Sign in and verify your phone. Without this, Kaggle hides the GPU and Internet settings | 5 min |
| 3. Notebook A | Create a notebook called `uav-<you>-a`; set **GPU T4 x2** and **Internet on** | 3 min |
| 4. Session 0 | Paste the cell and start it. It installs everything, checks the dataset and runs the **smoke test** (proof this account's setup is sound). srikr's Session 0 also runs the **probe**, which measures the batch size everyone must use | 40 min (srikr: ~1.5 h) of waiting |
| 5. Session 0 result | Download the `bring-home` folder; the wizard checks the smoke test passed | 2 min |
| 6. Batch sizes | srikr pushes the probe's numbers to GitHub. The teammate waits until they arrive (the wizard checks for them) | 2 min |
| 7. Notebook B | Create a second notebook `uav-<you>-b` that reads notebook A's output | 3 min |
| 8. Session 1 | Your first training session starts, in notebook B | 2 min |
| 9. What happens now | When to come back | |

**The teammate can do stages 1–5 at the same time as srikr.** Only stage 6 waits for srikr's probe.

### Why two notebooks?

A training run takes longer than one Kaggle session (12 hours at most), so it has to continue in the next
session from its saved checkpoint. **A Kaggle notebook can't read its own earlier output**, but it *can*
read another notebook's output. So each person has two notebooks and uses them in turn:

```
Session 0 in A ──> Session 1 in B ──> Session 2 in A ──> Session 3 in B ──> ...
                   (reads A)          (reads B)          (reads A)
```

Each session copies every checkpoint and result it finds into its own output, so the newest version of
either notebook always holds everything done so far. The wizard tracks which notebook is next.

---

## 3. After every session: run the wizard again

When a session has finished (check the notebook's page on Kaggle), run:

```bash
bash study/kaggle/wizard.sh next
```

It has 4 stages, about 5 minutes in total:

1. **Bring the results home.** Download the `bring-home` folder from the finished version's Output. The
   wizard files the result records into the repo and copies the checkpoints into your checkpoint folder.
2. **Share the records.** It offers to commit and push finished runs' records. If you can't push, send
   the files to Krishna.
3. **Get the other notebook ready.** The first time you go back to notebook A, it shows you how to attach
   notebook B as an input. That's needed once only.
4. **Start the next session.** It works out the right `TASK` line (which runs, and whether any start
   fresh), copies the cell to your clipboard, and you click **Save & Run All**.

Repeat until the wizard says **"Every run on your list is finished"**.

---

## What a session does

Every session runs the same single notebook cell ([`notebook_cell.txt`](notebook_cell.txt)). Only its
`TASK` line changes:

```python
TASK = "train A1 A3 --new A1 A3"

!rm -rf /kaggle/working/uav && git clone -q --depth 1 https://github.com/Sri-Krishna-V/uav-yolo-edge-compression.git /kaggle/working/uav
!cd /kaggle/working/uav && git log -1 --format="code version: %h %s"
!python /kaggle/working/uav/study/kaggle/session.py setup && python /kaggle/working/uav/study/kaggle/session.py {TASK}
```

Inside the session, in order:

1. **Get the code.** It downloads the latest version of this repo.
2. **`setup`** (~15–20 min, every session, because each session starts on a blank machine):
   - checks you have 2× T4 and Internet;
   - installs the pinned PyTorch (2.14.0, CUDA 13) and Ultralytics (8.4.149);
   - downloads VisDrone and checks every zip against the SHA-256 recorded in
     [`INVARIANTS.md`](../INVARIANTS.md) §2;
   - converts the dataset to YOLO format once and checks every label row against the raw files.

   If anything is wrong it prints a **STOP** box and trains nothing.
3. **The TASK.** For a training session:
   - copies in all checkpoints and records from the attached input;
   - starts one run per GPU;
   - prints progress every 15 minutes;
   - after any epoch that would leave the next one finishing past ~11 hours, **pauses cleanly** with the
     checkpoint saved;
   - evaluates every run that finished, with the study's own evaluator;
   - writes a `bring-home` folder;
   - ends with a **SESSION RESULT** box.

### Reading the SESSION RESULT

| It says | Meaning | What you do |
|---|---|---|
| `FINISHED: record written` | Trained 100 epochs and evaluated | Nothing. `wizard.sh next` files the record |
| `PAUSED: resume it next session` | Ran out of session time, checkpoint saved | Nothing. The next session continues it |
| `already finished earlier` | That run was already done | Nothing |
| `CRASHED (exit N)` | Something failed mid-run. The last 40 log lines are printed above | Start the next session normally; it resumes from the last saved epoch. If the same run crashes twice, send Krishna the log |
| `INVALID: GPU ran out of memory and the batch changed` | The run broke the same-batch rule and was set aside | Stop and tell Krishna. **Do not** change the batch size yourself |

### STOP messages

The session stops early, with a box that says what to do. The common ones:

| STOP says | Fix |
|---|---|
| `No GPU` / `This session has [...]` | Session options → Accelerator → **GPU T4 x2**, then Save & Run All again |
| `No internet` | Session options → **Internet ON**. If the switch is missing, verify your phone (Kaggle Settings) |
| `runs.yaml has no probe results yet` | srikr hasn't pushed the probe numbers. Wait, then start a new session |
| `No checkpoint found for [...]` | The other notebook's output isn't attached, or isn't the latest version. Fix the Input panel and run again. Add `--new` **only** if that run truly has never started |
| `The VisDrone download did not verify` | Don't train. Send Krishna the log |
| `smoke test FAILED` | Don't train. Send Krishna `bring-home/smoke_results-kaggle-<you>.json` |

---

## Rules that keep the results valid

The paper compares numbers across runs, people and accounts. These rules are what make that comparison
fair. The code enforces most of them. Please don't work around the ones it can't enforce.

1. **Change only the `TASK` line**, and only to what the wizard gives you. Everything else (recipe,
   seeds, batch size, evaluation) is frozen in the repo on purpose.
2. **GPU T4 x2, never P100.** Our PyTorch build is CUDA 13, and CUDA 13 can't run on the P100.
3. **One session at a time per account.** No interactive GPU session while a training session runs:
   both burn the same weekly quota, and two sessions can't safely share one checkpoint chain.
4. **Never lower the batch size yourself**, even after an out-of-memory crash. The batch is the same for
   every run at a given resolution, or the runs aren't comparable. Report it instead.
5. **Keep the checkpoints off Kaggle.** `wizard.sh next` copies them into your checkpoint folder. Put
   that folder in Google Drive or OneDrive, so the only copy is never on one account or one laptop.
6. **Attach only your own two notebooks as inputs.** A session loads every checkpoint it finds in its inputs,
   and loading a checkpoint can run code stored in it. Never attach someone else's notebook or dataset.
7. **Don't evaluate or report numbers any other way** (no `model.val()`, no Kaggle-side tweaks). The
   only reported numbers are the ones in `study/verification/runs/*.json`.

---

## Budget and schedule

- Each account gets **about 30 GPU-hours a week**. It resets **Saturday 00:00 UTC (05:30 IST)**. A session
  uses quota only while it runs, so a session that finishes early stops costing.
- One session is at most ~11 hours of training, so that's **2–3 sessions per account per week**.
- A 640 run needs very roughly 5 GPU-hours alone; a 960 run ~2.25× that; a 1280 run ~4×. The probe
  measures the real numbers on Kaggle.
- If the probe says **pair**, each session trains two runs at once, one per GPU. If it says **single**,
  one run per session; then the 4 CPU cores are the bottleneck, and pairing would only slow each run down.
- Check your remaining quota on Kaggle's notebook editor (Session options) or on your profile.

If the schedule slips, the cut order is in [`../RUNS.md`](../RUNS.md): the 1280 runs go first. Never cut A1–A4.

---

## Troubleshooting

**The wizard says `No Python 3.9+ found`.** Install Python (section 1). On Windows, reopen Git Bash
afterwards.

**The wizard can't reach the clipboard.** It saves the cell to `study/kaggle/.cell-to-paste.txt` as well;
open that file and copy from it.

**I can't find a button the wizard names.** Kaggle renames things now and then. The ideas stay the same:
*Session options* is in the editor's right-hand panel; *Save Version* is at the top right; *Output* and
*Logs* are on the notebook's page for a finished version. If you're stuck, take a screenshot and ask.

**The session's log is huge and I can't find the result.** Scroll to the very end. The SESSION RESULT
box is always last. Each run's full training log is in `bring-home/logs/`.

**The session ran past 12 hours / the version failed.** It shouldn't: training stops at ~11 h, and
anything still running at 11.6 h is killed so the output is still saved. If it happens, run
`wizard.sh next` anyway and tell Krishna. A version Kaggle cancelled may not be usable as an input.

**I started a session in the wrong notebook.** Stop it on its notebook page (cancel the running version)
and start it in the right one. If it already finished, nothing is lost: the furthest checkpoint wins when
the notebooks are next chained.

**I'm the teammate and can't push to GitHub.** That's fine. Send the new files in `study/verification/`
to Krishna after each session; your checkpoints stay in your checkpoint folder.

---

## Doing it by hand, without the wizard

The wizard only automates the steps below.

1. Kaggle: verify your phone (Settings → Phone verification).
2. Create notebook `uav-<you>-a`. Set **Session options → Accelerator: GPU T4 x2** and **Internet: on**.
   Replace the notebook's content with the single cell from [`notebook_cell.txt`](notebook_cell.txt).
3. Session 0: set `TASK = "smoke --who <you>"` (srikr: `"smoke --who srikr --probe"`).
   **Save Version → Save & Run All (Commit) → Save.**
4. When it finishes: download `bring-home` from the version's Output.
   - srikr: run `python study/kaggle/session.py apply-probe <path>/kaggle_probe.json`, then commit and push
     `study/kaggle/runs.yaml`.
5. Create notebook `uav-<you>-b`, same settings, and **Add Input → Your Work → uav-<you>-a**.
6. Find your next TASK with
   `python study/kaggle/session.py next-task --owner <srikr|teammate> --mirror <checkpoint folder>`.
   Put it in the cell and Save & Run All in notebook B.
7. After each session:
   - run `python study/kaggle/session.py import-bring-home <downloaded folder> --mirror <checkpoint folder>`;
   - get the next TASK the same way;
   - run it in the *other* notebook. Each notebook needs the other attached as an input, once.

---

## Why it is built this way

- **The dataset lives outside `/kaggle/working`.** Kaggle saves everything in `/kaggle/working` (up to
  20 GB) as the version's output. The ~4 GB dataset is re-downloaded each session instead, and verified
  every time.
- **Training pauses at ~11 h, not 12.** Kaggle may not save the output of a session that hits its 12-hour
  limit, and a cancelled version may not be accepted as an input. The pause happens right after a
  checkpoint is written, and *not* through Ultralytics' own stop. That stop would run the end-of-training
  step, which marks the checkpoint "finished" and makes it impossible to resume.
- **Fresh runs must be named with `--new`.** If the other notebook's output is missing, the code would
  otherwise quietly restart a run from zero and burn hours of quota. Now it stops and says why.
- **One run per GPU, not one run across both GPUs.** Ultralytics' two-GPU mode runs training in a
  subprocess that our pause and batch checks can't reach. It also brings little speed on two T4s with
  only 4 CPU cores.
- **The batch size has 20% memory headroom.** The probe's short test can't see VisDrone's densest images.
  Ultralytics responds to an out-of-memory error by silently halving the batch, which would break the
  same-batch rule. The training code now catches that and marks the run INVALID instead.

## Facts this guide relies on

Checked on 2026-10-12. Kaggle changes its setup every few weeks, so `setup` checks the important ones
itself on every session.

| Fact | Confidence | Source |
|---|---|---|
| T4 x2 = 2 GPUs, 4 CPU cores, 29 GB RAM, 12 h max per session, 20 GB saved output | High | [Kaggle docs](https://www.kaggle.com/docs/notebooks) |
| ~30 GPU-hours a week per account, reset Saturday 00:00 UTC, commits count | High / medium | [Kaggle docs](https://www.kaggle.com/docs/efficient-gpu-usage), staff forum posts |
| Kaggle's T4 driver (580) runs CUDA 13 builds; Kaggle's Python is 3.13 | Medium-high | User reports, Aug–Oct 2026; `setup` checks the driver and installs per its CUDA version |
| torch 2.14.0+cu130, torchvision 0.29.0+cu130 and faster-coco-eval 1.8.0 have Python 3.13 Linux wheels | High | Checked on download.pytorch.org and PyPI |
| A notebook can't attach its own output; notebook inputs follow the latest version | Medium | Kaggle docs (packages page), staff forum posts |
| Output past the 12 h limit is saved only "best effort" | Medium | Kaggle staff forum post |
