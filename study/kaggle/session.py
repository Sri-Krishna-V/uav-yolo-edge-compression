"""Every Kaggle step of the study, one command each. The notebook (kaggle/train.ipynb) calls these.

  python session.py setup                    # every session, first: install the pinned env, fetch + verify VisDrone
  python session.py smoke --who srikr        # Session 0, once per account (add --probe on the ONE account that probes)
  python session.py train A1 A3 --new A1 A3  # first session of two runs: one per GPU
  python session.py train A1 A3              # any later session: resumes them from the attached input
  python session.py status                   # every run: finished / paused at epoch N / not started
  python session.py apply-probe PROBE.json   # on your own computer, once: write the probe's numbers into runs.yaml
  python session.py import-bring-home PATH --mirror DIR   # on your computer, after each session (the wizard runs these)
  python session.py next-task --owner srikr --mirror DIR  # on your computer: the TASK line for the next session
  python session.py preflight TASK / wrap-up TASK         # the notebook's first and last commands (see below)
  python session.py seed-folder --owner X --mirror DIR --out DIR   # on your computer: `wizard.sh repair` uploads it

`train` first copies every checkpoint and record found in the attached inputs into this session (a Kaggle
notebook cannot attach its own output, so each person alternates two notebooks attached to each other).
Then, per run: a run with a record is skipped; a paused run resumes; a trained but unevaluated run is
evaluated. A run with nothing found starts from zero ONLY if listed in --new; otherwise it is BLOCKED (the
likeliest cause is a forgotten or outdated input, and restarting would burn hours of quota), and the other run
trains anyway. Training pauses after the last epoch that ends before TRAIN_STOP_H hours into the session, so the
session always finishes cleanly and its output is saved.

Every session's output also holds a stamp (session_stamp.json): its id and whether it trained. `preflight`, before
setup, checks that the inputs hold the session the wizard just brought home (TASK --input-from) and that each run
can start, so a wrong or outdated input costs a minute, not a session. `wrap-up` runs whatever happened: a session
that ended without its SESSION_RESULT still writes one (STOPPED before training, or CRASHED), relays its inputs'
runs into its output, and stamps it, so no notebook version is ever an empty link in the chain.

A run built on another run (runs.yaml prune_from: its pruned start; distill_from: its teacher) needs that run
finished: its record, and its best.pt in the inputs. A run with a batch_fallback that runs out of GPU memory in
its first epoch restarts at once, from zero, at that smaller batch, but only if it started from zero in this
session: one that trained in an earlier session is stopped and reported, never thrown away. Once started, a run
always resumes at the batch it started with, read from its own args.yaml.

Stdlib only at import time: `setup` runs before anything is installed, and `apply-probe` runs on a laptop.
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # study/
SCRIPTS = ROOT / "scripts"
RUNS_FILE = ROOT / "kaggle" / "runs.yaml"
RUNS_DIR = ROOT / "runs" / "baseline"
VERIFICATION = ROOT / "verification"
RECORD_DIR = VERIFICATION / "runs"

WORKING = Path("/kaggle/working")
INPUTS = Path("/kaggle/input")
BRING_HOME = WORKING / "bring-home"
LOGS = WORKING / "logs"
SESSION_START_FILE = Path("/tmp/study_session_start")
PRUNED_DIR = Path("/tmp/study-pruned")  # re-made every session from the source run (deterministic), never saved
STOP_FILE = Path("/tmp/study_stop.txt")  # why this session stopped early, for wrap-up's SESSION_RESULT
SESSION_ID_FILE = Path("/tmp/study_session_id")
STAMP_NAME = "session_stamp.json"  # in uav/study/ of every session's output and in its bring-home
SEED_DIR = Path("/tmp/study-seed")  # an attached seed dataset (wizard.sh repair) unpacked, if Kaggle left it zipped

SESSION_LIMIT_H = 12.0  # Kaggle's cap on one GPU session
TRAIN_STOP_H = 11.0  # no epoch may end later than this; leaves time for evaluation and saving the output
HARD_KILL_H = 11.6  # anything still running now is killed, so the version still saves its output
CUDA_TAGS = [("cu130", 13.0), ("cu126", 12.6)]  # torch 2.12+ ships only these two; cu130 matches the laptop
EXIT_PAUSED, EXIT_BATCH_CHANGED = 75, 76  # from scripts/studytrain.py


def banner(text):
    print("\n" + "=" * 78 + f"\n  {text}\n" + "=" * 78, flush=True)


def fail(text):
    banner(f"STOP: {text}")
    if WORKING.is_dir():  # on Kaggle: wrap-up puts the reason in this session's SESSION_RESULT
        with open(STOP_FILE, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    sys.exit(1)


class Blocked(Exception):
    """One run cannot start this session (something it needs is missing); the other run still can."""


def run(cmd, **kw):
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], **kw)


def session_start():
    if not SESSION_START_FILE.exists():
        SESSION_START_FILE.write_text(str(time.time()))
    return float(SESSION_START_FILE.read_text())


def hours_into_session():
    return (time.time() - session_start()) / 3600


def load_runs():
    """runs.yaml, without needing PyYAML: the laptop-side commands run on machines without the study env."""
    text = RUNS_FILE.read_text(encoding="utf-8")
    value = lambda v: None if v == "null" else int(v) if v.isdigit() else v  # noqa: E731
    cfg = {"batch_per_gpu": {}, "pairing": value(re.search(r"^pairing: (\S+)", text, re.M).group(1)), "runs": {}}
    for imgsz, b in re.findall(r"^  (\d+): (\S+)", text, re.M):
        cfg["batch_per_gpu"][int(imgsz)] = value(b)
    for rid, body in re.findall(r"^  ([A-Z]\d+): \{(.*)\}", text, re.M):
        cfg["runs"][rid] = {k.strip(): value(v.strip()) for k, v in (kv.split(":") for kv in body.split(","))}
    for r in cfg["runs"].values():
        if r.get("prune_from"):  # a pruning-recovery run is named after the run it prunes
            r["name"] = f"{run_name(cfg['runs'][r['prune_from']])}-prune{r['prune']}"
        if r.get("distill_from"):  # a run trained with a teacher: <run>-distill
            r["name"] = f"{run_name(r)}-distill"
    return cfg


def run_name(r):
    return r.get("name") or f"{Path(r['model']).stem}-{r['imgsz']}-s{r['seed']}"


def sources(r):
    """The runs this run is built on: the one it prunes and its teacher. It starts only once they have finished."""
    return [r[k] for k in ("prune_from", "distill_from") if r.get(k)]


MISSING_HINT = ("It is in neither input. Likely fix: under Input, update the other notebook to its newest version. "
                "If that does not help, run on your computer: bash study/kaggle/wizard.sh repair")


def source_blockers(cfg, r, has_best, has_record):
    """Why the runs r is built on (its pruned start, its teacher) are not usable here; [] if they are."""
    out = []
    for key, role in (("prune_from", "the run it prunes"), ("distill_from", "its teacher")):
        if r.get(key):
            sid, sname = r[key], run_name(cfg["runs"][r[key]])
            if not has_record(sname):
                out.append(f"{sid} ({role}) has not finished: it must finish first")
            elif not has_best(sname):
                out.append(f"{sid}'s best.pt ({role}) is missing. {MISSING_HINT}")
    return out


def make_pruned(cfg, r):
    """Prune the source run's best.pt for a recovery run; the same source always gives the same model."""
    source = RUNS_DIR / run_name(cfg["runs"][r["prune_from"]]) / "weights" / "best.pt"
    if not source.exists():
        raise Blocked(f"{r['prune_from']}'s best.pt is missing. {MISSING_HINT}")
    out = PRUNED_DIR / f"{run_name(r)}.pt"
    if not out.exists():
        if run([sys.executable, SCRIPTS / "studyprune.py", "--source", source, "--ratio", r["prune"] / 100, "--imgsz", r["imgsz"], "--out", out]).returncode:
            raise Blocked(f"pruning {r['prune_from']} failed (see the log above). Send this log to Krishna")
    return out


def teacher(cfg, r):
    """The distill_from run's best.pt, the teacher of a distilled or combined run. Only a finished teacher will do."""
    tid = r["distill_from"]
    name = run_name(cfg["runs"][tid])
    best = RUNS_DIR / name / "weights" / "best.pt"
    if not best.exists():
        raise Blocked(f"its teacher {tid}'s best.pt is missing. {MISSING_HINT}")
    state, detail = state_of(name)
    if state != "done":
        raise Blocked(f"its teacher {tid} has not finished ({detail}): {tid} must finish first")
    return best


def started_batch(name):
    """The per-GPU batch a started run trains at, from its own args.yaml (one GPU per run, so batch is per GPU)."""
    f = RUNS_DIR / name / "args.yaml"
    m = re.search(r"^batch: (\d+)", f.read_text(encoding="utf-8"), re.M) if f.exists() else None
    return int(m.group(1)) if m else None


# ── setup ────────────────────────────────────────────────────────────────────


def check_gpus():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout
        header = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        fail("No GPU. In the notebook's right-hand panel set Session options -> Accelerator -> GPU T4 x2, then run again.")
    gpus = [line.strip() for line in q.splitlines() if line.strip()]
    if len(gpus) != 2 or not all("T4" in g for g in gpus):
        fail(f"This session has {gpus}. The study needs Accelerator = GPU T4 x2 (never P100: it cannot run CUDA 13).")
    m = re.search(r"CUDA Version:\s*([\d.]+)", header)
    driver_cuda = float(m.group(1)) if m else 0.0
    print(f"GPUs: {gpus}; the driver supports CUDA up to {driver_cuda}", flush=True)
    return driver_cuda


def check_internet():
    try:
        urllib.request.urlopen("https://github.com", timeout=15)
    except OSError:
        fail("No internet. In the right-hand panel turn Session options -> Internet ON (needs a phone-verified account).")


def pins():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    return {name: ver for name, ver in re.findall(r"^(torch|torchvision)==([\w.]+)", text, re.M)}


def torch_ok(want):
    probe = "import torch, json; print(json.dumps([torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count()]))"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    if out.returncode:
        return None
    ver, cuda, available, count = json.loads(out.stdout.strip().splitlines()[-1])
    return (ver, cuda) if ver.split("+")[0] == want and available and count == 2 else None


def setup():
    t0 = time.time()
    session_start()
    driver_cuda = check_gpus()
    check_internet()
    free_gb = shutil.disk_usage("/tmp").free / 2**30
    if free_gb < 15:
        fail(f"Only {free_gb:.0f} GB free disk; the dataset needs ~5 GB and the environment ~8 GB.")

    want = pins()
    have = torch_ok(want["torch"])
    if not have:
        for tag, needs in CUDA_TAGS:
            if needs > driver_cuda:
                continue
            print(f"Installing torch {want['torch']} ({tag}) - several minutes, ~3 GB.", flush=True)
            r = run([sys.executable, "-m", "pip", "install", "-q", "--progress-bar", "off", f"torch=={want['torch']}", f"torchvision=={want['torchvision']}",
                     "--index-url", f"https://download.pytorch.org/whl/{tag}"])
            if r.returncode == 0 and (have := torch_ok(want["torch"])):
                break
        if not have:
            fail(f"Could not install torch {want['torch']} for this driver (CUDA {driver_cuda}). Send this log to Krishna.")
    print(f"torch {have[0]} (CUDA {have[1]}) works on both GPUs.", flush=True)

    r = run([sys.executable, "-m", "pip", "install", "-q", "--progress-bar", "off", "-r", ROOT / "requirements.txt"])
    if r.returncode:
        fail("pip could not install study/requirements.txt (see the lines above).")
    print("Note: red 'pip's dependency resolver' warnings about other Kaggle packages are normal and harmless.", flush=True)

    if run([sys.executable, SCRIPTS / "fingerprint_visdrone.py", "--download", "--convert"]).returncode:
        fail("The VisDrone download did not verify. Do NOT train: send this log to Krishna.")
    banner(f"SETUP OK in {(time.time() - t0) / 60:.0f} min. Environment pinned, dataset verified and converted.")


# ── smoke / probe ────────────────────────────────────────────────────────────


def smoke(who, also_probe):
    who = re.sub(r"[^a-z0-9-]", "", who.lower())
    if not who:
        fail("--who needs your name, e.g. --who srikr")
    ok = run([sys.executable, SCRIPTS / "smoke.py", "--tag", f"kaggle-{who}"]).returncode == 0
    if ok and also_probe:
        run([sys.executable, ROOT / "kaggle" / "probe_kaggle.py"])
    bring_home([])
    if not ok:
        fail(f"smoke test FAILED: this account must not produce numbers yet. Send bring-home/smoke_results-kaggle-{who}.json to Krishna.")
    banner("SMOKE TEST PASSED" + (" + PROBE DONE" if also_probe else ""))


def apply_probe(path, force):
    probe = json.loads(Path(path).read_text(encoding="utf-8"))
    batches = {int(k): v for k, v in probe.get("batch_per_gpu", {}).items()}
    verdict = probe.get("pairing", {}).get("verdict")
    if not batches or not verdict:
        fail(f"{path} has no batch sizes or no pairing verdict: the probe did not finish. Rerun it.")
    text = RUNS_FILE.read_text(encoding="utf-8")
    for imgsz, b in batches.items():
        current = re.search(rf"^  {imgsz}: (\S+)", text, re.M).group(1)
        if current not in ("null", str(b)) and not force:
            fail(f"runs.yaml already freezes batch {current} at {imgsz}; the probe says {b}. Changing it invalidates runs. Use --force only if no run at {imgsz} has started.")
        text = re.sub(rf"^  {imgsz}: \S+", f"  {imgsz}: {b}", text, flags=re.M)
    text = re.sub(r"^pairing: \S+", f"pairing: {verdict}", text, flags=re.M)
    RUNS_FILE.write_text(text, encoding="utf-8")
    VERIFICATION.mkdir(exist_ok=True)
    if Path(path).resolve() != (VERIFICATION / "kaggle_probe.json").resolve():
        shutil.copy(path, VERIFICATION / "kaggle_probe.json")
    print(f"runs.yaml: batch_per_gpu {batches}, pairing {verdict} (gain {probe['pairing'].get('gain')}x). Commit and push it.")


# ── train ────────────────────────────────────────────────────────────────────


def checkpoint_epoch(path):
    import torch

    # Unpickles (Ultralytics checkpoints hold model objects, so weights_only=True cannot load them), which can run
    # code from the file. Only checkpoints this study wrote reach here: find_in_inputs() reads nothing but the
    # uav/study tree our own notebook cell creates, and Ultralytics unpickles the same last.pt again to resume.
    return torch.load(path, map_location="cpu", weights_only=False).get("epoch", -1)


def find_in_inputs(pattern):
    """Files under an attached input's uav/study/ tree: only the output of our own notebook cell, or the seed dataset
    `wizard.sh repair` uploads from the person's own checkpoints, is ever read."""
    unpack_seeds()
    return sorted(f for root in (INPUTS, SEED_DIR) if root.is_dir() for f in root.glob(f"**/uav/study/{pattern}"))


def unpack_seeds():
    """A seed dataset arrives as uav.zip (or uav.tar) if Kaggle does not unpack it: unpack it once, outside /kaggle/input."""
    if SEED_DIR.exists() or not INPUTS.is_dir():
        return
    SEED_DIR.mkdir(parents=True)
    for n, archive in enumerate(sorted([*INPUTS.glob("**/uav.zip"), *INPUTS.glob("**/uav.tar")])):
        import tarfile
        import zipfile

        dest = SEED_DIR / str(n)
        (zipfile.ZipFile(archive) if archive.suffix == ".zip" else tarfile.open(archive)).extractall(dest)
        print(f"unpacked the seed {archive} -> {dest}", flush=True)


def carry_forward():
    """Copy every run, evaluation and record found in the attached inputs into this session's output.

    A Kaggle notebook cannot attach its own earlier output, so each person alternates two notebooks, each
    attached to the other (kaggle/README.md). Copying EVERYTHING forward each session keeps the latest
    version of either notebook a complete copy of the study so far, whichever run it trained.
    """
    runs = {}
    for last in find_in_inputs("runs/baseline/*/weights/last.pt"):
        runs.setdefault(last.parents[1].name, []).append(last)
    for name, candidates in runs.items():
        local = RUNS_DIR / name / "weights" / "last.pt"
        # -1 (finished) beats any epoch; among unfinished, the furthest-trained checkpoint wins.
        pool = candidates + ([local] if local.exists() else [])
        best = max(pool, key=lambda f: (checkpoint_epoch(f) == -1, checkpoint_epoch(f)))
        if best != local:
            shutil.copytree(best.parents[1], RUNS_DIR / name, dirs_exist_ok=True)
            print(f"carried forward {name} (epoch {'final' if checkpoint_epoch(best) == -1 else checkpoint_epoch(best) + 1}) from {best.parents[1]}", flush=True)
    carry_records()


def carry_records():
    """The evaluations and records found in the attached inputs (stdlib: wrap-up's relay uses it too)."""
    for d in find_in_inputs("runs/baseline/*-eval-*"):
        if d.is_dir() and not (RUNS_DIR / d.name).exists():
            shutil.copytree(d, RUNS_DIR / d.name)
    for pattern, dest in (("verification/runs/*.json", RECORD_DIR), ("verification/smoke_results-kaggle-*.json", VERIFICATION),
                          ("verification/kaggle_probe.json", VERIFICATION)):
        for f in find_in_inputs(pattern):
            if not (dest / f.name).exists():
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, dest / f.name)


def state_of(name):
    """'done' | 'eval-only' | 'resume' | 'fresh', from this session's files (call carry_forward first)."""
    if (RECORD_DIR / f"{name}.json").exists():
        return "done", "finished and evaluated"
    last = RUNS_DIR / name / "weights" / "last.pt"
    if not last.exists():
        return "fresh", "not started"
    e = checkpoint_epoch(last)
    return ("eval-only", "trained, needs evaluation") if e == -1 else ("resume", f"paused after epoch {e + 1}")


def progress(name):
    f = RUNS_DIR / name / "results.csv"
    if not f.exists():
        return "starting"
    rows = list(csv.DictReader(f.read_text(encoding="utf-8").splitlines()))
    if not rows:
        return "epoch 0"
    row = {k.strip(): v for k, v in rows[-1].items()}
    return f"epoch {row.get('epoch', len(rows)).strip()}, val mAP50-95 {float(row.get('metrics/mAP50-95(B)', 'nan')):.4f}"


def plan(cfg, rid, state, gpu):
    """One run's launch this session: its studytrain arguments and its batch. Stops if anything it needs is missing."""
    r = cfg["runs"][rid]
    name = run_name(r)
    args = ["--model", make_pruned(cfg, r) if r.get("prune_from") else r["model"], "--imgsz", r["imgsz"], "--seed", r["seed"], "--name", name]
    if r.get("prune_from"):
        args += ["--pruned"]
    if r.get("epochs"):
        args += ["--epochs", r["epochs"]]
    if r.get("distill_from") and state != "eval-only":  # resume rebuilds the teacher from this path too
        args += ["--distill", teacher(cfg, r)]
    args += {"resume": ["--resume"], "eval-only": ["--eval-only"], "fresh": []}[state]
    planned = r.get("batch") or cfg["batch_per_gpu"][r["imgsz"]]
    # A started run goes on at the batch it started with (its fallback, if it fell back), never runs.yaml's.
    batch = planned if state == "fresh" else (started_batch(name) or planned)
    if batch not in (planned, r.get("batch_fallback")):
        raise Blocked(f"it started at batch {batch}, but runs.yaml plans {planned}; resuming at another batch would "
                      "break the same-batch rule. Send this log to Krishna")
    return {"name": name, "args": args, "state": state, "gpu": gpu, "planned": planned, "batch": batch,
            "fallback": r.get("batch_fallback"), "fallback_from": planned if batch != planned else None}


def launch(p, stop_by):
    """Start (or restart) one run's studytrain on its GPU at p['batch']; output goes to its log."""
    cmd = [sys.executable, SCRIPTS / "studytrain.py", *p["args"], "--device", p["gpu"], "--batch-per-gpu", p["batch"], "--stop-by", stop_by]
    if p["fallback_from"]:
        cmd += ["--fallback-from", p["fallback_from"]]  # the record says the run trained at its fallback batch
    # A run planned close to the memory limit: expandable segments cut fragmentation, and so the risk of running out
    # of memory after the first epoch, which Ultralytics does not recover from. The allocator does not change results.
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"} if p["fallback"] else None
    p["proc"] = subprocess.Popen([str(c) for c in cmd], stdout=p["log"], stderr=subprocess.STDOUT, env=env)


def train(ids, new):
    cfg = load_runs()
    unknown = [i for i in ids if i not in cfg["runs"]]
    if unknown:
        fail(f"Unknown run id(s) {unknown}. Valid: {list(cfg['runs'])}")
    if not 1 <= len(ids) <= 2:
        fail("Give one or two run ids: one per GPU.")
    if cfg.get("pairing") is None:
        fail("runs.yaml has no probe results yet. Wait for Krishna's probe commit, then start a new session.")
    if len(ids) == 2 and cfg["pairing"] == "single":
        fail("The probe found two runs per session are no faster than one (4 CPU cores). Run ONE id per session.")
    for i in ids:
        if not cfg["runs"][i].get("batch") and cfg["batch_per_gpu"].get(cfg["runs"][i]["imgsz"]) is None:
            fail(f"No batch size for imgsz {cfg['runs'][i]['imgsz']} in runs.yaml (the probe found none that fits). Ask Krishna.")
    if hours_into_session() > TRAIN_STOP_H - 1:
        fail("Less than an hour of this session is left. Start a new session.")
    if set(new) - set(ids):
        fail(f"--new lists {sorted(set(new) - set(ids))}, which are not being trained in this session.")

    carry_forward()
    states = {i: state_of(run_name(cfg["runs"][i])) for i in ids}
    for i, (state, detail) in states.items():
        print(f"{i} {run_name(cfg['runs'][i])}: {detail}", flush=True)
    # A run that cannot start is BLOCKED and reported; the other run still trains, so a session is never wasted on
    # one missing input. Each run is planned (pruned start, teacher, batch) before the first one starts.
    blocked, runs = {}, {}
    for gpu, rid in enumerate(ids):
        state = states[rid][0]
        if state == "done":
            continue
        if state == "fresh" and rid not in new:
            blocked[rid] = ("no checkpoint of it is in the inputs, so it would restart from zero and throw its earlier "
                            f"epochs away. {MISSING_HINT}")
            continue
        if rid in new and state != "fresh":
            print(f"{rid}: listed in --new, but it has already started ({states[rid][1]}); continuing it instead.", flush=True)
        try:
            runs[rid] = plan(cfg, rid, state, gpu)
        except Blocked as e:
            blocked[rid] = str(e)
    for rid, why in blocked.items():
        print(f"{rid} BLOCKED: {why}", flush=True)
    if blocked and not runs:
        fail("Nothing could start: " + "; ".join(f"{rid}: {why}" for rid, why in blocked.items()))

    stop_by = session_start() + TRAIN_STOP_H * 3600
    kill_at = session_start() + HARD_KILL_H * 3600
    LOGS.mkdir(parents=True, exist_ok=True)
    for n, (rid, p) in enumerate(runs.items()):
        if n:
            time.sleep(90)  # let the first run download weights and build the label cache alone
        p["log"] = open(LOGS / f"{rid}-{p['name']}.log", "a", encoding="utf-8")
        print(f"{rid}: GPU {p['gpu']}, batch {p['batch']}, log in /kaggle/working/logs/{rid}-{p['name']}.log", flush=True)
        launch(p, stop_by)
    # If Kaggle stops this session from outside (the weekly quota runs out), wrap-up never runs. A stamp and a
    # bring-home kept fresh every 15 minutes make whatever Kaggle saves of the output usable as the next input.
    write_stamp("ok", ["training in progress"])

    last_report = 0
    while True:
        for rid, p in runs.items():
            # Out of memory in the first epoch of a run that started from zero here: restart it at its fallback batch,
            # once. Its directory is free (renamed *-INVALID-batch-*) and the other GPU's run carries on.
            if p["proc"].poll() == EXIT_BATCH_CHANGED and p["fallback"] and p["state"] == "fresh" and p["batch"] == p["planned"]:
                p["batch"], p["fallback_from"] = p["fallback"], p["planned"]
                print(f"{rid}: ran out of GPU memory at batch {p['planned']}; restarting it from zero at its fallback "
                      f"batch {p['batch']} on GPU {p['gpu']}, as planned.", flush=True)
                launch(p, stop_by)
        if all(p["proc"].poll() is not None for p in runs.values()):
            break
        if time.time() > kill_at:
            for rid, p in runs.items():
                if p["proc"].poll() is None:
                    print(f"{rid}: still running at the hard limit; killing it so the output is saved.", flush=True)
                    p["proc"].kill()
            break
        if time.time() - last_report > 900:
            last_report = time.time()
            status = "; ".join(f"{rid} {progress(p['name'])}" + ("" if p["proc"].poll() is None else " (exited)") for rid, p in runs.items())
            print(f"[{hours_into_session():.1f} h into session] {status}", flush=True)
            bring_home([])  # records and logs only: copying a last.pt while its run writes it could tear the copy
        time.sleep(60)

    outcomes = {}
    for rid, p in runs.items():
        p["proc"].wait()
        p["log"].close()
        code = p["proc"].returncode
        fell_back = p["state"] == "fresh" and p["fallback_from"] is not None
        if code == EXIT_BATCH_CHANGED and fell_back:
            outcomes[rid] = f"INVALID: GPU ran out of memory at batch {p['planned']}, and again at its fallback batch {p['batch']}. Tell Krishna"
        elif code == EXIT_BATCH_CHANGED and p["fallback"] and p["state"] != "fresh":
            outcomes[rid] = ("INVALID: GPU ran out of memory, and the run had trained in earlier sessions, so it was not "
                             "restarted from zero (that would throw those hours away). Tell Krishna")
        else:
            outcomes[rid] = {0: "FINISHED: record written", EXIT_PAUSED: "PAUSED: resume it next session",
                             EXIT_BATCH_CHANGED: "INVALID: GPU ran out of memory and the batch changed. Tell Krishna"}.get(
                code, f"CRASHED (exit {code}): resume it next session; if it crashes again, send its log to Krishna")
            if fell_back:
                outcomes[rid] += (f" (it ran out of GPU memory at batch {p['planned']} and restarted from zero at its "
                                  f"fallback batch {p['batch']}, as planned)")
        if code not in (0, EXIT_PAUSED):
            tail = (LOGS / f"{rid}-{p['name']}.log").read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
            print(f"\n--- last lines of {rid}'s log ---\n" + "\n".join(tail), flush=True)
    outcomes.update({rid: f"BLOCKED: {why}" for rid, why in blocked.items()})
    bring_home([run_name(cfg["runs"][i]) for i in ids])
    lines = [f"{rid} {run_name(cfg['runs'][rid])}: {outcomes.get(rid, 'already finished earlier: nothing to do')}" for rid in ids]
    (BRING_HOME / "SESSION_RESULT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_stamp("ok", lines)
    banner("SESSION RESULT")
    print("\n".join(f"  {line}" for line in lines))
    print("\n  Next: on your computer, run the wizard. It downloads these results by itself:\n"
          "  bash study/kaggle/wizard.sh next", flush=True)


# ── preflight / wrap-up: the notebook's first and last commands ──────────────


def session_id():
    """This session's id: UTC start time plus a random suffix. Its stamp carries it to the next session."""
    if not SESSION_ID_FILE.exists():
        SESSION_ID_FILE.write_text(time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + os.urandom(2).hex())
    return SESSION_ID_FILE.read_text().strip()


def write_stamp(result, lines):
    """Leave this session's id and outcome in its output (the next session checks its input against it) and bring-home.

    result: "ok" when the output holds this session's work, "stopped" when it stopped before any training, so the
    wizard re-runs the same notebook instead of building on a version that added nothing.
    """
    stamp = json.dumps({"session_id": session_id(), "result": result, "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "lines": lines}, indent=1)
    for d in (ROOT, BRING_HOME):
        d.mkdir(parents=True, exist_ok=True)
        (d / STAMP_NAME).write_text(stamp, encoding="utf-8")


def input_stamps():
    """{session id: stamp} of every session whose output is attached."""
    out = {}
    for f in find_in_inputs(STAMP_NAME):
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
            out[s["session_id"]] = s
        except (OSError, ValueError, KeyError):
            pass
    return out


def task_parser():
    p = argparse.ArgumentParser(prog="train")
    p.add_argument("ids", nargs="+")
    p.add_argument("--new", nargs="+", default=[], metavar="ID", help="runs starting from zero this session (guards against a missing input)")
    p.add_argument("--input-from", default=None, metavar="SESSION", help="the session id the inputs must hold (the wizard sets it)")
    return p


def preflight(task):
    """The notebook's first command, before ~10 minutes of setup: stop at once if the inputs cannot serve the TASK.

    It checks, with file lookups only, that the attached input is the session the wizard just brought home (its stamp)
    and that every run can start: its checkpoint when it continues, the best.pt of the runs it is built on. One
    blocked run does not stop the session: train reports it and trains the other.
    """
    if WORKING.is_dir():
        session_id()
    if not task or task[0] != "train":
        return
    a = task_parser().parse_args(task[1:])
    cfg = load_runs()
    if any(i not in cfg["runs"] for i in a.ids):
        return  # train names the unknown id
    stamps = input_stamps()
    if a.input_from and a.input_from not in stamps:
        held = ", ".join(sorted(stamps)) or "no session's output (nothing attached, or a version older than the stamps)"
        fail(f"The input is not the session the wizard just brought home. It expects session {a.input_from}; the inputs "
             f"hold {held}. In the right panel under Input, update the other notebook to its newest version (or add "
             "it), then Save Version again.")
    has_best = lambda name: bool(find_in_inputs(f"runs/baseline/{name}/weights/best.pt"))  # noqa: E731
    has_last = lambda name: bool(find_in_inputs(f"runs/baseline/{name}/weights/last.pt"))  # noqa: E731
    has_record = lambda name: (RECORD_DIR / f"{name}.json").exists() or bool(find_in_inputs(f"verification/runs/{name}.json"))  # noqa: E731
    blocked, ready = {}, []
    for rid in a.ids:
        r = cfg["runs"][rid]
        if has_record(run_name(r)):
            continue
        why = source_blockers(cfg, r, has_best, has_record)
        if rid not in a.new and not has_last(run_name(r)):
            why.append(f"no checkpoint of it is in the inputs, so it would restart from zero. {MISSING_HINT}")
        if why:
            blocked[rid] = "; ".join(why)
        else:
            ready.append(rid)
    for rid, why in blocked.items():
        print(f"{rid} BLOCKED: {why}", flush=True)
    if blocked and not ready:
        fail("Nothing can start: " + "; ".join(f"{rid}: {why}" for rid, why in blocked.items()))
    print(f"Preflight OK: {', '.join(ready) or 'nothing left to train'} can start" + (f" ({', '.join(blocked)} blocked)" if blocked else "") + ".", flush=True)


def relay():
    """carry_forward without torch, for a session that stops before setup has installed it: the version it leaves
    must still hold everything its inputs held, never a hole in the chain of notebook versions."""
    def epochs(d):
        f = d / "results.csv"
        return len(f.read_text(encoding="utf-8", errors="replace").splitlines()) if f.exists() else 0

    best = {}
    for last in find_in_inputs("runs/baseline/*/weights/last.pt"):
        d = last.parents[1]
        if d.name not in best or epochs(d) > epochs(best[d.name]):
            best[d.name] = d
    for name, d in best.items():
        if not (RUNS_DIR / name).exists():
            shutil.copytree(d, RUNS_DIR / name)
    carry_records()


def wrap_up(task):
    """The notebook's last command, run whatever happened before it. A session that ended without its SESSION_RESULT
    (a STOP, a failed setup, a crash) still leaves a complete output: its inputs' runs, a bring-home, and why."""
    if not task or task[0] != "train" or (BRING_HOME / "SESSION_RESULT.txt").exists():
        return
    try:
        cfg = load_runs()
        ids = [i for i in task_parser().parse_args(task[1:]).ids if i in cfg["runs"]]
    except SystemExit:
        cfg, ids = {"runs": {}}, []
    why = STOP_FILE.read_text(encoding="utf-8").strip() if STOP_FILE.exists() else \
        "it ended without a STOP message (setup failed, or a crash): see the end of the notebook's log"
    trained = LOGS.is_dir() and any(LOGS.glob("*.log"))  # a run was launched: this output holds its progress
    relay()
    bring_home([run_name(cfg["runs"][i]) for i in ids])
    if trained:
        lines = [f"CRASHED during the session: {why}"] + [f"{i} {run_name(cfg['runs'][i])}: resume it next session" for i in ids]
    else:
        lines = [f"STOPPED before training: {why}"] + [f"{i} {run_name(cfg['runs'][i])}: not started this session" for i in ids]
    (BRING_HOME / "SESSION_RESULT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_stamp("ok" if trained else "stopped", lines)
    banner("SESSION RESULT")
    print("\n".join(f"  {line}" for line in lines))
    print("\n  Next: on your computer, run the wizard. It reads this result and tells you what to do:\n"
          "  bash study/kaggle/wizard.sh next", flush=True)


# ── laptop side: used by kaggle/wizard.sh ────────────────────────────────────


def read_stamp(path):
    """'<session id> <result>' of the session stamp in a downloaded output, or 'none none' (older sessions had none)."""
    found = sorted(Path(path).expanduser().rglob(STAMP_NAME)) if Path(path).expanduser().is_dir() else []
    try:
        s = json.loads(found[0].read_text(encoding="utf-8"))
        print(f"{s['session_id']} {s['result']}")
    except (IndexError, OSError, ValueError, KeyError):
        print("none none")


def seed_folder(owner, mirror, out):
    """Rebuild, from this computer's checkpoint mirror, what this person's next runs need (their sources and their own
    started runs), laid out like a session's output, for `wizard.sh repair` to upload as a Kaggle dataset."""
    cfg = load_runs()
    mirror, out = Path(mirror).expanduser(), Path(out)
    done = lambda rid: (RECORD_DIR / f"{run_name(cfg['runs'][rid])}.json").exists()  # noqa: E731
    pending = [rid for rid, r in cfg["runs"].items() if r["owner"] == owner and not done(rid)]
    need = sorted({s for rid in pending for s in sources(cfg["runs"][rid])} | set(pending))
    shutil.rmtree(out, ignore_errors=True)
    total, missing = 0, []
    for rid in need:
        name = run_name(cfg["runs"][rid])
        src = mirror / name
        # A finished run's best.pt is final too, so it can stand in for a missing last.pt.
        last = next((f for f in (src / "last.pt", src / "best.pt" if done(rid) else None) if f and f.exists()), None)
        if last is None:
            if rid not in pending:  # a pending run that never started has nothing to seed
                missing.append(f"{rid} {name}")
            continue
        dest = out / "uav" / "study" / "runs" / "baseline" / name
        (dest / "weights").mkdir(parents=True)
        shutil.copy(last, dest / "weights" / "last.pt")
        if (src / "best.pt").exists():
            shutil.copy(src / "best.pt", dest / "weights" / "best.pt")
        for f in ("results.csv", "args.yaml", "sessions.jsonl"):
            if (src / f).exists():
                shutil.copy(src / f, dest / f)
        size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
        total += size
        print(f"seed       {rid} {name} ({size / 2**20:.0f} MB)")
    if missing:
        fail(f"Not in your checkpoint folder {mirror}: {', '.join(missing)}. Their runs cannot be rebuilt from this "
             "computer: send Krishna this message.")
    if not total:
        fail("Nothing to seed: no run on your list needs a checkpoint from this computer.")
    print(f"seed folder: {out} ({total / 2**20:.0f} MB)")



def import_bring_home(path, mirror):
    """Put a downloaded bring-home folder (or .zip) where it belongs: records into the repo, checkpoints into the mirror."""
    import tempfile
    import zipfile

    src = Path(path).expanduser()
    if src.suffix == ".zip":
        tmp = Path(tempfile.mkdtemp())
        zipfile.ZipFile(src).extractall(tmp)
        src = tmp
    if not src.is_dir():
        fail(f"{path} is neither a folder nor a .zip")
    homes = [d for d in [src, *src.rglob("bring-home")] if d.is_dir() and ((d / "records").is_dir() or (d / "checkpoints").is_dir() or list(d.glob("*.json")))]
    if not homes:
        fail(f"No bring-home contents under {src}. Is this the output of a session that finished?")
    home = homes[0]
    mirror = Path(mirror).expanduser()
    for f in sorted((home / "records").glob("*.json")):
        RECORD_DIR.mkdir(parents=True, exist_ok=True)
        # A session writes a run's record once and never again, so a copy already here is current. The download's
        # may be older: it holds every record GitHub had when the session started, the teammate's included.
        if (RECORD_DIR / f.name).exists():
            continue
        shutil.copy(f, RECORD_DIR / f.name)
        print(f"record     -> study/verification/runs/{f.name}")
    for f in sorted([*home.glob("smoke_results-kaggle-*.json"), *home.glob("kaggle_probe.json")]):
        VERIFICATION.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, VERIFICATION / f.name)
        print(f"record     -> study/verification/{f.name}")
    for d in sorted((home / "checkpoints").glob("*")) if (home / "checkpoints").is_dir() else []:
        shutil.copytree(d, mirror / d.name, dirs_exist_ok=True)
        print(f"checkpoint -> {mirror / d.name}")
    result = home / "SESSION_RESULT.txt"
    if result.exists():
        print("\nLast session:\n" + "".join(f"  {line}\n" for line in result.read_text(encoding="utf-8").splitlines()))


def next_task(owner, mirror):
    """The TASK line for this person's next training session, from runs.yaml order and what is finished."""
    cfg = load_runs()
    if cfg["pairing"] is None:
        fail("runs.yaml has no probe results yet (git pull, or wait for Krishna's probe commit).")
    cap = 2 if cfg["pairing"] == "pair" else 1
    queue = [rid for rid, r in cfg["runs"].items() if r["owner"] == owner]
    if not queue:
        fail(f"No runs for owner {owner!r} in runs.yaml.")
    mirror = Path(mirror).expanduser()
    done = lambda rid: (RECORD_DIR / f"{run_name(cfg['runs'][rid])}.json").exists()  # noqa: E731
    pending = [rid for rid in queue if not done(rid)]
    if not pending:
        print("ALL-DONE")
        return
    # A run built on another run (its pruned start, its teacher) waits until that run is finished and evaluated.
    ready = [rid for rid in pending if all(done(s) for s in sources(cfg["runs"][rid]))]
    started = [rid for rid in ready if (mirror / run_name(cfg["runs"][rid]) / "last.pt").exists()]
    pick = (started + [rid for rid in ready if rid not in started])[:cap]
    if not pick:
        fail(f"Nothing can start yet: {pending} wait for {sorted({s for rid in pending for s in sources(cfg['runs'][rid]) if not done(s)})} to finish.")
    new = [rid for rid in pick if rid not in started]
    print("train " + " ".join(pick) + (" --new " + " ".join(new) if new else ""))


# ── bring-home / status ──────────────────────────────────────────────────────


def bring_home(names):
    """Copy the small files a person must keep off Kaggle into one folder of the output."""
    BRING_HOME.mkdir(parents=True, exist_ok=True)
    for f in [*RECORD_DIR.glob("*.json"), *VERIFICATION.glob("smoke_results-kaggle-*.json"), VERIFICATION / "kaggle_probe.json"]:
        if f.exists():
            dest = BRING_HOME / ("records" if f.parent == RECORD_DIR else "") / f.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dest)
    for name in names:
        src = RUNS_DIR / name
        if not src.exists():
            continue
        dest = BRING_HOME / "checkpoints" / name
        dest.mkdir(parents=True, exist_ok=True)
        for rel in ("weights/last.pt", "weights/best.pt", "results.csv", "args.yaml", "sessions.jsonl"):
            if (src / rel).exists():
                shutil.copy(src / rel, dest / Path(rel).name)
    if LOGS.exists():
        shutil.copytree(LOGS, BRING_HOME / "logs", dirs_exist_ok=True)


def status():
    cfg = load_runs()
    carry_forward()
    print(f"batch_per_gpu {cfg['batch_per_gpu']}, pairing {cfg['pairing']}")
    for rid, r in cfg["runs"].items():
        state, detail = state_of(run_name(r))
        print(f"  {rid:3} {run_name(r):30} {r['owner']:9} {state:10} {detail}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    s = sub.add_parser("smoke")
    s.add_argument("--who", required=True)
    s.add_argument("--probe", action="store_true", help="also run the Kaggle probe (one account only)")
    t = sub.add_parser("train", parents=[task_parser()], add_help=False, conflict_handler="resolve")
    for name in ("preflight", "wrap-up"):  # the notebook's first and last commands; they take the TASK as it is
        sub.add_parser(name).add_argument("task", nargs=argparse.REMAINDER)
    sub.add_parser("read-stamp").add_argument("path")
    sf = sub.add_parser("seed-folder")
    sf.add_argument("--owner", required=True)
    sf.add_argument("--mirror", required=True)
    sf.add_argument("--out", required=True)
    sub.add_parser("status")
    ap = sub.add_parser("apply-probe")
    ap.add_argument("path")
    ap.add_argument("--force", action="store_true")
    ib = sub.add_parser("import-bring-home")
    ib.add_argument("path")
    ib.add_argument("--mirror", required=True, help="local folder that keeps a copy of every checkpoint")
    nt = sub.add_parser("next-task")
    nt.add_argument("--owner", required=True, help="srikr | teammate (the owner column of runs.yaml)")
    nt.add_argument("--mirror", required=True)
    a = p.parse_args()

    if a.cmd == "apply-probe":
        apply_probe(a.path, a.force)
    elif a.cmd == "setup":
        setup()
    elif a.cmd == "smoke":
        smoke(a.who, a.probe)
    elif a.cmd == "train":
        train(a.ids, a.new)
    elif a.cmd == "preflight":
        preflight(a.task)
    elif a.cmd == "wrap-up":
        wrap_up(a.task)
    elif a.cmd == "read-stamp":
        read_stamp(a.path)
    elif a.cmd == "seed-folder":
        seed_folder(a.owner, a.mirror, a.out)
    elif a.cmd == "status":
        status()
    elif a.cmd == "import-bring-home":
        import_bring_home(a.path, a.mirror)
    elif a.cmd == "next-task":
        next_task(a.owner, a.mirror)


if __name__ == "__main__":
    main()
