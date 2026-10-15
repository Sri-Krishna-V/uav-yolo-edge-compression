"""Every Kaggle step of the study, one command each. The notebook (kaggle/train.ipynb) calls these.

  python session.py setup                    # every session, first: install the pinned env, fetch + verify VisDrone
  python session.py smoke --who srikr        # Session 0, once per account (add --probe on the ONE account that probes)
  python session.py train A1 A3 --new A1 A3  # first session of two runs: one per GPU
  python session.py train A1 A3              # any later session: resumes them from the attached input
  python session.py status                   # every run: finished / paused at epoch N / not started
  python session.py apply-probe PROBE.json   # on your own computer, once: write the probe's numbers into runs.yaml
  python session.py import-bring-home PATH --mirror DIR   # on your computer, after each session (the wizard runs these)
  python session.py next-task --owner srikr --mirror DIR  # on your computer: the TASK line for the next session

`train` first copies every checkpoint and record found in the attached inputs into this session (a Kaggle
notebook cannot attach its own output, so each person alternates two notebooks attached to each other).
Then, per run: a run with a record is skipped; a paused run resumes; a trained but unevaluated run is
evaluated. A run with nothing found starts from zero ONLY if listed in --new; otherwise the session stops,
because the likeliest cause is a forgotten input, and restarting would burn hours of quota. Training pauses after the last epoch that ends before
TRAIN_STOP_H hours into the session, so the session always finishes cleanly and its output is saved.

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

SESSION_LIMIT_H = 12.0  # Kaggle's cap on one GPU session
TRAIN_STOP_H = 11.0  # no epoch may end later than this; leaves time for evaluation and saving the output
HARD_KILL_H = 11.6  # anything still running now is killed, so the version still saves its output
CUDA_TAGS = [("cu130", 13.0), ("cu126", 12.6)]  # torch 2.12+ ships only these two; cu130 matches the laptop
EXIT_PAUSED, EXIT_BATCH_CHANGED = 75, 76  # from scripts/studytrain.py


def banner(text):
    print("\n" + "=" * 78 + f"\n  {text}\n" + "=" * 78, flush=True)


def fail(text):
    banner(f"STOP: {text}")
    sys.exit(1)


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
    for rid, body in re.findall(r"^  ([ABP]\d+): \{(.*)\}", text, re.M):
        cfg["runs"][rid] = {k.strip(): value(v.strip()) for k, v in (kv.split(":") for kv in body.split(","))}
    for r in cfg["runs"].values():
        if r.get("prune_from"):  # a pruning-recovery run is named after the run it prunes
            r["name"] = f"{run_name(cfg['runs'][r['prune_from']])}-prune{r['prune']}"
    return cfg


def run_name(r):
    return r.get("name") or f"{Path(r['model']).stem}-{r['imgsz']}-s{r['seed']}"


def make_pruned(cfg, r):
    """Prune the source run's best.pt for a recovery run; the same source always gives the same model."""
    source = RUNS_DIR / run_name(cfg["runs"][r["prune_from"]]) / "weights" / "best.pt"
    if not source.exists():
        fail(f"{run_name(r)} prunes {r['prune_from']}, but {r['prune_from']}'s best.pt is not in the attached inputs. "
             "Attach the notebook output that holds it (Input panel -> Add Input).")
    out = PRUNED_DIR / f"{run_name(r)}.pt"
    if not out.exists():
        if run([sys.executable, SCRIPTS / "studyprune.py", "--source", source, "--ratio", r["prune"] / 100, "--imgsz", r["imgsz"], "--out", out]).returncode:
            fail(f"Pruning {r['prune_from']} for {run_name(r)} failed (see above). Send this log to Krishna.")
    return out


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
    """Files under an attached input's uav/study/ tree: only the output of our own notebook cell is ever read."""
    return sorted(INPUTS.glob(f"**/uav/study/{pattern}")) if INPUTS.is_dir() else []


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
        if cfg["batch_per_gpu"].get(cfg["runs"][i]["imgsz"]) is None:
            fail(f"No batch size for imgsz {cfg['runs'][i]['imgsz']} in runs.yaml (the probe found none that fits). Ask Krishna.")
    if hours_into_session() > TRAIN_STOP_H - 1:
        fail("Less than an hour of this session is left. Start a new session.")
    if set(new) - set(ids):
        fail(f"--new lists {sorted(set(new) - set(ids))}, which are not being trained in this session.")

    carry_forward()
    states = {i: state_of(run_name(cfg["runs"][i])) for i in ids}
    for i, (state, detail) in states.items():
        print(f"{i} {run_name(cfg['runs'][i])}: {detail}", flush=True)
    lost = [i for i in ids if states[i][0] == "fresh" and i not in new]
    if lost:
        fail(f"No checkpoint found for {lost} in the attached inputs, so they would restart from zero. Most likely the "
             "other notebook's output is not attached (Input panel -> Add Input). If they truly have never started, "
             f"put --new {' '.join(lost)} in TASK.")
    started_already = [i for i in new if states[i][0] != "fresh"]
    if started_already:
        fail(f"--new {started_already}, but they already exist ({[states[i][1] for i in started_already]}). Remove them from --new.")

    stop_by = session_start() + TRAIN_STOP_H * 3600
    kill_at = session_start() + HARD_KILL_H * 3600
    LOGS.mkdir(parents=True, exist_ok=True)
    procs = {}
    for gpu, rid in enumerate(ids):
        r = cfg["runs"][rid]
        name = run_name(r)
        state = states[rid][0]
        if state == "done":
            continue
        if procs:
            time.sleep(90)  # let the first run download weights and build the label cache alone
        cmd = [sys.executable, SCRIPTS / "studytrain.py", "--model", r["model"], "--imgsz", r["imgsz"], "--seed", r["seed"],
               "--device", gpu, "--batch-per-gpu", cfg["batch_per_gpu"][r["imgsz"]], "--stop-by", stop_by]
        if r.get("prune_from"):
            cmd[3] = make_pruned(cfg, r)
            cmd += ["--pruned", "--epochs", r["epochs"], "--name", name]
        cmd += {"resume": ["--resume"], "eval-only": ["--eval-only"], "fresh": []}[state]
        log = open(LOGS / f"{rid}-{name}.log", "a", encoding="utf-8")
        print(f"{rid}: GPU {gpu}, log in /kaggle/working/logs/{rid}-{name}.log", flush=True)
        procs[rid] = (name, subprocess.Popen([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT), log)

    last_report = 0
    while any(p.poll() is None for _, p, _ in procs.values()):
        if time.time() > kill_at:
            for rid, (name, p, _) in procs.items():
                if p.poll() is None:
                    print(f"{rid}: still running at the hard limit; killing it so the output is saved.", flush=True)
                    p.kill()
            break
        if time.time() - last_report > 900:
            last_report = time.time()
            status = "; ".join(f"{rid} {progress(name)}" + ("" if p.poll() is None else " (exited)") for rid, (name, p, _) in procs.items())
            print(f"[{hours_into_session():.1f} h into session] {status}", flush=True)
        time.sleep(60)

    outcomes = {}
    for rid, (name, p, log) in procs.items():
        p.wait()
        log.close()
        code = p.returncode
        outcomes[rid] = {0: "FINISHED: record written", EXIT_PAUSED: "PAUSED: resume it next session",
                         EXIT_BATCH_CHANGED: "INVALID: GPU ran out of memory and the batch changed. Tell Krishna"}.get(
            code, f"CRASHED (exit {code}): resume it next session; if it crashes again, send its log to Krishna")
        if code not in (0, EXIT_PAUSED):
            tail = (LOGS / f"{rid}-{name}.log").read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
            print(f"\n--- last lines of {rid}'s log ---\n" + "\n".join(tail), flush=True)
    bring_home([run_name(cfg["runs"][i]) for i in ids])
    lines = [f"{rid} {run_name(cfg['runs'][rid])}: {outcomes.get(rid, 'already finished earlier: nothing to do')}" for rid in ids]
    (BRING_HOME / "SESSION_RESULT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    banner("SESSION RESULT")
    print("\n".join(f"  {line}" for line in lines))
    print("\n  Next: on your computer, run the wizard. It downloads these results by itself:\n"
          "  bash study/kaggle/wizard.sh next", flush=True)


# ── laptop side: used by kaggle/wizard.sh ────────────────────────────────────


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
    pending = [rid for rid in queue if not (RECORD_DIR / f"{run_name(cfg['runs'][rid])}.json").exists()]
    started = [rid for rid in pending if (mirror / run_name(cfg["runs"][rid]) / "last.pt").exists()]
    pick = (started + [rid for rid in pending if rid not in started])[:cap]
    if not pick:
        print("ALL-DONE")
        return
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
        print(f"  {rid:3} {run_name(r):18} {r['owner']:9} {state:10} {detail}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    s = sub.add_parser("smoke")
    s.add_argument("--who", required=True)
    s.add_argument("--probe", action="store_true", help="also run the Kaggle probe (one account only)")
    t = sub.add_parser("train")
    t.add_argument("ids", nargs="+")
    t.add_argument("--new", nargs="+", default=[], metavar="ID", help="runs starting from zero this session (guards against a missing input)")
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
    elif a.cmd == "status":
        status()
    elif a.cmd == "import-bring-home":
        import_bring_home(a.path, a.mirror)
    elif a.cmd == "next-task":
        next_task(a.owner, a.mirror)


if __name__ == "__main__":
    main()
