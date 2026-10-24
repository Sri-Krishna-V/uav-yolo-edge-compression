"""Exercise session.py's preflight / train-blocking / wrap-up / seed / stamp paths on a fake Kaggle tree."""
import io
import json
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "study" / "kaggle"))
import session as s  # noqa: E402

T = Path(tempfile.mkdtemp())
s.WORKING = T / "working"
s.INPUTS = T / "input"
s.ROOT = s.WORKING / "uav" / "study"
s.RUNS_DIR = s.ROOT / "runs" / "baseline"
s.RECORD_DIR = s.ROOT / "verification" / "runs"
s.VERIFICATION = s.ROOT / "verification"
s.BRING_HOME = s.WORKING / "bring-home"
s.LOGS = s.WORKING / "logs"
s.STOP_FILE = T / "stop.txt"
s.SESSION_ID_FILE = T / "sid"
s.SEED_DIR = T / "seed-unpacked"
s.WORKING.mkdir(parents=True)
cfg = s.load_runs()
# The records the checks below assume: the teammate's baselines finished; T2, and every run built on another run, not
# yet. Fixed here, not copied from the clone, whose records move on as runs finish (T2's landed on 23 October 2026).
s.RECORD_DIR.mkdir(parents=True)
for rid in ("A2", "A5", "A6", "B2", "B4"):
    (s.RECORD_DIR / f"{s.run_name(cfg['runs'][rid])}.json").write_text("{}")
MAKE_PRUNED, RUN, LAUNCH = s.make_pruned, s.run, s.launch


def fake_run(root, name, best=True, rows=101):
    d = root / "uav" / "study" / "runs" / "baseline" / name
    (d / "weights").mkdir(parents=True, exist_ok=True)
    (d / "weights" / "last.pt").write_bytes(b"x")
    if best:
        (d / "weights" / "best.pt").write_bytes(b"x")
    (d / "results.csv").write_text("h\n" + "r\n" * rows)


def stamp(root, sid, result="ok"):
    p = root / "uav" / "study" / s.STAMP_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"session_id": sid, "result": result}))


def call(fn, *args):
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            fn(*args)
        rc = 0
    except SystemExit as e:
        rc = e.code
    return rc, buf.getvalue()


def reset():
    for p in (s.INPUTS, s.BRING_HOME, s.LOGS, s.RUNS_DIR, s.SEED_DIR):
        shutil.rmtree(p, ignore_errors=True)
    for f in (s.STOP_FILE, s.SESSION_ID_FILE, s.ROOT / s.STAMP_NAME):
        f.unlink(missing_ok=True)


results = []


def check(label, cond, detail=""):
    results.append((label, bool(cond)))
    print(("PASS " if cond else "FAIL ") + label + ("" if cond else f"\n     {detail[-600:]}"))


# 1. Vidhyarth's case: the input lacks B2's best.pt. T2 must still start; P4 is blocked, not the session.
reset()
nb_b = s.INPUTS / "uav-vidhyarth-b"
fake_run(nb_b, "yolo26s-640-s0")  # A2
stamp(nb_b, "S-old")
rc, out = call(s.preflight, ["train", "T2", "P4", "--new", "T2", "P4"])
check("1 preflight: T2 ready, P4 blocked, session goes on", rc == 0 and "P4 BLOCKED" in out and "T2" in out.split("Preflight OK:")[1], out)

# 2. A stale input: the wizard expects session S-new, the input holds S-old -> stop in preflight, reason recorded.
rc, out = call(s.preflight, ["train", "T2", "P4", "--new", "T2", "P4", "--input-from", "S-new"])
check("2 preflight: stale input stops the session", rc == 1 and "S-new" in s.STOP_FILE.read_text(), out)

# 3. wrap-up after that stop: STOPPED result, stamp 'stopped', the input's runs relayed into the output.
rc, out = call(s.wrap_up, ["train", "T2", "P4", "--new", "T2", "P4", "--input-from", "S-new"])
res = (s.BRING_HOME / "SESSION_RESULT.txt").read_text()
st = json.loads((s.ROOT / s.STAMP_NAME).read_text())
check("3 wrap-up: STOPPED result + stopped stamp + relay", rc == 0 and res.startswith("STOPPED before training") and st["result"] == "stopped"
      and (s.RUNS_DIR / "yolo26s-640-s0" / "weights" / "best.pt").exists() and (s.BRING_HOME / s.STAMP_NAME).exists(), res + str(st))

# 4. The right input (stamp matches) and B2 present -> both ready.
reset()
fake_run(nb_b, "yolo26s-640-s0")
fake_run(nb_b, "yolo26s-960-s0")
stamp(nb_b, "S-new")
rc, out = call(s.preflight, ["train", "T2", "P4", "--new", "T2", "P4", "--input-from", "S-new"])
check("4 preflight: matching stamp, both ready", rc == 0 and "BLOCKED" not in out and "T2, P4" in out, out)

# 5. A run to continue whose checkpoint is missing, alone in the TASK -> nothing can start.
reset()
stamp(nb_b, "S-new")
rc, out = call(s.preflight, ["train", "T2"])
check("5 preflight: continuing run without checkpoint stops", rc == 1 and "restart from zero" in s.STOP_FILE.read_text(), out)

# 6. D2 before T2 has a record -> blocked as 'has not finished'.
reset()
fake_run(nb_b, "yolo26m-640-s0")  # a teacher checkpoint but no record
rc, out = call(s.preflight, ["train", "D2", "--new", "D2"])
check("6 preflight: teacher without a record blocks D2", rc == 1 and "has not finished" in s.STOP_FILE.read_text(), out)

# 7. wrap-up is a no-op when train already wrote SESSION_RESULT; and it reports CRASHED when a run had launched.
reset()
s.BRING_HOME.mkdir(parents=True)
(s.BRING_HOME / "SESSION_RESULT.txt").write_text("T2 x: PAUSED\n")
rc, out = call(s.wrap_up, ["train", "T2"])
check("7a wrap-up leaves a normal result alone", (s.BRING_HOME / "SESSION_RESULT.txt").read_text() == "T2 x: PAUSED\n", out)
reset()
s.LOGS.mkdir(parents=True)
(s.LOGS / "T2-yolo26m-640-s0.log").write_text("Traceback ...")
fake_run(T / "fakeworking", "x")
rc, out = call(s.wrap_up, ["train", "T2", "--new", "T2"])
st = json.loads((s.ROOT / s.STAMP_NAME).read_text())
check("7b wrap-up after a crash mid-training: CRASHED, stamp ok", (s.BRING_HOME / "SESSION_RESULT.txt").read_text().startswith("CRASHED") and st["result"] == "ok", out)

# 8. read-stamp on a downloaded bring-home, and on an old one without a stamp.
rc, out = call(s.read_stamp, str(s.BRING_HOME))
check("8a read-stamp", out.strip().endswith(" ok") and out.split()[0] == st["session_id"], out)
rc, out = call(s.read_stamp, str(T / "nothing"))
check("8b read-stamp without a stamp", out.strip() == "none none", out)

# 9. train's own per-run blocking (after carry_forward), without launching anything: P4 blocked, T2 planned.
reset()
fake_run(nb_b, "yolo26s-640-s0")
stamp(nb_b, "S-new")
s.carry_forward = lambda: s.relay()  # no torch here: relay is carry_forward's stdlib twin
s.checkpoint_epoch = lambda p: -1
launched = []
s.launch = lambda p, stop_by: launched.append(p["name"]) or setattr_proc(p)


class Done:
    returncode = 0

    def poll(self):
        return 0

    def wait(self):
        return 0


def setattr_proc(p):
    p["proc"] = Done()


s.make_pruned = lambda cfg, r: (_ for _ in ()).throw(s.Blocked("B2's best.pt is missing")) if r["prune_from"] == "B2" else "x.pt"
s.time.sleep = lambda n: None
s.SESSION_START_FILE = T / "start"
rc, out = call(s.train, ["T2", "P4"], ["T2", "P4"])
res = (s.BRING_HOME / "SESSION_RESULT.txt").read_text()
check("9 train: T2 launched, P4 BLOCKED in the result", rc == 0 and launched == ["yolo26m-640-s0"] and "P4 yolo26s-960-s0-prune30: BLOCKED" in res, out + res)

# 10. seed-folder from a fake mirror: sources (A2, B2) and a started run (T2 paused) are seeded; never-started runs skipped.
mirror = T / "mirror"
for name, files in {"yolo26s-640-s0": ["last.pt", "best.pt", "results.csv"], "yolo26s-960-s0": ["best.pt"], "yolo26m-640-s0": ["last.pt", "results.csv"]}.items():
    (mirror / name).mkdir(parents=True)
    for f in files:
        (mirror / name / f).write_bytes(b"x" * 1024)
out_dir = T / "seed"
rc, out = call(s.seed_folder, "teammate", str(mirror), str(out_dir))
base = out_dir / "uav" / "study" / "runs" / "baseline"
check("10 seed-folder: A2, B2 (best.pt as last.pt), paused T2", rc == 0 and (base / "yolo26s-960-s0" / "weights" / "last.pt").exists()
      and (base / "yolo26m-640-s0" / "weights" / "last.pt").exists() and (base / "yolo26s-640-s0" / "weights" / "best.pt").exists(), out)

# 11. A zipped seed dataset (uav.zip) next to the notebook input supplies B2's best.pt: P4 is no longer blocked.
# The zip is built the way `kaggle datasets create --dir-mode zip` builds it (kaggle 2.2.4, DirectoryArchive):
# shutil.make_archive with the folder itself as root, so its entries start at study/, not at uav/.
reset()
fake_run(nb_b, "yolo26s-640-s0")
stamp(nb_b, "S-new")
src = T / "seedsrc"
fake_run(src, "yolo26s-960-s0")
(s.INPUTS / "uav-vidhyarth-seed").mkdir(parents=True)
shutil.make_archive(str(s.INPUTS / "uav-vidhyarth-seed" / "uav"), "zip", src / "uav")
rc, out = call(s.preflight, ["train", "T2", "P4", "--new", "T2", "P4", "--input-from", "S-new"])
check("11 zipped seed dataset unblocks P4", rc == 0 and "BLOCKED" not in out and "unpacked the seed" in out, out)

# 11b. The layout Kaggle really gave the teammate's notebook (24 October 2026): inputs sorted into notebooks/<owner>/ and
# datasets/<owner>/, and the seed archive unpacked into the dataset's root, so the name uav is gone: B2's best.pt is
# at <seed dataset>/study/runs/..., not under any uav/. P4 can start.
reset()
nb_kaggle = s.INPUTS / "notebooks" / "vidhyarthse" / "uav-vidhyarth-b"
fake_run(nb_kaggle, "yolo26s-640-s0")
stamp(nb_kaggle, "S-new")
shutil.copytree(src / "uav", s.INPUTS / "datasets" / "vidhyarthse" / "uav-vidhyarth-seed")
rc, out = call(s.preflight, ["train", "T2", "P4", "--new", "T2", "P4", "--input-from", "S-new"])
check("11b seed dataset unpacked by Kaggle (no uav/ level) unblocks P4", rc == 0 and "BLOCKED" not in out and "T2, P4" in out, out)

# 11c. The same tree at the root of any other input (not uav/, not a uav-*-seed dataset) is never read: P4 stays blocked.
shutil.rmtree(s.INPUTS / "datasets")
shutil.copytree(src / "uav", s.INPUTS / "datasets" / "someone" / "yolo-weights")
rc, out = call(s.preflight, ["train", "T2", "P4", "--new", "T2", "P4", "--input-from", "S-new"])
check("11c a study/ tree at another input's root is not read", rc == 0 and "P4 BLOCKED" in out, out)

# 12. train, not preflight, is the gate: a run must not be pruned from a source that has no record, even when the
# source's best.pt is there (a paused run has one). B2 paused after epoch 6 and no record: P4 BLOCKED, T2 trains.
reset()
fake_run(nb_b, "yolo26s-960-s0")
stamp(nb_b, "S-new")
b2_record = s.RECORD_DIR / "yolo26s-960-s0.json"
b2_saved = b2_record.read_bytes()
b2_record.unlink()
s.make_pruned = MAKE_PRUNED
s.run = lambda cmd, **kw: Path(cmd[cmd.index("--out") + 1]).parent.mkdir(parents=True, exist_ok=True) or Done()  # studyprune
s.checkpoint_epoch = lambda p: 5 if "960-s0" in str(p) else -1
launched.clear()
rc, out = call(s.train, ["T2", "P4"], ["T2", "P4"])
res = (s.BRING_HOME / "SESSION_RESULT.txt").read_text()
check("12 train: P4 blocked while B2 (paused) has no record", rc == 0 and launched == ["yolo26m-640-s0"]
      and "P4 yolo26s-960-s0-prune30: BLOCKED" in res and "has not finished" in res, out + res)
b2_record.write_bytes(b2_saved)
s.make_pruned, s.run, s.checkpoint_epoch = (lambda cfg, r: "x.pt"), RUN, (lambda p: -1)

# 13-16. train's launches through the real launch(), with only subprocess.Popen faked: each run's studytrain exits
# with the next code in its script. A launch records its command line, its environment, and the stamp it found.
launches = []


class FakePopen:
    script = {}

    def __init__(self, cmd, stdout=None, stderr=None, env=None):
        args = dict(zip(cmd, cmd[1:]))
        codes = FakePopen.script[args["--name"]]
        self.name, self.code = args["--name"], codes.pop(0) if codes else 0  # an unplanned launch fails its check
        stamp_file = s.ROOT / s.STAMP_NAME
        launches.append({"name": self.name, "device": args["--device"], "batch": args["--batch-per-gpu"],
                         "fallback_from": args.get("--fallback-from"), "resume": "--resume" in cmd,
                         "alloc": (env or {}).get("PYTORCH_CUDA_ALLOC_CONF"), "stamp": stamp_file.read_text() if stamp_file.exists() else ""})
        self.returncode = None

    def poll(self):
        self.returncode = self.code
        return self.code

    def wait(self):
        return self.poll()

    def kill(self):
        pass


def train_with(ids, new, script):
    launches.clear()
    FakePopen.script = script
    rc, out = call(s.train, ids, new)
    res = (s.BRING_HOME / "SESSION_RESULT.txt").read_text() if (s.BRING_HOME / "SESSION_RESULT.txt").exists() else ""
    return rc, out + res, res


s.launch, s.subprocess.Popen = LAUNCH, FakePopen
T2, P4 = "yolo26m-640-s0", "yolo26s-960-s0-prune30"

# 13. T2 (fresh) runs out of memory in its first epoch: restarted once, from zero, at batch 8 on its own GPU 0, the
# record told (--fallback-from 12). P4 on GPU 1 is launched once and untouched. The provisional stamp is written
# once both runs are launched.
reset()
stamp(nb_b, "S-new")
rc, out, res = train_with(["T2", "P4"], ["T2", "P4"], {T2: [s.EXIT_BATCH_CHANGED, 0], P4: [s.EXIT_PAUSED]})
t2 = [x for x in launches if x["name"] == T2]
p4 = [x for x in launches if x["name"] == P4]
check("13a train: T2 restarted at its fallback batch, from zero, on GPU 0",
      [(x["device"], x["batch"], x["fallback_from"], x["resume"]) for x in t2] == [("0", "12", None, False), ("0", "8", "12", False)], out + str(launches))
check("13b train: expandable segments for T2 only; P4 launched once on GPU 1 at the 960 batch",
      all(x["alloc"] == "expandable_segments:True" for x in t2) and [(x["device"], x["batch"], x["alloc"]) for x in p4] == [("1", "8", None)], str(launches))
check("13c train: result says T2 finished at its fallback batch, P4 paused",
      f"T2 {T2}: FINISHED: record written (it ran out of GPU memory at batch 12 and restarted from zero at its fallback batch 8" in res
      and f"P4 {P4}: PAUSED" in res, out)
check("13d train: the relaunch saw the provisional 'training in progress' stamp", "training in progress" in t2[1]["stamp"], str(launches))

# 14. The fallback batch runs out of memory too: no third launch, INVALID.
reset()
stamp(nb_b, "S-new")
rc, out, res = train_with(["T2"], ["T2"], {T2: [s.EXIT_BATCH_CHANGED, s.EXIT_BATCH_CHANGED]})
check("14 train: out of memory at the fallback too -> INVALID, two launches only",
      len(launches) == 2 and f"T2 {T2}: INVALID: GPU ran out of memory at batch 12, and again at its fallback batch 8" in res, out)

# 15. A run that trained in an earlier session is never restarted from zero: resumed at 12, exit 76 -> INVALID, once.
reset()
fake_run(nb_b, T2)
(nb_b / "uav" / "study" / "runs" / "baseline" / T2 / "args.yaml").write_text("batch: 12\n")
stamp(nb_b, "S-new")
s.checkpoint_epoch = lambda p: 40 if T2 in str(p) else -1
rc, out, res = train_with(["T2"], [], {T2: [s.EXIT_BATCH_CHANGED]})
check("15 train: a resumed run that runs out of memory is not restarted",
      [(x["batch"], x["resume"]) for x in launches] == [("12", True)] and "had trained in earlier sessions" in res, out)

# 16. A run that fell back in an earlier session resumes at the batch it started with (8), its record still told.
reset()
fake_run(nb_b, T2)
(nb_b / "uav" / "study" / "runs" / "baseline" / T2 / "args.yaml").write_text("batch: 8\n")
stamp(nb_b, "S-new")
rc, out, res = train_with(["T2"], [], {T2: [s.EXIT_PAUSED]})
check("16 train: resumes at its started batch 8 with --fallback-from 12",
      [(x["batch"], x["fallback_from"], x["resume"]) for x in launches] == [("8", "12", True)] and f"T2 {T2}: PAUSED" in res, out)
s.checkpoint_epoch = lambda p: -1

print(f"\n{sum(ok for _, ok in results)}/{len(results)} passed")
shutil.rmtree(T, ignore_errors=True)
sys.exit(0 if all(ok for _, ok in results) else 1)
