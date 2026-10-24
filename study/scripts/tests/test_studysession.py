"""Check the platform session runner's rules with a fake clock, fake temperatures and a fake benchmark. Runs on any machine:

  python study/scripts/tests/test_studysession.py
"""
import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import studysession as ss  # noqa: E402

FAILED = []


def check(label, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        FAILED.append(label)
        if detail:
            print(f"     {detail}")


def bench_meta(ms=100.0, throttled=False, e2e_deploy=None, jetson=None):
    """A complete studybench meta.json, cut down to the keys the runner reads."""
    timing = {"t_total_ms": {"median": ms, "p95": ms * 1.2}}
    if e2e_deploy is not None:
        timing["e2e_deploy"] = {"median": e2e_deploy, "p95": e2e_deploy * 1.2}
    flags = {"under_voltage": False, "freq_capped": throttled, "throttled": throttled, "soft_temp_limit": False}
    return {"status": "complete", "timing_ms": timing, "jetson": jetson,
            "throttle": {"start": "0x0", "end": "0x0", "or_of_samples": "0x6" if throttled else "0x0",
                         "during_run": flags, "since_boot": flags}}


GOOD_PI = {"governors": ["performance"] * 4, "desktop": [], "update_units": {"apt-daily.timer": "inactive", "apt-daily-upgrade.timer": "inactive"},
           "packages": {"numpy": "2.4.6", "opencv-python-headless": "5.0.0.93", "onnxruntime": "1.31.0", "openvino": "2026.4.1",
                        "ncnn": "1.0.20260526"},
           "python": "3.11.2", "os": "Debian GNU/Linux 12 (bookworm)", "pmic_w": 2.9, "arm_mhz": 2400}


class FakeEnv:
    """The platform as the runner sees it. Time moves only when the runner sleeps or a benchmark runs.

    script: {unit name: [outcome of the 1st call, of the 2nd call, ...]}; an outcome is a dict:
      {"ms": 103.0}  a complete run with that median latency (the default is {"ms": 100.0})
      {"crash": True}  the benchmark exits 1
      {"throttled": True}  a complete run that throttled
      {"interrupt": True}  Ctrl-C while the benchmark runs
      {"meta": {...}}  this exact meta.json
    An energy run (a spec with "seconds") loads for load_s, then runs `seconds` at `fps` frames per second; its outcome
    may also set {"fps": 4.0}, or {"no_power": True} (the sampler read nothing).

    The fake sampler: power(t, phase) gives the rail sum in W at time t, phase "idle", "load" (model load and warm-up)
    or "run"; it is read every SAMPLE_S, during the idle window and through the whole of a sampled bench.
    """

    SAMPLE_S = 0.5

    def __init__(self, temps=lambda t: 40.0, script=None, bench_s=60.0, battery=None, board=None, facts=None,
                 power=lambda t, phase: 3.0, fps=5.0, load_s=10.0):
        self.t, self.temps, self.script, self.bench_s = 0.0, temps, script or {}, bench_s
        self.battery, self.board, self.facts = battery, board, GOOD_PI if facts is None else facts  # a Pi set up as the protocol asks
        self.power, self.fps, self.load_s = power, fps, load_s
        self.calls = []  # (unit name, attempt directory, clock at the call, CPU temperature at the call)
        self.idles = []  # (clock at the start of an idle window, its length)

    def readings(self, t0, t1, phase, end=False):
        n = round((t1 - t0) / self.SAMPLE_S) + (1 if end else 0)
        return [[t0 + k * self.SAMPLE_S, self.power(t0 + k * self.SAMPLE_S, phase(t0 + k * self.SAMPLE_S) if callable(phase) else phase)]
                for k in range(n)]

    def idle_window(self, seconds):
        self.idles.append((self.t, seconds))
        samples = self.readings(self.t, self.t + seconds, "idle")
        self.t += seconds
        return samples

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def cpu_temp_c(self):
        return self.temps(self.t)

    def battery_temp_c(self):
        return self.battery

    def board_state(self):
        return self.board

    def setup_facts(self):
        return self.facts

    def bench(self, spec, out_dir):
        name = spec["name"]
        k = sum(1 for c in self.calls if c[0] == name)
        self.calls.append((name, Path(out_dir), self.t, self.cpu_temp_c(), spec))
        todo = self.script.get(name, [])
        outcome = todo[k] if k < len(todo) else {}
        if spec.get("seconds"):
            return self.energy_bench(spec, outcome)
        self.t += self.bench_s
        if outcome.get("interrupt"):
            raise KeyboardInterrupt
        if outcome.get("crash"):
            return {"exit": 1, "meta": {"status": "interrupted", "error": "RuntimeError: boom"}, "tail": "RuntimeError: boom"}
        meta = outcome.get("meta") or bench_meta(outcome.get("ms", 100.0), outcome.get("throttled", False))
        return {"exit": 0, "meta": meta, "tail": "done"}

    def energy_bench(self, spec, outcome):
        t0, fps = self.t, outcome.get("fps", self.fps)
        w0, w1 = t0 + self.load_s, t0 + self.load_s + spec["seconds"]
        self.t = w1
        if outcome.get("interrupt"):
            raise KeyboardInterrupt
        if outcome.get("crash"):
            return {"exit": 1, "meta": {"status": "interrupted", "error": "RuntimeError: boom"}, "tail": "RuntimeError: boom", "power": []}
        meta = bench_meta(0.0, outcome.get("throttled", False))
        del meta["timing_ms"]
        meta.update(mode="energy", precision="FP32", energy={"frames": round(fps * spec["seconds"]), "window": [w0, w1]})
        power = [] if outcome.get("no_power") or not spec["sample_power"] else \
            self.readings(t0, w1, lambda t: "load" if t < w0 else "run", end=True)
        return {"exit": 0, "meta": meta, "tail": "done", "power": power}


A = "exports/yolo11s-640-s0/640/onnxruntime-int8"
B = "exports/yolo11s-640-s0/640/openvino-fp32"


def plan(*artifacts, platform="laptop", **kw):
    return {"platform": platform, "cells": [{"artifact": a, "runtime": a.rsplit("/", 1)[1].split("-")[0], "imgsz": 640} for a in artifacts], **kw}


def name(artifact):
    return artifact.split("exports/", 1)[1]


def summary(out):
    with open(out / "summary.csv", newline="", encoding="utf-8") as f:
        return {r["cell"]: r for r in csv.DictReader(f)}


def run(p, env, **kw):
    out = Path(tempfile.mkdtemp())
    return out, ss.run_session(p, out, env, **kw)


# ---------------------------------------------------------------- 1: one pass over the plan
env = FakeEnv()
out, state = run(plan(A, B), env)
called = [c[0] for c in env.calls]
check("1 the reference cell runs first and last, every cell once in between",
      called[0] == "drift-start" and called[-1] == "drift-end" and sorted(called[1:-1]) == sorted([name(A), name(B)]), called)
s = summary(out)
check("1 one summary row per cell and per reference run", set(s) == {name(A), name(B), "drift-start", "drift-end"}, list(s))
check("1 a clean cell: one attempt, outcome ok, its median latency", s[name(A)]["attempts"] == "1" and s[name(A)]["outcome"] == "ok"
      and float(s[name(A)]["latency_ms"]) == 100.0, s[name(A)])
check("1 the session is complete", state["status"] == "complete" and json.loads((out / "session.json").read_text())["status"] == "complete")
spec = env.calls[0][4]
check("1 the reference cell: YOLO11s baseline, ONNX Runtime FP32, 640, first 200 images",
      spec["artifact"] == "exports/yolo11s-640-s0/640/onnxruntime-fp32" and spec["runtime"] == "onnxruntime" and spec["imgsz"] == 640
      and spec["limit"] == 200, spec)
check("1 every run: 4 threads and 20 warm-up images", all(c[4]["threads"] == 4 and c[4]["warmup"] == 20 for c in env.calls))
log = (out / "session.log").read_text(encoding="utf-8")
check("1 the log states the threads and the warm-up", "4 threads" in log and "20 warm-up" in log, log[:400])

# ---------------------------------------------------------------- 2: the cell order is shuffled once, with a fixed seed
SIX = {"platform": "laptop", "cells": [{"name": f"c{i}", "artifact": A, "runtime": "onnxruntime", "imgsz": 640} for i in range(6)]}
env = FakeEnv()
out, state = run(SIX, env)
ran = [c[0] for c in env.calls][1:-1]
check("2 seed 0 (the default): the order Python gives (checked on 3.11 and 3.12)", ran == ["c4", "c2", "c1", "c0", "c5", "c3"], ran)
check("2 the order and its seed are in session.json", state["order"] == ran and state["seed"] == 0, (state.get("order"), state.get("seed")))
log = (out / "session.log").read_text(encoding="utf-8")
check("2 the order and its seed are in the log", "seed 0" in log and "c4, c2, c1, c0, c5, c3" in log, log[:600])
env = FakeEnv()
run({**SIX, "seed": 1}, env)
check("2 another seed gives another order", [c[0] for c in env.calls][1:-1] == ["c2", "c3", "c5", "c0", "c4", "c1"])


# ---------------------------------------------------------------- 3: the start gate
def gates(state, unit):
    return [a["gate"] for a in state["units"][unit]["attempts"]]


env = FakeEnv(temps=lambda t: 40.0)
out, state = run(plan(A), env)
check("3 a cool CPU: every run starts at once", [c[2] for c in env.calls] == [0, 60, 120], [c[2] for c in env.calls])
g = gates(state, name(A))[0]
check("3 the gate records the CPU temperature and why it opened", g["waited_s"] == 0 and g["cpu_temp_c"] == 40.0 and "below 50" in g["opened"], g)

env = FakeEnv(temps=lambda t: 60.0 if t < 70 else 45.0)
out, state = run(plan(A), env)
check("3 a hot CPU: the first run starts when it falls below 50 C", env.calls[0][2] == 70 and env.calls[0][3] == 45.0, env.calls[0][2:4])
check("3 ... and the gate records the wait", gates(state, "drift-start")[0]["waited_s"] == 70, gates(state, "drift-start"))

env = FakeEnv(temps=lambda t: 50.0)
out, state = run(plan(A), env)
check("3 a CPU that stays at 50 C or above: each run starts after 3 minutes", [c[2] for c in env.calls] == [180, 420, 660],
      [c[2] for c in env.calls])
g = gates(state, name(A))[0]
check("3 ... and the gate says it opened on time, at 50 C", g["waited_s"] == 180 and g["cpu_temp_c"] == 50.0 and "3 min" in g["opened"], g)

env = FakeEnv(temps=lambda t: None)
out, state = run(plan(A), env)
g = gates(state, name(A))[0]
check("3 no CPU temperature: each run waits the 3 minutes", env.calls[0][2] == 180 and g["waited_s"] == 180
      and g["cpu_temp_c"] is None and "no CPU temperature" in g["opened"], g)

env = FakeEnv(temps=lambda t: 30.0, battery=31.5)
out, state = run(plan(A, platform="m34"), env)
g = gates(state, name(A))[0]
check("3 the phone waits a fixed 3 minutes, even when cool", [c[2] for c in env.calls] == [180, 420, 660] and "fixed" in g["opened"], g)
check("3 the phone logs its battery temperature", g["battery_temp_c"] == 31.5
      and "battery 31.5 C" in (out / "session.log").read_text(encoding="utf-8"), g)

BOARD = {"nvpmodel_mode": [2, "MAXN_SUPER"], "highest_mode": [2, "MAXN_SUPER"], "jetson_clocks": True,
         "gpu_clocks": {"cur": 1020, "min": 1020, "max": 1020}}
env = FakeEnv(temps=lambda t: 60.0 if t < 30 else 45.0, board=BOARD)
out, state = run(plan(A, platform="jetson"), env)
g = gates(state, "drift-start")[0]
check("3 the Jetson: the same 50 C gate", env.calls[0][2] == 30 and g["waited_s"] == 30, g)
check("3 the Jetson: each gate records the power mode and clocks", all(x["board"] == BOARD for u in state["units"] for x in gates(state, u)), g)

env = FakeEnv(temps=lambda t: 60.0)
out, state = run(plan(A), env, dry_run=True)
check("3 a dry run: the gate waits 5 s at most", [c[2] for c in env.calls] == [5, 70, 135], [c[2] for c in env.calls])
check("3 a dry run says so in session.json and the log", state["dry_run"] is True
      and "DRY RUN" in (out / "session.log").read_text(encoding="utf-8"))

# ---------------------------------------------------------------- 4: a crashed cell runs again from the start, once
env = FakeEnv(script={name(A): [{"crash": True}]})
out, state = run(plan(A, B), env)
calls = [c for c in env.calls if c[0] == name(A)]
check("4 a crash: the cell runs again at once, in a fresh directory", len(calls) == 2 and calls[0][1] != calls[1][1]
      and [c[0] for c in env.calls].index(name(A)) + 1 == [c[0] for c in env.calls].index(name(A), [c[0] for c in env.calls].index(name(A)) + 1),
      [c[0] for c in env.calls])
r = summary(out)[name(A)]
check("4 ... its summary: 2 attempts, outcome ok, the second attempt's latency and directory", r["attempts"] == "2" and r["outcome"] == "ok"
      and r["latency_ms"] == "100.0" and r["run_dir"].endswith("attempt-2") and "a1 crashed" in r["attempt_log"], r)
check("4 ... the crash's error is kept", "boom" in state["units"][name(A)]["attempts"][0]["error"], state["units"][name(A)]["attempts"][0])

env = FakeEnv(script={name(A): [{"crash": True}, {"crash": True}, {}]})
out, state = run(plan(A, B), env)
r = summary(out)[name(A)]
check("4 a second crash: the cell is failed and not run a third time", r["outcome"] == "failed" and r["attempts"] == "2"
      and r["latency_ms"] == "" and sum(c[0] == name(A) for c in env.calls) == 2, r)
check("4 ... and the session goes on", summary(out)[name(B)]["outcome"] == "ok" and state["status"] == "complete")

env = FakeEnv(script={"drift-start": [{"crash": True}]})
out, state = run(plan(A), env)
check("4 the reference run follows the same crash rule", summary(out)["drift-start"]["attempts"] == "2"
      and summary(out)["drift-start"]["outcome"] == "ok")

# ---------------------------------------------------------------- 5: a throttled cell runs again at the end of the session
C = "exports/yolo11s-640-s0/640/ncnn-fp32"
env = FakeEnv(script={name(A): [{"throttled": True, "ms": 130.0}, {"ms": 101.0}]})
out, state = run(plan(A, B, C), env)
order = [c[0] for c in env.calls]
check("5 a throttled cell runs again after every other cell, before the closing reference run",
      order[:-2].count(name(A)) == 1 and order[-2:] == [name(A), "drift-end"] and set(order[1:4]) == {name(A), name(B), name(C)}, order)
r = summary(out)[name(A)]
check("5 ... both attempts are kept; the clean one is the cell's latency", r["attempts"] == "2" and r["outcome"] == "ok"
      and r["latency_ms"] == "101.0" and r["throttle"] == "none" and r["run_dir"].endswith("attempt-2")
      and "a1 complete throttled" in r["attempt_log"], r)
check("5 ... the re-run waited at the start gate", "gate" in state["units"][name(A)]["attempts"][1])

env = FakeEnv(script={name(A): [{"throttled": True, "ms": 130.0}, {"throttled": True, "ms": 125.0}, {}]})
out, state = run(plan(A, B), env)
r = summary(out)[name(A)]
check("5 throttled again: outcome throttled, no third run", r["outcome"] == "throttled" and r["attempts"] == "2"
      and r["latency_ms"] == "125.0" and sum(c[0] == name(A) for c in env.calls) == 2, r)

env = FakeEnv(script={name(A): [{"throttled": True, "ms": 130.0}, {"crash": True}, {"crash": True}]})
out, state = run(plan(A), env)
r = summary(out)[name(A)]
check("5 a re-run that crashes twice leaves the throttled attempt", r["outcome"] == "throttled" and r["attempts"] == "3"
      and r["latency_ms"] == "130.0" and state["status"] == "complete", r)

quiet = bench_meta(100.0)
quiet["throttle"] = None
env = FakeEnv(script={name(A): [{"meta": quiet}]})
out, state = run(plan(A), env)
r = summary(out)[name(A)]
check("5 no throttle signal on this platform: unknown, and no re-run", r["throttle"] == "unknown" and r["attempts"] == "1", r)

sticky = bench_meta(100.0)  # no sample saw a throttle, but bit 18 (throttled since boot) was set between the first and last
sticky["throttle"].update(start="0x10000", end="0x50000")
env = FakeEnv(script={name(A): [{"meta": sticky}]})
out, state = run(plan(A, platform="pi5"), env)
check("5 the Pi: a since-boot throttle bit that appeared during the run counts", summary(out)[name(A)]["attempts"] == "2",
      summary(out)[name(A)])

for label, jet in (("an over-current counter rose", {"over_current_throttled": True, "images_gpu_below_max": 0}),
                   ("the GPU clock fell below its pinned maximum", {"over_current_throttled": False, "images_gpu_below_max": 3})):
    m = bench_meta(100.0, jetson=jet)
    m["throttle"] = None
    env = FakeEnv(script={name(A): [{"meta": m}]}, board=BOARD)
    out, state = run(plan(A, platform="jetson"), env)
    check(f"5 the Jetson: {label}: throttled, run again", summary(out)[name(A)]["attempts"] == "2", summary(out)[name(A)])
m = bench_meta(100.0, jetson={"over_current_throttled": False, "images_gpu_below_max": 0})
m["throttle"] = None
env = FakeEnv(script={name(A): [{"meta": m}]}, board=BOARD)
out, state = run(plan(A, platform="jetson"), env)
check("5 the Jetson: neither: not throttled", summary(out)[name(A)]["throttle"] == "none" and summary(out)[name(A)]["attempts"] == "1")

# ---------------------------------------------------------------- 6: the drift verdict
for end_ms, verdict in ((102.0, "ok"), (103.0, "ok"), (103.5, "re-run"), (96.0, "re-run")):
    env = FakeEnv(script={"drift-start": [{"ms": 100.0}], "drift-end": [{"ms": end_ms}]})
    out, state = run(plan(A, B), env)
    s = summary(out)
    check(f"6 the reference cell moves 100 -> {end_ms} ms: {verdict} on every cell", all(r["drift"] == verdict for r in s.values()),
          {k: r["drift"] for k, r in s.items()})
    check(f"6 ... session.json keeps both medians and the change ({end_ms})", state["drift"]["start_ms"] == 100.0
          and state["drift"]["end_ms"] == end_ms and state["drift"]["verdict"] == verdict, state["drift"])
log = (out / "session.log").read_text(encoding="utf-8")
check("6 a drift over 3% says, in the log, that every cell runs again", "-4.0%" in log and "every cell of this session" in log, log[-600:])

env = FakeEnv(script={"drift-end": [{"throttled": True, "ms": 101.0}]})
out, state = run(plan(A), env)
check("6 a throttled reference run: the drift record and the log say so", state["drift"]["throttled"] == ["drift-end"]
      and "drift-end throttled" in (out / "session.log").read_text(encoding="utf-8"), state["drift"])

env = FakeEnv(script={"drift-start": [{"crash": True}, {"crash": True}]})
out, state = run(plan(A), env)
check("6 a reference run that failed: the verdict is unknown", state["drift"]["verdict"] == "unknown"
      and summary(out)[name(A)]["drift"] == "unknown", state["drift"])


class Peek(FakeEnv):
    """Keeps the summary as it stood when each benchmark started."""
    seen = []

    def bench(self, spec, out_dir):
        self.seen.append(summary(Path(out_dir).parents[len(Path(spec["name"]).parts) + 1]))  # <out>/runs/<name>/attempt-<n>
        return super().bench(spec, out_dir)


env = Peek()
out, state = run(plan(A), env)
check("6 before the closing reference run, the drift verdict is pending", env.seen[-1][name(A)]["drift"] == "pending", env.seen[-1][name(A)])


# ---------------------------------------------------------------- 7: an interrupted session resumes
def interrupted(p, script, **kw):
    """A session stopped by Ctrl-C at the first call the script marks {"interrupt": True}: (out, the env)."""
    env = FakeEnv(script=script)
    out = Path(tempfile.mkdtemp())
    try:
        ss.run_session(p, out, env, **kw)
        check("7 (setup) the interrupt stops the session", False)
    except KeyboardInterrupt:
        pass
    return out, env


P = plan(A, B, C)
out, env = interrupted(P, {name(B): [{"interrupt": True}]})
state = json.loads((out / "session.json").read_text())
check("7 Ctrl-C: the half-done run is recorded as interrupted", state["status"] == "interrupted"
      and state["units"][name(B)]["attempts"][0]["result"] == "interrupted", state["units"][name(B)])
done_before = [c[0] for c in env.calls[:-1]]
env2 = FakeEnv(script={name(B): [{"crash": True}]})
state = ss.run_session(P, out, env2)
called = [c[0] for c in env2.calls]
check("7 the resumed session skips every run that finished", not set(done_before) & set(called) - {"drift-end"}
      and called[0] == name(B), (done_before, called))
check("7 the half-done cell restarts from the start, in a fresh directory", env2.calls[0][1].name == "attempt-2")
r = summary(out)[name(B)]
check("7 an interruption is not a crash: one crash after it still gets its re-run", r["outcome"] == "ok" and r["attempts"] == "3"
      and r["attempt_log"].startswith("a1 interrupted; a2 crashed; a3 complete"), r)
check("7 the resumed session completes, in the order it started with", state["status"] == "complete"
      and [c[0] for c in env.calls][1:] == state["order"] and called[-1] == "drift-end", (state["order"], env.calls, called))
log = (out / "session.log").read_text(encoding="utf-8")
check("7 the log keeps both invocations and says it resumed", "resum" in log and log.count("cell order") >= 1, log[-800:])


class PowerCut(FakeEnv):
    """At its interrupt, keeps session.json as it stood mid-run: what a power cut leaves on disk."""

    def bench(self, spec, out_dir):
        self.on_disk = (Path(out_dir).parents[len(Path(spec["name"]).parts) + 1] / "session.json").read_text()
        return super().bench(spec, out_dir)


env = PowerCut(script={name(A): [{"interrupt": True}]})
out = Path(tempfile.mkdtemp())
try:
    ss.run_session(P, out, env)
except KeyboardInterrupt:
    (out / "session.json").write_text(env.on_disk)
state = ss.run_session(P, out, FakeEnv())
check("7 a power cut: the run left 'running' is taken as interrupted, and restarts", state["status"] == "complete"
      and [a["result"] for a in state["units"][name(A)]["attempts"]] == ["interrupted", "complete"], state["units"][name(A)])

env3 = FakeEnv()
state = ss.run_session(P, out, env3)
check("7 a complete session run again: nothing runs", env3.calls == [] and state["status"] == "complete")
for label, other, kw in (("another plan", plan(A, B), {}), ("a dry run of the same plan", P, {"dry_run": True})):
    env3 = FakeEnv()
    try:
        ss.run_session(other, out, env3, **kw)
        check(f"7 {label} in the same directory is refused", False)
    except SystemExit as e:
        check(f"7 {label} in the same directory is refused", env3.calls == [] and "--out" in str(e), str(e))

# ---------------------------------------------------------------- 8: the Pi setup checks
PI = plan(A, B, platform="pi5")


def checks(state):
    return {c["check"]: c for c in state["invocations"][-1]["checks"]}


env = FakeEnv(facts=GOOD_PI)
out, state = run(PI, env)
check("8 a Pi set up as the protocol asks: every check passes, and the session runs",
      state["status"] == "complete" and all(c["status"] == "pass" for c in checks(state).values()), checks(state))
check("8 the checks cover the governor, the desktop, automatic updates and the packages the plan needs",
      {"governor", "desktop", "automatic updates", "numpy", "opencv-python-headless", "onnxruntime", "openvino", "vcgencmd clock"} <= set(checks(state))
      and "ncnn" not in checks(state), list(checks(state)))
check("8 the facts the checks read are recorded", state["invocations"][-1]["facts"]["os"] == "Debian GNU/Linux 12 (bookworm)")

for label, bad, fix in (("a CPU governor that is not performance", {"governors": ["performance", "ondemand"] * 2}, "scaling_governor"),
                        ("a desktop session", {"desktop": ["labwc"]}, "multi-user.target"),
                        ("an automatic-update timer on", {"update_units": {"apt-daily.timer": "active", "apt-daily-upgrade.timer": "inactive"}},
                         "systemctl disable"),
                        ("onnxruntime at another version", {"packages": {**GOOD_PI["packages"], "onnxruntime": "1.30.0"}}, "onnxruntime==1.31.0"),
                        ("a runtime the plan needs, missing", {"packages": {**GOOD_PI["packages"], "openvino": None}}, "openvino==2026.4.1"),
                        ("vcgencmd unable to read the CPU clock (#28)", {"arm_mhz": None}, "usermod -aG video")):
    env = FakeEnv(facts={**GOOD_PI, **bad})
    out = Path(tempfile.mkdtemp())
    try:
        ss.run_session(PI, out, env)
        check(f"8 {label}: refused", False)
    except SystemExit as e:
        check(f"8 {label}: refused before any run, with the fix", env.calls == [] and fix in str(e), str(e))
    state = json.loads((out / "session.json").read_text())
    check(f"8 {label}: the refusal is recorded", state["status"] == "refused"
          and any(c["status"] == "fail" for c in checks(state).values()), checks(state))

env = FakeEnv(facts={**GOOD_PI, "desktop": ["labwc"]})
out, state = run(PI, env, override="the borrowed Pi's desktop cannot be stopped")
check("8 an explicit override runs the session anyway", state["status"] == "complete" and len(env.calls) == 4)
check("8 ... and the override and the failed check are recorded", state["invocations"][-1]["override"] == "the borrowed Pi's desktop cannot be stopped"
      and checks(state)["desktop"]["status"] == "fail" and "OVERRIDE" in (out / "session.log").read_text(encoding="utf-8"))

env = FakeEnv(facts={"governors": None, "desktop": None, "update_units": None, "packages": GOOD_PI["packages"], "arm_mhz": 2400})
out, state = run(PI, env)
check("8 a check it cannot read is recorded as unreadable, and does not refuse", state["status"] == "complete"
      and checks(state)["governor"]["status"] == "unreadable", checks(state))

env = FakeEnv(facts={**GOOD_PI, "desktop": ["labwc"]})
out, state = run(plan(A), env)
check("8 the laptop has no setup checks to refuse on", state["status"] == "complete" and state["invocations"][-1]["checks"] == [])

out, env = interrupted(PI, {name(A): [{"interrupt": True}]}, override="first sitting")
state = ss.run_session(PI, out, FakeEnv(facts=GOOD_PI))
check("8 every invocation checks again, and records its own result", len(state["invocations"]) == 2
      and state["invocations"][1]["override"] is None, state["invocations"])

# ---------------------------------------------------------------- 9: latency is e2e_deploy when the benchmark gives it
E2E = {"meta": bench_meta(100.0, e2e_deploy=80.0)}
env = FakeEnv(script={u: [E2E] for u in ("drift-start", "drift-end", name(A))})
out, state = run(plan(A), env)
r = summary(out)[name(A)]
check("9 the benchmark gives e2e_deploy: that is the latency", r["latency_ms"] == "80.0" and r["latency_source"] == "e2e_deploy", r)
check("9 ... and the log does not speak of a fallback", "pre + net + post" not in (out / "session.log").read_text(encoding="utf-8"))
env = FakeEnv()
out, state = run(plan(A), env)
r = summary(out)[name(A)]
check("9 no e2e_deploy: pre + net + post, and the summary says so", r["latency_ms"] == "100.0" and r["latency_source"] == "pre+net+post", r)
check("9 ... and so does the log", "pre + net + post" in (out / "session.log").read_text(encoding="utf-8"))
env = FakeEnv(script={"drift-start": [E2E]})
out, state = run(plan(A), env)
check("9 the two reference runs measured different latencies: the drift verdict is unknown", state["drift"]["verdict"] == "unknown", state["drift"])

# ---------------------------------------------------------------- 10: a plan with a mistake is refused before anything runs
CELL = {"artifact": A, "runtime": "onnxruntime", "imgsz": 640}
for label, bad, says in (("a misspelt key", {"platform": "pi5", "seeed": 3, "cells": [CELL]}, "seeed"),
                         ("a misspelt cell key", {"platform": "pi5", "cells": [{**CELL, "limt": 200}]}, "limt"),
                         ("an unknown platform", {"platform": "pi3", "cells": [CELL]}, "pi3"),
                         ("an unknown runtime", {"platform": "pi5", "cells": [{**CELL, "runtime": "tflite"}]}, "tflite"),
                         ("two cells with one name", {"platform": "pi5", "cells": [CELL, {**CELL, "limit": 200}]}, "name"),
                         ("a cell named like a reference run", {"platform": "pi5", "cells": [{**CELL, "name": "drift-end"}]}, "drift-end"),
                         ("a limit that is not a positive integer", {"platform": "pi5", "cells": [{**CELL, "limit": "200"}]}, "limit"),
                         ("no cells", {"platform": "pi5", "cells": []}, "cells"),
                         ("a plan that moves the reference cell", {"platform": "pi5", "reference": B, "cells": [CELL]}, "reference")):
    env = FakeEnv()
    try:
        run(bad, env)
        check(f"10 {label}: refused", False)
    except SystemExit as e:
        check(f"10 {label}: refused, naming the problem", env.calls == [] and says in str(e), str(e))
env = FakeEnv()
out, state = run({"platform": "laptop", "cells": [CELL, {**CELL, "limit": 200, "name": "a1-int8-200"}]}, env)
check("10 the same artifact twice, with names of its own: fine", state["status"] == "complete" and "a1-int8-200" in summary(out))
check("10 a cell's limit reaches the benchmark", [c[4]["limit"] for c in env.calls if c[0] == "a1-int8-200"] == [200])

# ---------------------------------------------------------------- 11: what the wizard copies: the paths a plan needs on the platform
needs = ss.needed_paths({"platform": "pi5", "cells": [CELL, {**CELL, "limit": 200, "name": "a-200"}, {**CELL, "artifact": B}]})
check("11 a plan needs its images, the reference cell and each artifact once, in that order",
      needs == ["datasets/VisDrone/images/test", "exports/yolo11s-640-s0/640/onnxruntime-fp32", A, B], needs)
needs = ss.needed_paths({"platform": "pi5", "images": "imgs", "cells": [{**CELL, "artifact": "exports\\x\\640\\onnxruntime-fp32"},
                                                                         {**CELL, "artifact": "exports/yolo11s-640-s0/640/onnxruntime-fp32"}]})
check("11 a plan's own images; Windows separators; the reference artifact as a cell is not listed twice",
      needs == ["imgs", "exports/yolo11s-640-s0/640/onnxruntime-fp32", "exports/x/640/onnxruntime-fp32"], needs)

# ---------------------------------------------------------------- 12: the setup checks alone, before a session (the wizard's install step)
env = FakeEnv(facts={**GOOD_PI, "desktop": ["labwc"], "packages": {**GOOD_PI["packages"], "openvino": None}})
report = {c["check"]: c for c in ss.setup_report(PI, env)}
check("12 a Pi's checks, without a session: the failed ones with their fix, the plan's runtimes only",
      report["desktop"]["status"] == "fail" and "multi-user.target" in report["desktop"]["fix"]
      and report["openvino"]["status"] == "fail" and "openvino==2026.4.1" in report["openvino"]["fix"]
      and report["governor"]["status"] == "pass" and "ncnn" not in report, report)
check("12 ... and no run starts", env.calls == [])
check("12 the drift check's onnxruntime is checked even when no cell uses it",
      "onnxruntime" in {c["check"] for c in ss.setup_report(plan(B, platform="pi5"), FakeEnv())})
check("12 the laptop has no setup checks", ss.setup_report(plan(A), FakeEnv(facts={**GOOD_PI, "desktop": ["labwc"]})) == [])


# ---------------------------------------------------------------- 13: an energy session (#26): idle, then a 3-minute run, per cell
def energy(*artifacts, **kw):
    return {**plan(*artifacts, platform="pi5"), "mode": "energy", **kw}


def windows(state, unit, k=0):
    a = state["units"][unit]["attempts"][k]
    return a.get("idle"), a.get("run")


env = FakeEnv(power=lambda t, load: 6.0 if load == "run" else 4.5 if load == "load" else 3.0, fps=5.0)
out, state = run(energy(A, B), env)
called = [c[0] for c in env.calls]
first = state["order"].index(name(A)) + 1
check("13 each cell runs once, in the shuffled order, with the sampler-off run right after the plan's first cell; no reference runs",
      called == state["order"][:first] + ["sampler-off"] + state["order"][first:] and sorted(state["order"]) == sorted([name(A), name(B)]),
      (called, state["order"]))
sampled = [c for c in env.calls if c[0] != "sampler-off"]
check("13 each energy run: a 1-minute idle reading right after the gate, then a 3-minute run with the sampler on",
      env.idles[0] == (0, 60) and [t + s for t, s in env.idles] == [c[2] for c in sampled]
      and all(c[4]["seconds"] == 180 and c[4]["sample_power"] for c in sampled), (env.idles, [c[2] for c in env.calls]))
idle, run_ = windows(state, name(A))
check("13 the idle window and the run window are recorded next to each other, as rail sums",
      idle["mean_w"] == 3.0 and idle["seconds"] == 60 and idle["samples"] == 120
      and run_["mean_w"] == 6.0 and run_["seconds"] == 180 and run_["samples"] == 361, (idle, run_))
a = state["units"][name(A)]["attempts"][0]
check("13 the model load and warm-up (between the windows) are in neither", run_["mean_w"] == 6.0 and idle["mean_w"] == 3.0)
check("13 J per frame: the run's rail energy over its frames, and net of the idle reading",
      a["frames"] == 900 and a["fps"] == 5.0 and a["rail_j_per_frame"] == 1.2 and a["rail_j_per_frame_net"] == 0.6, a)
s = summary(out)
check("13 the summary: one row per cell, with FPS, both windows, J per frame, and the boundary",
      s[name(A)]["outcome"] == "ok" and s[name(A)]["fps"] == "5.0" and s[name(A)]["rail_idle_w"] == "3.0"
      and s[name(A)]["rail_run_w"] == "6.0" and s[name(A)]["rail_j_per_frame"] == "1.2" and s[name(A)]["rail_j_per_frame_net"] == "0.6"
      and s[name(A)]["measures"] == "rail sum, not wall power", s.get(name(A)))
check("13 ... and no latency column: an energy run is never a latency", "latency_ms" not in s[name(A)], list(s[name(A)]))
log = (out / "session.log").read_text(encoding="utf-8")
check("13 the log labels the energy as the rail sum, not wall power", "rail sum, not wall power" in log, log[-800:])
power = (out / a["dir"] / "power.csv").read_text(encoding="utf-8").splitlines()
check("13 every reading is kept in the run's power.csv, with its window",
      power[0] == "window,t_s,rail_w" and sum(r.startswith("idle,") for r in power) == 120 and sum(r.startswith("run,") for r in power) == 361
      and any(r.startswith("between,") for r in power), power[:3])

# ---------------------------------------------------------------- 14: the sampler's cost on FPS, measured once per energy session
env = FakeEnv(script={name(A): [{"fps": 4.9}], "sampler-off": [{"fps": 5.0}]})
out, state = run(energy(A, B), env)
off = next(c for c in env.calls if c[0] == "sampler-off")
on = next(c for c in env.calls if c[0] == name(A))
check("14 the sampler-off run: the plan's first cell, the same 3 minutes, the sampler off",
      off[0] == "sampler-off" and off[4]["artifact"] == A and off[4]["seconds"] == 180 and off[4]["sample_power"] is False, off[4])
check("14 ... right after the sampled run of that cell, and it rests the same minute first, without readings",
      env.calls.index(off) == env.calls.index(on) + 1 and len(env.idles) == 2 and off[2] - on[2] == 10 + 180 + 60,
      (env.idles, [c[2] for c in env.calls]))
c = state["sampler_cost"]
check("14 the cost: FPS with the sampler against FPS without it", c == {"cell": name(A), "fps_sampler_on": 4.9, "fps_sampler_off": 5.0,
                                                                    "cost_pct": 2.0, "throttled": []}, c)
s = summary(out)
check("14 the sampler-off row has its FPS and no energy", s["sampler-off"]["sampler"] == "off" and s["sampler-off"]["fps"] == "5.0"
      and s["sampler-off"]["rail_j_per_frame"] == "" and s["sampler-off"]["measures"] == "", s["sampler-off"])
check("14 ... and the log gives the cost", "the sampler's cost on FPS" in (out / "session.log").read_text(encoding="utf-8")
      and "+2.0%" in (out / "session.log").read_text(encoding="utf-8"))

# ---------------------------------------------------------------- 15: an energy run never shares a pass with a latency measurement
env = FakeEnv()
out, state = run(plan(A, B, platform="pi5"), env)
check("15 a latency session never reads the PMIC and never runs timed", env.idles == []
      and not any(c[4].get("sample_power") or c[4].get("seconds") for c in env.calls), [c[4] for c in env.calls])
env = FakeEnv()
out, state = run(energy(A, B), env)
check("15 an energy session never runs a latency pass: every run is timed, with no reference runs", all(c[4].get("seconds") for c in env.calls)
      and "drift-start" not in state["units"] and "drift" not in state, list(state["units"]))
check("15 an energy session needs no reference cell on the Pi", ss.needed_paths(energy(A, B)) == ["datasets/VisDrone/images/test", A, B])
for label, bad, says in (("on another platform", {**energy(A), "platform": "laptop"}, "Pi 5"),
                         ("a cell with a limit", {**energy(A), "cells": [{**CELL, "limit": 200}]}, "limit"),
                         ("an unknown mode", {**plan(A, platform="pi5"), "mode": "power"}, "mode"),
                         ("a cell named like the sampler-off run", {**energy(A), "cells": [{**CELL, "name": "sampler-off"}]}, "sampler-off")):
    try:
        run(bad, FakeEnv())
        check(f"15 an energy plan {label}: refused", False)
    except SystemExit as e:
        check(f"15 an energy plan {label}: refused, naming the problem", says in str(e), str(e))
P0 = plan(A, B, platform="pi5")
check("15 a latency plan validates as before #26, so a session started before it resumes", "mode" not in ss.load_plan(P0))

# ---------------------------------------------------------------- 16: the PMIC check, and the run rules in an energy session
for label, w in (("unreadable", None), ("reading 0 W", 0.0)):
    env = FakeEnv(facts={**GOOD_PI, "pmic_w": w})
    try:
        run(energy(A), env)
        check(f"16 PMIC rails {label}: an energy session is refused", False)
    except SystemExit as e:
        check(f"16 PMIC rails {label}: an energy session is refused before any run, with the fix", env.calls == [] and "pmic_read_adc" in str(e), str(e))
check("16 a latency session does not check the PMIC", "PMIC rails" not in {c["check"] for c in ss.setup_report(P0, FakeEnv())})

env = FakeEnv(script={name(A): [{"no_power": True}]})
out, state = run(energy(A, B), env)
r = summary(out)[name(A)]
check("16 a run the sampler read nothing in counts as a crash, and runs again", r["attempts"] == "2" and r["outcome"] == "ok"
      and "readings" in state["units"][name(A)]["attempts"][0]["error"], (r, state["units"][name(A)]["attempts"][0]))
env = FakeEnv(script={name(A): [{"crash": True}, {"crash": True}]})
out, state = run(energy(A, B), env)
check("16 two crashes: the cell is failed, and the sampler's cost is unknown", summary(out)[name(A)]["outcome"] == "failed"
      and state["sampler_cost"]["cost_pct"] is None and state["status"] == "complete", state["sampler_cost"])
env = FakeEnv(script={name(B): [{"throttled": True, "fps": 4.0}, {}]})
out, state = run(energy(A, B), env)
order = [c[0] for c in env.calls]
check("16 a throttled energy run runs again at the end, with its own idle reading",
      order[-1] == name(B) and order.count(name(B)) == 2 and len(env.idles) == 3, (order, env.idles))
r = summary(out)[name(B)]
check("16 ... and the clean attempt gives the cell's energy", r["outcome"] == "ok" and r["fps"] == "5.0" and r["run_dir"].endswith("attempt-2"), r)
env = FakeEnv(script={"sampler-off": [{"throttled": True, "fps": 4.0}, {"fps": 5.1}]})
out, state = run(energy(A, B), env)
check("16 a throttled sampler-off run runs again at the end too, and the clean one gives the cost",
      [c[0] for c in env.calls][-1] == "sampler-off" and state["sampler_cost"]["fps_sampler_off"] == 5.1
      and state["sampler_cost"]["throttled"] == [], state["sampler_cost"])
env = FakeEnv(script={"sampler-off": [{"throttled": True, "fps": 4.0}, {"throttled": True, "fps": 4.2}]})
out, state = run(energy(A, B), env)
check("16 throttled twice: the cost record and the log say so", state["sampler_cost"]["throttled"] == ["sampler-off"]
      and "sampler-off throttled" in (out / "session.log").read_text(encoding="utf-8"), state["sampler_cost"])
env = FakeEnv()
out, state = run(energy(A), env, dry_run=True)
check("16 a dry run: a 10 s idle reading and 15 s runs", [s for _, s in env.idles] == [10] and all(c[4]["seconds"] == 15 for c in env.calls),
      (env.idles, [c[4]["seconds"] for c in env.calls]))

# ---------------------------------------------------------------- 17: the PMIC reading and the sampler on the real platform
PMIC = """     3V7_WL_SW_A current(0)=0.00000000A
       3V3_SYS_A current(1)=0.10000000A
      VDD_CORE_A current(7)=2.00000000A
          HDMI_A current(22)=0.02000000A
     3V7_WL_SW_V volt(8)=3.74639000V
       3V3_SYS_V volt(9)=3.30000000V
      VDD_CORE_V volt(15)=0.75000000V
          HDMI_V volt(23)=5.00000000V
         EXT5V_V volt(24)=5.13072000V
          BATT_V volt(25)=0.00000000V
"""
w = ss.parse_pmic(PMIC)
check("17 the rail sum: each rail's current times its own voltage, summed; EXT5V and BATT have no current",
      w is not None and abs(w - (0.1 * 3.3 + 2.0 * 0.75 + 0.02 * 5.0)) < 1e-9, w)
check("17 a rail with a current and no voltage is not counted", abs(ss.parse_pmic(PMIC + "       1V8_SYS_A current(2)=1.0A\n") - w) < 1e-9)
check("17 output with no rails (another Pi, an error) gives no reading", ss.parse_pmic("error=1 error_msg=\"Command not registered\"") is None
      and ss.parse_pmic("") is None)


class FakePmic(ss.Platform):
    """The real platform with a fake PMIC: every reading is 2.5 W, every third fails."""
    sample_s = 0.02
    n = 0

    def read_power(self):
        self.n += 1
        return None if self.n % 3 == 0 else 2.5


p = FakePmic("pi5")
t0 = ss.time.monotonic()
samples = p.idle_window(0.5)
t1 = ss.time.monotonic()
check("17 an idle window: the sampler reads for its length, at its interval, and keeps the readings that worked",
      10 <= len(samples) <= 25 and all(x == 2.5 for _, x in samples), len(samples))
check("17 ... each reading timed on the monotonic clock the bench's window uses", all(t0 <= t <= t1 for t, _ in samples)
      and [t for t, _ in samples] == sorted(t for t, _ in samples) and t1 - t0 >= 0.5, (t0, samples[:2], t1))
sampler = ss.PowerSampler(lambda: 1 / 0, 0.01)
sampler.start()
ss.time.sleep(0.1)
check("17 a reader that raises: the sampler keeps going and gives no readings", sampler.stop() == [] and sampler.failed > 3, sampler.failed)

# ---------------------------------------------------------------- 18: the room and the cooling, for every invocation (#28)
COOLING = {"fan_rpm": "none", "arm_freq": 2400, "arm_freq_min": 1500, "temp_soft_limit": 60, "idle_temp_c": 52.3}
out, env = interrupted(PI, {name(A): [{"interrupt": True}]}, ambient_c=24.5)
state = ss.run_session(PI, out, FakeEnv(facts={**GOOD_PI, "cooling": COOLING}), ambient_c=27.0)
check("18 every invocation records the room temperature it was given", [i["ambient_c"] for i in state["invocations"]] == [24.5, 27.0],
      state["invocations"])
check("18 ... and the cooling its platform reads", state["invocations"][-1]["facts"]["cooling"] == COOLING)
log = (out / "session.log").read_text(encoding="utf-8")
check("18 the log gives the room and the cooling", "room 27 C" in log and "fan none" in log and "arm_freq 2400" in log, log[-1500:])
out, state = run(plan(A), FakeEnv())
check("18 a session given no room temperature records none", state["invocations"][-1]["ambient_c"] is None)


class FakeVcgencmd:
    """subprocess.run for `vcgencmd get_config <name>` and `vcgencmd measure_clock arm`, printing what a throttled Pi 5
    prints; anything else gets vcgencmd's error line."""

    CONFIG = {"arm_freq": "2400", "arm_freq_min": "1500", "temp_soft_limit": "60"}

    def __call__(self, cmd, **kw):
        if cmd[0] != "vcgencmd":
            raise FileNotFoundError(cmd[0])
        out = (f"{cmd[2]}={self.CONFIG[cmd[2]]}\n" if cmd[1:2] == ["get_config"] and cmd[2] in self.CONFIG
               else "frequency(0)=1500004992\n" if cmd[1:] == ["measure_clock", "arm"] else 'error=1 error_msg="Command not registered"\n')
        return ss.subprocess.CompletedProcess(cmd, 0, out, "")


def pi_root(fan_rpm=None, temp_mc=52300):
    root = Path(tempfile.mkdtemp())
    zone = root / "sys/class/thermal/thermal_zone0"
    zone.mkdir(parents=True)
    (zone / "type").write_text("cpu-thermal\n")
    (zone / "temp").write_text(f"{temp_mc}\n")
    if fan_rpm is not None:
        hw = root / "sys/devices/platform/cooling_fan/hwmon/hwmon3"
        hw.mkdir(parents=True)
        (hw / "fan1_input").write_text(f"{fan_rpm}\n")
    return root


saved = ss.shutil.which, ss.subprocess.run
ss.shutil.which = lambda n, *a, **kw: "vcgencmd" if n == "vcgencmd" else None
ss.subprocess.run = FakeVcgencmd()
try:
    bare = ss.Platform("pi5", root=pi_root()).setup_facts()
    fanned = ss.Platform("pi5", root=pi_root(fan_rpm=3200, temp_mc=41000)).setup_facts()
    pi4 = ss.Platform("pi4b", root=pi_root()).setup_facts()
    laptop = ss.Platform("laptop", root=pi_root()).setup_facts()
finally:
    ss.shutil.which, ss.subprocess.run = saved
check("18 a Pi with no fan: the cooling says none, with the firmware's clock and temperature settings and the idle temperature",
      bare.get("cooling") == COOLING, bare.get("cooling"))
check("18 a Pi with a fan: its rpm", (fanned.get("cooling") or {}).get("fan_rpm") == 3200
      and fanned["cooling"]["idle_temp_c"] == 41.0, fanned.get("cooling"))
check("18 the Pi 4B records its cooling too, its fan unknown (it cannot see one); the laptop records none",
      (pi4.get("cooling") or {}).get("fan_rpm", "missing") is None and pi4["cooling"]["arm_freq"] == 2400 and "cooling" not in laptop,
      (pi4.get("cooling"), laptop.get("cooling")))
check("18 the setup check reads the firmware's clock through vcgencmd", bare.get("arm_mhz") == 1500 and "arm_mhz" not in laptop,
      (bare.get("arm_mhz"), laptop.get("arm_mhz")))

if FAILED:
    sys.exit(f"{len(FAILED)} failed: {FAILED}")
print("all passed")
