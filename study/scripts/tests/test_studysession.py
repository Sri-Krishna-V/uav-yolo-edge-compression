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
           "python": "3.11.2", "os": "Debian GNU/Linux 12 (bookworm)"}


class FakeEnv:
    """The platform as the runner sees it. Time moves only when the runner sleeps or a benchmark runs.

    script: {unit name: [outcome of the 1st call, of the 2nd call, ...]}; an outcome is a dict:
      {"ms": 103.0}  a complete run with that median latency (the default is {"ms": 100.0})
      {"crash": True}  the benchmark exits 1
      {"throttled": True}  a complete run that throttled
      {"interrupt": True}  Ctrl-C while the benchmark runs
      {"meta": {...}}  this exact meta.json
    """

    def __init__(self, temps=lambda t: 40.0, script=None, bench_s=60.0, battery=None, board=None, facts=None):
        self.t, self.temps, self.script, self.bench_s = 0.0, temps, script or {}, bench_s
        self.battery, self.board, self.facts = battery, board, GOOD_PI if facts is None else facts  # a Pi set up as the protocol asks
        self.calls = []  # (unit name, attempt directory, clock at the call, CPU temperature at the call)

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
        self.t += self.bench_s
        if outcome.get("interrupt"):
            raise KeyboardInterrupt
        if outcome.get("crash"):
            return {"exit": 1, "meta": {"status": "interrupted", "error": "RuntimeError: boom"}, "tail": "RuntimeError: boom"}
        meta = outcome.get("meta") or bench_meta(outcome.get("ms", 100.0), outcome.get("throttled", False))
        return {"exit": 0, "meta": meta, "tail": "done"}


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
      {"governor", "desktop", "automatic updates", "numpy", "opencv-python-headless", "onnxruntime", "openvino"} <= set(checks(state))
      and "ncnn" not in checks(state), list(checks(state)))
check("8 the facts the checks read are recorded", state["invocations"][-1]["facts"]["os"] == "Debian GNU/Linux 12 (bookworm)")

for label, bad, fix in (("a CPU governor that is not performance", {"governors": ["performance", "ondemand"] * 2}, "scaling_governor"),
                        ("a desktop session", {"desktop": ["labwc"]}, "multi-user.target"),
                        ("an automatic-update timer on", {"update_units": {"apt-daily.timer": "active", "apt-daily-upgrade.timer": "inactive"}},
                         "systemctl disable"),
                        ("onnxruntime at another version", {"packages": {**GOOD_PI["packages"], "onnxruntime": "1.30.0"}}, "onnxruntime==1.31.0"),
                        ("a runtime the plan needs, missing", {"packages": {**GOOD_PI["packages"], "openvino": None}}, "openvino==2026.4.1")):
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

env = FakeEnv(facts={"governors": None, "desktop": None, "update_units": None, "packages": GOOD_PI["packages"]})
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
                         ("no cells", {"platform": "pi5", "cells": []}, "cells")):
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

if FAILED:
    sys.exit(f"{len(FAILED)} failed: {FAILED}")
print("all passed")
