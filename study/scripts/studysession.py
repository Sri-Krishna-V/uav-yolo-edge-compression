"""Run one platform session: every cell of a platform plan through studybench.py, under the platform-run rules of #8.

  python3 scripts/studysession.py plans/pi5-speedtest.json --out bench/pi5/speedtest

Runs ON the platform, from study/ (relative paths in the plan are relative to it). Standard library only: it starts
studybench.py once per run, as a child process, and reads the meta.json it writes.

The rules it applies, so that nobody applies them by hand:
  setup     on a Pi (pi5, pi4b), every invocation first checks what it can read: the CPU governor is "performance" on
            every CPU, no desktop session runs, the automatic-update timers are off, and the packages the plan needs are
            at PLATFORMS.md's versions. It refuses to start on a failed check, unless --override-checks gives a reason,
            which is recorded. A check it cannot read is recorded as unreadable and does not refuse.
  threads   4 runtime threads and 20 untimed warm-up images per run, recorded.
  order     the plan's cells are shuffled once with the plan's seed (default 0), and the order is logged.
  gate      a run starts only when the CPU is below 50 C, or after 3 minutes (polled every 5 s). The phone (m34) shows no
            CPU temperature: it waits a fixed 3 minutes, and logs its battery temperature if Termux can read it. A Jetson
            records its power mode and clocks at every gate.
  crash     a run that exits non-zero or leaves no complete meta.json runs again from the start, once, in a fresh
            directory. A second crash marks the cell failed, and the session goes on.
  throttle  a cell that throttled runs again at the end of the session (after the gate lets the platform cool), once.
            Both attempts are kept; the clean one gives the cell's latency. Throttled means: on a Pi, a vcgencmd flag
            seen during the run, or a since-boot flag that appeared between its first and last sample; on a Jetson,
            an over-current counter that rose, or images whose GPU clock fell below its pinned maximum. Elsewhere
            there is no signal, and the summary says "unknown".
  drift     the reference cell (YOLO11s baseline, ONNX Runtime FP32, 640, the first 200 test-dev images) runs at the start
            and at the end of the session. If its median latency moves by more than 3%, every cell of the session is
            flagged for a re-run: the summary's drift column says "re-run" ("unknown" if a reference run failed).

Writes to --out: session.log (every decision, appended), session.json (the state, rewritten after every step) and
summary.csv (one row per cell and per reference run: attempts, outcome, median latency, throttle state, drift verdict,
and the run directory to score). The same command resumes an interrupted session: it skips the runs that finished and
restarts a half-done one from the start, in a fresh directory. An interruption is not a crash. A plan whose artifacts,
reference cell or images are missing stops before anything runs.

--needs prints the paths the plan needs on the platform (what to copy there), and --check runs only the setup checks:
neither runs a cell or writes anything.

Latency is the median of studybench's e2e_deploy stage (latency at the deployment threshold, #22) when the benchmark
gives it. A benchmark without it gives pre + net + post at the evaluation threshold instead; the log and the summary's
latency_source say which.

--dry-run is a plumbing check: the gate waits 5 s at most, and the log and session.json say DRY RUN.
"""

import argparse
import collections
import csv
import hashlib
import json
import os
import platform as pyplatform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent

THREADS, WARMUP = 4, 20
PLATFORMS = ("pi5", "pi4b", "m34", "jetson", "laptop")
RUNTIMES = ("onnxruntime", "openvino", "ncnn", "tensorrt")
REFERENCE = "exports/yolo11s-640-s0/640/onnxruntime-fp32"  # YOLO11s baseline, ONNX Runtime FP32, 640
DRIFT_IMAGES = 200
GATE_C, GATE_S, POLL_S, DRY_GATE_S = 50.0, 180, 5, 5
CRASHES = 2  # a second crash fails the cell
PIS = ("pi5", "pi4b")
# PLATFORMS.md's Pi install (the bench build of 2026-10-21). Copied, not read: keep the two in step.
PI_PINS = {"numpy": "2.4.6", "opencv-python-headless": "5.0.0.93", "onnxruntime": "1.31.0", "openvino": "2026.4.1", "ncnn": "1.0.20260526"}
# What starts an automatic update. Not unattended-upgrades.service: it is "active" whenever the package is installed, and
# only finishes an upgrade at shutdown; with these two timers off, nothing starts one.
UPDATE_UNITS = ("apt-daily.timer", "apt-daily-upgrade.timer")
DESKTOP = {"Xorg", "Xwayland", "labwc", "wayfire", "lxsession", "lxpanel", "pcmanfm", "lightdm", "gnome-shell", "xfce4-session"}
DRIFT_MAX = 0.03
DRIFT_UNITS = ("drift-start", "drift-end")
THROTTLE_WORDS = {"throttled": "throttled", "none": "not throttled", "unknown": "throttle unknown"}
SUMMARY = ["cell", "runtime", "imgsz", "images", "attempts", "outcome", "latency_ms", "latency_source", "throttle", "drift",
           "run_dir", "attempt_log"]


# ---------------------------------------------------------------- the plan


PLAN_KEYS = {"platform", "note", "seed", "images", "cells"}
CELL_KEYS = {"name", "artifact", "runtime", "imgsz", "limit"}


def load_plan(plan):
    """A plan dict (a platform plan's JSON) -> the validated plan, with every cell named. A mistake stops it before anything runs."""
    def positive_int(v):
        return isinstance(v, int) and not isinstance(v, bool) and v > 0

    if set(plan) - PLAN_KEYS:
        raise SystemExit(f"plan: unknown keys {sorted(set(plan) - PLAN_KEYS)}; a plan has {sorted(PLAN_KEYS)}")
    if plan.get("platform") not in PLATFORMS:
        raise SystemExit(f"plan: platform must be one of {PLATFORMS}, not {plan.get('platform')!r}")
    cells = []
    for c in plan.get("cells") or []:
        if set(c) - CELL_KEYS:
            raise SystemExit(f"plan: unknown cell keys {sorted(set(c) - CELL_KEYS)}; a cell has {sorted(CELL_KEYS)}")
        if c.get("runtime") not in RUNTIMES or not positive_int(c.get("imgsz")) or not c.get("artifact"):
            raise SystemExit(f"plan: a cell needs artifact, runtime (one of {RUNTIMES}) and imgsz: {c}")
        if c.get("limit") is not None and not positive_int(c["limit"]):
            raise SystemExit(f"plan: a cell's limit is a number of images, not {c['limit']!r}")
        artifact = str(c["artifact"]).replace("\\", "/")
        cells.append({"name": c.get("name") or artifact.split("exports/", 1)[-1], "artifact": artifact, "runtime": c["runtime"],
                      "imgsz": c["imgsz"], "limit": c.get("limit")})
    if not cells:
        raise SystemExit("plan: no cells")
    names = [c["name"] for c in cells]
    if len(set(names)) < len(names) or set(names) & set(DRIFT_UNITS):
        raise SystemExit(f"plan: every cell needs a name of its own, and not {DRIFT_UNITS}: give the cell a \"name\". Names: {names}")
    if not isinstance(plan.get("seed", 0), int):
        raise SystemExit(f"plan: seed must be an integer, not {plan['seed']!r}")
    return {"platform": plan["platform"], "note": plan.get("note"), "seed": plan.get("seed", 0),
            "images": plan.get("images", "datasets/VisDrone/images/test"), "cells": cells}


def needed_paths(plan):
    """What a plan needs on the platform, relative to study/: its images, the reference cell's artifact, each artifact once."""
    plan = load_plan(plan)
    return list(dict.fromkeys([plan["images"], REFERENCE, *(c["artifact"] for c in plan["cells"])]))


def shuffled(plan):
    """The cell order of the session: the plan's order, shuffled once with the plan's seed."""
    order = [c["name"] for c in plan["cells"]]
    random.Random(plan["seed"]).shuffle(order)
    return order


def unit_specs(plan):
    """{unit name: what studybench runs}: the plan's cells and the reference cell at both ends of the session."""
    ref = {"artifact": REFERENCE, "runtime": "onnxruntime", "imgsz": 640, "limit": DRIFT_IMAGES}
    specs = {u: {"name": u, **ref} for u in DRIFT_UNITS}
    specs.update({c["name"]: dict(c) for c in plan["cells"]})
    for s in specs.values():
        s.update(images=plan["images"], threads=THREADS, warmup=WARMUP, platform=plan["platform"])
    return specs


# ---------------------------------------------------------------- the Pi setup checks


def pi_checks(facts, runtimes):
    """What the platform's setup_facts() could read -> [{check, status (pass, fail or unreadable), value, fix}]."""
    def check(name, value, ok, fix):
        return {"check": name, "status": "unreadable" if value is None else "pass" if ok else "fail", "value": value, "fix": fix}

    gov, desktop, units, packages = facts.get("governors") or None, facts.get("desktop"), facts.get("update_units"), facts.get("packages") or {}
    out = [check("governor", gov, gov and all(g == "performance" for g in gov),
                 "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"),
           check("desktop", desktop, desktop == [],
                 "sudo systemctl isolate multi-user.target (this boot), and sudo systemctl set-default multi-user.target (every boot)"),
           check("automatic updates", units, units is not None and "active" not in units.values(),
                 f"sudo systemctl disable --now {' '.join(UPDATE_UNITS)}")]
    for name in ["numpy", "opencv-python-headless", *sorted(set(runtimes) & set(PI_PINS))]:
        have = packages.get(name)
        out.append({"check": name, "status": "pass" if have == PI_PINS[name] else "fail", "value": have,
                    "fix": f"pip install {'--no-deps ' if name == 'ncnn' else ''}{name}=={PI_PINS[name]}"})
    return out


def platform_checks(plan, facts):
    """A validated plan and its platform's setup_facts() -> its setup checks: a Pi's, for every runtime its runs use; none elsewhere."""
    return pi_checks(facts, {s["runtime"] for s in unit_specs(plan).values()}) if plan["platform"] in PIS else []


def setup_report(plan, env):
    """The setup checks of a plan's platform, read now, without a session: nothing runs and nothing is written."""
    plan = load_plan(plan)
    return platform_checks(plan, env.setup_facts())


# ---------------------------------------------------------------- the start gate


def _dur(s):
    return f"{s // 60:g} min" if s % 60 == 0 else f"{s:g} s"


def start_gate(env, platform, max_wait):
    """Wait until a run may start: the CPU below 50 C or max_wait passed; the phone always waits max_wait. -> the gate record."""
    t0 = env.now()
    if platform == "m34":
        env.sleep(max_wait)
        temp, opened = env.cpu_temp_c(), f"fixed {_dur(max_wait)} wait (the phone shows no CPU temperature)"
    else:
        while True:
            temp, waited = env.cpu_temp_c(), env.now() - t0
            if temp is not None and temp < GATE_C:
                opened = f"CPU below {GATE_C:g} C"
                break
            if waited >= max_wait:
                opened = f"{_dur(max_wait)} passed" + (" (no CPU temperature)" if temp is None else f", CPU still at {temp} C")
                break
            env.sleep(min(POLL_S, max_wait - waited))
    gate = {"waited_s": round(env.now() - t0, 1), "opened": opened, "cpu_temp_c": temp, "battery_temp_c": env.battery_temp_c()}
    board = env.board_state()
    if board is not None:
        gate["board"] = board
    return gate


# ---------------------------------------------------------------- one run's result


def throttle_state(meta):
    """studybench's meta.json -> "throttled", "none" or "unknown" (the platform gave no throttle signal)."""
    signals = []
    t = meta.get("throttle")  # a Raspberry Pi's vcgencmd get_throttled samples
    if t:
        appeared = int(t["end"], 16) & ~int(t["start"], 16) & 0xF0000  # bits 16-19: "has occurred since boot"
        signals.append(any(t["during_run"].values()) or bool(appeared))
    j = meta.get("jetson")
    if j:
        below = j.get("images_gpu_below_max")
        signals += [j.get("over_current_throttled"), None if below is None else below > 0]
    known = [x for x in signals if x is not None]
    return "throttled" if any(known) else "none" if known else "unknown"


def judge(result):
    """What studybench left behind -> the attempt's result, median latency and throttle state."""
    meta = result.get("meta") or {}
    if result.get("exit") != 0 or meta.get("status") != "complete":
        return {"result": "crashed", "exit": result.get("exit"), "error": meta.get("error") or result.get("tail")}
    timing = meta["timing_ms"]
    source = "e2e_deploy" if "e2e_deploy" in timing else "pre+net+post"
    return {"result": "complete", "latency_ms": timing["e2e_deploy" if source == "e2e_deploy" else "t_total_ms"]["median"],
            "latency_source": source, "throttle": throttle_state(meta)}


def pass_state(attempts, pass_="main"):
    """A unit's attempts in one pass ("main", or "rerun" after a throttle) -> "complete", "failed" (crashed twice) or "todo"."""
    attempts = [a for a in attempts if a["pass"] == pass_]
    if any(a["result"] == "complete" for a in attempts):
        return "complete"
    return "failed" if sum(a["result"] == "crashed" for a in attempts) >= CRASHES else "todo"


# ---------------------------------------------------------------- the session


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class Session:
    def __init__(self, plan, out, env, dry_run, override):
        self.plan, self.out, self.env, self.dry_run, self.override = plan, Path(out), env, dry_run, override
        self.specs = unit_specs(plan)
        sha = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        old = json.loads((self.out / "session.json").read_text(encoding="utf-8")) if (self.out / "session.json").exists() else None
        if old and (old["plan_sha256"], old["dry_run"]) != (sha, dry_run):
            raise SystemExit(f"{self.out} holds a session of another plan{' or a dry run' if old['dry_run'] != dry_run else ''}: "
                             "give this one its own --out")
        self.out.mkdir(parents=True, exist_ok=True)
        self.state = old or {"tool": "studysession.py", "plan": plan, "plan_sha256": sha, "dry_run": dry_run, "seed": plan["seed"],
                             "order": shuffled(plan), "units": {u: {"attempts": []} for u in self.specs}}
        self.resumed = old is not None
        for u in self.state["units"]:
            for a in self.attempts(u):
                if a["result"] == "running":  # the runner itself died mid-run (power cut, killed): no exit code to judge
                    a.update(result="interrupted", error="the session stopped during this run")

    def log(self, msg):
        line = f"{now_iso()} {msg}"
        print(line, flush=True)
        with open(self.out / "session.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def save(self):
        self.state["drift"] = self.drift()
        write_json(self.out / "session.json", self.state)
        self.write_summary()

    def attempts(self, u):
        return self.state["units"][u]["attempts"]

    def needs_rerun(self, u):
        """A cell whose main pass completed throttled gets a re-run pass."""
        return u not in DRIFT_UNITS and any(a["pass"] == "main" and a.get("throttle") == "throttled" for a in self.attempts(u))

    def next_run(self):
        """(unit, pass) to run next, or None: the opening reference run, the cells, the throttle re-runs, the closing reference run."""
        for u in [DRIFT_UNITS[0], *self.state["order"]]:
            if pass_state(self.attempts(u)) == "todo":
                return u, "main"
        for u in self.state["order"]:
            if self.needs_rerun(u) and pass_state(self.attempts(u), "rerun") == "todo":
                return u, "rerun"
        return (DRIFT_UNITS[1], "main") if pass_state(self.attempts(DRIFT_UNITS[1])) == "todo" else None

    def outcome(self, u):
        """(outcome, the attempt that gives the cell's latency or None)."""
        attempts = self.attempts(u)
        done = [a for a in attempts if a["result"] == "complete"]
        clean = [a for a in done if a["throttle"] != "throttled"]
        if clean:
            return "ok", clean[-1]
        if done:
            pending = self.needs_rerun(u) and pass_state(attempts, "rerun") == "todo"
            return ("re-run queued" if pending else "throttled"), done[-1]
        return ("failed" if pass_state(attempts) == "failed" else "pending"), None

    def setup_checks(self):
        """Record this invocation's platform facts and, on a Pi, refuse a failed setup check unless overridden."""
        facts = self.env.setup_facts()
        checks = platform_checks(self.plan, facts)
        self.state.setdefault("invocations", []).append({"start": now_iso(), "facts": facts, "checks": checks, "override": self.override})
        for c in checks:
            self.log(f"setup check {c['check']}: {c['status']} ({c['value']})" + (f"; fix: {c['fix']}" if c["status"] == "fail" else ""))
        failed = [c for c in checks if c["status"] == "fail"]
        if failed and not self.override:
            self.state["status"] = "refused"
            self.save()
            self.log("REFUSED: a setup check failed")
            raise SystemExit("setup checks failed:\n" + "\n".join(f"  {c['check']} is {c['value']}: {c['fix']}" for c in failed)
                             + "\nfix them, or pass --override-checks \"<why>\" to run anyway (the reason is recorded)")
        if self.override:
            self.log(f"OVERRIDE of the setup checks: {self.override}")

    def drift(self):
        """The drift check's record: the two reference medians and the verdict (pending until the closing run is done)."""
        (_, start), (_, end) = (self.outcome(u) for u in DRIFT_UNITS)
        rec = {"start_ms": start and start["latency_ms"], "end_ms": end and end["latency_ms"], "change_pct": None,
               "throttled": [u for u, a in zip(DRIFT_UNITS, (start, end)) if a and a["throttle"] == "throttled"]}
        if pass_state(self.attempts(DRIFT_UNITS[1])) == "todo":
            return {**rec, "verdict": "pending"}
        if not start or not end or start["latency_source"] != end["latency_source"]:
            return {**rec, "verdict": "unknown"}
        moved = abs(end["latency_ms"] - start["latency_ms"]) > DRIFT_MAX * start["latency_ms"]
        return {**rec, "change_pct": round(100 * (end["latency_ms"] / start["latency_ms"] - 1), 1), "verdict": "re-run" if moved else "ok"}

    def run(self):
        if self.state.get("status") == "complete":
            self.log("this session is already complete")
            return self.state
        self.log(f"session on {self.plan['platform']}" + (f" ({self.plan['note']})" if self.plan["note"] else "")
                 + f": {len(self.plan['cells'])} cells; every run uses {THREADS} threads and {WARMUP} warm-up images, untimed")
        if self.resumed:
            done = sum(a["result"] != "interrupted" for u in self.state["units"] for a in self.attempts(u))
            self.log(f"resuming the session in {self.out}: {done} runs recorded; a half-done run restarts from the start")
        if self.dry_run:
            self.log(f"DRY RUN: the start gate waits {DRY_GATE_S} s at most; no number from this session is reported")
        self.setup_checks()
        self.state["status"] = "running"
        self.log(f"cell order, shuffled with seed {self.state['seed']}: {', '.join(self.state['order'])}")
        try:
            while (step := self.next_run()) is not None:
                self.attempt(*step)
        except BaseException as e:  # Ctrl-C, a dropped SSH session: the same command resumes
            self.state["status"] = "interrupted"
            self.save()
            self.log(f"INTERRUPTED ({type(e).__name__}): run the same command again to resume")
            raise
        self.state["status"] = "complete"
        self.save()
        d = self.state["drift"]
        if d["verdict"] == "unknown":
            self.log("drift check: UNKNOWN, a reference run failed: this session's latencies are unchecked")
        else:
            self.log(f"drift check: reference median {d['start_ms']} -> {d['end_ms']} ms ({d['change_pct']:+.1f}%): "
                     + ("ok" if d["verdict"] == "ok" else f"moved more than {DRIFT_MAX:.0%}: every cell of this session is flagged for a re-run")
                     + "".join(f" ({u} throttled)" for u in d["throttled"]))
        for r in self.summary_rows():
            self.log(f"  {r['cell']}: {r['outcome']}, {r['attempts']} attempt(s), median {r['latency_ms'] or '-'} ms "
                     f"({r['latency_source'] or '-'}), throttle {r['throttle'] or '-'}, drift {r['drift']}, {r['run_dir'] or 'no run to score'}")
        self.log(f"session complete: {self.out / 'summary.csv'}")
        return self.state

    def attempt(self, u, pass_):
        attempts = self.attempts(u)
        n = len(attempts) + 1
        gate = start_gate(self.env, self.plan["platform"], DRY_GATE_S if self.dry_run else GATE_S)
        self.log(f"{u}: gate opened after {gate['waited_s']:g} s: {gate['opened']}"
                 + (f"; battery {gate['battery_temp_c']} C" if gate["battery_temp_c"] is not None else "")
                 + (f"; board {json.dumps(gate['board'])}" if "board" in gate else ""))
        run_dir = self.out / "runs" / u / f"attempt-{n}"
        rec = {"n": n, "pass": pass_, "dir": run_dir.relative_to(self.out).as_posix(), "gate": gate, "start": now_iso(), "result": "running"}
        attempts.append(rec)
        self.save()
        self.log(f"{u}: attempt {n} starts" + (" (the throttle re-run)" if pass_ == "rerun" else ""))
        try:
            result = self.env.bench(self.specs[u], run_dir)
        except BaseException as e:  # the next invocation restarts this run; run() records the session as interrupted
            rec.update(result="interrupted", error=f"{type(e).__name__}: {e}", end=now_iso())
            self.log(f"{u}: attempt {n} INTERRUPTED")
            raise
        rec.update(judge(result), end=now_iso())
        self.save()
        if rec["result"] == "complete":
            self.log(f"{u}: attempt {n} complete, median "
                     + (f"latency {rec['latency_ms']} ms" if rec["latency_source"] == "e2e_deploy" else
                        f"pre + net + post {rec['latency_ms']} ms (at the evaluation threshold, so not latency: this studybench "
                        "gives no e2e_deploy, #22)") + f", throttle {rec['throttle']}"
                     + ("; it runs again at the end of the session" if pass_ == "main" and self.needs_rerun(u) else ""))
        else:
            then = "it runs again from the start" if pass_state(attempts, pass_) == "todo" else "a second crash: the cell is failed"
            self.log(f"{u}: attempt {n} CRASHED (exit {rec['exit']}): {rec['error']}; {then}")

    def summary_rows(self):
        rows = []
        for u, spec in self.specs.items():
            attempts = self.attempts(u)
            outcome, counted = self.outcome(u)
            rows.append({"cell": u, "runtime": spec["runtime"], "imgsz": spec["imgsz"], "images": spec["limit"] or "all",
                         "attempts": len(attempts), "outcome": outcome,
                         **{k: (counted or {}).get(k, "") for k in ("latency_ms", "latency_source", "throttle")},
                         "drift": self.state["drift"]["verdict"], "run_dir": counted["dir"] if counted else "",
                         "attempt_log": "; ".join(f"a{a['n']}{' re-run' if a['pass'] == 'rerun' else ''} {a['result']}"
                                                  + (" " + THROTTLE_WORDS[a["throttle"]] if a["result"] == "complete" else "") for a in attempts)})
        return rows

    def write_summary(self):
        with open(self.out / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, SUMMARY)
            w.writeheader()
            w.writerows(self.summary_rows())


def run_session(plan, out, env, dry_run=False, override=None):
    """Run (or resume) the session of `plan` in `out` on the platform `env`. Returns the session state."""
    return Session(load_plan(plan), out, env, dry_run, override).run()


# ---------------------------------------------------------------- the real platform (the tests use a fake one)


def _read(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _int(path):
    v = _read(path)
    return int(v) if v and v.lstrip("-").isdigit() else None


def _version(dist):
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


class Platform:
    """The platform this runs on, read without root: its clock, temperatures and setup, and studybench.py as a child."""

    def __init__(self, platform, root="/"):
        self.platform, self.root = platform, Path(root)

    now = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)

    def cpu_temp_c(self):
        """The thermal zone named cpu ("cpu-thermal" on the Pis and the Jetson), else zone 0, in C; None when unreadable."""
        zones = sorted((self.root / "sys/class/thermal").glob("thermal_zone*"), key=lambda z: int(z.name[12:] or 0))
        zone = next((z for z in zones if "cpu" in (_read(z / "type") or "").lower()), zones[0] if zones else None)
        mc = _int(zone / "temp") if zone else None
        return round(mc / 1000, 1) if mc is not None and mc > -40000 else None

    def battery_temp_c(self):
        """The battery's temperature in C: the power-supply node, or Termux:API's termux-battery-status on the phone."""
        tenths = _int(self.root / "sys/class/power_supply/battery/temp")
        if tenths is not None:
            return tenths / 10
        if self.platform == "m34" and shutil.which("termux-battery-status"):
            try:  # it hangs when the Termux:API app is missing, hence the timeout
                out = subprocess.run(["termux-battery-status"], capture_output=True, text=True, timeout=10).stdout
                return json.loads(out).get("temperature")
            except (OSError, ValueError, subprocess.SubprocessError):
                return None
        return None

    def _jetson(self):
        sys.path.insert(0, str(HERE))
        import jetson_probe  # beside this file; stdlib only

        return jetson_probe

    def board_state(self):
        """On a Jetson: the power mode and clocks each gate records. None elsewhere."""
        if self.platform != "jetson":
            return None
        b = self._jetson().board_state()
        return {k: b.get(k) for k in ("nvpmodel_mode", "highest_mode", "jetson_clocks", "gpu_clocks", "cpu_clocks", "emc_rate_hz", "thermal")}

    def setup_facts(self):
        """What the setup checks read, and the versions and OS each invocation records."""
        cpus = sorted((self.root / "sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor"), key=lambda f: int(f.parts[-3][3:]))
        proc = self.root / "proc"
        comms = {_read(f) for f in proc.glob("[0-9]*/comm")} if (proc / "self").exists() else None
        units = None
        if shutil.which("systemctl"):
            try:  # one line per unit; a unit that is not installed is "inactive"
                out = subprocess.run(["systemctl", "is-active", *UPDATE_UNITS], capture_output=True, text=True, timeout=20).stdout.split()
                units = dict(zip(UPDATE_UNITS, out)) if len(out) == len(UPDATE_UNITS) else None
            except (OSError, subprocess.SubprocessError):
                pass
        os_release = dict(line.split("=", 1) for line in (_read(self.root / "etc/os-release") or "").splitlines() if "=" in line)
        facts = {"governors": [_read(f) for f in cpus] or None, "desktop": sorted(DESKTOP & comms) if comms is not None else None,
                 "update_units": units, "packages": {d: _version(d) for d in (*PI_PINS, "opencv-python")},
                 "python": pyplatform.python_version(), "os": os_release.get("PRETTY_NAME", "").strip('"') or pyplatform.platform(),
                 "kernel": pyplatform.release(), "rpi_issue": ((_read(self.root / "etc/rpi-issue") or "").splitlines() or [None])[0],
                 "device_model": (_read(self.root / "proc/device-tree/model") or "").rstrip("\x00") or None}
        if self.platform == "jetson":
            facts["jetson"] = self._jetson().board_state()
        return facts

    def bench(self, spec, out_dir):
        """One studybench.py run into out_dir, its output shown and kept in bench.log: {exit, meta (its meta.json), tail}."""
        cmd = [sys.executable, "-u", str(HERE / "studybench.py"), "--artifact", spec["artifact"], "--runtime", spec["runtime"],
               "--imgsz", str(spec["imgsz"]), "--images", spec["images"], "--out", str(out_dir), "--threads", str(spec["threads"]),
               "--warmup", str(spec["warmup"]), "--platform-tag", spec["platform"], *(["--limit", str(spec["limit"])] if spec["limit"] else [])]
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        tail = collections.deque(maxlen=20)
        with open(Path(out_dir) / "bench.log", "w", encoding="utf-8") as log:
            log.write(" ".join(cmd) + "\n")
            child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                     env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            try:
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    print("    " + line, end="", flush=True)
                    tail.append(line.rstrip())
                code = child.wait()
            except BaseException:  # Ctrl-C reaches the child too: let it record "interrupted" in its meta.json, then make sure it is gone
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        try:
            meta = json.loads((Path(out_dir) / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = None
        return {"exit": code, "meta": meta, "tail": "\n".join(tail)}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("plan", help="the platform plan, a JSON file (PLATFORMS.md, \"Running a session\")")
    p.add_argument("--out", help="the session directory (default bench/<platform>/<plan file name>); the same --out resumes")
    p.add_argument("--dry-run", action="store_true", help="a plumbing check: the gate waits 5 s at most; nothing from it is reported")
    p.add_argument("--override-checks", metavar="REASON", help="run although a Pi setup check failed; the reason is recorded")
    p.add_argument("--needs", action="store_true", help="only print the paths the plan needs on the platform, one per line")
    p.add_argument("--check", action="store_true", help="only run the setup checks and print them; exit 1 if one fails")
    a = p.parse_args()
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    checked = load_plan(plan)
    if a.needs:
        print("\n".join(needed_paths(plan)))
        return
    if a.check:
        checks = setup_report(plan, Platform(checked["platform"]))
        failed = any(c["status"] == "fail" for c in checks)
        for c in checks:
            print(f"{c['status']:>10}  {c['check']}: {c['value']}" + (f"\n            fix: {c['fix']}" if c["status"] == "fail" else ""))
        print(f"{checked['platform']}: " + ("no setup checks" if not checks else "a setup check FAILED" if failed else "setup checks passed"))
        sys.exit(1 if failed else 0)
    missing = [x for x in needed_paths(plan) if not Path(x).exists()]
    if missing:
        sys.exit(f"missing on this platform (paths are relative to {Path.cwd()}): {missing}" + (
            f"\nthe drift check runs the reference cell (YOLO11s baseline, ONNX Runtime FP32, 640) at the start and the end of "
            f"every session: copy {REFERENCE} here, and install onnxruntime" if REFERENCE in missing else ""))
    out = Path(a.out) if a.out else Path("bench") / checked["platform"] / Path(a.plan).stem
    try:
        run_session(plan, out, Platform(plan.get("platform")), a.dry_run, a.override_checks)
    except KeyboardInterrupt:
        sys.exit(f"interrupted: run the same command again to resume the session in {out}")


if __name__ == "__main__":
    main()
