"""Check the CPU clock the bench logs (#28): on a Pi, the clock the firmware really runs at (vcgencmd measure_clock arm),
not Linux's scaling_cur_freq, which a firmware throttle leaves at 2400 MHz. Runs studybench.py end to end, on a fake
runtime and a fake vcgencmd, on any machine:

  python study/scripts/tests/test_cpu_clock.py
"""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import studybench as sb  # noqa: E402

FAILED = []


def check(label, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        FAILED.append(label)
        if detail:
            print(f"     {detail}")


class FakeRuntime:
    """A 64 px model whose raw head finds nothing: the bench's plumbing without a real runtime."""

    def __init__(self, model, threads):
        self.input_shape, self.version, self.info, self.precision = [1, 3, 64, 64], "fake", {}, "FP32"

    def __call__(self, x):
        return np.zeros((1, 14, 84), np.float32)


class FakeVcgencmd:
    """subprocess.run for `vcgencmd`, answering with what a Pi 5 prints (a Pi 4 with index 48); every other command is
    refused. With hz None, measure_clock prints nothing, as for a user outside the video group."""

    def __init__(self, hz, index=0):
        self.hz, self.index, self.calls = hz, index, []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd[1:])
        if Path(cmd[0]).name != "vcgencmd" or tuple(cmd[1:]) not in (("measure_clock", "arm"), ("get_throttled",)):
            raise FileNotFoundError(cmd[0])
        if cmd[1] != "measure_clock":
            out = "throttled=0x0\n"
        else:
            out = "" if self.hz is None else f"frequency({self.index})={next(self.hz)}\n"
        return subprocess.CompletedProcess(cmd, 0, out, "")


def bench(*args, vcgencmd=None):
    """studybench.py's main() on 4 tiny images and the fake runtime, on a Pi when vcgencmd is given -> (out dir, meta, rows)."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "images").mkdir()
    for k in range(4):
        cv2.imwrite(str(tmp / "images" / f"{k:04d}.jpg"), np.full((48, 64, 3), 60 * k, np.uint8))
    (tmp / "model.onnx").write_bytes(b"fake")
    saved = sb.RUNTIMES["onnxruntime"], sb.shutil.which, sb.subprocess.run, sys.argv
    sb.RUNTIMES["onnxruntime"] = FakeRuntime
    if vcgencmd:
        sb.shutil.which = lambda name, *a, **kw: "/usr/bin/vcgencmd" if name == "vcgencmd" else None
        sb.subprocess.run = vcgencmd
    sys.argv = ["studybench.py", "--artifact", str(tmp / "model.onnx"), "--runtime", "onnxruntime", "--imgsz", "64",
                "--images", str(tmp / "images"), "--out", str(tmp / "out"), "--warmup", "1", *args]
    try:
        sb.main()
    finally:
        sb.RUNTIMES["onnxruntime"], sb.shutil.which, sb.subprocess.run, sys.argv = saved
    out = tmp / "out"
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    rows = []
    if (out / "timings.csv").exists():
        with open(out / "timings.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    return out, meta, rows


def hz(*mhz):
    """A fake clock that gives these readings, in MHz, and then the last one forever."""
    def gen():
        yield from (m * 1_000_000 for m in mhz)
        while True:
            yield mhz[-1] * 1_000_000
    return gen()


# ---------------------------------------------------------------- 1: a latency run on a Pi logs the firmware's clock
vc = FakeVcgencmd(hz(2400, 1500, 1500, 1800, 1500))
out, meta, rows = bench(vcgencmd=vc)
check("1 on a Pi, every image's clock comes from vcgencmd measure_clock arm: a throttled 1500 MHz reaches timings.csv",
      [r["cpu_mhz"] for r in rows] == ["2400", "1500", "1500", "1800"], [r["cpu_mhz"] for r in rows])
check("1 meta.json names the clock's source", meta["cpu_clock"]["source"] == "vcgencmd measure_clock arm", meta.get("cpu_clock"))
check("1 meta.json gives the clock's min, median and max over the run",
      meta["device_state"]["cpu_mhz"] == {"min": 1500.0, "max": 2400.0, "median": 1650.0}, meta["device_state"]["cpu_mhz"])
check("1 meta.json gives what one vcgencmd call costs, from every call", meta["cpu_clock"]["call_ms"]["n"] == 4
      and meta["cpu_clock"]["call_ms"]["median"] >= 0, meta["cpu_clock"].get("call_ms"))

out, meta, rows = bench(vcgencmd=FakeVcgencmd(hz(1500), index=48))
check("1 a Pi 4's vcgencmd prints frequency(48)=...: read the same way", [r["cpu_mhz"] for r in rows] == ["1500"] * 4,
      [r["cpu_mhz"] for r in rows])
out, meta, rows = bench(vcgencmd=FakeVcgencmd(None))
check("1 vcgencmd that cannot read the clock: no clock, and meta.json counts the failed reads",
      [r["cpu_mhz"] for r in rows] == [""] * 4 and meta["cpu_clock"]["failed_reads"] == 4 and meta["device_state"]["cpu_mhz"] is None,
      ([r["cpu_mhz"] for r in rows], meta["cpu_clock"]))

# ---------------------------------------------------------------- 2: off a Pi, scaling_cur_freq as before
saved = sb.cpu_mhz
sb.cpu_mhz = lambda: 3800
try:
    out, meta, rows = bench()
finally:
    sb.cpu_mhz = saved
check("2 off a Pi, the clock is scaling_cur_freq, as before", [r["cpu_mhz"] for r in rows] == ["3800"] * 4
      and meta["device_state"]["cpu_mhz"] == {"min": 3800.0, "max": 3800.0, "median": 3800.0}, ([r["cpu_mhz"] for r in rows], meta["device_state"]))
check("2 ... meta.json names it, and there is no vcgencmd call to cost", meta["cpu_clock"]["source"] == "scaling_cur_freq (cpu0)"
      and meta["cpu_clock"]["call_ms"] is None, meta["cpu_clock"])

# ---------------------------------------------------------------- 3: an energy run records the clock too
vc = FakeVcgencmd(hz(2400, 1500))
out, meta, rows = bench("--seconds", "0.3", "--limit", "2", vcgencmd=vc)
check("3 an energy run reads vcgencmd's clock", meta["cpu_clock"]["source"] == "vcgencmd measure_clock arm"
      and meta["cpu_clock"]["call_ms"]["n"] >= 2 and "frame" in meta["cpu_clock"]["read"], meta.get("cpu_clock"))
check("3 ... and gives its min, median and max over the run, a throttled 1500 MHz included",
      meta.get("device_state", {}).get("cpu_mhz", {}).get("min") == 1500.0 and meta["device_state"]["cpu_mhz"]["max"] == 2400.0,
      meta.get("device_state"))

c = sb.energy_loop(["a"], lambda im: None, 120, clock=iter(range(1000)).__next__, throttle=lambda: None, mhz=lambda: 1500)
check("3 the energy loop reads the clock where it reads the throttle flags: at the start, every 50 frames and the end",
      [i for i, _ in c["mhz_samples"]] == [0, 50, 100, c["frames"]] == [i for i, _ in c["throttle_samples"]], c)

if FAILED:
    sys.exit(f"{len(FAILED)} failed: {FAILED}")
print("all passed")
