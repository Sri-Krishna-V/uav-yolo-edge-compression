"""Check studybench's energy run (#26): frames for a fixed time over images decoded beforehand, the window on the
monotonic clock the session's PMIC sampler uses. Runs in the study venv:

  python study/scripts/tests/test_energy_loop.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import studybench as sb  # noqa: E402

FAILED = []


def check(label, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        FAILED.append(label)
        if detail:
            print(f"     {detail}")


class Clock:
    """Time moves only when a frame runs: `per_frame` seconds each."""

    def __init__(self, start=1000.0, per_frame=0.25):
        self.t, self.per_frame, self.seen = start, per_frame, []

    def __call__(self):
        return self.t

    def infer(self, im):
        self.seen.append(im)
        self.t += self.per_frame


# ---------------------------------------------------------------- 1: frames for a fixed time
c = Clock()
r = sb.energy_loop(["a", "b", "c"], c.infer, 2.0, clock=c, throttle=lambda: None)
check("1 the run counts the frames that fit in its time: 2 s at 0.25 s a frame is 8", r["frames"] == 8, r)
check("1 the window is on the clock: from before the first frame to after the last", r["window"] == [1000.0, 1002.0], r["window"])
check("1 the images are cycled, in order", c.seen == ["a", "b", "c", "a", "b", "c", "a", "b"], c.seen)

c = Clock(per_frame=0.3)
r = sb.energy_loop(["a"], c.infer, 1.0, clock=c, throttle=lambda: None)
check("1 the last frame finishes past the time: it counts, and the window ends after it", r["frames"] == 4
      and abs(r["window"][1] - 1001.2) < 1e-9, r)

# ---------------------------------------------------------------- 2: the throttle flags, as in a latency run
c = Clock(per_frame=1 / 64)  # exact in binary, so 120 frames are exactly 1.875 s
calls = []
r = sb.energy_loop(["a"], c.infer, 1.875, clock=c, throttle=lambda: calls.append(1) or 0x50000)
check("2 the throttle state is read at the start, every 50 frames and at the end",
      [i for i, _ in r["throttle_samples"]] == [0, 50, 100, 120] and all(v == 0x50000 for _, v in r["throttle_samples"]), r["throttle_samples"])

if FAILED:
    sys.exit(f"{len(FAILED)} failed: {FAILED}")
print("all passed")
