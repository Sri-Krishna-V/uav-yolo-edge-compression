"""Check the laptop's memory cap (#20): a capped step that goes over its cap or its timeout is stopped and says so, and
off Windows the cap does nothing. Standard library only; the stopping checks need Windows and allocate at most 1 GB:

  python study/scripts/tests/test_studycap.py
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import studycap  # noqa: E402

FAILED = []


def check(label, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        FAILED.append(label)
        if detail:
            print(f"     {detail}")


def stopped(argv, **kw):
    """Run a capped child: (the Stopped message or None, the exit code or None, seconds)."""
    t = time.time()
    try:
        return None, studycap.run(argv, "the test step", **kw), time.time() - t
    except studycap.Stopped as e:
        return str(e), None, time.time() - t


def restore(saved):
    """Put environment variables back as they were: {name: value, or None for unset}."""
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


ALLOC_1GB = ["-c", "b = bytearray(1 << 30); import time; time.sleep(30)"]

# ---------------------------------------------------------------- 1: over the cap
if sys.platform == "win32":
    msg, code, s = stopped(ALLOC_1GB, cap_gb=0.25, timeout_min=1)
    check("1 a child that goes over its cap is stopped", msg is not None, f"exit {code}")
    check("1 the report names the step and the cap", msg is not None and "the test step" in msg and "0.25 GB memory cap" in msg, msg)
    check("1 it is stopped at once, not at the timeout", s < 20, f"{s:.1f} s")

    # pnnx, or any process a step starts, is in the same job: the cap is on all of them together
    msg, code, s = stopped(["-c", f"import subprocess, sys; sys.exit(subprocess.run([sys.executable, *{ALLOC_1GB!r}]).returncode)"], cap_gb=0.25, timeout_min=1)
    check("1 a process the child starts is under the same cap", msg is not None and "0.25 GB memory cap" in msg, msg or f"exit {code}")

    # ---------------------------------------------------------------- 2: over the timeout, or the laptop running out of free RAM
    msg, code, s = stopped(["-c", "import time; time.sleep(60)"], cap_gb=1, timeout_min=0.05)
    check("2 a child that runs past its timeout is stopped", msg is not None and "the test step" in msg and "timeout" in msg, msg or f"exit {code}")
    check("2 it is stopped at the timeout, not when it ends", s < 15, f"{s:.1f} s")

    saved = {"STUDY_RAM_FLOOR_GB": os.environ.get("STUDY_RAM_FLOOR_GB")}
    os.environ["STUDY_RAM_FLOOR_GB"] = "100000"  # more than any laptop has free: the floor is crossed at once
    msg, code, s = stopped(["-c", "import time; time.sleep(60)"], cap_gb=1, timeout_min=1)
    restore(saved)
    check("2 a child is stopped when the laptop's free RAM falls below the floor", msg is not None and "free RAM" in msg, msg or f"exit {code}")
    check("2 it is stopped at once", s < 15, f"{s:.1f} s")

    # ---------------------------------------------------------------- 3: a step that stays inside its limits is untouched
    msg, code, s = stopped(["-c", "import sys; print('child output'); sys.exit(3)"], cap_gb=1, timeout_min=1)
    check("3 a child inside its limits keeps its exit code", msg is None and code == 3, msg or f"exit {code}")

    # a capped step runs its own steps inside its cap: it must not start another, or a script that caps itself never ends
    here = str(Path(studycap.__file__).parent)
    check("3 outside a capped step, the cap applies", studycap.applies())
    msg, code, s = stopped(["-c", f"import sys; sys.path.insert(0, {here!r}); import studycap; sys.exit(7 if studycap.applies() else 0)"], cap_gb=1, timeout_min=1)
    check("3 inside a capped step, the cap does not apply again", msg is None and code == 0, msg or f"exit {code}")
    msg, code, s = stopped(["-c", f"import sys; sys.path.insert(0, {here!r}); import studycap; "
                                  f"sys.exit(studycap.run({ALLOC_1GB!r}, 'inner', cap_gb=100, timeout_min=1))"], cap_gb=0.25, timeout_min=1)
    check("3 a step started inside a capped step is held by the outer cap", msg is not None and "the test step" in msg and "0.25 GB" in msg, msg or f"exit {code}")
else:
    print("skip 1-3: the cap acts only on Windows")

# ---------------------------------------------------------------- 4: off Windows the cap does nothing
on_windows, studycap.WINDOWS = studycap.WINDOWS, False  # what the Pis, the Jetson and Kaggle see
check("4 off Windows the cap does not apply", not studycap.applies())
msg, code, s = stopped(["-c", "b = bytearray(1 << 29); import time; time.sleep(4)"], cap_gb=0.1, timeout_min=0.01)
check("4 off Windows a child over its cap and its timeout runs to the end", msg is None and code == 0 and s >= 4, msg or f"exit {code}, {s:.1f} s")
msg, code, s = stopped(["-c", "import sys; sys.exit(3)"], cap_gb=0.1, timeout_min=0.01)
check("4 off Windows the child keeps its exit code", msg is None and code == 3, msg or f"exit {code}")
studycap.WINDOWS = on_windows

# ---------------------------------------------------------------- 5: the defaults, and the flag or variable that changes them
saved = {k: os.environ.pop(k, None) for k in ("STUDY_MEM_CAP_GB", "STUDY_TIMEOUT_MIN")}
check("5 the defaults are a 4 GB cap and a 60-minute timeout", studycap.limits() == (4.0, 60.0), studycap.limits())
os.environ.update(STUDY_MEM_CAP_GB="2.5", STUDY_TIMEOUT_MIN="90")
check("5 the environment variables change both", studycap.limits() == (2.5, 90.0), studycap.limits())
p = argparse.ArgumentParser()
studycap.add_arguments(p)
studycap.use(p.parse_args(["--mem-cap-gb", "0.25", "--timeout-min", "0.05"]))
check("5 the flags change both, over the variables", studycap.limits() == (0.25, 0.05), studycap.limits())
studycap.use(p.parse_args([]))
check("5 no flag leaves them as they were", studycap.limits() == (0.25, 0.05), studycap.limits())
if sys.platform == "win32":
    msg, code, s = stopped(ALLOC_1GB)
    check("5 run() uses them when the caller names none", msg is not None and "0.25 GB memory cap" in msg, msg or f"exit {code}")
restore(saved)

print(f"\n{len(FAILED)} failed" if FAILED else "\nall passed")
sys.exit(1 if FAILED else 0)
