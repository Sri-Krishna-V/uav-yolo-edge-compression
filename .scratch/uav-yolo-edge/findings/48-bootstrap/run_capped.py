"""Run a probe (e.g. probe_real.py) as a child of the repo's studycap Job Object (cap GB, timeout min). Usage:
  run_capped.py CAP_GB TIMEOUT_MIN probe_real.py args...
"""
import sys, time
from pathlib import Path
sys.dont_write_bytecode = True  # leave study/scripts free of __pycache__
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "study" / "scripts"))  # the repo's study/scripts
import studycap
cap, tmo, script, *rest = sys.argv[1:]
t = time.time()
code = studycap.run([script, *rest], "bootstrap probe", cap_gb=float(cap), timeout_min=float(tmo))
print(f"[run_capped] exit {code} after {time.time() - t:.0f} s")
sys.exit(code)
