"""Import this module BEFORE ultralytics in every study script.

- Isolates Ultralytics settings to study/.ultralytics, so datasets/runs/weights never land in
  whatever project last wrote the machine-global settings.json.
- Disables Ultralytics runtime auto-install, so a missing package fails loudly instead of
  silently mutating the pinned environment mid-run.
- Refuses to run below ultralytics 8.4.133, where the area-split metrics do not exist.
- Exposes the invariants from cfg/invariants.yaml as ready-to-splat argument dicts.
"""

import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / ".ultralytics"
DATA = "VisDrone.yaml"

os.environ["YOLO_CONFIG_DIR"] = str(CONFIG_DIR)
os.environ["YOLO_AUTOINSTALL"] = "false"

_floor = (8, 4, 133)
_installed = tuple(int(p) for p in version("ultralytics").split(".")[:3])
if _installed < _floor:
    sys.exit(f"ultralytics {version('ultralytics')} < 8.4.133: AP-small/medium/large would be missing. See INVARIANTS.md.")

WANTED_SETTINGS = {
    "datasets_dir": str(ROOT / "datasets"),
    "weights_dir": str(ROOT / "weights"),
    "runs_dir": str(ROOT / "runs"),
    "sync": False,
}

# Ultralytics binds DATASETS_DIR etc. at import time, so settings must be correct on disk before
# the first import in this process - write them from a throwaway child process.
_settings_file = CONFIG_DIR / "Ultralytics" / "settings.json"  # Ultralytics appends "Ultralytics" to YOLO_CONFIG_DIR
_current = json.loads(_settings_file.read_text(encoding="utf-8")) if _settings_file.exists() else {}
if any(_current.get(k) != v for k, v in WANTED_SETTINGS.items()):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-c", f"from ultralytics import settings; settings.update({WANTED_SETTINGS!r})"], check=True, env=os.environ.copy())

import yaml  # noqa: E402  (after env setup on purpose)

INVARIANTS = yaml.safe_load((ROOT / "cfg" / "invariants.yaml").read_text(encoding="utf-8"))
TRAIN_ARGS = {**INVARIANTS["shared"], **INVARIANTS["train"]}
EVAL_ARGS = {**INVARIANTS["shared"], **{k: v for k, v in INVARIANTS["eval"].items() if k != "coco_max_dets"}}
COCO_MAX_DETS = list(INVARIANTS["eval"]["coco_max_dets"])
AREA_KEYS = ["metrics/mAP_small(B)", "metrics/mAP_medium(B)", "metrics/mAP_large(B)"]
