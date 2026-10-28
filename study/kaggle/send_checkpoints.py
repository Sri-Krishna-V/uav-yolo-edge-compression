"""Find finished runs' best.pt on this computer and pack them into one tar to send to the teammate (#37).

  python study/kaggle/send_checkpoints.py            # the four YOLO26s runs the Pi 5 matrix still needs
  python study/kaggle/send_checkpoints.py RUN...     # other runs, by run name (e.g. yolo26s-640-s0-distill)

Where it looks, in order: your checkpoint folder (KAGGLE_MIRROR in study/kaggle/.env.kaggle, where `wizard.sh next`
copies every checkpoint; else ~/uav-yolo-checkpoints), then Downloads and Desktop. A file counts as a run's checkpoint
only if it was trained under the run's name: an Ultralytics checkpoint keeps its training arguments, `name` among them,
in its data.pkl. That is read as bytes, so neither torch nor ultralytics is needed, and nothing in the file runs.
A checkpoint saved as best.pt.zip (some browser downloads) is the same file under another name.

It writes ~/checkpoints-for-krishna.tar: <run>/best.pt for each run found, and SHA256SUMS.txt.
Stdlib only.
"""

import argparse
import hashlib
import io
import itertools
import re
import sys
import tarfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # study/
ENV_FILE = ROOT / "kaggle" / ".env.kaggle"
MISSING = ["yolo26s-640-s0-prune30", "yolo26s-960-s0-prune30", "yolo26s-640-s0-distill", "yolo26s-640-s0-prune30-distill"]


def mirror():
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if line.startswith("KAGGLE_MIRROR="):
                return Path(line.split("=", 1)[1].strip().strip("'\"")).expanduser()
    return Path.home() / "uav-yolo-checkpoints"


# The training argument `name` as torch.save pickles it: the key, an optional memo opcode, then the value's length
NAME_ARG = re.compile(rb"X\x04\x00\x00\x00name(?:q.|r....)?X(....)", re.DOTALL)


def names_inside(path):
    """Every string stored under a `name` key in a checkpoint: the run it was trained under, and its layers' class
    names. Other run names appear in it too (a pruned run's source, a teacher's path), so only `name` values count.
    Empty when the file is not a torch checkpoint."""
    try:
        with zipfile.ZipFile(path) as z:
            pkl = next((n for n in z.namelist() if n.endswith("/data.pkl")), None)
            data = z.read(pkl) if pkl else b""
    except (zipfile.BadZipFile, OSError):
        return set()
    return {data[m.end():m.end() + int.from_bytes(m.group(1), "little")].decode(errors="replace") for m in NAME_ARG.finditer(data)}


def find(run, home):
    """The first checkpoint of RUN: the usual places first, then a search of the whole folders."""
    usual = [home / run / n for n in ("best.pt", "best.pt.zip", "weights/best.pt")]
    roots = [home, Path.home() / "Downloads", Path.home() / "Desktop"]
    searched = (p for r in roots if r.is_dir() for p in r.rglob("best.pt*") if p.suffix in (".pt", ".zip"))
    for p in itertools.chain(usual, searched):
        if p.is_file() and run in names_inside(p):
            return p
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", default=MISSING, help="run names (default: the four the Pi 5 matrix needs)")
    p.add_argument("--out", default=str(Path.home() / "checkpoints-for-krishna.tar"))
    a = p.parse_args()
    home = mirror()
    print(f"Your checkpoint folder: {home}{'' if home.is_dir() else '  (it does not exist)'}\n")
    found = {}
    for run in a.runs:
        f = find(run, home)
        print(f"  {'found  ' if f else 'MISSING'} {run}" + (f"  <- {f} ({f.stat().st_size / 2**20:.1f} MB)" if f else ""))
        if f:
            found[run] = f
    if found:
        sums = []
        with tarfile.open(a.out, "w") as tar:
            for run, f in found.items():
                sums.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {run}/best.pt")
                tar.add(f, arcname=f"{run}/best.pt")
            body = ("\n".join(sums) + "\n").encode()
            info = tarfile.TarInfo("SHA256SUMS.txt")
            info.size, info.mtime = len(body), time.time()
            tar.addfile(info, io.BytesIO(body))
        print(f"\nWrote {a.out} ({Path(a.out).stat().st_size / 2**20:.0f} MB, {len(found)} of {len(a.runs)} runs).")
        print("Upload it to Google Drive and send Krishna the link.")
    missing = [r for r in a.runs if r not in found]
    if missing:
        print(f"\nNot found: {', '.join(missing)}. They are in your last Kaggle session's output. Either:")
        print("  - run  bash study/kaggle/wizard.sh next  (it copies your last session's checkpoints into your")
        print("    checkpoint folder), then run this again; or")
        print("  - on kaggle.com, open your newest notebook version -> Output -> bring-home/checkpoints/<run>/best.pt,")
        print("    download each one into Downloads, then run this again.")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
