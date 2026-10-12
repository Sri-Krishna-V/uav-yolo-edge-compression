"""Fingerprint the raw VisDrone2019-DET zips BEFORE Ultralytics converts them.

  python fingerprint_visdrone.py                       # fingerprint zips already in place
  python fingerprint_visdrone.py --download --convert  # a fresh machine (Kaggle): fetch, fingerprint, convert, verify

The Ultralytics VisDrone.yaml deletes the raw annotation folders after conversion, so the raw
row counts can only be taken from the zips. Writes verification/visdrone_fingerprint.json and
exits non-zero if any check fails, including a zip whose SHA-256 differs from INVARIANTS.md section 2.

--convert runs the Ultralytics conversion once, here, and checks the YOLO labels row for row against
the zips. Two training processes started together on an unconverted dataset would both convert it.
"""

import argparse
import hashlib
import json
import platform
import sys
import urllib.request
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path

import studyenv  # noqa: F401  must precede ultralytics; also decides where the dataset lives

ROOT = studyenv.ROOT
DS = studyenv.VISDRONE_DIR
OUT = ROOT / "verification" / "visdrone_fingerprint.json"
MIRROR = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"  # the URL list in Ultralytics' VisDrone.yaml

SPLITS = {"train": "VisDrone2019-DET-train.zip", "val": "VisDrone2019-DET-val.zip", "test-dev": "VisDrone2019-DET-test-dev.zip"}
SPLIT_TO_FINGERPRINT = {"train": "train", "val": "val", "test": "test-dev"}  # Ultralytics split name -> zip split

# The download of 2026-09-12 (INVARIANTS.md section 2). A different hash is different data: stop.
EXPECTED_SHA256 = {
    "train": "86a77eba93137bfc16e4993860de9245b0675c0dba0d3ab98fb458699e256f84",
    "val": "abeea063037e5d20398837deb11084e652402a34ddf4f207bdf541a6f2a35ef9",
    "test-dev": "b28a94b06dfd9e36ce77ff8155fb82b9d2c030f198a76105933bd56c6ea6a68d",
}

# Independent counts from findings/02-visdrone.md (research download, 2026-09-12).
EXPECTED_IMAGES = {"train": 6471, "val": 548, "test-dev": 1610}
EXPECTED_IGNORED = {"train": (8813, 1532), "val": (1378, 32), "test-dev": (2180, 265)}  # (cat 0, cat 11), all score 0
EXPECTED_TESTDEV_CLASSES = [21006, 6376, 1302, 28074, 5771, 2659, 530, 599, 2940, 5845]  # categories 1..10


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(split, zip_path):
    crosstab = Counter()  # (score, category) -> rows
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        images = [n for n in names if n.endswith(".jpg")]
        annotations = [n for n in names if n.endswith(".txt") and "/annotations/" in n]
        for n in annotations:
            # Mirror visdrone2yolo exactly: file.read().strip().splitlines(), split on ","
            for row in [x.split(",") for x in z.read(n).decode("utf-8").strip().splitlines()]:
                crosstab[(row[4], int(row[5]))] += 1

    rows = sum(crosstab.values())
    kept = sum(c for (score, _), c in crosstab.items() if score != "0")  # rows the converter keeps
    per_class_kept = [sum(c for (s, cat), c in crosstab.items() if s != "0" and cat == k) for k in range(1, 11)]
    score0_cat0 = crosstab[("0", 0)]
    score0_cat11 = crosstab[("0", 11)]
    score0_evaluated = sum(c for (s, cat), c in crosstab.items() if s == "0" and 1 <= cat <= 10)
    kept_cat0_or_11 = sum(c for (s, cat), c in crosstab.items() if s != "0" and cat in (0, 11))

    return {
        "zip": zip_path.name,
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": sha256(zip_path),
        "images": len(images),
        "annotation_files": len(annotations),
        "raw_rows": rows,
        "rows_kept_by_converter": kept,
        "kept_per_class_1_to_10": per_class_kept,
        "score0_cat0": score0_cat0,
        "score0_cat11": score0_cat11,
        "score0_with_evaluated_category": score0_evaluated,
        "kept_rows_with_cat0_or_11": kept_cat0_or_11,
        "score_values": sorted({s for s, _ in crosstab}),
    }


def download_missing():
    """Fetch any zip not already in place. A partial download never sits under the final name."""
    DS.mkdir(parents=True, exist_ok=True)
    for name in SPLITS.values():
        dest = DS / name
        if dest.exists():
            print(f"{name}: already present", flush=True)
            continue
        part = dest.with_suffix(".zip.part")
        print(f"{name}: downloading from {MIRROR}", flush=True)
        with urllib.request.urlopen(MIRROR + name, timeout=60) as r, open(part, "wb") as f:
            total, done, shown = int(r.headers.get("Content-Length", 0)), 0, -1
            while chunk := r.read(1 << 22):
                f.write(chunk)
                done += len(chunk)
                pct = 100 * done // total if total else 0
                if pct // 20 != shown:
                    shown = pct // 20
                    print(f"  {pct}% of {total / 2**20:.0f} MiB", flush=True)
        part.rename(dest)


def verify_conversion(failures):
    """Convert with Ultralytics (once), then check the YOLO labels against the raw-zip fingerprint, row for row."""
    from ultralytics.data.utils import check_det_dataset

    fp_file = OUT
    if not fp_file.exists():
        sys.exit("Run fingerprint_visdrone.py first: conversion deletes the raw annotations.")
    fp = json.loads(fp_file.read_text(encoding="utf-8"))
    if not fp["checks_passed"]:
        sys.exit("Raw-zip fingerprint failed its checks; refusing to convert a suspect download.")
    if fp.get("host") != platform.node():
        # The repo ships the laptop's record, so a stale file must never vouch for this machine's download.
        sys.exit(f"The fingerprint record is from {fp.get('host')!r}, not this machine. Run fingerprint_visdrone.py here first.")

    data = check_det_dataset(studyenv.DATA)  # first call converts, reusing the already-downloaded zips
    base = Path(data["path"])
    report = {}
    for split, fp_split in SPLIT_TO_FINGERPRINT.items():
        raw = fp["splits"][fp_split]
        images = len(list((base / "images" / split).glob("*.jpg")))
        label_files = list((base / "labels" / split).glob("*.txt"))
        classes = Counter()
        max_objects = 0
        for f in label_files:
            lines = [line for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
            max_objects = max(max_objects, len(lines))
            classes.update(int(line.split()[0]) for line in lines)
        per_class = [classes[k] for k in range(10)]
        report[split] = {"images": images, "label_files": len(label_files), "label_rows": sum(classes.values()), "per_class": per_class, "max_objects_per_image": max_objects}
        if images != raw["images"] or len(label_files) != raw["annotation_files"]:
            failures.append(f"{split}: {images} images / {len(label_files)} label files, raw zip had {raw['images']}")
        if per_class != raw["kept_per_class_1_to_10"]:
            failures.append(f"{split}: converted per-class rows {per_class} != raw kept rows {raw['kept_per_class_1_to_10']}")
        if set(classes) - set(range(10)):
            failures.append(f"{split}: out-of-range class ids {sorted(set(classes) - set(range(10)))}")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true", help=f"fetch missing zips from {MIRROR}")
    p.add_argument("--convert", action="store_true", help="after a passing fingerprint, convert once and verify the labels")
    a = p.parse_args()

    if a.download:
        download_missing()
    missing = [name for name in SPLITS.values() if not (DS / name).exists()]
    if missing:
        sys.exit(f"Missing in {DS}: {missing}. Run with --download, or copy the zips there.")

    converted = (DS / "images").exists()
    if converted and not (DS / "VisDrone2019-DET-train").exists():
        # Conversion already ran here (it deletes the raw folders); the zips are still fingerprinted below.
        print(f"{DS} is already converted; fingerprinting the zips only.", flush=True)

    result = {"fingerprinted_on": date.today().isoformat(), "host": platform.node(), "zip_dir": str(DS), "splits": {}}
    failures = []
    for split, name in SPLITS.items():
        fp = fingerprint(split, DS / name)
        result["splits"][split] = fp
        if fp["zip_sha256"] != EXPECTED_SHA256[split]:
            failures.append(f"{split}: SHA-256 {fp['zip_sha256']} is not the study's download ({EXPECTED_SHA256[split]}). Different data: stop.")
        if fp["images"] != EXPECTED_IMAGES[split] or fp["annotation_files"] != EXPECTED_IMAGES[split]:
            failures.append(f"{split}: {fp['images']} images / {fp['annotation_files']} annotation files, expected {EXPECTED_IMAGES[split]}")
        if (fp["score0_cat0"], fp["score0_cat11"]) != EXPECTED_IGNORED[split]:
            failures.append(f"{split}: ignored/others rows {(fp['score0_cat0'], fp['score0_cat11'])}, expected {EXPECTED_IGNORED[split]}")
        if fp["score0_with_evaluated_category"] or fp["kept_rows_with_cat0_or_11"]:
            failures.append(f"{split}: score==0 is NOT exactly category in {{0, 11}} - converter would emit out-of-range labels")
    if result["splits"]["test-dev"]["kept_per_class_1_to_10"] != EXPECTED_TESTDEV_CLASSES:
        failures.append(f"test-dev per-class counts differ: {result['splits']['test-dev']['kept_per_class_1_to_10']}")

    result["checks_passed"] = not failures
    result["failures"] = failures
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"checks_passed": result["checks_passed"], "failures": failures, "written": str(OUT)}, indent=2))
    if failures:
        sys.exit(1)

    if a.convert:
        conv_failures = []
        report = verify_conversion(conv_failures)
        print(json.dumps({"conversion_checks_passed": not conv_failures, "failures": conv_failures, "rows": {k: v["label_rows"] for k, v in report.items()}}, indent=2))
        if conv_failures:
            sys.exit(1)


if __name__ == "__main__":
    main()
