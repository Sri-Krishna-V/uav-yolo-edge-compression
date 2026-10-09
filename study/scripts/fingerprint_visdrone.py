"""Fingerprint the raw VisDrone2019-DET zips BEFORE Ultralytics converts them.

The Ultralytics VisDrone.yaml deletes the raw annotation folders after conversion, so the raw
row counts can only be taken from the zips. Writes verification/visdrone_fingerprint.json and
exits non-zero if any check fails.
"""

import hashlib
import json
import sys
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "datasets" / "VisDrone"
OUT = ROOT / "verification" / "visdrone_fingerprint.json"

SPLITS = {"train": "VisDrone2019-DET-train.zip", "val": "VisDrone2019-DET-val.zip", "test-dev": "VisDrone2019-DET-test-dev.zip"}

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


def main():
    result = {"fingerprinted_on": date.today().isoformat(), "splits": {}}
    failures = []
    for split, name in SPLITS.items():
        fp = fingerprint(split, DS / name)
        result["splits"][split] = fp
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
    print(json.dumps(result, indent=2))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
