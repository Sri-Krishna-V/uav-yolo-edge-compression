"""Results C and D (#47), the paper's accuracy and latency tables and the small-object section, from results/cells.csv.
Runs nothing on a Pi and reads nothing but cells.csv (studyresults.py, #44): re-run it after studyresults.py when a session lands.

  python study/scripts/studytables.py [--cells study/results/cells.csv] [--out study/results]

Which cells. studypairs.usable's: every scored cells.csv row, a cell run again in a later session counted once, by its latest run.
INT8. The per-channel cells (onnxruntime-int8pc) are the matrix's ONNX Runtime INT8 cells (INVARIANTS.md section 8): a stock INT8
cell is the headline only while no per-channel cell of the same platform, runtime, checkpoint, input resolution and head is
scored (studypairs.int8_notes), and stays as its own row after. A session that adds per-channel cells moves the headline with no
code change.
Session 6. Its latencies are about 6 % high against the other sessions' (measured by session 7) and stay as measured (#46): its
cells carry a footnote, and a comparison with one cell from it is flagged (studypairs.spans_offset).

Outputs, all under --out:
  tables/accuracy.csv, .tex           configuration x study model at the matrix resolution (640) on the Pi 5, one-to-many head: AP,
                                      AP50, AP-small, AP-medium, AP-large. Baseline, pruned and distilled from the ONNX Runtime FP32
                                      cells, quantized and combined from ONNX Runtime INT8 (headline, then stock)
  tables/latency.csv, .tex            every cell, platform x runtime x model x configuration x input resolution: median latency,
                                      pre, net and post at 0.25 (and at 0.001 in the CSV), precision, the throttle columns of
                                      INVARIANTS.md section 4.7 (images below 2000 MHz, median clock), session 6's footnote. The .tex
                                      is a longtable
  tables/result_c.csv, .tex           result C, YOLO11s against YOLO26s: per platform, configuration, runtime, precision, INT8 recipe,
                                      input resolution and head, YOLO26s's latency over YOLO11s's and the AP and AP-small differences
                                      (YOLO26s minus YOLO11s); clean when neither cell ran an image below 2000 MHz. The .tex holds the
                                      Pi 5's baseline and quantized pairs
  tables/result_d_int8.csv, .tex      result D: per platform, checkpoint, input resolution and head, ONNX Runtime INT8's speed-up over
                                      FP32 (FP32 latency over INT8's) and its AP and AP-small change (INT8 minus FP32, in % of FP32),
                                      per-channel and stock side by side; the Pi 5 and the Galaxy M34
  tables/result_d_runtimes.csv, .tex  result D: the Pi 5's runtimes side by side per checkpoint and input resolution (NCNN FP16,
                                      OpenVINO FP32, ONNX Runtime FP32 and INT8), each with its speed-up over ONNX Runtime FP32
  tables/small_object.csv, .tex       the supporting section: quantized, pruned and combined against the reference baseline on the Pi
                                      5, the relative drop per bucket and the AP-small drop over the AP drop, banded by #9's
                                      pre-registered bar; beside #9's laptop numbers (LAPTOP_9). #9's OpenVINO FP32 reference ran in
                                      bf16 and #23 has not scored it again, so that row is footnoted
  figures/result_c_pi5.pdf, .png      YOLO26s's latency over YOLO11s's against input resolution, one line per runtime and precision
Standard library, numpy and matplotlib only (studynms.py's palette and figure saving, studypairs.py's loading and INT8 rule); no
torch, no ultralytics.
"""

import argparse
from pathlib import Path

import studynms
import studypairs as sp

MATRIX_IMGSZ = 640  # the matrix resolution (GLOSSARY.md)
CONFIGURATIONS = ["baseline", "quantized", "pruned", "distilled", "combined"]  # GLOSSARY.md's order
INT8_CONFIGURATIONS = {"quantized", "combined"}


def _headline(int8, c):
    """(headline, note) of a cell, from studypairs.int8_notes' map: (True, '') for a cell that is not a stock INT8 cell."""
    return int8.get(id(c), (True, ""))


def accuracy_rows(cells, platform="pi5", imgsz=MATRIX_IMGSZ):
    """The accuracy table: configuration x study model at the matrix resolution on one platform, the one-to-many head. Baseline,
    pruned and distilled are the ONNX Runtime FP32 cells; quantized and combined the ONNX Runtime INT8 cells, the headline first
    and each stock cell after its per-channel one. [{cell, headline, note}]."""
    int8 = sp.int8_notes(cells)
    here = [c for c in cells if c["platform"] == platform and c["imgsz"] == imgsz and c["head"] == "one-to-many" and c["runtime"] == "onnxruntime"]
    rows = []
    for model in sorted({c["model"] for c in here}):
        for conf in CONFIGURATIONS:
            precision = "INT8" if conf in INT8_CONFIGURATIONS else "FP32"
            found = [dict(zip(("cell", "headline", "note"), (c, *_headline(int8, c))))
                     for c in here if c["model"] == model and c["configuration"] == conf and c["precision"] == precision]
            rows += sorted(found, key=lambda r: not r["headline"])
    return rows


RECIPES = {"per-channel": "per_channel", "stock": "stock"}  # cells.csv's int8_recipe, and the column prefix


def _platform_first(platform):
    return (platform != "pi5", platform)


def int8_rows(cells):
    """Result D: one row per platform, runtime, checkpoint, input resolution and head with an INT8 cell. Its FP32 cell on the same
    runtime (None when no FP32 export ran, as for the combined checkpoints), its per-channel and stock INT8 cells, and per recipe
    the speed-up (FP32 latency over INT8's), the AP and AP-small change (INT8 minus FP32, and that in % of FP32) and whether the
    two cells span session 6 (studypairs.spans_offset). speedup is the headline one, speedup_recipe its recipe: the per-channel
    speed-up, but the stock one where only the per-channel one spans session 6 (#47's comment of 1 November: at 1120 the
    per-channel cells against session 6's slow FP32 cells give 1.58-1.59x, the stock cells within session 6 1.48-1.49x), and the
    stock one where there is no per-channel cell."""
    key = lambda c: (c["platform"], c["runtime"], c["checkpoint"], c["imgsz"], c["head"])  # noqa: E731
    fp32 = {key(c): c for c in cells if c["precision"] == "FP32"}
    groups = {}
    for c in cells:
        if c["precision"] == "INT8":
            groups.setdefault(key(c), {})[c["int8_recipe"]] = c
    rows = []
    for k, int8 in groups.items():
        fp, first = fp32.get(k), next(iter(int8.values()))
        row = {col: first[col] for col in ("platform", "runtime", "model", "checkpoint", "configuration", "imgsz", "trained_imgsz", "untrained_imgsz", "head")}
        row.update(fp32=fp, note="" if fp else "no FP32 cell of this checkpoint (no FP32 export of it was run)")
        for recipe, name in RECIPES.items():
            x = int8.get(recipe)
            both = fp is not None and x is not None
            row[name] = x
            row[f"{name}_speedup"] = fp["latency_ms"] / x["latency_ms"] if both else None
            for col in ("ap", "ap_small"):
                row[f"{name}_{col}_diff"] = x[col] - fp[col] if both else None
                row[f"{name}_{col}_diff_pct"] = 100 * (x[col] - fp[col]) / fp[col] if both else None
            row[f"{name}_spans_session_6"] = both and sp.spans_offset(fp, x)
        stock_within = row["stock_speedup"] is not None and not row["stock_spans_session_6"]
        recipe = ("per-channel" if row["per_channel_speedup"] is not None and not (row["per_channel_spans_session_6"] and stock_within)
                  else "stock" if row["stock_speedup"] is not None else None)
        row.update(speedup_recipe=recipe, speedup=row[f"{RECIPES[recipe]}_speedup"] if recipe else None)
        rows.append(row)
    return sorted(rows, key=lambda r: (_platform_first(r["platform"]), r["runtime"], r["model"], r["imgsz"], r["checkpoint"], r["head"] != "one-to-many"))


BUCKETS = ["ap", "ap_small", "ap_medium", "ap_large"]
SMALL_OBJECT_SIDE = {"quantized", "pruned", "combined"}  # the compressions the supporting section is about (#9)


def band(ratio):
    """#9's pre-registered bar on the AP-small drop over the AP drop: below 1.3x absent, 1.3x to 2x weak. 2x or more is #9's
    'confirmed' only when the paired bootstrap's CI of (small drop - all drop) excludes 0, and there is no CI here (#48)."""
    return "absent" if ratio < 1.3 else "weak" if ratio < 2 else "2x or more"


def small_object_rows(cells, platform="pi5", imgsz=MATRIX_IMGSZ):
    """The supporting section: each quantized, pruned and combined row of the accuracy table against its model's reference baseline
    (the baseline row: ONNX Runtime FP32 at the matrix resolution). Per bucket the relative drop, (reference - compressed) /
    reference in %, as #9 pre-registered it; small_over_all, the AP-small drop over the AP drop; band, by #9's bar."""
    acc = accuracy_rows(cells, platform, imgsz)
    ref = {r["cell"]["model"]: r["cell"] for r in acc if r["cell"]["configuration"] == "baseline"}
    rows = []
    for r in acc:
        c, b = r["cell"], ref.get(r["cell"]["model"])
        if c["configuration"] not in SMALL_OBJECT_SIDE or b is None:
            continue
        row = dict(r, reference=b)
        row.update({f"{k}_drop_pct": 100 * (b[k] - c[k]) / b[k] for k in BUCKETS})
        row["small_over_all"] = row["ap_small_drop_pct"] / row["ap_drop_pct"]
        row["band"] = band(row["small_over_all"])
        rows.append(row)
    return rows


STUDY_MODELS = ("YOLO11s", "YOLO26s")
RECIPE_ORDER = {"": 0, "per-channel": 1, "stock": 2}


def _runtime_order(runtime):
    return list(sp.RUNTIMES).index(runtime) if runtime in sp.RUNTIMES else len(sp.RUNTIMES)


def model_pairs(cells):
    """Result C: YOLO11s against YOLO26s, the same configuration on the same platform, runtime, precision, INT8 recipe, input
    resolution and head. latency_ratio is YOLO26s's latency over YOLO11s's (latency_pct: that minus 1, in %); ap_diff and
    ap_small_diff are YOLO26s minus YOLO11s. clean: neither cell ran an image below 2000 MHz (None off the Pi 5, which has no
    clock reading); headline: neither is a stock INT8 cell whose per-channel cell exists (studypairs.int8_notes)."""
    int8 = sp.int8_notes(cells)
    key = lambda c: (c["platform"], c["runtime"], c["precision"], c["int8_recipe"], c["configuration"], c["imgsz"], c["head"])  # noqa: E731
    by = {}
    for c in cells:
        if c["model"] in STUDY_MODELS:
            by.setdefault(key(c), {})[c["model"]] = c
    pairs = []
    for k, m in by.items():
        if len(m) < 2:
            continue
        a, b = (m[x] for x in STUDY_MODELS)
        below = [a["below_2000_mhz"], b["below_2000_mhz"]]
        pairs.append({"platform": k[0], "runtime": k[1], "precision": k[2], "int8_recipe": k[3], "configuration": k[4], "imgsz": k[5], "head": k[6],
                      "yolo11s": a, "yolo26s": b, "latency_ratio": b["latency_ms"] / a["latency_ms"], "latency_pct": 100 * (b["latency_ms"] / a["latency_ms"] - 1),
                      "ap_diff": b["ap"] - a["ap"], "ap_small_diff": b["ap_small"] - a["ap_small"],
                      "clean": None if None in below else below == [0, 0], "spans_session_6": sp.spans_offset(a, b),
                      "headline": all(_headline(int8, c)[0] for c in (a, b))})
    return sorted(pairs, key=lambda p: (_platform_first(p["platform"]), CONFIGURATIONS.index(p["configuration"]), _runtime_order(p["runtime"]),
                                        RECIPE_ORDER.get(p["int8_recipe"], 3), p["head"] != "one-to-many", p["imgsz"]))


# the runtime columns of result D's runtime table: (runtime, precision, INT8 recipe) on the Pi 5
RUNTIME_COLUMNS = {"ncnn_fp16": ("ncnn", "FP16", ""), "openvino_fp32": ("openvino", "FP32", ""), "onnxruntime_fp32": ("onnxruntime", "FP32", ""),
                   "onnxruntime_int8_per_channel": ("onnxruntime", "INT8", "per-channel"), "onnxruntime_int8_stock": ("onnxruntime", "INT8", "stock")}


def runtime_rows(cells, platform="pi5"):
    """Result D: the platform's runtimes side by side, one row per checkpoint and input resolution (one-to-many head), with the cell
    of each RUNTIME_COLUMNS column (None where none ran), its speed-up over ONNX Runtime FP32 (that latency over its own) and
    whether the two span session 6. The row's configuration is its float cell's (baseline, pruned, distilled), or its INT8 cells'
    where it has none (combined)."""
    here = [c for c in cells if c["platform"] == platform and c["head"] == "one-to-many"]
    by = {}
    for c in here:
        by.setdefault((c["model"], c["checkpoint"], c["imgsz"]), []).append(c)
    rows = []
    for (model, checkpoint, imgsz), cs in by.items():
        row = {"platform": platform, "model": model, "checkpoint": checkpoint, "imgsz": imgsz, "trained_imgsz": cs[0]["trained_imgsz"],
               "untrained_imgsz": cs[0]["untrained_imgsz"],
               "configuration": min(cs, key=lambda c: c["precision"] == "INT8")["configuration"]}
        for name, (rt, precision, recipe) in RUNTIME_COLUMNS.items():
            row[name] = next((c for c in cs if (c["runtime"], c["precision"], c["int8_recipe"]) == (rt, precision, recipe)), None)
        ref = row["onnxruntime_fp32"]
        for name in RUNTIME_COLUMNS:
            row[f"{name}_speedup"] = ref["latency_ms"] / row[name]["latency_ms"] if ref and row[name] else None
            row[f"{name}_spans_session_6"] = bool(ref and row[name]) and sp.spans_offset(ref, row[name])
        rows.append(row)
    return sorted(rows, key=lambda r: (r["model"], r["imgsz"], CONFIGURATIONS.index(r["configuration"]), r["checkpoint"]))


def figure_lines(pairs, platform="pi5"):
    """Result C's figure: one line per RUNTIME_COLUMNS column (FP32 and INT8 apart), through model_pairs' baseline and quantized
    pairs on the platform with the one-to-many head, in input-resolution order. headline: no point is a stock pair whose
    per-channel pair exists."""
    lines = []
    for name, (rt, precision, recipe) in RUNTIME_COLUMNS.items():
        points = sorted((p for p in pairs if (p["platform"], p["runtime"], p["precision"], p["int8_recipe"]) == (platform, rt, precision, recipe)
                         and p["configuration"] in ("baseline", "quantized") and p["head"] == "one-to-many"), key=lambda p: p["imgsz"])
        if points:
            lines.append({"name": name, "runtime": rt, "precision": precision, "int8_recipe": recipe, "points": points,
                          "headline": all(p["headline"] for p in points)})
    return lines


def latency_rows(cells):
    """The latency table: every cell, by platform (the Pi 5 first), runtime, model, configuration, INT8 recipe, input resolution and
    head. [{cell, headline, session_6, note}]: headline as studypairs.int8_notes, session_6 for a cell of a session whose latencies
    are offset (studypairs.SESSION_OFFSET), and note the INT8 note and cells.csv's own (which carries session 6's)."""
    int8 = sp.int8_notes(cells)
    rows = [{"cell": c, "headline": _headline(int8, c)[0], "session_6": c["session"] in sp.SESSION_OFFSET,
             "note": "; ".join(n for n in (_headline(int8, c)[1], c["note"]) if n)} for c in cells]
    return sorted(rows, key=lambda r: (_platform_first(r["cell"]["platform"]), _runtime_order(r["cell"]["runtime"]), r["cell"]["model"],
                                       CONFIGURATIONS.index(r["cell"]["configuration"]), RECIPE_ORDER.get(r["cell"]["int8_recipe"], 3),
                                       r["cell"]["imgsz"], r["cell"]["head"] != "one-to-many"))


# #9's laptop numbers (its comments of 15 and 17 October 2026): A1 at 640 on the laptop CPU, test-dev, relative drops in %, and the
# AP-small drop over the AP drop with its paired-bootstrap 95 % CI (1,000 replicates). No file in the repo holds them.
OPENVINO_BF16 = ("its FP32 reference ran in bf16 (INVARIANTS.md section 8); #23, which scores it again at true FP32, has not run, "
                 "so this row is the bf16 one")
LAPTOP_9 = [
    {"arm": "ONNX Runtime INT8 (stock), against its FP32 export", "ap_drop_pct": 10.2, "ap_small_drop_pct": 14.3, "ap_medium_drop_pct": 9.6,
     "ap_large_drop_pct": 1.9, "small_over_all": 1.40, "ci": (1.27, 1.64), "verdict": "weak", "note": ""},
    {"arm": "OpenVINO INT8 (NNCF), against its FP32 export", "ap_drop_pct": 2.3, "ap_small_drop_pct": 3.5, "ap_medium_drop_pct": 1.9,
     "ap_large_drop_pct": -2.1, "small_over_all": 1.51, "ci": (1.05, 1.97), "verdict": "weak", "note": OPENVINO_BF16},
    {"arm": "pruned 30 % and recovered, against A1's .pt", "ap_drop_pct": 17.1, "ap_small_drop_pct": 20.0, "ap_medium_drop_pct": 16.6,
     "ap_large_drop_pct": 20.7, "small_over_all": 1.17, "ci": (0.99, 1.32), "verdict": "absent", "note": ""}]


# ---------------------------------------------------------------- tables

PLATFORM_NAMES = {"pi5": "Raspberry Pi 5", "m34": "Galaxy M34"}  # the tables' wording (studypairs.PLATFORMS is the figures')
CONFIG_LABELS = {"baseline": "Baseline", "quantized": "Quantized", "pruned": "Pruned", "distilled": "Distilled", "combined": "Combined"}
BS = sp.BS
DAGGER, S6 = f"$^{{{BS}dagger}}$", f"$^{{{BS}mathrm{{S}}}}$"
CELL_KEYS = ["session", "checkpoint", "precision", "int8_recipe", "latency_ms", "ap", "ap_small", "throttled", "below_2000_mhz", "clock_median_mhz"]


def _config(c):
    """'Quantized, stock' for a stock INT8 cell, else the configuration's name."""
    return CONFIG_LABELS[c["configuration"]] + (", stock" if c["int8_recipe"] == "stock" else "")


def _cell_tex(c, s6=False):
    """'B1 INT8 at 800', a dagger at an untrained input resolution, S for session 6's latency where s6."""
    return sp.label(c) + (DAGGER if c["untrained_imgsz"] else "") + (S6 if s6 and c["session"] in sp.SESSION_OFFSET else "")


def _count(n):
    """1,373 as LaTeX writes a thousands comma; -- for no reading (the phone)."""
    return "--" if n is None else f"{n:,}".replace(",", "{,}")


def _num(v, nd):
    return "--" if v is None else f"{v:.{nd}f}".replace("-", "$-$")


def _plain(v, nd):
    return "--" if v is None else f"{v:.{nd}f}"


def _signed(v, nd):
    return "--" if v is None else f"{v:+.{nd}f}".replace("-", "$-$")


def _ms(c, below=False):
    """A latency in ms, then t when the cell was flagged throttled; or, with below, the images it ran below 2000 MHz in square
    brackets (INVARIANTS.md section 4.7: the paper reports throttling by the clock), for a table with no clock columns."""
    if c is None:
        return "--"
    if below:
        return f"{c['latency_ms']:.1f}" + (f" [{_count(c['below_2000_mhz'])}]" if c["below_2000_mhz"] else "")
    return f"{c['latency_ms']:.1f}" + (" t" if c["throttled"] == "throttled" else "")


def _times(v):
    return "--" if v is None else f"{v:.2f}${BS}times$"


def _tex(path, comment, colspec, head, body, notes, env="tabular"):
    """A booktabs table: body rows are lists of cells, or a string for a group's heading (a midrule, then the heading in italics)."""
    n = len(colspec)
    lines = [f"% {comment} Written by studytables.py from results/cells.csv.", f"{BS}begin{{{env}}}{{{colspec}}}", f"{BS}toprule",
             " & ".join(head) + f" {BS}{BS}", f"{BS}midrule"] + ([f"{BS}endhead"] if env == "longtable" else [])
    for i, row in enumerate(body):
        if isinstance(row, str):
            lines += ([f"{BS}midrule"] if i else []) + [f"{BS}multicolumn{{{n}}}{{l}}{{{BS}textit{{{row}}}}} {BS}{BS}"]
        else:
            lines.append(" & ".join(row) + f" {BS}{BS}")
    lines += [f"{BS}bottomrule"] + [f"{BS}multicolumn{{{n}}}{{l}}{{{BS}footnotesize {x}}} {BS}{BS}" for x in notes] + [f"{BS}end{{{env}}}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


NOTE_INT8 = "INT8 is ONNX Runtime's, per-channel (INVARIANTS.md section 8) unless marked stock; a stock row follows its per-channel row."
NOTE_DAGGER = (f"{DAGGER} A checkpoint run at an input resolution it was not trained at (800 from the 960 models, 1120 from the "
               f"1280 ones): its accuracy there is a little lower ({BS}#8).")
NOTE_S6 = f"{S6} Session 6's cell: its latencies are about 6{BS}% high against the other sessions' (measured by session 7), kept as measured."
NOTE_T = "t: the cell was flagged throttled (INVARIANTS.md section 4.7)."
NOTE_BELOW = "[n]: the cell ran n of its 1,610 images below 2000 MHz (INVARIANTS.md section 4.7); the median clocks are in the latency table."


def write_accuracy(rows, out):
    sp.write_csv(out / "tables" / "accuracy.csv", [
        {"model": r["cell"]["model"], "configuration": r["cell"]["configuration"], "checkpoint": r["cell"]["checkpoint"], "imgsz": r["cell"]["imgsz"],
         "runtime": r["cell"]["runtime"], "precision": r["cell"]["precision"], "int8_recipe": r["cell"]["int8_recipe"], "headline": r["headline"],
         **{k: r["cell"][k] for k in ("ap", "ap50", "ap_small", "ap_medium", "ap_large", "session")}, "note": r["note"]} for r in rows])
    body, last = [], None
    for r in rows:
        c = r["cell"]
        if c["model"] != last:
            body.append(c["model"])
        body.append([_config(c), c["checkpoint"], c["precision"]] + [_num(c[k], 4) for k in ("ap", "ap50", "ap_small", "ap_medium", "ap_large")])
        last = c["model"]
    _tex(out / "tables" / "accuracy.tex", "The accuracy table (#47): the Pi 5, the matrix resolution (640), one-to-many head.", "lllrrrrr",
         ["Configuration", "Cell", "Precision", "AP", "AP50", "AP-small", "AP-medium", "AP-large"], body,
         ["COCO AP at maxDets 500 on VisDrone test-dev, scored from the Pi 5's detections at the evaluation threshold (0.001).",
          "Baseline, pruned and distilled: the ONNX Runtime FP32 cells. " + NOTE_INT8,
          "Combined: pruned, recovered by distillation, then INT8."])


def write_latency(rows, out):
    sp.write_csv(out / "tables" / "latency.csv", [
        {**{k: r["cell"][k] for k in ("platform", "runtime", "model", "checkpoint", "configuration", "imgsz", "trained_imgsz", "untrained_imgsz",
                                      "precision", "int8_recipe", "head")}, "cell_name": sp.label(r["cell"]), "headline": r["headline"],
         **{k: r["cell"][k] for k in ("latency_ms", "pre_ms", "net_ms", "post_deploy_ms", "post_eval_ms", "throttled", "below_2000_mhz",
                                      "clock_median_mhz", "session")}, "session_6": r["session_6"], "note": r["note"]} for r in rows])
    body, last = [], None
    for r in rows:
        c = r["cell"]
        group = f"{PLATFORM_NAMES.get(c['platform'], c['platform'])}, {sp.RUNTIMES.get(c['runtime'], c['runtime'])}"
        if group != last:
            body.append(group)
        body.append([c["model"], _cell_tex(c, s6=True), _config(c), c["precision"], _ms(c)] + [_num(c[k], 1) for k in ("pre_ms", "net_ms")]
                    + [_num(c["post_deploy_ms"], 2), _count(c["below_2000_mhz"]), "--" if c["clock_median_mhz"] is None else str(c["clock_median_mhz"])])
        last = group
    _tex(out / "tables" / "latency.tex", "The latency table (#47): every cell.", "llllrrrrrr",
         ["Model", "Cell", "Configuration", "Precision", "Latency (ms)", "Pre", "Net", "Post (0.25)", "Below 2000 MHz", "Median clock (MHz)"], body,
         ["Medians over the 1,610 test-dev images at batch 1. Latency: pre-processing + network + post-processing at the deployment threshold, 0.25.",
          "The Pi 5 at a 2000 MHz clock lock (INVARIANTS.md section 4.7). Below 2000 MHz: images, of 1,610. The phone has no clock reading.",
          NOTE_T, NOTE_INT8, NOTE_DAGGER, NOTE_S6], env="longtable")


def write_result_c(pairs, out):
    sp.write_csv(out / "tables" / "result_c.csv", [
        {**{k: p[k] for k in ("platform", "runtime", "precision", "int8_recipe", "configuration", "imgsz", "head", "headline")},
         "yolo11s": sp.label(p["yolo11s"]), "yolo26s": sp.label(p["yolo26s"]),
         **{f"{m}_{k}": p[m][k] for m in ("yolo11s", "yolo26s") for k in CELL_KEYS},
         **{k: p[k] for k in ("latency_ratio", "latency_pct", "ap_diff", "ap_small_diff", "clean", "spans_session_6")},
         "note": sp.SPANS_NOTE if p["spans_session_6"] else ""} for p in pairs])
    shown = [p for p in pairs if p["platform"] == "pi5" and p["configuration"] in ("baseline", "quantized") and p["head"] == "one-to-many"]
    body, last = [], None
    for p in shown:
        group = "INT8 (quantized)" if p["precision"] == "INT8" else "Baselines at their runtime's precision"
        if group != last:
            body.append(group)
        a, b = p["yolo11s"], p["yolo26s"]
        body.append([sp.RUNTIMES.get(p["runtime"], p["runtime"]), p["precision"] + (", stock" if p["int8_recipe"] == "stock" else ""), str(p["imgsz"]),
                     _ms(a), _ms(b), f"{p['latency_ratio']:.3f}" + (S6 if p["spans_session_6"] else ""), _signed(p["ap_diff"], 4),
                     _signed(p["ap_small_diff"], 4), f"{_count(a['below_2000_mhz'])} / {_count(b['below_2000_mhz'])}"])
        last = group
    lines = figure_lines(pairs)
    clean = [_line_label(ln) for ln in lines if all(p["clean"] for p in ln["points"])]
    spans = sum(p["spans_session_6"] for p in shown)
    _tex(out / "tables" / "result_c.tex", "Result C (#47): YOLO11s against YOLO26s on the Pi 5.", "llrrrrrrc",
         ["Runtime", "Precision", "Input res.", "YOLO11s (ms)", "YOLO26s (ms)", "YOLO26s / YOLO11s", f"${BS}Delta$AP", f"${BS}Delta$AP-small",
          "Below 2000 MHz"], body,
         ["The same configuration, runtime, precision and input resolution for both study models; one-to-many head.",
          "Ratio: YOLO26s's median latency over YOLO11s's. Differences: YOLO26s minus YOLO11s. Below 2000 MHz: images, YOLO11s / YOLO26s.",
          "Clean latency comparisons, with no image below 2000 MHz in either cell: " + (", ".join(clean) or "none") + ". " + NOTE_T,
          (f"{S6} The pair spans session 6 (about 6{BS}% high): {spans} of them." if spans else
           "No pair spans session 6: its cells pair only with each other, so its offset cancels in the ratio."),
          f"INT8: per-channel unless marked stock. The pruned, distilled and combined pairs are in result{BS}_c.csv."])


def write_result_d(int8, runtimes, out):
    flat = []
    for r in int8:
        row = {k: r[k] for k in ("platform", "runtime", "model", "checkpoint", "configuration", "imgsz", "trained_imgsz", "untrained_imgsz", "head")}
        for side in ("fp32", "per_channel", "stock"):
            row.update({f"{side}_{k}": None if r[side] is None else r[side][k] for k in CELL_KEYS if k not in ("checkpoint", "int8_recipe", "precision")})
            if side != "fp32":
                row.update({f"{side}_{k}": r[f"{side}_{k}"] for k in ("speedup", "ap_diff", "ap_diff_pct", "ap_small_diff", "ap_small_diff_pct", "spans_session_6")})
        flat.append(dict(row, speedup=r["speedup"], speedup_recipe=r["speedup_recipe"], note=r["note"]))
    sp.write_csv(out / "tables" / "result_d_int8.csv", flat)
    body, last = [], None
    for r in int8:
        group = f"{PLATFORM_NAMES.get(r['platform'], r['platform'])}, {sp.RUNTIMES.get(r['runtime'], r['runtime'])}"
        if group != last:
            body.append(group)
        name = f"{r['checkpoint']} at {r['imgsz']}" + (" NMS-free" if r["head"] == "NMS-free" else "") + (DAGGER if r["untrained_imgsz"] else "")
        per_channel = _times(r["per_channel_speedup"])
        if r["per_channel_spans_session_6"]:  # not the one to use (#47's comment of 1 November): bracketed, and the stock one stands in
            per_channel = f"({per_channel}){S6}"
        fp32_s6 = S6 if r["fp32"] and r["fp32"]["session"] in sp.SESSION_OFFSET else ""
        body.append([r["model"], name, _ms(r["fp32"], below=True) + fp32_s6, f"{_ms(r['per_channel'], below=True)} / {_ms(r['stock'], below=True)}",
                     f"{per_channel} / {_times(r['stock_speedup'])}"]
                    + [f"{_signed(r[f'per_channel_{k}'], 1)} / {_signed(r[f'stock_{k}'], 1)}" for k in ("ap_diff_pct", "ap_small_diff_pct")])
        last = group
    no_fp32 = sorted({r["checkpoint"] for r in int8 if r["fp32"] is None})
    spanning = [r for r in int8 if r["per_channel_spans_session_6"]]
    _tex(out / "tables" / "result_d_int8.tex", "Result D (#47): ONNX Runtime INT8 against FP32, per-channel and stock.", "llrrrrr",
         ["Model", "Checkpoint", "FP32 (ms)", "INT8 (ms)", "Speed-up", f"${BS}Delta$AP ({BS}%)", f"${BS}Delta$AP-small ({BS}%)"], body,
         ["Each INT8 cell against the FP32 cell of the same checkpoint, input resolution and runtime. INT8 columns: per-channel / stock.",
          f"Speed-up: FP32 latency over INT8's. Changes: INT8 minus FP32, in {BS}% of FP32. --: no such cell"
          + (f" (no FP32 export of {' and '.join(no_fp32)} was run)." if no_fp32 else "."),
          NOTE_BELOW, NOTE_DAGGER]
         + ([f"{S6} Session 6's FP32 cell (about 6{BS}% high). A bracketed per-channel speed-up sets session 7's cell against it, so use the stock "
             f"one, within session 6: {', '.join(f'''{r['checkpoint']} at {r['imgsz']} {_times(r['stock_speedup'])}''' for r in spanning)}."]
            if spanning else []))
    sp.write_csv(out / "tables" / "result_d_runtimes.csv", [
        {**{k: r[k] for k in ("platform", "model", "checkpoint", "configuration", "imgsz", "trained_imgsz", "untrained_imgsz")},
         **{f"{name}_{k}": None if r[name] is None else r[name][k] for name in RUNTIME_COLUMNS
            for k in ("latency_ms", "ap", "ap_small", "throttled", "below_2000_mhz", "session")},
         **{f"{name}_{k}": r[f"{name}_{k}"] for name in RUNTIME_COLUMNS for k in ("speedup", "spans_session_6")}} for r in runtimes])

    def cell_ms(r, name):
        """The cell's latency, [images below 2000 MHz], S for a session 6 latency, and its speed-up over ONNX Runtime FP32, S when
        that spans session 6."""
        c = r[name]
        if c is None:
            return "--"
        speedup = "" if r[f"{name}_speedup"] is None or name == "onnxruntime_fp32" else f" ({_times(r[f'{name}_speedup'])})"
        return (_ms(c, below=True) + (S6 if c["session"] in sp.SESSION_OFFSET else "") + speedup
                + (S6 if r[f"{name}_spans_session_6"] else "") + (", stock" if c["int8_recipe"] == "stock" else ""))

    body, last = [], None
    for r in runtimes:
        if r["model"] != last:
            body.append(r["model"])
        int8_name = "onnxruntime_int8_per_channel" if r["onnxruntime_int8_per_channel"] else "onnxruntime_int8_stock"
        aps = [None if r[n] is None else r[n]["ap"] for n in ("ncnn_fp16", "onnxruntime_fp32", int8_name)]
        body.append([f"{r['checkpoint']} at {r['imgsz']}" + (DAGGER if r["untrained_imgsz"] else ""), CONFIG_LABELS[r["configuration"]]]
                    + [cell_ms(r, n) for n in ("ncnn_fp16", "openvino_fp32", "onnxruntime_fp32", int8_name)] + [" / ".join(_num(v, 4) for v in aps)])
        last = r["model"]
    _tex(out / "tables" / "result_d_runtimes.tex", "Result D (#47): the three Pi 5 runtimes side by side.", "llrrrrr",
         ["Cell", "Configuration", "NCNN FP16 (ms)", "OpenVINO FP32 (ms)", "ONNX Runtime FP32 (ms)", "ONNX Runtime INT8 (ms)", "AP: NCNN / FP32 / INT8"],
         body, ["Median latency on the Pi 5 at 2000 MHz; in brackets, the speed-up over ONNX Runtime FP32 (its latency over the cell's). One-to-many head.",
                "OpenVINO FP32 and ONNX Runtime FP32 give the same AP to 4 digits. INT8: per-channel, or stock where marked; the stock cells are in the CSV.",
                NOTE_BELOW, NOTE_DAGGER,
                f"{S6} After a latency: session 6's cell (about 6{BS}% high). After a speed-up: its two cells span session 6, so it is about 6{BS}% off."])


def write_small_objects(rows, out):
    flat = [{"platform": r["cell"]["platform"], "model": r["cell"]["model"], "compression": _config(r["cell"]), "cell": sp.label(r["cell"]),
             "reference": sp.label(r["reference"]), "runtime": r["cell"]["runtime"], "int8_recipe": r["cell"]["int8_recipe"], "headline": r["headline"],
             **{f"reference_{k}": r["reference"][k] for k in BUCKETS}, **{k: r["cell"][k] for k in BUCKETS},
             **{f"{k}_drop_pct": r[f"{k}_drop_pct"] for k in BUCKETS}, "small_over_all": r["small_over_all"], "ci_low": None, "ci_high": None,
             "band": r["band"], "source": "cells.csv", "note": r["note"]} for r in rows]
    flat += [{"platform": "laptop", "model": "YOLO11s", "compression": x["arm"], "cell": "A1 at 640", "reference": "", "runtime": "", "int8_recipe": "",
              "headline": True, **{f"reference_{k}": None for k in BUCKETS}, **{k: None for k in BUCKETS},
              **{f"{k}_drop_pct": x[f"{k}_drop_pct"] for k in BUCKETS}, "small_over_all": x["small_over_all"], "ci_low": x["ci"][0],
              "ci_high": x["ci"][1], "band": x["verdict"], "source": "#9", "note": x["note"]} for x in LAPTOP_9]
    sp.write_csv(out / "tables" / "small_object.csv", flat)
    body, last = [], None
    for r in rows:
        if r["cell"]["model"] != last:
            body.append(f"Raspberry Pi 5, {r['cell']['model']}, against {sp.label(r['reference'])} (ONNX Runtime)")
        body.append([_config(r["cell"]), sp.label(r["cell"])] + [_num(r[f"{k}_drop_pct"], 1) for k in BUCKETS]
                    + [f"{r['small_over_all']:.2f}${BS}times$", r["band"]])
        last = r["cell"]["model"]
    body.append(f"Laptop CPU, {BS}#9: YOLO11s A1 at 640 (95{BS}% CI of the ratio, paired bootstrap)")
    for x in LAPTOP_9:
        body.append([x["arm"].replace("%", f"{BS}%") + (f"$^{{{BS}ddagger}}$" if x["note"] else ""), "", *[_num(x[f"{k}_drop_pct"], 1) for k in BUCKETS],
                     f"{x['small_over_all']:.2f}${BS}times$ [{x['ci'][0]:.2f}, {x['ci'][1]:.2f}]", x["verdict"]])
    _tex(out / "tables" / "small_object.tex", "Small-object degradation (#47, #9): the Pi 5 cells beside #9's laptop numbers.", "llrrrrrl",
         ["Compression", "Cell", f"AP drop ({BS}%)", f"AP-small ({BS}%)", f"AP-medium ({BS}%)", f"AP-large ({BS}%)", "Small / all", f"{BS}#9's bar"], body,
         ["Relative drop per bucket: (reference $-$ compressed) / reference, on test-dev. Small / all: the AP-small drop over the AP drop.",
          f"{BS}#9's pre-registered bar: below 1.3${BS}times$ absent, 1.3--2${BS}times$ weak, 2${BS}times$ or more and a CI that excludes 0 confirmed.",
          f"The Pi 5 rows have no CI yet ({BS}#48). Reference: the reference baseline, ONNX Runtime FP32 at 640. " + NOTE_INT8,
          f"$^{{{BS}ddagger}}$ Its FP32 reference ran in bf16 (INVARIANTS.md section 8). {BS}#23, which scores it again at true FP32, has not run."])


# ---------------------------------------------------------------- the figure

LINE_STYLE = {  # runtime column: (colour, marker, label); the dataviz reference palette's slots 3, 4, 7 and 5 (slots 1 and 2 are the models')
    "ncnn_fp16": ("#1baf7a", "o", "NCNN FP16"), "openvino_fp32": ("#eda100", "s", "OpenVINO FP32"),
    "onnxruntime_fp32": ("#6250d6", "D", "ONNX Runtime FP32"), "onnxruntime_int8_per_channel": ("#e87ba4", "^", "ONNX Runtime INT8"),
    "onnxruntime_int8_stock": ("#e87ba4", "v", "ONNX Runtime INT8, stock")}


def _line_label(ln):
    """'ONNX Runtime INT8 (per-channel)', as the figure's legend and result C's table notes name a line."""
    return LINE_STYLE[ln["name"]][2] + (" (per-channel)" if ln["int8_recipe"] == "per-channel" else "")


def figure(lines, platform, out):
    """Result C's figure: YOLO26s's latency over YOLO11s's against input resolution, one line per runtime and precision. A hollow
    marker is a pair with an image below 2000 MHz, and a line with one is drawn thinner (the throttle moves its ratio); the stock
    INT8 line is dashed, behind the per-channel one."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ink, muted = studynms.INK, studynms.MUTED
    fig, ax = plt.subplots(figsize=(8.2, 6.4), facecolor="white")
    ax.axhline(1.0, color=muted, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    style = {}
    for ln in lines:
        colour, marker, name = LINE_STYLE[ln["name"]]
        clean = all(p["clean"] for p in ln["points"])
        style[ln["name"]] = {"color": colour, "linewidth": 2.0 if clean and ln["headline"] else 1.3, "alpha": 1 if clean else 0.8,
                             "linestyle": "-" if ln["headline"] else (0, (5, 3))}
        z = 3 if ln["headline"] else 2
        ax.plot([p["imgsz"] for p in ln["points"]], [p["latency_ratio"] for p in ln["points"]], zorder=z, **style[ln["name"]])
        for p in ln["points"]:
            ax.scatter([p["imgsz"]], [p["latency_ratio"]], s=54, marker=marker, zorder=z + 0.5, facecolors=colour if p["clean"] else "white",
                       edgecolors="white" if p["clean"] else colour, linewidths=1.0 if p["clean"] else 1.6)
    ax.set_xticks([640, 800, 960, 1120, 1280])
    ax.set_xlim(600, 1320)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_xlabel("Input resolution (px)", color=muted)
    ax.set_ylabel("YOLO26s's median latency / YOLO11s's", color=muted)
    ax.grid(color=studynms.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(muted)
    ax.tick_params(colors=muted)
    ax.text(1316, 1.0, "equal latency", fontsize=8, color=muted, ha="right", va="bottom")
    handles = [Line2D([], [], marker=LINE_STYLE[ln["name"]][1], markeredgecolor="white", markersize=7, **style[ln["name"]],
                      label=_line_label(ln) + f": {100 * (ln['points'][-1]['latency_ratio'] - 1):+.1f} % at {ln['points'][-1]['imgsz']}")
               for ln in lines]
    handles.append(Line2D([], [], color=muted, marker="o", markerfacecolor="white", markeredgewidth=1.6, linestyle="none", markersize=7,
                          label="Hollow: a cell of the pair ran images below 2000 MHz"))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.5, labelcolor=ink)
    fig.suptitle(f"YOLO26s's latency against YOLO11s's on the {PLATFORM_NAMES.get(platform, platform)}, by input resolution",
                 x=0.01, ha="left", fontsize=11.5, color=ink)
    clean = [_line_label(ln) for ln in lines if all(p["clean"] for p in ln["points"])]
    fig.text(0.01, 0.94, chr(10).join(["Same configuration, runtime and precision: the baselines (FP32, NCNN FP16) and the quantized cells (INT8).",
                                       "One-to-many head, 2000 MHz.",
                                       "The clean comparisons, with no image below 2000 MHz: " + (", ".join(clean) or "none") + "."]),
             fontsize=8.5, color=muted, va="top")
    fig.tight_layout(rect=(0, 0.13, 1, 0.87))
    studynms.save_figure(fig, out / "figures" / f"result_c_{platform}")


# ---------------------------------------------------------------- the script

STUDY = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cells", type=Path, default=STUDY / "results" / "cells.csv")
    ap.add_argument("--out", type=Path, default=STUDY / "results")
    a = ap.parse_args(argv)
    cells, aside = sp.usable(sp.load_cells(a.cells))
    for c, why in aside:
        print(f"set aside: {c['session']} {c['cell']}: {why}")
    acc, lat, pairs = accuracy_rows(cells), latency_rows(cells), model_pairs(cells)
    int8, runtimes, small = int8_rows(cells), runtime_rows(cells), small_object_rows(cells)
    write_accuracy(acc, a.out)
    write_latency(lat, a.out)
    write_result_c(pairs, a.out)
    write_result_d(int8, runtimes, a.out)
    write_small_objects(small, a.out)
    lines = figure_lines(pairs)
    figure(lines, "pi5", a.out)
    for ln in lines:
        print(f"result C, {ln['name']}{'' if ln['headline'] else ' (stock)'}: " + ", ".join(f"{p['imgsz']} {p['latency_pct']:+.1f} %" for p in ln["points"]))
    for r in int8:
        print(f"result D, {r['platform']} {r['checkpoint']} at {r['imgsz']} {r['head']}: speed-up per-channel {_plain(r['per_channel_speedup'], 2)}, "
              f"stock {_plain(r['stock_speedup'], 2)}; AP {_plain(r['per_channel_ap_diff_pct'], 1)} / {_plain(r['stock_ap_diff_pct'], 1)} %")
    for r in small:
        print(f"small objects, {sp.label(r['cell'])} {r['cell']['int8_recipe']}: AP -{r['ap_drop_pct']:.1f} %, AP-small -{r['ap_small_drop_pct']:.1f} %, "
              f"{r['small_over_all']:.2f}x ({r['band']})")
    print(f"{len(acc)} accuracy rows, {len(lat)} latency rows, {len(pairs)} result C pairs, {len(int8)} result D INT8 rows, "
          f"{len(runtimes)} runtime rows, {len(small)} small-object rows (+{len(LAPTOP_9)} from #9); written under {a.out}")


if __name__ == "__main__":
    main()
