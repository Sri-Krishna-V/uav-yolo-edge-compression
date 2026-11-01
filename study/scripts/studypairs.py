"""Result A (#46): the equal-latency comparisons on the Pi 5, their table and the paper's main figure, from results/cells.csv.
Runs nothing on a Pi and reads nothing but cells.csv (studyresults.py, #44): re-run it after studyresults.py when a session lands.

  python study/scripts/studypairs.py [--cells study/results/cells.csv] [--out study/results]

Which cells. Every scored cells.csv row; a cell that ran again in a later session (the phone's matrix sessions repeat its
full-cell check) counts once, by its latest run, and the ones set aside are printed.
The pairs (GLOSSARY.md, Equal-latency comparison). A compressed configuration (quantized, pruned or combined; distilled cells
save no latency, so they are never the compressed side) at a larger input resolution against a baseline at a smaller one, on
the same platform, model, runtime and head, whose median latencies differ by 10 % or less of the smaller (#8). So the INT8
cells, which are ONNX Runtime cells, pair with ONNX Runtime FP32 baselines. Latencies are as measured.
INT8. The per-channel cells (onnxruntime-int8pc) are the matrix's ONNX Runtime INT8 cells (INVARIANTS.md section 8). A stock INT8
cell's pair is the headline only while no per-channel cell of the same checkpoint, input resolution and head is scored; after
that it stays as its own row, headline 0. A session that adds per-channel cells moves the headline with no code change.
Session 6. Its latencies are about 6 % high against the other sessions' (measured by session 7). Decided on #46: they stay as
measured, flagged. A pair with one cell from session 6 spans it; a pair inside it does not.

Outputs, all under --out:
  tables/equal_latency.csv      every platform's pairs, one row each: platform, runtime, model; compressed and baseline (the
                                cells as the records name them, 'B1 INT8 at 800') and per cell its session, cell, checkpoint,
                                configuration, imgsz, trained_imgsz, untrained_imgsz, precision, int8_recipe, latency_ms, ap,
                                ap_small and the throttle columns (throttled, below_2000_mhz, clock_median_mhz: INVARIANTS.md
                                section 4.7); latency_ratio (compressed over baseline), apart_pct (the larger over the smaller,
                                minus 1), ap_small_diff, ap_diff, ap_medium_diff, ap_large_diff (compressed minus baseline);
                                spans_session_6, headline (1/0) and note
  tables/equal_latency.tex      the Pi 5's pairs (booktabs): the headline pairs, each stock INT8 row after its per-channel one
  tables/equal_latency_all.csv  for the writer, not a paper table: every pair of cells on one platform within 10 %, across
                                models, configurations, runtimes and heads, a and b (a at the larger input resolution, the
                                slower at the same), differences a minus b, within_limit, paper_pair; and the pairs spanning
                                session 6 that are within 10 % only with its offset taken out (apart_pct_without_offset), as
                                rows with within_limit 0 and a note
  figures/result_a_<platform>.pdf, .png
                                AP-small against median latency: one panel per runtime, both study models, the marker's shape
                                the configuration (hollow at an untrained input resolution), a line through each model's
                                baselines from 640 to 1280, an arrow from baseline to compressed cell for each headline pair.
                                The Pi 5's, and any other platform's once it has three cells (the phone has 2 now)
Standard library, numpy and matplotlib only (studynms.py's palette and figure saving); no torch, no ultralytics.
"""

import argparse
import csv
from pathlib import Path

import studynms

COMPRESSED_SIDE = {"quantized", "pruned", "combined"}  # distilled cells save no latency, so they are never the compressed side
LIMIT = 0.10  # equal latency: the medians differ by 10 % or less of the smaller (#8)
EPS = 1e-9  # so that exactly 10 % is in (110 / 100 - 1 is 0.10000000000000009); medians are kept to 0.001 ms
DIFFS = {"ap_small_diff": "ap_small", "ap_diff": "ap", "ap_medium_diff": "ap_medium", "ap_large_diff": "ap_large"}
# Session 6's latencies are about 6 % high against the other sessions' (measured by session 7; kept as measured, flagged: #46)
SESSION_OFFSET = {"pi5-matrix-6-1120": 1.06}
SPANS_NOTE = "spans session 6, whose latencies are about 6 % high against the other sessions'"
INTS = ["trained_imgsz", "imgsz", "untrained_imgsz", "images", "throttle_reads", "soft_limit_reads", "below_2000_mhz", "clock_min_mhz",
        "clock_median_mhz", "attempts", "attempt"]
FLOATS = ["ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large", "ar500", "latency_ms", "latency_p95_ms", "pre_ms", "net_ms",
          "post_deploy_ms", "post_eval_ms", "peak_temp_c"]
VERSIONS = ["python_version", "numpy_version", "opencv_version", "runtime_version"]


def load_cells(path):
    """cells.csv (studyresults.py's columns), with its numbers as numbers and an empty field as None."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in INTS + FLOATS:
            r[k] = None if r[k] == "" else (int(r[k]) if k in INTS else float(r[k]))
    return rows


def usable(cells):
    """The cells a pair or a figure can use: scored (an AP and a latency), and one per platform and cell name, the latest by its
    start (the phone's matrix sessions run its full-cell check's cells again). Returns (kept, [(cell, why set aside)])."""
    latest, aside = {}, []
    for c in cells:
        if c["ap"] is None or c["latency_ms"] is None:
            aside.append((c, "not scored or no latency: " + (c["note"] or c["outcome"])))
            continue
        k = (c["platform"], c["cell"])
        if k in latest:
            old, c = sorted((latest[k], c), key=lambda x: x["start_utc"])
            aside.append((old, f"the same cell ran again later, in {c['session']}"))
        latest[k] = c
    return [c for c in cells if any(c is v for v in latest.values())], aside


def label(c):
    """'B1 INT8 at 800', as the session records name a cell (its runtime and INT8 recipe are columns of their own)."""
    return f"{c['checkpoint']} {c['precision']}{' NMS-free' if c['head'] == 'NMS-free' else ''} at {c['imgsz']}"


def apart(a, b):
    """How far apart two cells' median latencies are: the larger over the smaller, minus 1."""
    lo, hi = sorted((a["latency_ms"], b["latency_ms"]))
    return hi / lo - 1


def spans_offset(a, b):
    """Whether exactly one of the two cells is from a session whose latencies are offset (session 6): a pair inside it is not."""
    return (a["session"] in SESSION_OFFSET) != (b["session"] in SESSION_OFFSET)


def apart_without_offset(a, b):
    """apart(), with each cell's latency divided by its session's offset (session 6's 1.06; 1 for the others)."""
    return apart(*({"latency_ms": c["latency_ms"] / SESSION_OFFSET.get(c["session"], 1.0)} for c in (a, b)))


def versions_note(a, b):
    """A note when the two cells ran different software (the phone's sessions of 25 October and 1 November do). Two runtimes'
    versions are not compared: they always differ."""
    keys = VERSIONS if a["runtime"] == b["runtime"] else [k for k in VERSIONS if k != "runtime_version"]
    diff = [f"{k.replace('_version', '')} {a.get(k)} / {b.get(k)}" for k in keys if a.get(k) != b.get(k)]
    return "the two cells ran different software: " + ", ".join(diff) if diff else ""


def int8_notes(cells):
    """The INT8 rule (INVARIANTS.md section 8): the per-channel cells are the matrix's ONNX Runtime INT8 cells. A stock INT8 cell
    is the headline only while no per-channel cell of the same platform, runtime, checkpoint, input resolution and head is scored, so a
    session that adds per-channel cells moves the headline with no code change. {id(cell): (headline, note)} for the INT8 cells."""
    same = lambda c: (c["platform"], c["checkpoint"], c["imgsz"], c["head"], c["runtime"])  # noqa: E731
    per_channel = {same(c) for c in cells if c["int8_recipe"] == "per-channel"}
    out = {}
    for c in cells:
        if c["int8_recipe"] == "stock":
            out[id(c)] = ((False, "stock INT8 recipe: the per-channel cell is the headline (INVARIANTS.md section 8)") if same(c) in per_channel
                          else (True, "stock INT8 recipe: no per-channel cell scored yet"))
    return out


def equal_latency_pairs(cells, limit=LIMIT):
    """GLOSSARY.md's equal-latency comparisons: a compressed configuration (quantized, pruned or combined) at a larger input
    resolution against a baseline at a smaller one, on the same platform, model, runtime and head, whose median latencies are
    within limit of the smaller. Differences are the compressed cell minus the baseline. headline is False for a stock INT8 pair
    whose per-channel pair exists (int8_notes); spans_session_6 when one cell is session 6's; note holds what the table must footnote.
    Latencies are as measured: session 6's offset never moves a pair in or out (#46)."""
    pairs, int8 = [], int8_notes(cells)
    for c in cells:
        if c["configuration"] not in COMPRESSED_SIDE:
            continue
        headline, int8_note = int8.get(id(c), (True, ""))
        for b in cells:
            same = all(c[k] == b[k] for k in ("platform", "model", "runtime", "head"))
            if same and b["configuration"] == "baseline" and c["imgsz"] > b["imgsz"] and apart(c, b) <= limit + EPS:
                pair = {"compressed": c, "baseline": b, "latency_ratio": c["latency_ms"] / b["latency_ms"], "apart_pct": 100 * apart(c, b)}
                pair.update({k: c[col] - b[col] for k, col in DIFFS.items()})
                spans = spans_offset(c, b)
                pair.update(headline=headline, spans_session_6=spans,
                            note="; ".join(n for n in (int8_note, SPANS_NOTE if spans else "", versions_note(c, b)) if n))
                pairs.append(pair)
    return pairs


def all_pairs(cells, limit=LIMIT):
    """Every pair of cells on one platform whose median latencies are within limit, across models, configurations, runtimes and
    heads (for the writer, not a paper table); and the pairs that span session 6 and are within limit only once its offset is
    taken out, as rows with within_limit False and a note. a is the cell at the larger input resolution (the slower one at the
    same), differences are a minus b, and paper_pair says whether equal_latency_pairs has the pair."""
    paper = {(id(p["compressed"]), id(p["baseline"])) for p in equal_latency_pairs(cells, limit)}
    out = []
    for i, x in enumerate(cells):
        for y in cells[i + 1:]:
            if x["platform"] != y["platform"]:
                continue
            a, b = sorted((x, y), key=lambda c: (c["imgsz"], c["latency_ms"]), reverse=True)
            spans = spans_offset(a, b)
            measured, offset = apart(a, b), (apart_without_offset(a, b) if spans else None)
            within = measured <= limit + EPS
            if not (within or (spans and offset <= limit + EPS)):
                continue
            notes = [SPANS_NOTE if spans else "", "" if within else f"not within {100 * limit:.0f} % as measured ({100 * measured:.1f} %); "
                     f"{100 * offset:.1f} % with session 6's latency taken as 6 % high; kept as measured, so not a pair (#46)", versions_note(a, b)]
            row = {"a": a, "b": b, "latency_ratio": a["latency_ms"] / b["latency_ms"], "apart_pct": 100 * measured,
                   "apart_pct_without_offset": None if offset is None else 100 * offset, "within_limit": within,
                   "paper_pair": (id(a), id(b)) in paper, "spans_session_6": spans, "note": "; ".join(n for n in notes if n)}
            row.update({k: a[col] - b[col] for k, col in DIFFS.items()})
            out.append(row)
    return out


# ---------------------------------------------------------------- the figure's data

RUNTIMES = {"onnxruntime": "ONNX Runtime", "openvino": "OpenVINO", "ncnn": "NCNN"}  # the panels, in this order
MIN_CELLS = 3  # a platform gets the figure once it has this many cells (the phone's has 2 now)


def headline_cells(cells):
    """The cells the figure draws: the one-to-many head (the NMS-free head is result B's), and no stock INT8 cell whose
    per-channel cell exists."""
    int8 = int8_notes(cells)
    return [c for c in cells if c["head"] == "one-to-many" and int8.get(id(c), (True, ""))[0]]


def figure_platforms(cells):
    """The platforms that get the figure: those with MIN_CELLS cells or more."""
    n = {}
    for c in cells:
        n[c["platform"]] = n.get(c["platform"], 0) + 1
    return [p for p in n if n[p] >= MIN_CELLS]


def figure_data(cells, pairs, platform):
    """One platform's panels, one per runtime it has: per study model its headline cells (points) and its baselines in
    input-resolution order (the line), and the panel's headline pairs (connectors)."""
    shown = [c for c in headline_cells(cells) if c["platform"] == platform]
    panels = []
    for rt in [r for r in RUNTIMES if any(c["runtime"] == r for c in shown)] + sorted({c["runtime"] for c in shown} - set(RUNTIMES)):
        here = [c for c in shown if c["runtime"] == rt]
        models = {m: {"points": [c for c in here if c["model"] == m],
                      "baselines": sorted((c for c in here if c["model"] == m and c["configuration"] == "baseline"), key=lambda c: c["imgsz"])}
                  for m in sorted({c["model"] for c in here})}
        connectors = [p for p in pairs if p["headline"] and p["compressed"]["platform"] == platform and p["compressed"]["runtime"] == rt]
        panels.append({"runtime": rt, "models": models, "connectors": connectors})
    return panels


# ---------------------------------------------------------------- tables

CELL_COLUMNS = ["session", "cell", "checkpoint", "configuration", "imgsz", "trained_imgsz", "untrained_imgsz", "precision", "int8_recipe",
                "latency_ms", "ap", "ap_small", "throttled", "below_2000_mhz", "clock_median_mhz"]
ORDER = {"quantized": 0, "pruned": 1, "combined": 2}


def _value(v):
    return "" if v is None else int(v) if isinstance(v, bool) else round(v, 6) if isinstance(v, float) else v


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows({k: _value(v) for k, v in r.items()} for r in rows)


def sort_pairs(pairs):
    """By platform (the Pi 5 first), runtime (the panels' order), model, the baseline's input resolution, the compressed
    configuration and input resolution, and each stock INT8 row after its headline row."""
    rt = list(RUNTIMES)
    return sorted(pairs, key=lambda p: (p["compressed"]["platform"] != "pi5", p["compressed"]["platform"],
                                        rt.index(p["compressed"]["runtime"]) if p["compressed"]["runtime"] in rt else len(rt),
                                        p["compressed"]["model"], p["baseline"]["imgsz"], ORDER[p["compressed"]["configuration"]],
                                        p["compressed"]["imgsz"], not p["headline"]))


def pair_row(p):
    """One equal_latency.csv row."""
    c, b = p["compressed"], p["baseline"]
    row = {"platform": c["platform"], "runtime": c["runtime"], "model": c["model"], "compressed": label(c), "baseline": label(b)}
    for side, x in (("compressed", c), ("baseline", b)):
        row.update({f"{side}_{k}": x[k] for k in CELL_COLUMNS})
    row.update({k: p[k] for k in ["latency_ratio", "apart_pct"] + list(DIFFS) + ["spans_session_6", "headline", "note"]})
    return row


def all_row(p):
    """One equal_latency_all.csv row."""
    a, b = p["a"], p["b"]
    row = {"platform": a["platform"], "a": label(a), "b": label(b)}
    for side, x in (("a", a), ("b", b)):
        row.update({f"{side}_{k}": x[k] for k in ["runtime", "model", "head"] + CELL_COLUMNS})
    row.update({k: p[k] for k in ["latency_ratio", "apart_pct", "apart_pct_without_offset", "within_limit", "paper_pair", "spans_session_6"]
                + list(DIFFS) + ["note"]})
    return row


BS = chr(92)


def _signed(v, nd):
    return f"{v:+.{nd}f}".replace("-", "$-$")


def _tex_cell(c):
    """'B1 INT8 at 800, stock' with a dagger at an untrained input resolution."""
    return label(c) + (", stock" if c["int8_recipe"] == "stock" else "") + (f"$^{{{BS}dagger}}$" if c["untrained_imgsz"] else "")


def _tex_ms(c, spans):
    """The latency, t when the cell was flagged throttled, S when it is session 6's in a pair that spans it."""
    return (f"{c['latency_ms']:.1f}" + (" t" if c["throttled"] == "throttled" else "")
            + (f"$^{{{BS}S}}$" if spans and c["session"] in SESSION_OFFSET else ""))


def write_tex(pairs, path):
    """The Pi 5's pairs as a booktabs table: a block per runtime, each stock INT8 row after its headline pair."""
    delta = f"${BS}Delta$"
    head = (f"Runtime & Model & Compressed cell & ms & Baseline & ms & Ratio & Apart ({BS}%) & {delta}AP-small & {delta}AP & {delta}AP-medium & "
            f"{delta}AP-large & Images below 2000 MHz & Median clock (MHz) {BS}{BS}")
    lines = ["% Result A (#46): the equal-latency pairs on the Pi 5. Written by studypairs.py from results/cells.csv.",
             f"{BS}begin{{tabular}}{{lllrlrrrrrrrcc}}", f"{BS}toprule", head]
    last = None
    for p in sort_pairs([p for p in pairs if p["compressed"]["platform"] == "pi5"]):
        c, b = p["compressed"], p["baseline"]
        if c["runtime"] != last:
            lines.append(f"{BS}midrule")
        cells = [RUNTIMES.get(c["runtime"], c["runtime"]) if c["runtime"] != last else "", c["model"], _tex_cell(c),
                 _tex_ms(c, p["spans_session_6"]), _tex_cell(b), _tex_ms(b, p["spans_session_6"]), f"{p['latency_ratio']:.3f}",
                 f"{p['apart_pct']:.1f}"] + [_signed(p[k], 4) for k in DIFFS]
        cells += [f"{c['below_2000_mhz']} / {b['below_2000_mhz']}", f"{c['clock_median_mhz']} / {b['clock_median_mhz']}"]
        lines.append(" & ".join(cells) + f" {BS}{BS}")
        last = c["runtime"]
    notes = ["Equal latency: median latencies 10\\% apart or less, the difference divided by the smaller of the two (Apart).",
             "Ratio: the compressed cell's latency over the baseline's. Differences: compressed cell minus baseline.",
             "INT8 cells are ONNX Runtime per-channel unless marked stock; a stock row follows its per-channel pair.",
             f"$^{{{BS}dagger}}$ A checkpoint run at an input resolution it was not trained at (800 from the 960 models, 1120 from the "
             "1280 ones): its accuracy there is a little lower.",
             "t: the cell was flagged throttled. Images below 2000 MHz and the median clock: compressed / baseline.",
             f"$^{{{BS}S}}$ The pair spans session 6, whose latencies are about 6\\% high against the other sessions' (kept as measured)."]
    lines += [f"{BS}bottomrule"] + [f"{BS}multicolumn{{14}}{{l}}{{{BS}footnotesize {n}}} {BS}{BS}" for n in notes] + [f"{BS}end{{tabular}}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- the figure

PLATFORMS = {"pi5": "the Pi 5", "m34": "the Galaxy M34"}
COLOURS = {"YOLO11s": studynms.BLUE, "YOLO26s": studynms.ORANGE}  # the dataviz reference palette, slots 1 and 2, as result B
MARKERS = {"baseline": "o", "quantized": "s", "pruned": "^", "distilled": "D", "combined": "P"}
CONFIG_NAMES = {"baseline": "Baseline", "quantized": "Quantized (INT8)", "pruned": "Pruned", "distilled": "Distilled", "combined": "Combined (INT8)"}


def _point(ax, c):
    """One cell: the marker's shape is its configuration, the colour its model; hollow at an untrained input resolution."""
    colour = COLOURS.get(c["model"], studynms.INK)
    hollow = bool(c["untrained_imgsz"])
    ax.scatter([c["latency_ms"]], [c["ap_small"]], s=58, marker=MARKERS[c["configuration"]], zorder=4, facecolors="white" if hollow else colour,
               edgecolors=colour if hollow else "white", linewidths=1.6 if hollow else 1.0)


def _axes(fig, n):
    """With three runtimes the first (ONNX Runtime, the one with the INT8 pairs) gets a large panel on the left and the other two
    stack on its right; otherwise the panels sit side by side. All share the AP-small axis."""
    if n == 3:
        gs = fig.add_gridspec(2, 2, width_ratios=(1.75, 1))
        first = fig.add_subplot(gs[:, 0])
        return [first, fig.add_subplot(gs[0, 1], sharey=first), fig.add_subplot(gs[1, 1], sharey=first)]
    axes = [fig.add_subplot(1, n, 1)]
    return axes + [fig.add_subplot(1, n, i + 1, sharey=axes[0]) for i in range(1, n)]


def figure(panels, platform, out_dir):
    """AP-small against median latency, figure_data's panels. Each baseline is labelled with its input resolution (the first of a
    line at its left, then YOLO11s's below its line and YOLO26s's above), and every other cell not at 640 on the side away from
    its pair's baseline (else to its right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ink, muted = studynms.INK, studynms.MUTED
    fig = plt.figure(figsize=(13, 8.2) if len(panels) == 3 else (5.5 * len(panels) + 0.5, 6), facecolor="white")
    for i, (ax, panel) in enumerate(zip(_axes(fig, len(panels)), panels)):
        left = {id(p["compressed"]) for p in panel["connectors"] if p["baseline"]["latency_ms"] > p["compressed"]["latency_ms"]}
        for j, (model, s) in enumerate(panel["models"].items()):
            line = s["baselines"]
            ax.plot([c["latency_ms"] for c in line], [c["ap_small"] for c in line], color=COLOURS.get(model, ink), linewidth=1.6, zorder=2)
            for k, c in enumerate(line):
                at = {"xytext": (-8, 0), "ha": "right", "va": "center"} if k == 0 else {"xytext": (0, 8 if j else -14), "ha": "center"}
                ax.annotate(str(c["imgsz"]), (c["latency_ms"], c["ap_small"]), textcoords="offset points", fontsize=7.5, color=muted, **at)
            for c in s["points"]:
                _point(ax, c)
                if c["configuration"] != "baseline" and c["imgsz"] != 640:
                    ax.annotate(str(c["imgsz"]), (c["latency_ms"], c["ap_small"]), textcoords="offset points", va="center", fontsize=7.5,
                                color=ink, **({"xytext": (-8, 0), "ha": "right"} if id(c) in left else {"xytext": (8, 0), "ha": "left"}))
        for p in panel["connectors"]:
            b, c = p["baseline"], p["compressed"]
            ax.annotate("", xy=(c["latency_ms"], c["ap_small"]), xytext=(b["latency_ms"], b["ap_small"]), zorder=3,
                        arrowprops={"arrowstyle": "-|>", "color": ink, "linewidth": 1.2, "shrinkA": 4, "shrinkB": 5, "mutation_scale": 11})
        precisions = sorted({c["precision"] for s in panel["models"].values() for c in s["baselines"]})
        ax.set_title(f"({chr(97 + i)}) {RUNTIMES.get(panel['runtime'], panel['runtime'])}, baselines at {' / '.join(precisions)}",
                     loc="left", fontsize=10.5, color=ink)
        ax.set_xlabel("Median latency per image (ms)" + (", at 2000 MHz" if platform == "pi5" else ""), color=muted)
        ax.grid(color=studynms.GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(muted)
        ax.tick_params(colors=muted)
        ax.set_ylabel("AP-small (COCO, VisDrone test-dev)", color=muted)
    handles = [Line2D([], [], color=COLOURS[m], marker="o", markeredgecolor="white", linewidth=1.6, label=f"{m}: baselines, 640 to 1280")
               for m in COLOURS if any(m in p["models"] for p in panels)]
    shown = {c["configuration"] for p in panels for s in p["models"].values() for c in s["points"]}
    handles += [Line2D([], [], color=muted, marker=MARKERS[k], linestyle="none", markersize=7, label=CONFIG_NAMES[k]) for k in MARKERS if k in shown]
    handles += [Line2D([], [], color=muted, marker="o", markerfacecolor="white", markeredgewidth=1.6, linestyle="none", markersize=7,
                       label="Run at an input resolution it was not trained at"),
                Line2D([], [], color=ink, linewidth=1.2, marker=">", markevery=[1], markersize=5, label="Equal-latency pair: baseline to compressed cell")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5, labelcolor=muted)
    fig.suptitle(f"AP-small against latency on {PLATFORMS.get(platform, platform)}: compression spent on input resolution",
                 x=0.01, ha="left", fontsize=11.5, color=ink)
    fig.text(0.01, 0.935, "Numbers: input resolution; a compressed cell with none is at 640. Arrow: a compressed cell at a larger input resolution "
             "within 10 % of a baseline's median latency" + chr(10) + "(the headline pairs: INT8 is per-channel). One-to-many head."
             + (" The baselines at 1120 are session 6's, whose latencies are about 6 % high; the INT8 cells at 1120 are session 7's."
                if platform == "pi5" else ""),
             fontsize=8.5, color=muted, va="top")
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    studynms.save_figure(fig, out_dir / "figures" / f"result_a_{platform}")


# ---------------------------------------------------------------- the script

STUDY = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cells", type=Path, default=STUDY / "results" / "cells.csv")
    ap.add_argument("--out", type=Path, default=STUDY / "results")
    a = ap.parse_args(argv)
    cells, aside = usable(load_cells(a.cells))
    for c, why in aside:
        print(f"set aside: {c['session']} {c['cell']}: {why}")
    pairs = sort_pairs(equal_latency_pairs(cells))
    _write(a.out / "tables" / "equal_latency.csv", [pair_row(p) for p in pairs])
    write_tex(pairs, a.out / "tables" / "equal_latency.tex")
    every = sorted(all_pairs(cells), key=lambda p: (p["a"]["platform"], p["a"]["latency_ms"], p["b"]["latency_ms"]))
    _write(a.out / "tables" / "equal_latency_all.csv", [all_row(p) for p in every])
    platforms = figure_platforms(cells)
    for platform in platforms:
        figure(figure_data(cells, pairs, platform), platform, a.out)
    for p in pairs:
        c, b = p["compressed"], p["baseline"]
        print(f"{'' if p['headline'] else '(stock) '}{c['platform']} {c['runtime']}: {label(c)} {c['latency_ms']:.1f} against {label(b)} "
              f"{b['latency_ms']:.1f} ms, {p['apart_pct']:.1f} % apart, AP-small {p['ap_small_diff']:+.4f}, AP {p['ap_diff']:+.4f}")
    print(f"{sum(p['headline'] for p in pairs)} headline pairs, {sum(not p['headline'] for p in pairs)} stock rows; "
          f"{len(every)} rows in equal_latency_all.csv; figures for {', '.join(platforms)}; written under {a.out}")


if __name__ == "__main__":
    main()
