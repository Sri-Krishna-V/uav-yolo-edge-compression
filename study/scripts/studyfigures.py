"""The paper's figures (#49), in studyfig.py's style, from results/cells.csv and the tables studypairs.py, studytables.py and
studynms.py write. Runs nothing on a Pi and computes nothing new: re-run it after those three when a session lands.

  python study/scripts/studyfigures.py [--results study/results] [--pdf]

Outputs, under <results>/figures/paper/, each a 600 dpi .png, and with --pdf its .pdf (specs: .scratch/uav-yolo-edge/findings/
49-research-figures.md, section 9):
  fig_a_pi5          174 mm. (a) AP-small against median latency (log), ONNX Runtime on the Pi 5: each model's baselines from 640
                     to 1280 as a line, the compressed cells, a connector per headline pair. (b) Every Pi 5 headline pair's
                     AP-small difference (compressed minus baseline) and how far apart its latencies are; a bootstrap interval
                     per pair once tables/equal_latency.csv has ap_small_ci_low and ap_small_ci_high (#48)
  fig_a_m34          88 mm. The same for the phone, on a grey ground (its latency is provisional, #34), each latency with a
                     +/-5 % bar for the drift its checks failed by
  fig_s1_runtimes    174 mm. (a) of fig_a_pi5 for OpenVINO and NCNN
  fig_b_nms          174 mm. Post-processing time (log) against objects per image and against the candidates entering NMS, per
                     head and threshold, for one pair (default: YOLO26s A2 per-channel INT8, session 7): median and IQR
  fig_c_ratio        88 mm. YOLO26s's median latency over YOLO11s's against input resolution, per runtime (baselines; INT8
                     per-channel; stock INT8 thin grey)
  fig_d_int8         174 mm. Per checkpoint and input resolution: INT8's speed-up, AP change and AP-small change, per-channel
                     and stock, on the Pi 5 and the phone
  fig_small_object   88 mm. AP-small drop against AP drop per compressed configuration at 640 on the Pi 5, with y = x, 1.3x, 2x
Standard library and matplotlib only; no torch, no ultralytics.
"""

import argparse
import csv
import math
import re
from pathlib import Path

import studyfig as sf
import studypairs as sp

STUDY = Path(__file__).resolve().parents[1]
DRIFT = 0.05  # the bound of the +/- bar on the phone's latencies (#34); main sets it from the drift notes in cells.csv
DRIFT_MOVES = []  # the phone sessions' drift checks, smallest first, as their cells.csv notes give them
DRIFT_NOTE = re.compile(r"^drift ([+-][\d.]+) %")


def phone_drift(cells):
    """The phone sessions' failed drift checks, from the notes studyresults.SESSION_NOTES gives their cells, smallest first,
    and the bar's bound: the largest, rounded up to a whole per cent."""
    moves = sorted({float(m[1]) for c in cells if c["platform"] == "m34" and (m := DRIFT_NOTE.match(c.get("note") or ""))})
    return moves, math.ceil(max(moves, default=5)) / 100
B_CELL = ("pi5-matrix-7-int8pc", "onnxruntime-int8pc")  # result B's pair: the matrix's INT8 (INVARIANTS.md section 8)


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v):
    return None if v in ("", None) else float(v)


def res(n):
    """An input resolution as the paper prints it: '1120' (MDPI's thousands comma is for five digits or more)."""
    return str(int(n))


# ---------------------------------------------------------------- result A


def _log_latency(ax, lo, hi):
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ticks = [t for t in (100, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000) if lo <= t <= hi]
    if len(ticks) < 3:  # a narrow range (the phone's): every 50 or 100 ms, at most five ticks
        step = next(s for s in (50, 100, 200) if (hi - lo) / s <= 5)
        ticks = [t for t in range(step, 2001, step) if lo <= t <= hi]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Median latency per image (ms, log scale)")


def _cell_marker(ax, c, x=None, size=26, xerr=None):
    colour = sf.MODEL_COLOR[c["model"]]
    hollow = bool(c["untrained_imgsz"])
    x = c["latency_ms"] if x is None else x
    if xerr:
        ax.errorbar([x], [c["ap_small"]], xerr=[[x - x / (1 + xerr)], [x * (1 + xerr) - x]], fmt="none", ecolor=colour,
                    elinewidth=0.75, capsize=1.5, zorder=3)
    ax.scatter([x], [c["ap_small"]], s=size, marker=sf.CONFIG_MARKER[c["configuration"]], zorder=5,
               facecolors="white" if hollow else colour, edgecolors=colour if hollow else "white", linewidths=1.0 if hollow else 0.6)


def _tag(c):
    return sf.DAGGER if c.get("throttled") == "throttled" else ""


def scatter_panel(ax, panel, ylim, xlim, models=None, drift=None):
    """AP-small against latency for one runtime: each model's baselines as a line labelled with their input resolutions, and
    only the compressed cells in a headline pair (the rest are results C and D's), labelled with theirs and joined to their
    baseline by a thin connector. With one model the labels sit above the points; with two, YOLO11s's below and YOLO26s's above."""
    models = sorted(panel["models"]) if models is None else models
    paired = {id(p["compressed"]) for p in panel["connectors"]}
    for p in panel["connectors"]:
        b, c = p["baseline"], p["compressed"]
        if c["model"] in models:
            ax.plot([b["latency_ms"], c["latency_ms"]], [b["ap_small"], c["ap_small"]], color=sf.MUTED, linewidth=0.75, zorder=4)
    for j, model in enumerate(models):
        s = panel["models"].get(model)
        if not s:
            continue
        colour = sf.MODEL_COLOR[model]
        line = s["baselines"]
        above = len(models) > 1 and j == 1  # with two models YOLO26s's baselines are labelled above-left, else below-right
        ax.plot([c["latency_ms"] for c in line], [c["ap_small"] for c in line], color=colour, linewidth=1.25, zorder=2)
        # a baseline whose loss pair lies to its right would have its label under that pair's: it is labelled above-left
        flip = {id(p["baseline"]) for p in panel["connectors"]
                if p["ap_small_diff"] < 0 and p["compressed"]["latency_ms"] > p["baseline"]["latency_ms"]}
        for c in line:
            _cell_marker(ax, c, xerr=drift)
            up = above or id(c) in flip
            ax.annotate(res(c["imgsz"]) + _tag(c), (c["latency_ms"], c["ap_small"]), textcoords="offset points",
                        xytext=(-4, 7) if up else (5, -8), ha="right" if up else "left", va="center", color=colour,
                        path_effects=sf.halo(), zorder=6)
        gain = {id(p["compressed"]): p["ap_small_diff"] >= 0 for p in panel["connectors"]}
        drawn = [p["compressed"] for p in panel["connectors"] if p["compressed"]["model"] in models]
        for c in s["points"]:
            if id(c) in paired:  # a gain sits above the baselines' line, labelled above-left; a loss below it, labelled under it
                _cell_marker(ax, c, xerr=drift)
                # a gain with another drawn paired cell just to its left (within 10 % latency, 0.005 AP-small), whose label
                # takes the space above-left, is labelled straight above
                top = gain[id(c)] and any(o is not c and o["latency_ms"] < c["latency_ms"] < 1.1 * o["latency_ms"]
                                          and abs(o["ap_small"] - c["ap_small"]) < 0.005 for o in drawn)
                ax.annotate(res(c["imgsz"]) + _tag(c), (c["latency_ms"], c["ap_small"]), textcoords="offset points",
                            xytext=(0, 9) if top else (-5, 6) if gain[id(c)] else (0, -10),
                            ha="center" if top else "right" if gain[id(c)] else "center", va="center",
                            color=sf.INK, path_effects=sf.halo(), zorder=6)
    _log_latency(ax, *xlim)
    ax.set_ylim(*ylim)
    ax.set_ylabel("AP-small")


def _pair_name(p, two_lines=False):
    c, b = p["compressed"], p["baseline"]
    def side(x):
        return f"{'pruned ' if x['configuration'] == 'pruned' else ''}{x['precision']} {res(x['imgsz'])}"
    return f"{c['model']}:{chr(10) if two_lines else ' '}{side(c)} vs {side(b)}"


def _throttled(p):
    return "throttled" in (p["compressed"].get("throttled"), p["baseline"].get("throttled"))


def forest_panel(ax, pairs, ci, nbins=7, drift=None):
    """One row per headline pair: its AP-small difference (a dot, the compressed cell's marker in its model's colour), a
    bootstrap interval when #48's columns exist, and on the right how far apart the two latencies are (a dagger when either
    cell throttled, a section sign when the pair spans session 6). With a drift bound, a pair whose gap plus the bound passes
    the 10 % rule is labelled not robust to the drift."""
    from matplotlib.transforms import blended_transform_factory
    rows, labels, groups = [], [], []
    last = None
    for p in pairs:
        rt = p["compressed"]["runtime"]
        if rt != last:
            groups.append((len(rows) + len(groups), sf.RUNTIME_NAME.get(rt, rt)))
            last = rt
        rows.append((len(rows) + len(groups), p))
    if len(groups) == 1:  # one runtime: no group row
        rows, groups = [(i, p) for i, (_, p) in enumerate(rows)], []
    n = len(rows) + len(groups)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for y, p in rows:
        c = p["compressed"]
        d = p["ap_small_diff"]
        key = (c["cell"], p["baseline"]["cell"])
        if key in ci:
            lo, hi = ci[key]
            ax.plot([lo, hi], [y, y], color=sf.MODEL_COLOR[c["model"]], linewidth=1.25, zorder=3)
        ax.scatter([d], [y], s=24, marker=sf.CONFIG_MARKER[c["configuration"]], color=sf.MODEL_COLOR[c["model"]],
                   edgecolors="white", linewidths=0.5, zorder=4)
        flag = (sf.DAGGER if _throttled(p) else "") + (sf.SECTION if p["spans_session_6"] else "")
        note = ""
        if drift is not None and p["apart_pct"] + drift * 100 > sp.LIMIT * 100:  # the gap plus the drift passes the 10 % rule
            note = "\nnot robust\nto drift"
        ax.text(1.02, y, f"{p['apart_pct']:.1f} %{flag}{note}", transform=trans, ha="left", va="center")
    for y, name in groups:  # in the label column, right-aligned with the row labels, clear of the spine and the zero line
        ax.text(-0.015, y, name, transform=trans, ha="right", va="center", fontweight="bold")
    two = drift is not None
    ax.set_yticks([y for y, _ in rows], [_pair_name(p, two_lines=two) for _, p in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(n - 0.4, -0.8)
    ax.axvline(0, color=sf.INK, linewidth=0.6, zorder=1)
    ax.grid(False)
    ax.grid(True, axis="x")
    ax.text(1.02, -0.8, "latency\napart" if two else "latency apart", transform=trans, ha="left", va="center", color=sf.MUTED)
    from matplotlib.ticker import MaxNLocator
    ax.set_xlabel(f"AP-small, compressed {sf.MINUS} baseline")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(lambda v, _: sf.signed(v, 3) if abs(v) > 1e-12 else "0")


def _ylim(cells):
    v = [c["ap_small"] for c in cells]
    return (math.floor(min(v) * 100) / 100 - 0.005, math.ceil(max(v) * 100) / 100 + 0.005)


def _xlim(panels):
    """The latency range of what scatter_panel draws: the baselines and the paired compressed cells."""
    v = [c["latency_ms"] for p in panels for s in p["models"].values() for c in s["baselines"]]
    v += [q["compressed"]["latency_ms"] for p in panels for q in p["connectors"]]
    return (min(v) / 1.15, max(v) * 1.28)  # the right pad leaves room for a top baseline's label (1280 and its dagger)


def _legend(fig, configurations, hollow=True, ncols=4, lines=True, marks=(), extra=()):
    """The models' baselines (a line through them, or markers alone where no model has two), the compressed configurations
    drawn, hollow = untrained input resolution, then the text marks (dagger, section sign) and any extra handles."""
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=sf.MODEL_COLOR[m], marker="o", markeredgecolor="white", linestyle="-" if lines else "none",
                label=f"{m} baseline{'s' if lines else ''}") for m in sf.MODEL_COLOR]
    h += [Line2D([], [], color=sf.MUTED, marker=sf.CONFIG_MARKER[k], linestyle="none", label=sf.CONFIG_NAME[k])
          for k in sf.CONFIG_MARKER if k in configurations and k != "baseline"]
    if hollow:
        h.append(Line2D([], [], color=sf.MUTED, marker="o", markerfacecolor="white", linestyle="none", label="Untrained input resolution"))
    h += list(extra)
    h += [sf.text_entry(m) for m in marks]
    fig.legend(handles=h, loc="outside lower center", ncols=ncols, handletextpad=0.5)


MARK_T = f"{sf.DAGGER} the cell (or a cell of the pair) throttled"
MARK_S = f"{sf.SECTION} the pair spans session 6"


def _ci(rows):
    """#48's intervals, when equal_latency.csv carries them: {(compressed cell, baseline cell): (low, high)}."""
    if not rows or "ap_small_ci_low" not in rows[0]:
        return {}
    return {(r["compressed_cell"], r["baseline_cell"]): (float(r["ap_small_ci_low"]), float(r["ap_small_ci_high"]))
            for r in rows if r["ap_small_ci_low"] not in ("", None)}


def result_a(cells, pairs, eq_rows, out):
    ci = _ci(eq_rows)
    shown = [c for c in sp.headline_cells(cells)]
    ylim = _ylim(shown)
    pi5 = sp.figure_data(cells, pairs, "pi5")
    def configs(panels):
        return {p["compressed"]["configuration"] for panel in panels for p in panel["connectors"]}
    headline = [p for p in pairs if p["headline"] and p["compressed"]["platform"] == "pi5"]
    order = {r: i for i, r in enumerate(sp.RUNTIMES)}
    headline.sort(key=lambda p: (order.get(p["compressed"]["runtime"], 9), p["compressed"]["model"], p["compressed"]["imgsz"],
                                 p["compressed"]["configuration"]))
    with sf.style():
        fig = sf.figure(sf.DOUBLE, 150)
        top, bottom = fig.subfigures(2, 1, height_ratios=(70, 80))
        a, b = top.subplots(1, 2, sharey=True)
        ort = next(p for p in pi5 if p["runtime"] == "onnxruntime")
        xl = _xlim([ort])
        for ax, letter, model in ((a, "a", "YOLO11s"), (b, "b", "YOLO26s")):
            scatter_panel(ax, ort, ylim, xl, models=[model])
            sf.panel_label(ax, letter, f"{model}, ONNX Runtime, Pi 5")
        b.set_ylabel("")
        c = bottom.subplots()
        forest_panel(c, headline, ci)
        sf.panel_label(c, "c", "Headline equal-latency pairs, Pi 5, all runtimes")
        marks = [MARK_T] + ([MARK_S] if any(p["spans_session_6"] for p in headline) else [])
        _legend(fig, configs(pi5), ncols=3, marks=marks)
        sf.save(fig, out / "fig_a_pi5")

        rest = [p for p in pi5 if p["runtime"] != "onnxruntime"]
        if rest:
            fig = sf.figure(sf.DOUBLE, 30 + 60 * len(rest))
            axes = fig.subplots(len(rest), 2, sharey=True, squeeze=False)
            letters = iter("abcdef")
            for row, p in zip(axes, rest):
                xl = _xlim([p])
                for ax, model in zip(row, sf.MODEL_COLOR):
                    scatter_panel(ax, p, ylim, xl, models=[model])
                    sf.panel_label(ax, next(letters), f"{model}, {sf.RUNTIME_NAME[p['runtime']]}, Pi 5")
                row[1].set_ylabel("")
            _legend(fig, configs(rest), marks=[f"{sf.DAGGER} the cell throttled"])
            sf.save(fig, out / "fig_s1_runtimes")

        m34 = sp.figure_data(cells, pairs, "m34")
        if m34:
            phone = [p for p in pairs if p["headline"] and p["compressed"]["platform"] == "m34"]
            fig = sf.figure(sf.SINGLE, 40 + 70 + 11 * len(phone))  # 11 mm a row: room for a three-line note on the right
            top, bottom = fig.subplots(2, 1, height_ratios=(70, 14 + 11 * len(phone)))
            panel = m34[0]
            scatter_panel(top, panel, ylim, _xlim(m34), drift=DRIFT)
            sf.tint(top)
            sf.panel_label(top, "a", "Galaxy M34, a consumer-grade phone")
            moves = f"{DRIFT_MOVES[0]:+.1f} % to {DRIFT_MOVES[-1]:+.1f} %" if len(DRIFT_MOVES) > 1 else f"{DRIFT * 100:+.0f} %"
            top.text(0.03, 0.97, f"Latency provisional:\ndrift {moves},\nbars {chr(177)}{DRIFT * 100:.0f} %", transform=top.transAxes,
                     ha="left", va="top", color=sf.MUTED)
            forest_panel(bottom, phone, ci, nbins=3, drift=DRIFT)
            sf.tint(bottom)
            sf.panel_label(bottom, "b", "Its equal-latency pair")
            from matplotlib.lines import Line2D
            drawn = [c for p in m34 for s in p["models"].values() for c in s["baselines"]]
            lines = any(len(s["baselines"]) > 1 for p in m34 for s in p["models"].values())
            bar = Line2D([], [], color=sf.MUTED, marker="|", markersize=6, linestyle="-", linewidth=0.75,
                         label=f"{chr(177)}{DRIFT * 100:.0f} % drift bar on latency")
            _legend(fig, configs(m34), hollow=any(c["untrained_imgsz"] for c in drawn), ncols=2, lines=lines, extra=[bar])
            sf.save(fig, out / "fig_a_m34")


# ---------------------------------------------------------------- result B


def result_b(bins, out, cell=B_CELL):
    rows = [r for r in bins if r["session"] == cell[0] and r["variant"] == cell[1]]
    if not rows:
        return
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator
    thresholds = (("0.001", "evaluation threshold 0.001", "-", "o"), ("0.25", "deployment threshold 0.25", "--", None))
    one = sf.HEAD_STYLE["one_to_many"]
    free = sf.HEAD_STYLE["nms_free"]
    # The NMS-free head's times sit at the timer's resolution (0.1-0.25 ms, bin medians jumping between steps), so its bins
    # carry no trend: it is drawn as one pooled band (all its bins, both thresholds) with the median of its bin medians.
    nf = [r for r in rows if r["x"] == "objects per image" and r["head"] == "nms_free"]
    nf_med = sorted(float(r["median"]) for r in nf)
    nf_mid = nf_med[len(nf_med) // 2] if nf_med else None
    nf_lo, nf_hi = (min(float(r["q1"]) for r in nf), max(float(r["q3"]) for r in nf)) if nf else (None, None)
    with sf.style():
        fig = sf.figure(sf.DOUBLE, 76)
        axes = fig.subplots(1, 2, sharey=True)
        xs = ["objects per image", "candidates entering NMS at 0.001"]
        xlabels = ["Objects per image (ground truth)", "Candidates entering NMS at the 0.001 evaluation threshold"]
        for i, (ax, x) in enumerate(zip(axes, xs)):
            for thr, _, ls, marker in thresholds:
                rs = sorted((r for r in rows if r["x"] == x and r["head"] == "one_to_many" and r["threshold"].startswith(thr)),
                            key=lambda r: float(r["x_median"]))
                if not rs:
                    continue
                xv = [float(r["x_median"]) for r in rs]
                ax.fill_between(xv, [float(r["q1"]) for r in rs], [float(r["q3"]) for r in rs], color="#D9D9D9", linewidth=0, zorder=1)
                ax.plot(xv, [float(r["median"]) for r in rs], color=one["color"], linestyle=ls, marker=marker or "none",
                        markersize=3, markeredgecolor="white", markeredgewidth=0.5, zorder=3)
            if i == 0 and nf_mid is not None:
                lo, hi = ax.get_xlim()
                ax.fill_between([lo, hi], [nf_lo] * 2, [nf_hi] * 2, color="#EBD3E1", linewidth=0, zorder=1)
                ax.plot([lo, hi], [nf_mid] * 2, color=free["color"], linewidth=1.25, zorder=3)
                ax.set_xlim(lo, hi)
                ax.annotate("NMS-free head, both thresholds:\nat the timer's resolution, no trend",
                            (0.98, nf_hi), xycoords=("axes fraction", "data"), textcoords="offset points", xytext=(0, 4),
                            ha="right", va="bottom", color=free["color"], path_effects=sf.halo())
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]))
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
            ax.set_xlabel(xlabels[i])
            ax.grid(True, axis="both")
        axes[0].set_ylabel("Post-processing time (ms, log scale)")
        sf.panel_label(axes[0], "a", "Against the objects in the image, both heads")
        sf.panel_label(axes[1], "b", "Against the candidates entering NMS, one-to-many head only")
        h = [Line2D([], [], color=one["color"], marker=marker or "none", markersize=3, markeredgecolor="white", linestyle=ls,
                    label=f"one-to-many head, {name}") for _, name, ls, marker in thresholds]
        h.append(Line2D([], [], color=free["color"], linestyle="-", label="NMS-free head, both thresholds (pooled median)"))
        h.append(Line2D([], [], color="#D9D9D9", linewidth=5, label="interquartile range"))
        fig.legend(handles=h, loc="outside lower center", ncols=2)
        sf.save(fig, out / "fig_b_nms")


# ---------------------------------------------------------------- result C


def result_c(rows, out, platform="pi5"):
    from matplotlib.lines import Line2D
    rows = [r for r in rows if r["platform"] == platform and r["configuration"] in ("baseline", "quantized")]
    lines = {}
    for r in rows:
        lines.setdefault((r["runtime"], r["precision"], r["int8_recipe"]), []).append(r)
    with sf.style():
        fig = sf.figure(sf.SINGLE, 100)
        ax = fig.subplots()
        ax.axhline(1, color=sf.INK, linewidth=0.6, zorder=1)
        # At 88 mm the lines are named in a one-column legend (colour, line style and marker together), not by end labels.
        h = []
        order = {("onnxruntime", "FP32"): 0, ("openvino", "FP32"): 1, ("ncnn", "FP16"): 2, ("onnxruntime", "INT8"): 3}
        for (rt, prec, recipe), rs in sorted(lines.items(), key=lambda kv: (kv[0][2] == "stock", order.get(kv[0][:2], 9))):
            rs.sort(key=lambda r: int(r["imgsz"]))
            x = [int(r["imgsz"]) for r in rs]
            y = [float(r["latency_ratio"]) for r in rs]
            marker = sf.CONFIG_MARKER[rs[0]["configuration"]]
            unclean = [r["clean"] == "0" for r in rs]
            line_tag = sf.DAGGER if all(unclean) else ""  # a whole line not clean takes one dagger, in its legend entry
            end = f"{sf.signed((y[-1] - 1) * 100, 1, ' %')} at {x[-1]}"
            if recipe == "stock":
                st = {"color": "#888888", "linestyle": ":", "linewidth": 0.9}
                ax.plot(x, y, marker="x", markersize=3.5, zorder=2, **st)
                h.append(Line2D([], [], marker="x", markersize=3.5, label=f"{sf.RUNTIME_NAME[rt]} INT8, stock{line_tag} ({end})", **st))
                continue
            st = dict(sf.RUNTIME_STYLE[(rt, prec)])
            if all(unclean):  # every pair of the line has a cell below 2000 MHz: drawn thinner, so it does not dominate
                st["linewidth"] = 0.9
            ax.plot(x, y, marker=marker, markersize=4, markeredgecolor="white", markeredgewidth=0.5, zorder=3, **st)
            if not all(unclean):
                for xi, yi, u in zip(x, y, unclean):
                    if u:
                        ax.annotate(sf.DAGGER, (xi, yi), textcoords="offset points", xytext=(3, 3), ha="left", va="bottom",
                                    color=st["color"])
            kind = "INT8, per-channel" if prec == "INT8" else f"{prec} baseline"
            h.append(Line2D([], [], marker=marker, markersize=4, markeredgecolor="white", markeredgewidth=0.5,
                            label=f"{sf.RUNTIME_NAME[rt]} {kind}{line_tag} ({end})", **st))
        ax.set_xticks([640, 800, 960, 1120, 1280])
        ax.set_xlim(600, 1320)
        ax.set_xlabel("Input resolution (pixels)")
        ax.set_ylabel("Latency ratio, YOLO26s / YOLO11s")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
        h.append(sf.text_entry(f"{sf.DAGGER} every pair of the line has a cell that ran below 2000 MHz"))
        fig.legend(handles=h, loc="outside lower center", ncols=1, handlelength=2.4)
        sf.save(fig, out / "fig_c_ratio")


# ---------------------------------------------------------------- result D


def result_d(rows, out):
    from matplotlib.lines import Line2D
    rows = [r for r in rows if r["head"] == "one-to-many" and num(r["per_channel_speedup"]) is not None]
    rows.sort(key=lambda r: (r["platform"] != "pi5", r["model"], int(r["imgsz"])))
    ys, labels = [], []
    y = 0
    last = None
    groups = []
    for r in rows:
        g = (r["platform"], r["model"])
        if g != last:
            groups.append((y, f"{r['model']}, {'Pi 5' if r['platform'] == 'pi5' else 'M34 phone*'}"))
            y += 1
            last = g
        ys.append(y)
        untrained = " (u)" if r["untrained_imgsz"] == "1" else ""  # not a dagger: the dagger means throttled elsewhere
        labels.append(f"{r['checkpoint']} at {res(r['imgsz'])}{untrained}")
        y += 1
    n = y
    cols = [("speedup", "INT8 speed-up over FP32 (×)"), ("ap_diff_pct", "AP change (%)"), ("ap_small_diff_pct", "AP-small change (%)")]
    DAG2 = "‡"  # the phone's stock speed-up: its INT8 and FP32 cells come from different phone sessions
    with sf.style():
        from matplotlib.transforms import blended_transform_factory
        fig = sf.figure(sf.DOUBLE, 40 + 4.2 * n)
        axes = fig.subplots(1, 3, sharey=True)
        phone = [yi for yi, r in zip(ys, rows) if r["platform"] == "m34"]
        for k, (ax, (col, xlabel)) in enumerate(zip(axes, cols)):
            for yi, r in zip(ys, rows):
                colour = sf.MODEL_COLOR[r["model"]]
                pc, st = num(r[f"per_channel_{col}"]), num(r[f"stock_{col}"])
                drift = col == "speedup" and r["platform"] == "m34"  # only the phone's latency, so only its speed-up, drifts
                if pc is not None and st is not None:
                    ax.plot([pc, st], [yi, yi], color=sf.RULE, linewidth=0.75, zorder=1)
                for v in (pc, st):
                    if drift and v is not None:
                        ax.errorbar([v], [yi], xerr=[[v - v / (1 + DRIFT)], [v * (1 + DRIFT) - v]], fmt="none", ecolor=colour,
                                    elinewidth=0.75, capsize=1.5, zorder=2)
                if pc is not None:
                    ax.scatter([pc], [yi], marker="s", s=20, color=colour, edgecolors="white", linewidths=0.5, zorder=4)
                    if drift and r["per_channel_session"] != r["fp32_session"]:
                        ax.annotate(DAG2, (pc, yi), textcoords="offset points", xytext=(3, 7), va="center", ha="left")
                if st is not None:  # drawn over the square, so a stock value equal to the per-channel one still shows
                    ax.scatter([st], [yi], marker="x", s=30, color=sf.INK, linewidths=1.1, zorder=5)
                    if drift and r["stock_session"] != r["fp32_session"]:
                        ax.annotate(DAG2, (st, yi), textcoords="offset points", xytext=(3, 7), va="center", ha="left")
                spans = r["per_channel_spans_session_6"] == "1" or r["stock_spans_session_6"] == "1"
                if spans:
                    ax.annotate(sf.SECTION, (max(v for v in (pc, st) if v is not None), yi), textcoords="offset points",
                                xytext=(6, 0), va="center")
            if col != "speedup":
                ax.axvline(0, color=sf.INK, linewidth=0.6, zorder=1)
                ax.xaxis.set_major_formatter(lambda v, _: sf.signed(v, 0) if v else "0")
            elif phone:  # the tint marks the phone's provisional latency: the speed-up panel only
                ax.axhspan(min(phone) - 1.5, n - 0.5, color=sf.TINT, zorder=0, linewidth=0)
            ax.set_xlabel(xlabel)
            ax.grid(False)
            ax.grid(True, axis="x")
            sf.panel_label(ax, "abc"[k])
        ax0 = axes[0]
        ax0.set_yticks(ys, labels)
        ax0.tick_params(axis="y", length=0)
        trans = blended_transform_factory(ax0.transAxes, ax0.transData)
        for yg, name in groups:
            ax0.text(-0.02, yg, name, transform=trans, ha="right", va="center", fontweight="bold")
        ax0.set_ylim(n - 0.5, -0.7)
        h = [Line2D([], [], color=sf.MUTED, marker="s", linestyle="none", label="Per-channel INT8 (the matrix's)"),  # column 1
             Line2D([], [], color=sf.INK, marker="x", linestyle="none", label="Stock INT8"),
             Line2D([], [], color=sf.MODEL_COLOR["YOLO11s"], marker="o", linestyle="none", label="YOLO11s"),
             Line2D([], [], color=sf.MODEL_COLOR["YOLO26s"], marker="o", linestyle="none", label="YOLO26s"),
             Line2D([], [], color=sf.TINT, marker="s", markersize=8, linestyle="none",  # column 2
                    label="* Galaxy M34, a consumer-grade phone:"),
             sf.text_entry(f"   latency provisional, bars {chr(177)}{DRIFT * 100:.0f} %"),
             sf.text_entry(f"(u) untrained input resolution · {sf.SECTION} spans session 6"),
             sf.text_entry(f"{DAG2} INT8 and FP32 from different phone sessions")]
        fig.legend(handles=h, loc="outside lower center", ncols=2)
        sf.save(fig, out / "fig_d_int8")


# ---------------------------------------------------------------- small objects


def small_object(rows, out):
    from matplotlib.lines import Line2D
    rows = [r for r in rows if r["platform"] == "pi5"]
    config = {"Quantized": "quantized", "Pruned": "pruned", "Combined": "combined"}
    with sf.style():
        fig = sf.figure(sf.SINGLE, 84)
        ax = fig.subplots()
        top = max(max(float(r["ap_small_drop_pct"]), float(r["ap_drop_pct"])) for r in rows) * 1.08
        # where each line is labelled (fraction of the height) and on which side: clear of the points and the other lines
        at = {1: (0.8, "right"), 1.3: (0.96, "right"), 2: (0.92, "left")}
        for k, ls in ((1, "-"), (1.3, "--"), (2, ":")):
            x_end = min(top, top / k)
            ax.plot([0, x_end], [0, x_end * k], color=sf.MUTED, linestyle=ls, linewidth=0.75, zorder=1)
            y_at, side = top * at[k][0], at[k][1]
            ax.annotate("y = x" if k == 1 else f"y = {k:g}x", (y_at / k, y_at), textcoords="offset points",
                        xytext=(4, 0) if side == "right" else (-4, 0), ha="left" if side == "right" else "right", va="center",
                        color=sf.MUTED, path_effects=sf.halo(), zorder=6)
        ax.text(0.97, 0.03, "Pi 5 · input resolution 640\nONNX Runtime", transform=ax.transAxes, ha="right", va="bottom",
                color=sf.MUTED, path_effects=sf.halo())
        stock_marker = {"Quantized": "x", "Combined": "+"}  # a thin mark = the stock recipe; its shape = the configuration
        for r in rows:
            base = r["compression"].split(",")[0]
            stock = "stock" in r["compression"]
            colour = sf.MODEL_COLOR[r["model"]]
            x, y = float(r["ap_drop_pct"]), float(r["ap_small_drop_pct"])
            if stock:
                ax.scatter([x], [y], marker=stock_marker[base], s=30, color=colour, linewidths=1.1, zorder=3)
            else:
                ax.scatter([x], [y], marker=sf.CONFIG_MARKER[config[base]], s=28, color=colour, edgecolors="white",
                           linewidths=0.5, zorder=4)
            if r["ci_low"] not in ("", None) and r.get("ci_high") not in ("", None):
                pass  # #48: the intervals of the ratio are drawn once they exist for the Pi 5 rows
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("AP drop against the FP32 baseline (%)")
        ax.set_ylabel("AP-small drop against the FP32 baseline (%)")
        ax.grid(True, axis="both")
        names = {"quantized": "Quantized, per-channel INT8", "pruned": "Pruned", "combined": "Combined, per-channel INT8"}
        h = [Line2D([], [], color=c, marker="o", linestyle="none", label=m) for m, c in sf.MODEL_COLOR.items()]
        h += [Line2D([], [], color=sf.MUTED, marker=sf.CONFIG_MARKER[v], linestyle="none", label=names[v]) for v in config.values()]
        h += [Line2D([], [], color=sf.MUTED, marker="x", markeredgewidth=1.1, linestyle="none", label="Quantized, stock INT8"),
              Line2D([], [], color=sf.MUTED, marker="+", markeredgewidth=1.1, linestyle="none", label="Combined, stock INT8")]
        fig.legend(handles=h, loc="outside lower center", ncols=2, columnspacing=0.8, handletextpad=0.3)
        sf.save(fig, out / "fig_small_object")


# ---------------------------------------------------------------- the script


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=STUDY / "results")
    ap.add_argument("--pdf", action="store_true", help="also write each figure's PDF (off for now; PNGs only)")
    a = ap.parse_args(argv)
    sf.VECTOR = a.pdf
    t = a.results / "tables"
    out = a.results / "figures" / "paper"
    cells, _ = sp.usable(sp.load_cells(a.results / "cells.csv"))
    global DRIFT, DRIFT_MOVES  # result_a's and result_d's phone bars
    DRIFT_MOVES, DRIFT = phone_drift(cells)
    pairs = sp.sort_pairs(sp.equal_latency_pairs(cells))
    result_a(cells, pairs, read(t / "equal_latency.csv"), out)
    result_b(read(t / "result_b_bins.csv"), out)
    result_c(read(t / "result_c.csv"), out)
    result_d(read(t / "result_d_int8.csv"), out)
    small_object(read(t / "small_object.csv"), out)
    print(f"written under {out}: " + ", ".join(sorted(p.stem for p in out.glob("*.png"))))


if __name__ == "__main__":
    main()
