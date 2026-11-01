"""The story figures (#49): ten simple charts for the results talk, one message each, drawn from results/cells.csv, the tables
the results scripts write, and the test-dev ground truth. Computes nothing new; re-run it after those when a session lands.

  python study/scripts/studystory.py [--results study/results] [--gt study/runs/gt/visdrone-test-640.json]

The rule for every chart: one highlight colour (BLUE) on what the slide's sentence is about, grey for the rest, the value
written on the mark, no legend where a label will do (Wong, "Salience to relevance", Nat Methods 8:889, 2011; Rougier et al.
2014, rule 8). Charts are 160 mm wide with 11 pt text (studyfig.SLIDE), for a 1,920 px slide.

Outputs, under <results>/figures/story/, each a 600 dpi .png:
  s01_objects      the test-dev objects by COCO area bucket: most are small
  s02_resolution   the AP gain from input resolution 640 to 1280, by object size, per model: small objects gain most
  s03_pairs        every Pi 5 headline equal-latency pair: AP-small change of the compressed cell, % of its baseline's
  s04_price        three ways to make a 960 model faster, and the AP-small each costs
  s05_phone        INT8's speed-up on the Pi 5 and on the phone (provisional)
  s06_time         where one image's latency goes, per head: NMS is a sliver
  s07_gap          YOLO26s's extra latency over YOLO11s by input resolution (NCNN FP16, the clean runtime)
  s08_runtime      YOLO11s at 640 on four runtime and precision choices: latency and AP-small
  s09_recipe       AP lost to INT8, per-channel against stock, per model
  s10_small        AP drop against AP-small drop per compressed configuration, with the ratio
Standard library and matplotlib only; no torch, no ultralytics.
"""

import argparse
import csv
import json
from pathlib import Path

import studyfig as sf
import studypairs as sp

STUDY = Path(__file__).resolve().parents[1]
BLUE, GREY, LIGHT, INK, MUTED = "#2563eb", "#9aa5b1", "#c9d8fb", "#0a1628", "#64748b"  # the deck's one hue, and its greys
LADDER = {"YOLO11s": {640: "A1", 800: "B1", 960: "B1", 1120: "B3", 1280: "B3"},
          "YOLO26s": {640: "A2", 800: "B2", 960: "B2", 1120: "B4", 1280: "B4"}}
PRUNED = {"YOLO11s": "P2", "YOLO26s": "P4"}
CHECKPOINT_640 = {"YOLO11s": "A1", "YOLO26s": "A2"}


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cell(cells, platform, checkpoint, imgsz, variant, head="one-to-many"):
    """The latest scored cell of that platform, checkpoint, input resolution and variant (as studypairs.usable keeps)."""
    m = [c for c in cells if c["platform"] == platform and c["checkpoint"] == checkpoint and c["imgsz"] == imgsz
         and c["variant"] == variant and c["head"] == head]
    return max(m, key=lambda c: c["start_utc"]) if m else None


def bare(ax, keep="left"):
    """A chart read from its labels: no grid, only the axis the bars stand on."""
    ax.grid(False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side == keep)
    if keep != "bottom":
        ax.tick_params(axis="x", length=0, labelbottom=False)
    if keep != "left":
        ax.tick_params(axis="y", length=0)


def pct(v, nd=0):
    return sf.signed(v, nd, " %")


def slide_figure(height_mm):
    """A SLIDE-wide figure with a margin, so no label touches the image's edge on the slide."""
    fig = sf.figure(sf.SLIDE, height_mm)
    fig.get_layout_engine().set(w_pad=8 / 72, h_pad=6 / 72)
    return fig


def note(ax, text, y=1.02):
    """The chart's quantity, in small muted type at its top left (relative changes say so)."""
    ax.text(0, y, text, transform=ax.transAxes, ha="left", va="bottom", color=MUTED, fontsize=sf.SLIDE_SIZE - 1)


# ---------------------------------------------------------------- the charts


def objects(gt_path, out):
    g = json.loads(Path(gt_path).read_text(encoding="utf-8"))["gdict"]
    ann = [a for a in g["annotations"] if not a.get("iscrowd", 0)]
    n = {"small": 0, "medium": 0, "large": 0}
    for a in ann:
        n["small" if a["area"] < 32 ** 2 else "medium" if a["area"] < 96 ** 2 else "large"] += 1
    total = len(ann)
    names = {"small": "Small\narea under 32² px", "medium": "Medium\n32² to 96² px", "large": "Large\narea over 96² px"}
    with sf.style(sf.SLIDE_SIZE):
        fig = slide_figure(70)
        ax = fig.subplots()
        keys = ["small", "medium", "large"]
        shares = [n[k] / total * 100 for k in keys]
        ax.barh(range(3), shares, color=[BLUE, GREY, GREY], height=0.62)
        for i, (k, s) in enumerate(zip(keys, shares)):
            ax.text(s + 1, i, f"{s:.1f} %  ({n[k]:,} objects)", va="center", color=BLUE if k == "small" else INK,
                    fontweight="bold" if k == "small" else "normal")
        ax.set_yticks(range(3), [names[k] for k in keys])
        ax.set_ylim(2.6, -0.6)
        ax.set_xlim(0, 105)
        bare(ax)
        note(ax, f"Share of the {total:,} test-dev objects by COCO area bucket\n(areas in original image pixels)")
        sf.save(fig, out / "s01_objects")
    return n, total


def resolution(cells, out):
    """Relative AP gain from 640 to 1280 per size bucket, per model (ONNX Runtime FP32 baselines on the Pi 5)."""
    rows = []
    for model, ladder in LADDER.items():
        lo = cell(cells, "pi5", ladder[640], 640, "onnxruntime-fp32")
        hi = cell(cells, "pi5", ladder[1280], 1280, "onnxruntime-fp32")
        rows.append((model, [(hi[k] / lo[k] - 1) * 100 for k in ("ap_small", "ap_medium", "ap_large")],
                     hi["latency_ms"] / lo["latency_ms"]))
    with sf.style(sf.SLIDE_SIZE):
        fig = slide_figure(78)
        ax = fig.subplots()
        width, buckets = 0.36, ["Small objects", "Medium objects", "Large objects"]
        for j, (model, gains, slow) in enumerate(rows):
            for i, gval in enumerate(gains):
                x = i + (j - 0.5) * (width + 0.04)
                ax.bar(x, gval, width, color=BLUE if i == 0 else GREY)
                ax.text(x, gval + 1.5, f"+{gval:.0f} %", ha="center", va="bottom", color=BLUE if i == 0 else INK,
                        fontweight="bold" if i == 0 else "normal")
                ax.text(x, -3, model, ha="center", va="top", color=MUTED, fontsize=sf.SLIDE_SIZE - 1)
        ax.set_xticks(range(3), buckets)
        ax.tick_params(axis="x", length=0, pad=22)
        ax.set_ylim(-1, max(g for _, gs, _ in rows for g in gs) * 1.18)
        ax.axhline(0, color=INK, linewidth=0.8)
        bare(ax, keep="none")
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=False)
        note(ax, "Relative AP gain from input resolution 640 to 1280\n(baselines; % of the 640 value)")
        sf.save(fig, out / "s02_resolution")
    return rows


def pairs_chart(cells, out):
    pairs = [p for p in sp.sort_pairs(sp.equal_latency_pairs(cells)) if p["headline"] and p["compressed"]["platform"] == "pi5"]
    order = {"quantized": 0, "pruned": 1}
    rt = {r: i for i, r in enumerate(sp.RUNTIMES)}
    pairs.sort(key=lambda p: (order.get(p["compressed"]["configuration"], 2), rt.get(p["compressed"]["runtime"], 9),
                              p["compressed"]["model"], p["compressed"]["imgsz"]))
    rows = []
    for p in pairs:
        c, b = p["compressed"], p["baseline"]
        rel = p["ap_small_diff"] / b["ap_small"] * 100
        if c["configuration"] == "quantized":  # the INT8 pairs are all ONNX Runtime: name the resolutions
            label = f"{c['model']}   {c['imgsz']} vs {b['imgsz']}"
        else:  # the pruned pairs are all 960 vs 800: name the runtime
            label = f"{c['model']}   {sf.RUNTIME_NAME[c['runtime']]}"
        rows.append((label, rel, p["ap_small_diff"], c["configuration"]))
    from matplotlib.transforms import blended_transform_factory
    with sf.style(sf.SLIDE_SIZE):
        fig = slide_figure(100)
        ax = fig.subplots()
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        y, ticks, labels = 0, [], []
        last = None
        for label, rel, diff, conf in rows:
            if conf != last:
                head = ("Per-channel INT8, one input resolution up" if conf == "quantized"
                        else "Pruned at 960 vs baseline at 800")
                ax.text(0.4, y, head, va="center", ha="left", fontweight="bold", color=BLUE if conf == "quantized" else INK)
                y += 1
                last = conf
            colour = BLUE if rel > 0 else GREY
            ax.barh(y, rel, color=colour, height=0.66)
            ax.text(rel + (0.4 if rel > 0 else -0.4), y, pct(rel), va="center", ha="left" if rel > 0 else "right",
                    color=BLUE if rel > 0 else INK, fontweight="bold" if rel > 0 else "normal")
            ticks.append(y)
            labels.append(label)
            y += 1
        ax.set_yticks(ticks, labels)
        ax.set_ylim(y - 0.4, -0.7)
        ax.set_xlim(-9, 21)
        ax.axvline(0, color=INK, linewidth=0.9)
        ax.text(0.99, y - 0.6, "AP-small change, % of the baseline’s", transform=trans, ha="right", va="center",
                color=MUTED, fontsize=sf.SLIDE_SIZE - 1)
        bare(ax, keep="none")
        ax.tick_params(axis="y", length=0)
        sf.save(fig, out / "s03_pairs")
    return rows


def price(cells, out):
    """At input resolution 960, from the ONNX Runtime FP32 baseline: INT8, input resolution 800, or pruning."""
    data = []
    for model, ladder in LADDER.items():
        base = cell(cells, "pi5", ladder[960], 960, "onnxruntime-fp32")
        ways = [("INT8, per-channel", cell(cells, "pi5", ladder[960], 960, "onnxruntime-int8pc")),
                ("Input 800, not 960", cell(cells, "pi5", ladder[800], 800, "onnxruntime-fp32")),
                ("Pruning", cell(cells, "pi5", PRUNED[model], 960, "onnxruntime-fp32"))]
        data.append((model, [(name, base["latency_ms"] / c["latency_ms"], (1 - c["ap_small"] / base["ap_small"]) * 100)
                             for name, c in ways]))
    with sf.style(sf.SLIDE_SIZE):
        fig = slide_figure(72)
        axes = fig.subplots(1, 2, sharey=True)
        top = max(lost for _, ws in data for _, _, lost in ws) * 1.75
        for ax, (model, ways) in zip(axes, data):
            for i, (name, speed, lost) in enumerate(ways):
                ax.barh(i, lost, color=BLUE if i == 0 else GREY, height=0.6)
                ax.text(lost + 0.4, i, f"−{lost:.0f} % of AP-small\nfor {speed:.2f}× speed", va="center",
                        color=BLUE if i == 0 else INK, fontweight="bold" if i == 0 else "normal", linespacing=1.15)
            ax.set_yticks(range(3), [w[0] for w in ways])
            ax.set_ylim(2.6, -0.6)
            ax.set_xlim(0, top)
            ax.set_title(model, loc="left", fontweight="bold")
            bare(ax)
        fig.supxlabel("From the ONNX Runtime FP32 baseline at 960 on the Pi 5:\nAP-small lost (% of the baseline’s) and the speed-up gained",
                      color=MUTED, fontsize=sf.SLIDE_SIZE - 1)
        sf.save(fig, out / "s04_price")
    return data


def phone(rows_d, out):
    d = {(r["platform"], r["model"]): float(r["per_channel_speedup"]) for r in rows_d
         if r["head"] == "one-to-many" and r["imgsz"] == "640" and r["configuration"] == "quantized" and r["per_channel_speedup"]}
    with sf.style(sf.SLIDE_SIZE):
        fig = slide_figure(70)
        ax = fig.subplots()
        width = 0.36
        for i, model in enumerate(("YOLO11s", "YOLO26s")):
            for j, (platform, name) in enumerate((("pi5", "Raspberry Pi 5"), ("m34", "Galaxy M34\n(provisional)"))):
                v = d.get((platform, model))
                if v is None:
                    continue
                x = i + (j - 0.5) * (width + 0.05)
                ax.bar(x, v, width, color=GREY if platform == "pi5" else BLUE)
                ax.text(x, v + 0.04, f"{v:.2f}×", ha="center", va="bottom", fontweight="bold" if platform == "m34" else "normal",
                        color=BLUE if platform == "m34" else INK)
                ax.text(x, -0.06, name, ha="center", va="top", color=MUTED)
        ax.set_xticks(range(2), ["YOLO11s at 640", "YOLO26s at 640"])
        ax.tick_params(axis="x", pad=44)
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.axhline(1, color=MUTED, linewidth=0.8, linestyle="--", zorder=0)
        ax.text(-0.59, 1.03, "1×", ha="left", va="bottom", color=MUTED, fontsize=sf.SLIDE_SIZE - 1)
        ax.set_ylim(0, 2.4)
        ax.set_xlim(-0.6, 1.6)
        bare(ax, keep="none")
        ax.tick_params(axis="x", labelbottom=True, length=0)
        ax.tick_params(axis="y", labelleft=False)
        note(ax, "Per-channel INT8 speed-up over FP32 on the same platform (ONNX Runtime)")
        sf.save(fig, out / "s05_phone")
    return d


def time_split(heads, out, cell_key=("pi5-matrix-7-int8pc", "onnxruntime-int8pc")):
    r = next(h for h in heads if h["session"] == cell_key[0] and h["variant"] == cell_key[1])
    parts = {}
    for head in ("one_to_many", "nms_free"):
        pre, net, post = (float(r[f"{head}_{k}"]) for k in ("pre_ms", "net_ms", "post_025_ms"))
        parts[head] = (pre, net, post)
    with sf.style(sf.SLIDE_SIZE):
        fig = slide_figure(62)
        ax = fig.subplots()
        for i, head in enumerate(("one_to_many", "nms_free")):
            pre, net, post = parts[head]
            total = pre + net + post
            ax.barh(i, pre, color=INK, height=0.5)
            ax.barh(i, net, left=pre, color=GREY, height=0.5)
            ax.barh(i, post, left=pre + net, color=BLUE, height=0.5)
            ax.text(pre + net / 2, i, f"network {net:.0f} ms", ha="center", va="center", color="white", fontweight="bold")
            ax.text(total + 4, i - 0.13, f"{total:.0f} ms in all", ha="left", va="center")
            ax.text(total + 4, i + 0.17, f"post-processing {post:.2f} ms ({post / total * 100:.1f} %)", ha="left",
                    va="center", color=BLUE, fontweight="bold" if head == "one_to_many" else "normal")
        ax.set_yticks(range(2), ["One-to-many head\n(with NMS)", "NMS-free head"])
        ax.set_ylim(1.6, -0.5)
        ax.set_xlim(0, max(sum(p) for p in parts.values()) * 1.95)
        ax.text(0, 1.47, f"dark sliver at left: pre-processing {parts['one_to_many'][0]:.1f} ms", color=MUTED, va="center",
                fontsize=sf.SLIDE_SIZE - 1)
        bare(ax)
        note(ax, "One image, YOLO26s at 640, per-channel INT8, ONNX Runtime, Pi 5;\ndeployment threshold 0.25")
        sf.save(fig, out / "s06_time")
    return parts


def gap(rows_c, out, platform="pi5", runtime="ncnn", precision="FP16"):
    rows = sorted((r for r in rows_c if r["platform"] == platform and r["runtime"] == runtime and r["precision"] == precision
                   and r["configuration"] == "baseline"), key=lambda r: int(r["imgsz"]))
    with sf.style(sf.SLIDE_SIZE):
        fig = slide_figure(66)
        ax = fig.subplots()
        xs = [int(r["imgsz"]) for r in rows]
        vs = [float(r["latency_pct"]) for r in rows]
        ax.bar(range(len(xs)), vs, color=[GREY] * (len(xs) - 1) + [BLUE], width=0.6)
        for i, v in enumerate(vs):
            ax.text(i, v + 0.2, pct(v, 1), ha="center", va="bottom", color=BLUE if i == len(vs) - 1 else INK,
                    fontweight="bold" if i == len(vs) - 1 else "normal")
        ax.set_xticks(range(len(xs)), [f"{x}" for x in xs])
        ax.set_xlabel("Input resolution (pixels)")
        ax.set_ylim(0, max(vs) * 1.25)
        bare(ax, keep="bottom")
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelleft=False, length=0)
        note(ax, "YOLO26s’s latency over YOLO11s’s, minus 1. NCNN FP16 baselines,\nPi 5 (the runtime with no throttled cell)")
        sf.save(fig, out / "s07_gap")
    return list(zip(xs, vs))


def runtime_chart(cells, out, model="YOLO11s"):
    ck = CHECKPOINT_640[model]
    choices = [("ONNX Runtime, FP32", "onnxruntime-fp32"), ("OpenVINO, FP32", "openvino-fp32"),
               ("ONNX Runtime, per-channel INT8", "onnxruntime-int8pc"), ("NCNN, FP16", "ncnn-fp32")]
    rows = [(name, cell(cells, "pi5", ck, 640, v)) for name, v in choices]
    best = min(range(len(rows)), key=lambda i: rows[i][1]["latency_ms"])
    ref = rows[0][1]["ap_small"]
    with sf.style(sf.SLIDE_SIZE):
        fig = slide_figure(70)
        ax = fig.subplots()
        top = max(c["latency_ms"] for _, c in rows)
        for i, (name, c) in enumerate(rows):
            ax.barh(i, c["latency_ms"], color=BLUE if i == best else GREY, height=0.6)
            change = (c["ap_small"] / ref - 1) * 100
            acc = ("AP-small: reference" if i == 0 else "AP-small ±0 %" if abs(change) < 0.5 else f"AP-small {pct(change)}")
            ax.text(c["latency_ms"] + top * 0.015, i, f"{c['latency_ms']:.0f} ms   {acc}", va="center",
                    color=BLUE if i == best else INK, fontweight="bold" if i in (best,) or change < -5 else "normal")
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.set_xlim(0, top * 1.95)
        bare(ax)
        note(ax, "Median latency, YOLO11s at 640 on the Pi 5;\nAP-small against ONNX Runtime FP32")
        sf.save(fig, out / "s08_runtime")
    return [(n, c["latency_ms"], c["ap_small"]) for n, c in rows]


def recipe(rows_d, out):
    d = {r["model"]: (-float(r["per_channel_ap_diff_pct"]), -float(r["stock_ap_diff_pct"])) for r in rows_d
         if r["platform"] == "pi5" and r["head"] == "one-to-many" and r["imgsz"] == "640" and r["configuration"] == "quantized"}
    with sf.style(sf.SLIDE_SIZE):
        fig = slide_figure(62)
        ax = fig.subplots()
        y = 0
        ticks, labels = [], []
        for model in ("YOLO11s", "YOLO26s"):
            pc, st = d[model]
            for name, v, colour in (("per-channel", pc, BLUE), ("stock", st, GREY)):
                ax.barh(y, v, color=colour, height=0.62)
                ax.text(v + 0.3, y, f"−{v:.1f} % AP", va="center", color=colour if colour == BLUE else INK,
                        fontweight="bold" if colour == BLUE else "normal")
                ticks.append(y)
                labels.append(f"{model}, {name}")
                y += 1
            y += 0.5
        ax.set_yticks(ticks, labels)
        ax.set_ylim(y - 0.9, -0.6)
        ax.set_xlim(0, max(max(v) for v in d.values()) * 1.35)
        bare(ax)
        note(ax, "AP lost to INT8 at 640, % of the FP32 baseline’s. Stock: one weight\nscale per layer; per-channel: one per output channel")
        sf.save(fig, out / "s09_recipe")
    return d


def small(rows_s, out):
    order = [("Quantized", "INT8"), ("Pruned", "Pruned"), ("Combined", "Combined")]
    rows = [r for r in rows_s if r["platform"] == "pi5" and r["headline"] == "1"]
    with sf.style(sf.SLIDE_SIZE):
        fig = slide_figure(80)
        ax = fig.subplots()
        width = 0.36
        x = 0
        ticks, labels = [], []
        for comp, short in order:
            for model in ("YOLO11s", "YOLO26s"):
                r = next(r for r in rows if r["compression"] == comp and r["model"] == model)
                a, s = float(r["ap_drop_pct"]), float(r["ap_small_drop_pct"])
                ax.bar(x - width / 2 - 0.02, a, width, color=GREY)
                ax.bar(x + width / 2 + 0.02, s, width, color=BLUE)
                ax.text(x - width / 2 - 0.02, a + 0.5, f"{a:.1f}", ha="center", va="bottom", fontsize=sf.SLIDE_SIZE - 1)
                ax.text(x + width / 2 + 0.02, s + 0.5, f"{s:.1f}", ha="center", va="bottom", fontsize=sf.SLIDE_SIZE - 1)
                ax.text(x, max(a, s) + 4.2, f"{float(r['small_over_all']):.2f}×", ha="center", va="bottom", fontweight="bold",
                        color=BLUE)
                ticks.append(x)
                labels.append(f"{short}\n{model}")
                x += 1.1
            x += 0.35
        ax.set_xticks(ticks, labels)
        ax.set_ylim(0, 47)
        ax.axhline(0, color=INK, linewidth=0.8)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=GREY, label="AP drop"), Patch(color=BLUE, label="AP-small drop"),
                           Patch(color="white", label="(% of the FP32 baseline’s; above: the ratio)")],
                  loc="upper left", ncols=3, handlelength=1.2)
        bare(ax, keep="none")
        ax.tick_params(axis="x", labelbottom=True, length=0)
        ax.tick_params(axis="y", labelleft=False)
        sf.save(fig, out / "s10_small")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=STUDY / "results")
    ap.add_argument("--gt", type=Path, default=STUDY / "runs" / "gt" / "visdrone-test-640.json")
    a = ap.parse_args(argv)
    t = a.results / "tables"
    out = a.results / "figures" / "story"
    cells, _ = sp.usable(sp.load_cells(a.results / "cells.csv"))
    n, total = objects(a.gt, out)
    print(f"s01: {total:,} objects; small {n['small']:,} ({n['small'] / total:.1%})")
    for model, gains, slow in resolution(cells, out):
        print(f"s02: {model} 640->1280 gain small/medium/large {gains[0]:.0f}/{gains[1]:.0f}/{gains[2]:.0f} %, latency {slow:.2f}x")
    for label, rel, diff, _ in pairs_chart(cells, out):
        print(f"s03: {label}: {rel:+.1f} % ({diff:+.4f})")
    for model, ways in price(cells, out):
        print("s04:", model, [(w, f"{s:.2f}x", f"-{lost:.1f}%") for w, s, lost in ways])
    print("s05:", phone(read(t / "result_d_int8.csv"), out))
    print("s06:", time_split(read(t / "result_b_heads.csv"), out))
    print("s07:", gap(read(t / "result_c.csv"), out))
    print("s08:", runtime_chart(cells, out))
    print("s09:", recipe(read(t / "result_d_int8.csv"), out))
    small(read(t / "small_object.csv"), out)
    print(f"written under {out}")


if __name__ == "__main__":
    main()
