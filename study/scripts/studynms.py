"""Result B (#45): what NMS costs on the Pi 5 and how that cost grows with the objects in an image, from cells already
benchmarked and scored. Runs nothing on a Pi, reads no model and no predictions.json (100+ MB each).

  python study/scripts/studynms.py [--bench study/bench/pi5] [--gt study/runs/gt/visdrone-test-640.json]
                                   [--out study/results] [--bins 10] [--figure-cell pi5-matrix-2:onnxruntime-int8]

Head pairs. Every cell whose variant ends in -e2e (the NMS-free head) is paired with the cell of the same run, input
resolution and variant minus -e2e (the one-to-many head) in the same session. Pairs are found in each pi5-matrix-*
session's summary.csv, never listed by hand: session 2 gives the three of A2 (yolo26s-640-s0 at 640: ONNX Runtime FP32,
ONNX Runtime INT8, OpenVINO FP32) and session 7 adds the per-channel INT8 one, so a later session with such a cell adds a
pair with no change here.
Which numbers. The attempt that counts is the one in summary.csv's run_dir. Medians are meta.json's timing_ms (the same the
session records quote): pre-processing, network, post-processing at 0.25 (post_deploy, the deployment threshold) and at 0.001
(t_post_ms, the evaluation threshold), and latency (e2e_deploy: pre + net + post-processing at 0.25). AP, AP-small, AP-large
and the precision are score.json's, never the folder name's. The share of the latency is post-processing at 0.25 over the
latency: for the one-to-many head that is NMS and its decoding, for the NMS-free head only the conversion of its 500 boxes.
Differences are the NMS-free head minus the one-to-many head. Whether a cell throttled is summary.csv's verdict; clock
statistics are not computed here.
Objects per image. studybench logs the images in the order of the ground truth: runs/gt/visdrone-test-640.json holds
{"files", "gdict"}, and image id i is files[i]. Objects per image is the number of that id's annotations in gdict (75,102 in all,
none crowd, so every annotation is an object the detector is scored on). It was checked against the
YOLO label files under datasets/VisDrone/labels/test for all 1,610 images: every count equals the file's line count. Images
join the timings by file name.
Candidates. n_cand in timings.csv is studybench's count of rows entering NMS in the evaluation pass: boxes with a class score
above 0.001, one row per (box, class), counted before the 30,000-row cap. The deployment pass (0.25) logs no count, so the
post-processing time at 0.25 is also plotted against the count at 0.001. The NMS-free head's own n_cand is its (up to 500)
boxes over the threshold, which says nothing about NMS, so its times are plotted against the count of the one-to-many head on
the same image (both heads ran the same 1,610 images).
Bins. Images are cut into bins of about equal size by quantiles of the x value (--bins, 10); a bin is [lo, hi), the last one
closed. The curve is the median of the bin's times at the median x of its images; the band is the interquartile range.
Slopes are least-squares lines through the 1,610 images (not the bins).

Outputs, all under --out:
  figures/result_b_nms.pdf, .png          --figure-cell's pair: post-processing time per image against objects per image
                                          (panel a) and against candidates at 0.001 (panel b), at both thresholds, both heads
  figures/result_b_nms_pooled.pdf, .png   the same curve for every one-to-many Pi 5 cell at 640 (post-processing times only)
  tables/result_b_heads.csv, .tex         per head pair: medians, share of latency, AP, AP-small, and the difference
  tables/result_b_bins.csv                the figure's bins (its source data)
  tables/result_b_slopes.csv              the straight-line slopes of each head's post-processing time
Standard library, numpy and matplotlib only; no torch, no ultralytics. It loads the 13 MB ground truth and about 50 small
timings.csv files, well under 200 MB.
"""

import argparse
import collections
import csv
import json
import re
from pathlib import Path

import numpy as np

FIELDS = ["pre_ms", "net_ms", "post_025_ms", "post_0001_ms", "latency_ms", "ap", "ap_small", "ap_large"]


def gt_counts(gt_path):
    """Ground-truth objects per image file: {file name: annotations}. See the module docstring for what counts."""
    rec = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    n = collections.Counter(a["image_id"] for a in rec["gdict"]["annotations"])
    return {f: n.get(img["id"], 0) for f, img in zip(rec["files"], rec["gdict"]["images"])}


def quantile_edges(xs, k):
    """Edges that cut xs into k bins of about as many images each: the k+1 quantiles, rounded down, repeats dropped."""
    return [int(e) for e in np.unique(np.floor(np.quantile(np.asarray(xs, float), np.linspace(0, 1, k + 1))))]


def bin_stats(xs, ys, edges):
    """Images in bins by x: [edges[i], edges[i+1]), the last bin closed at its upper edge, an image outside the edges
    left out, a bin with no image dropped. One dict per bin: lo, hi, n, x_median, and the median, q1 and q3 (the
    interquartile range, numpy's linear interpolation) of y."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    out = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        take = (xs >= lo) & ((xs <= hi) if last else (xs < hi))
        if take.any():
            q1, med, q3 = np.percentile(ys[take], [25, 50, 75])
            out.append({"lo": lo, "hi": hi, "n": int(take.sum()), "x_median": float(np.median(xs[take])),
                        "median": float(med), "q1": float(q1), "q3": float(q3)})
    return out


def _cell(session, row):
    """One summary.csv row -> the cell's medians, scores and attempt directory, or None while it has no meta.json and score.json
    (not scored yet). The attempt that counts is the one in run_dir."""
    d = session / row["run_dir"]
    if not ((d / "meta.json").exists() and (d / "score.json").exists()):
        return None
    t = json.loads((d / "meta.json").read_text(encoding="utf-8"))["timing_ms"]
    score = json.loads((d / "score.json").read_text(encoding="utf-8"))
    st = score["stats"]
    return {"session": session.name, "cell": row["cell"], "imgsz": int(row["imgsz"]), "throttle": row["throttle"], "head": score["head"], "dir": d, "runtime": score.get("runtime", row["runtime"]), "precision": score["precision"],
            "pre_ms": t["t_pre_ms"]["median"], "net_ms": t["t_net_ms"]["median"], "post_025_ms": t["post_deploy"]["median"],
            "post_0001_ms": t["t_post_ms"]["median"], "latency_ms": t["e2e_deploy"]["median"],
            "ap": st["coco/AP_all"], "ap_small": st["coco/AP_small"], "ap_large": st["coco/AP_large"]}


def head_pairs(session_dir):
    """Every pair of cells in one session that differ only in the head: a cell whose variant ends in -e2e (the NMS-free head)
    with the cell of the same run, input resolution and variant minus -e2e (the one-to-many head). Found from summary.csv,
    never listed by hand, so a session that gains such a cell gains a pair. Reference runs (drift-start, drift-end), cells
    without a partner and cells not scored yet make no pair."""
    session = Path(session_dir)
    with open(session / "summary.csv", newline="", encoding="utf-8") as f:
        rows = {r["cell"]: r for r in csv.DictReader(f)}
    pairs = []
    for name, row in rows.items():
        run, imgsz, variant = name.rsplit("/", 2) if name.count("/") == 2 else (None, None, None)
        if variant is None or not variant.endswith("-e2e") or f"{run}/{imgsz}/{variant[:-4]}" not in rows:
            continue
        many, free = _cell(session, rows[f"{run}/{imgsz}/{variant[:-4]}"]), _cell(session, row)
        if many and free:
            pairs.append({"session": session.name, "run": run, "imgsz": int(imgsz), "variant": variant[:-4], "one_to_many": many, "nms_free": free})
    return pairs


def pair_row(pair):
    """One table row for a head pair: per head the medians of pre-processing, network, post-processing at 0.25 (deployment
    threshold) and at 0.001 (evaluation threshold), the latency, post-processing at 0.25 as a share of the latency (percent),
    AP, AP-small and AP-large; then the NMS-free head minus the one-to-many head in each."""
    m, n = pair["one_to_many"], pair["nms_free"]
    row = {"session": pair["session"], "run": pair["run"], "imgsz": pair["imgsz"], "runtime": m["runtime"], "precision": m["precision"],
           "variant": pair["variant"]}
    for head, c in (("one_to_many", m), ("nms_free", n)):
        for k in FIELDS:
            row[f"{head}_{k}"] = c[k]
        row[f"{head}_throttle"] = c["throttle"]
        row[f"{head}_post_share_pct"] = round(100 * c["post_025_ms"] / c["latency_ms"], 3)
    for k in FIELDS + ["post_share_pct"]:
        row[f"diff_{k}"] = round(row[f"nms_free_{k}"] - row[f"one_to_many_{k}"], 6)
    return row


def timings(cell):
    """One attempt's timings.csv as {file: (post-processing at 0.001, post-processing at 0.25, n_cand)}, times in ms."""
    with open(cell["dir"] / "timings.csv", newline="", encoding="utf-8") as f:
        return {r["file"]: (float(r["t_post_ms"]), float(r["t_post_deploy_ms"]), int(r["n_cand"])) for r in csv.DictReader(f)}


def per_image(pair):
    """Per image file, from the two heads' timings.csv: the post-processing time of each head at 0.25 and at 0.001, and n_cand,
    the candidates entering NMS at 0.001 (studybench logs it from the evaluation pass; the deployment pass's count is not
    logged) - the one-to-many head's, which the NMS-free head's times are plotted against too, since both ran the same images."""
    out = {}
    for head in ("one_to_many", "nms_free"):
        for f, (p_eval, p_deploy, n_cand) in timings(pair[head]).items():
            d = out.setdefault(f, {})
            d[f"{head}_post_025_ms"], d[f"{head}_post_0001_ms"] = p_deploy, p_eval
            d["n_cand" if head == "one_to_many" else "n_cand_nms_free"] = n_cand
    return out


def one_to_many_cells(bench_dir, imgsz):
    """Every scored one-to-many cell at one input resolution in every pi5-matrix-* session under bench_dir (the optional pooled
    curve). The head is score.json's, so the reference runs and the NMS-free head's cells are out."""
    out = []
    for session in sorted(Path(bench_dir).glob("pi5-matrix-*")):
        with open(session / "summary.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["imgsz"] == str(imgsz) and not row["cell"].startswith("drift-"):
                    c = _cell(session, row)
                    if c and c["head"] == "raw":
                        out.append(c)
    return out


def line_fit(xs, ys):
    """Least-squares line through the images: slope (y per x), intercept and Pearson r."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    slope, intercept = np.polyfit(xs, ys, 1)
    return {"slope": float(slope), "intercept": float(intercept), "r": float(np.corrcoef(xs, ys)[0, 1])}


# ---------------------------------------------------------------- tables

RUNTIMES = {"onnxruntime": "ONNX Runtime", "openvino": "OpenVINO", "ncnn": "NCNN"}
COLUMNS = [("pre_ms", 1), ("net_ms", 1), ("post_025_ms", 2), ("post_0001_ms", 2), ("latency_ms", 1), ("post_share_pct", 2), ("ap", 4),
           ("ap_small", 4)]


def pair_label(row):
    """'ONNX Runtime INT8 (session 2)', 'ONNX Runtime INT8 per-channel (session 7)'."""
    n = re.search(r"matrix-(\d+)", row["session"])
    pc = " per-channel" if row["variant"].endswith("int8pc") else ""
    return f"{RUNTIMES.get(row['runtime'], row['runtime'])} {row['precision']}{pc} (session {n.group(1) if n else row['session']})"


def write_table(rows, out_dir):
    """tables/result_b_heads.csv (every column of pair_row) and result_b_heads.tex (booktabs: per pair the one-to-many head,
    the NMS-free head and their difference; a cell that throttled has t after its latency)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "result_b_heads.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bs = chr(92)
    head = (f"Cell & Head & Pre (ms) & Net (ms) & Post at 0.25 (ms) & Post at 0.001 (ms) & Latency (ms) & Post at 0.25, share of latency ({bs}%) "
            f"& AP & AP-small {bs}{bs}")
    lines = [f"% result B (#45): the one-to-many head against the NMS-free head, one block per pair. Written by studynms.py.",
             f"{bs}begin{{tabular}}{{ll" + "r" * len(COLUMNS) + "}", f"{bs}toprule", head]
    for r in rows:
        lines.append(f"{bs}midrule")
        for i, (key, name) in enumerate((("one_to_many", "One-to-many head"), ("nms_free", "NMS-free head"), ("diff", "Difference"))):
            cells = []
            for col, nd in COLUMNS:
                v = r[f"{key}_{col}"]
                s = f"{v:+.{nd}f}".replace("-", "$-$") if key == "diff" else f"{v:.{nd}f}"
                if col == "latency_ms" and key != "diff" and r[f"{key}_throttle"] == "throttled":
                    s += " t"
                cells.append(s)
            lines.append(f"{pair_label(r) if i == 0 else ''} & {name} & " + " & ".join(cells) + f" {bs}{bs}")
    lines += [f"{bs}bottomrule", f"{bs}end{{tabular}}", f"% t: the cell throttled (summary.csv). Difference: NMS-free head minus one-to-many head."]
    (out / "result_b_heads.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pair_points(pair, objects):
    """Per-image arrays for one pair: objects, n_cand (the one-to-many head's, at 0.001) and each head's two times."""
    im = per_image(pair)
    files = [f for f in im if f in objects and "n_cand" in im[f] and "nms_free_post_025_ms" in im[f]]
    cols = {k: np.array([im[f][k] for f in files]) for k in ("n_cand", "one_to_many_post_025_ms", "one_to_many_post_0001_ms",
                                                          "nms_free_post_025_ms", "nms_free_post_0001_ms")}
    cols["objects"] = np.array([objects[f] for f in files])
    return cols


XNAMES = {"objects": "objects per image", "n_cand": "candidates entering NMS at 0.001"}
THRESHOLDS = {"025": "0.25 (deployment threshold)", "0001": "0.001 (evaluation threshold)"}


def bins_and_slopes(pairs, objects, k):
    """The figure's bins and the slopes, one row per (pair, head, threshold, x)."""
    bins, slopes = [], []
    for pair in pairs:
        pts = pair_points(pair, objects)
        for xname in XNAMES:
            edges = quantile_edges(pts[xname], k)
            for head in ("one_to_many", "nms_free"):
                for thr in THRESHOLDS:
                    y = pts[f"{head}_post_{thr}_ms"]
                    key = {"session": pair["session"], "run": pair["run"], "imgsz": pair["imgsz"], "variant": pair["variant"],
                           "head": head, "threshold": THRESHOLDS[thr], "x": XNAMES[xname]}
                    for b in bin_stats(pts[xname], y, edges):
                        bins.append({**key, **b})
                    fit = line_fit(pts[xname], y)
                    scale = 100 if xname == "objects" else 1000
                    slopes.append({**key, "images": len(y), "slope_ms_per": f"{scale} {'objects' if scale == 100 else 'candidates'}",
                                   "slope_ms": fit["slope"] * scale, "intercept_ms": fit["intercept"], "r": fit["r"]})
    return bins, slopes


def write_rows(path, rows, nd=6):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, nd) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)


# ---------------------------------------------------------------- figures

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"  # dataviz reference palette, slots 1 and 2, light
STYLE = {"0001": ("-", "o"), "025": ("--", "s")}  # the threshold is the line style, the head the colour


def _axes(ax, xlabel):
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel("Post-processing time per image (ms)", color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.set_ylim(bottom=0)


def _curve(ax, bins, color, thr, label):
    ls, mk = STYLE[thr]
    x = [b["x_median"] for b in bins]
    ax.fill_between(x, [b["q1"] for b in bins], [b["q3"] for b in bins], color=color, alpha=0.14, linewidth=0)
    ax.plot(x, [b["median"] for b in bins], color=color, linestyle=ls, marker=mk, markersize=5, linewidth=1.6,
            markeredgecolor="white", markeredgewidth=0.8, label=label)


def figure_pair(pair, objects, k, out_dir):
    """Panels (a) objects per image and (b) candidates at 0.001, both heads at both thresholds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = pair_points(pair, objects)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor="white")
    for ax, xname, title in zip(axes, XNAMES, ("(a) Against objects per image (ground truth)",
                                               "(b) Against candidates entering NMS")):
        edges = quantile_edges(pts[xname], k)
        for head, color, hname in (("one_to_many", BLUE, "One-to-many head"), ("nms_free", ORANGE, "NMS-free head")):
            for thr in ("0001", "025"):
                _curve(ax, bin_stats(pts[xname], pts[f"{head}_post_{thr}_ms"], edges), color, thr,
                       f"{hname}, post-processing at {THRESHOLDS[thr]}")
        _axes(ax, "Objects per image (ground truth)" if xname == "objects" else
              "Candidates entering NMS, at 0.001 (evaluation threshold)" + chr(10) + "(the one-to-many head's count on the same image, for both heads)")
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=MUTED)
    c = pair["one_to_many"]
    fig.suptitle(f"NMS time against image content on the Pi 5: {pair['run']} at {pair['imgsz']}, {RUNTIMES.get(c['runtime'], c['runtime'])} {c['precision']}, "
                 f"{len(pts['objects']):,} VisDrone test-dev images", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.text(0.01, 0.905, f"Line: median of the images in a bin ({k} bins of about equal size); band: interquartile range.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.11, 1, 0.9))
    save_figure(fig, out_dir / "figures" / "result_b_nms")


def figure_pooled(cells, objects, k, out_dir):
    """The one-to-many head's post-processing time against objects per image, pooled over every cell, with each cell's own curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edges = quantile_edges(list(objects.values()), k)
    per_cell = []
    for c in cells:
        t = timings(c)
        files = [f for f in t if f in objects]
        per_cell.append((np.array([objects[f] for f in files]), np.array([t[f][0] for f in files]), np.array([t[f][1] for f in files])))
    xs = np.concatenate([p[0] for p in per_cell])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="white")
    for ax, idx, thr in zip(axes, (1, 2), ("0001", "025")):
        for p in per_cell:
            b = bin_stats(p[0], p[idx], edges)
            ax.plot([r["x_median"] for r in b], [r["median"] for r in b], color=MUTED, alpha=0.35, linewidth=0.9)
        _curve(ax, bin_stats(xs, np.concatenate([p[idx] for p in per_cell]), edges), BLUE, thr, f"All {len(cells)} cells pooled")
        _axes(ax, "Objects per image (ground truth)")
        ax.set_title(f"Post-processing at {THRESHOLDS[thr]}", loc="left", fontsize=10.5, color=INK)
    axes[0].plot([], [], color=MUTED, alpha=0.6, linewidth=0.9, label="One cell's median")
    h, lab = axes[0].get_legend_handles_labels()
    axes[0].legend(h, lab, frameon=False, loc="upper left", fontsize=9, labelcolor=MUTED)
    fig.suptitle(f"One-to-many head's NMS time against objects per image, all {len(cells)} Pi 5 cells at 640 "
                 f"({len({c['session'] for c in cells})} sessions)", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, out_dir / "figures" / "result_b_nms_pooled")


def save_figure(fig, stem):
    import matplotlib.pyplot as plt
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- the script

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench", type=Path, default=ROOT / "bench" / "pi5")
    ap.add_argument("--gt", type=Path, default=ROOT / "runs" / "gt" / "visdrone-test-640.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--figure-cell", default="pi5-matrix-2:onnxruntime-int8",
                    help="session-name prefix:one-to-many variant of the pair the main figure shows")
    a = ap.parse_args(argv)
    objects = gt_counts(a.gt)
    pairs = [p for s in sorted(a.bench.glob("pi5-matrix-*")) for p in head_pairs(s)]
    if not pairs:
        raise SystemExit(f"no head pair under {a.bench}")
    rows = [pair_row(p) for p in pairs]
    write_table(rows, a.out / "tables")
    bins, slopes = bins_and_slopes(pairs, objects, a.bins)
    write_rows(a.out / "tables" / "result_b_bins.csv", bins)
    write_rows(a.out / "tables" / "result_b_slopes.csv", slopes)
    prefix, variant = a.figure_cell.split(":")
    main_pair = next(p for p in pairs if p["session"].startswith(prefix) and p["variant"] == variant)
    figure_pair(main_pair, objects, a.bins, a.out)
    cells = one_to_many_cells(a.bench, 640)
    figure_pooled(cells, objects, a.bins, a.out)
    for r in rows:
        print(f"{pair_label(r)}: latency {r['one_to_many_latency_ms']} / {r['nms_free_latency_ms']} ms, post at 0.25 "
              f"{r['one_to_many_post_025_ms']} / {r['nms_free_post_025_ms']}, at 0.001 {r['one_to_many_post_0001_ms']} / {r['nms_free_post_0001_ms']}, "
              f"AP {r['one_to_many_ap']:.4f} / {r['nms_free_ap']:.4f}, AP-small {r['one_to_many_ap_small']:.4f} / {r['nms_free_ap_small']:.4f}")
    print(f"{len(pairs)} head pairs, {len(cells)} one-to-many cells at 640 pooled; written under {a.out}")


if __name__ == "__main__":
    main()
