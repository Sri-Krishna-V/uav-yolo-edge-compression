"""PROTOTYPE - throwaway, lives only on branch prototype/yolo26-pruning-feasibility. Map ticket: YOLO26 pruning feasibility.

Can YOLO26s be structurally pruned with the study's own pruner (scripts/studyprune.py, unchanged), and does the
pruned model still forward, train, and export to ONNX and NCNN with output parity? Checks run cheapest first:

  0  unpruned YOLO26s -> ONNX and NCNN, raw-output parity against PyTorch   (ultralytics #23365 said this failed)
  1  DepGraph trace + prune 5% and 30%, round_to=8: per-layer channels, attention integrity, dual-head consistency.
     YOLO11s first (the control: known to work, P1), then the YOLO26s ignore-list variants (VARIANTS), stock last.
  2  pruned -> ONNX (one-to-many and one-to-one heads) and NCNN, parity against the pruned PyTorch model
  3  laptop CPU latency, ONNX Runtime + NCNN, unpruned vs pruned (indicative only: the laptop is not a target)
  4  recovery-training smoke: prune a VisDrone (nc=10) YOLO26s, train it with studytrain.PrunedTrainer, reload, export

Every stage runs in its own child process under a hard memory cap and a timeout (memcap.py), because the stock
YOLO26s prune never terminates: DepGraph doubles an index list around the attention in layer 22 every lap, and on
Windows that pages the whole laptop for hours before any MemoryError (MEMORY-INCIDENT-2026-10-19.md). The cause is
measured by diagnose_groups.py. A stage that hits its cap or its timeout is recorded as that finding; a sibling
stage that would repeat the same failure is skipped, not re-run.

ncnn/pnnx are NOT in the pinned venv; install them to a side directory and put it on PYTHONPATH:
  uv pip install --python .venv/Scripts/python.exe --no-deps --target <dir> ncnn "pnnx==20260526"
  PYTHONPATH=<dir> .venv/Scripts/python prototype_yolo26_prune/feasibility.py           # every stage
  PYTHONPATH=<dir> .venv/Scripts/python prototype_yolo26_prune/feasibility.py --only 1_prune_yolo11s-prune5

Artifacts and per-stage logs land in runs/prototype-yolo26-prune/ (git-ignored); the record in
prototype_yolo26_prune/results/feasibility.json, rewritten before and after every stage.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT / "runs" / "prototype-yolo26-prune"
OUT = HERE / "results"
STAGE_OUT = OUT / "stages"
IMGSZ = 640
YOLO26S = ROOT / "yolo26s.pt"  # COCO-pretrained, nc=80: the "untrained" (not VisDrone-trained) model the ticket asks for
YOLO11S = ROOT / "yolo11s.pt"  # control: the family studyprune was built and used on (P1)
P1 = Path(r"C:\Users\srikr\uav-yolo-checkpoints\yolo11s-640-s0-prune30.pt")  # the real YOLO11s 30% prune, for comparison
SMOKE26 = ROOT / "runs" / "smoke" / "yolo26s-train" / "weights" / "best.pt"  # a VisDrone-trained (nc=10) YOLO26s

# Limits. The laptop has 15.2 GB and sits at ~10-11 GB with VS Code, Claude and Chrome open, so caps stay well under
# what is free; FLOOR_GB kills any stage once the whole system gets that close to paging, whatever the stage's own use.
# Measured peaks: prune 2.1-2.3 GB, export 2.2 GB, latency 2.6 GB. A 4 GB prune cap let the runaway stage take the
# system down to 1.15 GB available, so the caps sit just above the measured peaks.
CAP_GB = {"prune": 3, "export": 3.5, "latency": 3.5, "train": 6}
TIMEOUT_MIN = {"prune": 10, "export": 10, "latency": 15, "train": 30}  # the YOLO11s prune takes seconds
FLOOR_GB = 1.0

# Ignore-list variants for YOLO26s, passed through prune_ckpt's extra_ignored hook (studyprune.py stays unchanged):
#   attnblock  every C3k2 built with attn=True (layer 22), ignored whole, as studyprune already ignores C2PSA whole
#   attnonly   only the Attention modules: stops the runaway root, but pins the C3k2 half that feeds the attention's
#              residual stream while the other half is pruned, which studyprune's rechunk guard rejects
#   stock      studyprune's own list: the DepGraph group rooted at layer 22's attn.pe never terminates
VARIANTS = (("yolo11s", "YOLO11S", (), ""), ("yolo26s", "YOLO26S", ("AttnC3k2",), "-attnblock"),
            ("yolo26s", "YOLO26S", ("Attention",), "-attnonly"), ("yolo26s", "YOLO26S", (), ""))


class _HoldsAttention(type):
    def __instancecheck__(cls, m):
        from ultralytics.nn.modules.block import Attention, C3k2
        return isinstance(m, C3k2) and any(isinstance(s, Attention) for s in m.modules())


class AttnC3k2(metaclass=_HoldsAttention):
    """isinstance() matches a C3k2 that holds an Attention (YOLO26 layer 22), so studyprune can ignore it by 'class'."""


def S(name, kind, fn, *args, needs=(), same_failure_as=None):
    return {"name": name, "kind": kind, "fn": fn, "args": args, "needs": needs, "same_failure_as": same_failure_as}


def stages():
    """The plan, in run order: cheap and informative first, the known blow-up last among the prunes."""
    plan = [S("0_unpruned_export", "export", "check_unpruned_export")]
    for family, src, extra, suffix in VARIANTS:
        p5, p30 = f"{family}-prune5{suffix}", f"{family}-prune30{suffix}"
        plan.append(S(f"1_prune_{p5}", "prune", "prune_and_check", src, 0.05, p5, extra))
        plan.append(S(f"1_prune_{p30}", "prune", "prune_and_check", src, 0.30, p30, extra, same_failure_as=f"1_prune_{p5}"))
    for tag in ("yolo26s-prune5-attnblock", "yolo26s-prune30-attnblock"):
        plan.append(S(f"2_export_{tag}", "export", "export_and_check", tag, needs=(f"1_prune_{tag}",)))
    plan.append(S("3_laptop_latency", "latency", "check_latency"))
    plan.append(S("4_recovery_training_smoke", "train", "check_training", ("AttnC3k2",)))
    return plan


# =============================================================== child side: one stage, under the cap
def child(name):
    import memcap
    memcap.cap_from_env()
    sys.path.insert(0, str(ROOT / "scripts"))
    import studyenv  # noqa: F401  must precede ultralytics
    global studyprune, cv2, np, torch, YOLO, Attention
    import studyprune
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.modules.block import Attention

    spec = next(s for s in stages() if s["name"] == name)
    args = [globals()[a] if a in ("YOLO11S", "YOLO26S") else a for a in spec["args"]]
    t, oom = time.time(), False
    try:
        r = {"ok": True, **(globals()[spec["fn"]](*args) or {})}
    except MemoryError:
        oom = True  # allocate nothing here: the traceback's frames still hold the runaway lists until this block exits
    except Exception as e:  # noqa: BLE001  a failure IS a finding: record it, the parent decides what follows
        r = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}
    if oom:
        r = {"ok": False, "error": f"MemoryError at the {memcap_gb()} GB cap", "trace": ""}
    r["seconds"] = round(time.time() - t, 1)
    STAGE_OUT.mkdir(parents=True, exist_ok=True)
    (STAGE_OUT / f"{name}.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in r.items() if k != "trace"}, indent=2, default=str)[:4000])
    if not r["ok"]:
        print(r["trace"])


def memcap_gb():
    import os
    return os.environ.get("PROTO_CAP_GB", "?")


def image_tensor():
    import studyenv
    image = next((studyenv.VISDRONE_DIR / "images" / "val").glob("*.jpg"))
    im = cv2.resize(cv2.imread(str(image)), (IMGSZ, IMGSZ))[:, :, ::-1].transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(im)).float()[None] / 255


def torch_raw(model, x, end2end=False):
    """The tensor an export emits: (1, 4+nc, anchors) from the one-to-many head, or (1, max_det, 6) from one-to-one."""
    m = deepcopy(model).float().eval()
    m.model[-1].end2end = end2end
    with torch.no_grad():
        return m(x)[0].numpy()


def ort_raw(onnx_file, x):
    import onnxruntime as ort
    s = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    return s.run(None, {s.get_inputs()[0].name: x.numpy()})[0]


def ncnn_net(ncnn_dir):
    import ncnn
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.load_param(str(Path(ncnn_dir) / "model.ncnn.param"))
    net.load_model(str(Path(ncnn_dir) / "model.ncnn.bin"))
    return net


def ncnn_raw(ncnn_dir, x):
    import ncnn
    ex = ncnn_net(ncnn_dir).create_extractor()
    ex.input("in0", ncnn.Mat(x[0].numpy()).clone())
    _, out = ex.extract("out0")
    return np.array(out)[None]


def parity(ref, got):
    """Box and score agreement of two one-to-many outputs (1, 4+nc, A)."""
    got = got.reshape(ref.shape)
    boxes, scores = np.abs(ref[:, :4] - got[:, :4]), np.abs(ref[:, 4:] - got[:, 4:])
    top = np.argsort(-ref[0, 4:].max(0))[:100]  # the 100 most confident anchors: where detections actually come from
    return {"shape": list(ref.shape), "max_abs_box_px": float(boxes.max()), "max_abs_score": float(scores.max()),
            "top100_max_abs_score": float(scores[0][:, top].max()), "top100_max_abs_box_px": float(boxes[0][:, top].max())}


def export(pt, fmt, tag, **kw):
    d = WORK / tag
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    shutil.copy2(pt, d / "model.pt")
    return YOLO(d / "model.pt").export(format=fmt, imgsz=IMGSZ, batch=1, device="cpu", **kw)


def layer_channels(model, x):
    """Output channels of every top-level layer except Detect."""
    shapes = {}
    hooks = [m.register_forward_hook(lambda mod, i, o, n=n: shapes.__setitem__(n, o.shape[1] if torch.is_tensor(o) else None))
             for n, m in enumerate(model.model[:-1])]
    with torch.no_grad():
        model.float().eval()(x)
    for h in hooks:
        h.remove()
    return shapes


def attention_integrity(model):
    """Every Attention's qkv/proj/pe widths must still match its head arithmetic, or its view() mis-splits silently."""
    bad, seen = [], 0
    for name, m in model.named_modules():
        if isinstance(m, Attention):
            seen += 1
            dim = m.proj.conv.out_channels
            want = m.num_heads * (2 * m.key_dim + m.head_dim)
            if m.qkv.conv.out_channels != want or m.qkv.conv.in_channels != dim or m.pe.conv.out_channels != dim or m.head_dim * m.num_heads != dim:
                bad.append({"module": name, "qkv_out": m.qkv.conv.out_channels, "qkv_out_wanted": want, "dim": dim,
                            "heads": m.num_heads, "head_dim": m.head_dim, "key_dim": m.key_dim})
    return {"attention_modules": seen, "inconsistent": bad}


def head_consistency(model):
    """Both heads read the same P3/P4/P5 tensors; their first convs must agree on input width."""
    h = model.model[-1]
    rows = []
    for i in range(h.nl):
        row = {"level": f"P{i + 3}", "cv2_in": h.cv2[i][0].conv.in_channels, "cv3_in": h.cv3[i][0][0].conv.in_channels}
        if getattr(h, "one2one_cv2", None) is not None:
            row.update(o2o_cv2_in=h.one2one_cv2[i][0].conv.in_channels, o2o_cv3_in=h.one2one_cv3[i][0][0].conv.in_channels)
        rows.append(row)
    return rows


def forwards(model, x):
    """The three forwards the study needs: eval one-to-many, eval one-to-one, and train (both heads, for the loss)."""
    out = {}
    m = deepcopy(model).float()
    for e2e in (False, True):
        m.eval()
        m.model[-1].end2end = e2e
        with torch.no_grad():
            out[f"eval_{'one2one' if e2e else 'one2many'}"] = list(m(x)[0].shape)
    m.train()
    preds = m(x)
    out["train_keys"] = sorted(preds) if isinstance(preds, dict) else type(preds).__name__
    return out


def prune_ckpt(source, ratio, tag, extra_ignored=()):
    """studyprune.py's own CLI path, unchanged; `extra_ignored` (class names) widens its ignore list without editing it."""
    from ultralytics.nn.modules import block
    out = WORK / f"{tag}.pt"
    out.unlink(missing_ok=True)  # never let a stale file from an earlier run stand in for this one
    argv, keep = sys.argv, studyprune.C2PSA
    sys.argv = ["studyprune.py", "--source", str(source), "--ratio", str(ratio), "--out", str(out)]
    studyprune.C2PSA = (keep, *(AttnC3k2 if n == "AttnC3k2" else getattr(block, n) for n in extra_ignored))  # isinstance() accepts nested tuples
    try:
        studyprune.main()
    finally:
        sys.argv, studyprune.C2PSA = argv, keep
    return out, torch.load(out, map_location="cpu", weights_only=False)["pruning"]


# ---------------------------------------------------------------- 0
def check_unpruned_export():
    x = image_tensor()
    ref = torch_raw(YOLO(YOLO26S).model, x)
    onnx = export(YOLO26S, "onnx", "yolo26s-unpruned-onnx")
    ncnn_dir = export(YOLO26S, "ncnn", "yolo26s-unpruned-ncnn")
    return {"onnx": parity(ref, ort_raw(onnx, x)), "ncnn": parity(ref, ncnn_raw(ncnn_dir, x))}


# ---------------------------------------------------------------- 1, 2
def prune_and_check(source, ratio, tag, extra_ignored=()):
    x = image_tensor()
    base = YOLO(source).model
    pruned_pt, info = prune_ckpt(source, ratio, tag, extra_ignored)
    pruned = YOLO(pruned_pt).model.float().eval()
    before, after = layer_channels(deepcopy(base), x), layer_channels(deepcopy(pruned), x)
    return {
        "extra_ignored": list(extra_ignored),
        "params": [info["params_before"], info["params_after"], round(1 - info["params_after"] / info["params_before"], 3)],
        "gflops": [round(info["gflops_before"], 2), round(info["gflops_after"], 2)],
        "layer_channels": {n: [before[n], after[n]] for n in before},
        "attention": attention_integrity(pruned),
        "heads": head_consistency(pruned),
        "forwards": forwards(pruned, x),
        "pruned_pt": str(pruned_pt),
    }


def export_and_check(tag):
    x = image_tensor()
    pruned_pt = WORK / f"{tag}.pt"
    model = YOLO(pruned_pt).model
    ref = torch_raw(model, x)
    r = {"onnx": parity(ref, ort_raw(export(pruned_pt, "onnx", f"{tag}-onnx"), x)),
         "ncnn": parity(ref, ncnn_raw(export(pruned_pt, "ncnn", f"{tag}-ncnn"), x))}
    # the NMS-free head: export with nms=False, compare the (1, 300, 6) detections
    ref1 = torch_raw(model, x, end2end=True)
    got1 = ort_raw(export(pruned_pt, "onnx", f"{tag}-onnx-e2e", nms=False), x)
    r["onnx_one2one"] = {"shape": list(got1.shape), "max_abs_box_px": float(np.abs(ref1[..., :4] - got1[..., :4]).max()),
                         "max_abs_conf": float(np.abs(ref1[..., 4] - got1[..., 4]).max()),
                         "class_mismatches_top100": int((ref1[0, :100, 5] != got1[0, :100, 5]).sum())}
    return r


# ---------------------------------------------------------------- 3
def median_ms(fn, n=40, warm=8):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return round(float(np.median(ts)), 1)


def check_latency():
    import ncnn
    import onnxruntime as ort
    x = image_tensor()
    models = {"yolo26s-unpruned": YOLO26S, "yolo11s-unpruned": YOLO11S}
    for tag in ("yolo26s-prune30-attnblock", "yolo11s-prune30"):
        if (STAGE_OUT / f"1_prune_{tag}.json").exists() and json.loads((STAGE_OUT / f"1_prune_{tag}.json").read_text())["ok"]:
            models[tag] = WORK / f"{tag}.pt"
    rows = {}
    for tag, pt in models.items():
        s = ort.InferenceSession(str(export(pt, "onnx", f"lat-{tag}-onnx")), providers=["CPUExecutionProvider"])
        feed = {s.get_inputs()[0].name: x.numpy()}
        net, mat = ncnn_net(export(pt, "ncnn", f"lat-{tag}-ncnn")), ncnn.Mat(x[0].numpy()).clone()

        def run_ncnn():
            ex = net.create_extractor()
            ex.input("in0", mat)
            ex.extract("out0")

        rows[tag] = {"ort_ms": median_ms(lambda: s.run(None, feed)), "ncnn_ms": median_ms(run_ncnn)}
    return {"median_ms_640": rows, "note": "laptop CPU, plugged in, single image, median of 40 after 8 warm-up"}


# ---------------------------------------------------------------- 4
def check_training(extra_ignored):
    import studyenv
    import studytrain
    x = image_tensor()
    pruned_pt, info = prune_ckpt(SMOKE26, 0.30, "smoke26-prune30", extra_ignored)
    run = WORK / "train-smoke"
    if run.exists():
        shutil.rmtree(run)
    args = {**studyenv.TRAIN_ARGS, **studytrain.recipe()}
    args.update(epochs=1, workers=0)
    model = YOLO(str(pruned_pt))
    model.train(data=studyenv.DATA, imgsz=320, batch=4, device=0, fraction=0.02, project=str(WORK), name="train-smoke",
                exist_ok=True, trainer=studytrain.PrunedTrainer, **args)  # the recipe already sets plots: false
    csv = (run / "results.csv").read_text(encoding="utf-8").splitlines()
    last = run / "weights" / "last.pt"
    reloaded = YOLO(last).model
    n_reloaded = sum(p.numel() for p in reloaded.parameters())
    onnx = export(last, "onnx", "train-smoke-onnx")
    return {"extra_ignored": list(extra_ignored), "pruned_params": info["params_after"], "reloaded_params": n_reloaded,
            "still_pruned": n_reloaded == info["params_after"], "optimizer": model.trainer.optimizer.__class__.__name__,
            "results_csv": csv, "reloaded_forwards": forwards(reloaded, x), "onnx_parity": parity(torch_raw(reloaded, x), ort_raw(onnx, x))}


# =============================================================== parent side: the plan, each stage in a capped child
def pagefile():
    """Windows page file: allocated and in-use MB (did the run make Windows grow it?)."""
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_PageFileUsage | Select-Object AllocatedBaseSize,CurrentUsage,PeakUsage | ConvertTo-Json"],
                         capture_output=True, text=True).stdout
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return out.strip()


def parent(only):
    import memcap
    import psutil
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "logs").mkdir(exist_ok=True)
    out_file = OUT / "feasibility.json"
    results = json.loads(out_file.read_text(encoding="utf-8")) if only and out_file.exists() else {}
    results["env"] = {p: version(p) for p in ("ultralytics", "torch", "torch-pruning", "onnx", "onnxruntime", "ncnn", "pnnx")}
    results["system_before"] = {"available_gb": round(psutil.virtual_memory().available / memcap.GB, 2), "pagefile_mb": pagefile()}

    def save():
        out_file.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    for spec in stages():
        name = spec["name"]
        if only and name not in only:
            continue
        prior = results.get(spec["same_failure_as"]) if spec["same_failure_as"] else None
        missing = [n for n in spec["needs"] if not results.get(n, {}).get("ok")]
        if prior and not prior.get("ok") and (prior.get("killed") or "MemoryError" in prior.get("error", "")):
            results[name] = {"ok": False, "skipped": f"same failure as {spec['same_failure_as']}: {prior.get('killed') or prior['error']}"}
        elif missing:
            results[name] = {"ok": False, "skipped": f"needs {', '.join(missing)}, which did not succeed"}
        else:
            log = WORK / "logs" / f"{name}.log"
            log.unlink(missing_ok=True)
            (STAGE_OUT / f"{name}.json").unlink(missing_ok=True)
            results[name] = {"ok": None, "started": datetime.now().isoformat(timespec="seconds"), "log": str(log)}
            save()  # a killed run still shows which stage it was in
            print(f"=== {name} (cap {CAP_GB[spec['kind']]} GB, timeout {TIMEOUT_MIN[spec['kind']]} min) ...", flush=True)
            run = memcap.run_capped([str(Path(__file__)), "--stage", name], CAP_GB[spec["kind"]], TIMEOUT_MIN[spec["kind"]] * 60,
                                    log, floor_gb=FLOOR_GB)
            stage_file = STAGE_OUT / f"{name}.json"
            r = json.loads(stage_file.read_text(encoding="utf-8")) if stage_file.exists() and not run["killed"] else None
            if r is None:  # killed by the parent, or died without writing a result (e.g. a native allocation failure)
                at_cap = run["peak_private_gb"] >= 0.95 * CAP_GB[spec["kind"]]
                why = run["killed"] or (f"MemoryError at the {CAP_GB[spec['kind']]} GB cap (died before recording a result)" if at_cap
                                        else f"child exited with code {run['exit']} without a result")
                r = {"ok": False, "error": why, "killed": run["killed"], "log_tail": log.read_text(encoding="utf-8", errors="replace")[-1500:]}
            results[name] = {**r, "limits": {"cap_gb": CAP_GB[spec["kind"]], "timeout_min": TIMEOUT_MIN[spec["kind"]]},
                             "child": run, "log": str(log)}
        save()
        r = results[name]
        print(f"=== {name}: {'OK' if r['ok'] else 'FAILED'} {r.get('skipped') or r.get('error') or ''} "
              f"{'(peak %.2f GB, min available %.2f GB, %.0f s)' % (r['child']['peak_private_gb'], r['child']['min_available_gb'], r['child']['seconds']) if 'child' in r else ''}",
              flush=True)

    if P1.exists():
        import torch
        results["P1_reference"] = torch.load(P1, map_location="cpu", weights_only=False).get("pruning")
    results["system_after"] = {"available_gb": round(psutil.virtual_memory().available / memcap.GB, 2), "pagefile_mb": pagefile()}
    save()
    print("\nSUMMARY:", {k: v.get("ok") for k, v in results.items() if isinstance(v, dict) and "ok" in v})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", help="internal: run one stage in this (capped) process")
    ap.add_argument("--only", nargs="*", help="run just these stages, keeping the other recorded results")
    a = ap.parse_args()
    child(a.stage) if a.stage else parent(a.only)
