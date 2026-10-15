"""PROTOTYPE - throwaway, lives only on branch prototype/small-object-effect. Map ticket: Confirm the small-object effect.

Pruning arm, laptop half: uniform 30% structured channel prune of A1 (yolo11s-640-s0) with Torch-Pruning
DepGraph, round_to=8, L2-magnitude importance. Saves an Ultralytics-loadable pruned checkpoint (the input to
the Kaggle recovery fine-tune) and evaluates it UNRECOVERED on val and test-dev as a quick look.

  .venv/Scripts/python prototype_small_object/prune_arm.py

Ignored (never pruned): C2PSA (its attention reshapes by head count) and the Detect head, whose output
channels are fixed by reg_max and the class count.
"""

import json
import shutil
import types
import sys
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import studyenv  # noqa: E402  must precede ultralytics
import studyeval  # noqa: E402

import torch  # noqa: E402
import torch_pruning as tp  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import C2PSA, C2f, Detect  # noqa: E402
from ultralytics.utils.torch_utils import get_flops  # noqa: E402

A1 = Path(r"C:\Users\srikr\uav-yolo-checkpoints\yolo11s-640-s0\best.pt")
WORK = studyenv.ROOT / "runs" / "prototype-small-object"
OUT = Path(__file__).resolve().parent / "results"
IMGSZ, ROUND_TO = 640, 8
RATIO = float(sys.argv[1]) if len(sys.argv) > 1 else 0.30
TAG = f"prune{round(RATIO * 100)}"


def _split_forward(self, x):
    y = [self.cv0(x), self.cv1(x)]
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))


def unchunk_c2f(model):
    """Replace C2f/C3k2's cv1-then-chunk with two convs (cv0, cv1). DepGraph mis-indexes the chunk otherwise.

    The same trick as Torch-Pruning's YOLOv8 example (C2f_v2); numerically identical before pruning.
    """
    for m in model.modules():
        if isinstance(m, C2f) and not hasattr(m, "cv0"):
            c, old = m.c, m.cv1
            halves = []
            for sl in (slice(0, c), slice(c, 2 * c)):
                h = deepcopy(old)
                h.conv = torch.nn.Conv2d(old.conv.in_channels, c, old.conv.kernel_size, old.conv.stride, old.conv.padding, bias=False)
                h.conv.weight.data = old.conv.weight.data[sl].clone()
                h.bn = torch.nn.BatchNorm2d(c, eps=old.bn.eps, momentum=old.bn.momentum)
                for k in ("weight", "bias", "running_mean", "running_var"):
                    getattr(h.bn, k).data = getattr(old.bn, k).data[sl].clone()
                halves.append(h)
            m.cv0, m.cv1 = halves
            m.forward = types.MethodType(_split_forward, m)
    return model


def prune(model):
    model = unchunk_c2f(deepcopy(model).float()).eval()  # eval() after: the new BatchNorms start in train mode
    for p in model.parameters():
        p.requires_grad_(True)
    ignored = [m for m in model.modules() if isinstance(m, (C2PSA, Detect))]
    example = torch.randn(1, 3, IMGSZ, IMGSZ)
    pruner = tp.pruner.MagnitudePruner(
        model,
        example,
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=RATIO,
        global_pruning=False,  # uniform: the same ratio in every prunable layer
        round_to=ROUND_TO,
        ignored_layers=ignored,
    )
    pruner.step()
    with torch.no_grad():
        before = model(example)[0]
        rechunk_c2f(model).eval()
        after = model(example)[0]  # the pruned graph must still run end to end, on stock Ultralytics forwards
    print(f"re-chunk max abs diff {(before - after).abs().max():.2e}, max |out| {before.abs().max():.1f}")
    assert torch.allclose(before, after, rtol=1e-4, atol=1e-3), "re-chunking changed the pruned model output"
    return model


def rechunk_c2f(model):
    """Fold cv0/cv1 back into one cv1 so the pruned model uses only stock Ultralytics classes and forwards.

    Pickling the patched forward would tie the checkpoint to this script; the stock form loads anywhere (Kaggle).
    """
    for m in model.modules():
        if isinstance(m, C2f) and hasattr(m, "cv0"):
            a, b = m.cv0, m.cv1
            assert a.conv.out_channels == b.conv.out_channels, "unequal halves: chunk(2) would mis-split"
            c, merged = a.conv.out_channels, deepcopy(a)
            merged.conv = torch.nn.Conv2d(a.conv.in_channels, 2 * c, a.conv.kernel_size, a.conv.stride, a.conv.padding, bias=False)
            merged.conv.weight.data = torch.cat([a.conv.weight.data, b.conv.weight.data])
            merged.bn = torch.nn.BatchNorm2d(2 * c, eps=a.bn.eps, momentum=a.bn.momentum)
            for k in ("weight", "bias", "running_mean", "running_var"):
                getattr(merged.bn, k).data = torch.cat([getattr(a.bn, k).data, getattr(b.bn, k).data])
            m.cv1, m.c = merged, c
            del m.cv0, m.forward
    return model


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    yolo = YOLO(A1)
    base = yolo.model
    n0, f0 = sum(p.numel() for p in base.parameters()), get_flops(base, IMGSZ)
    pruned = prune(base)
    n1, f1 = sum(p.numel() for p in pruned.parameters()), get_flops(pruned, IMGSZ)
    print(f"params {n0 / 1e6:.2f}M -> {n1 / 1e6:.2f}M ({1 - n1 / n0:.1%} removed); GFLOPs {f0:.1f} -> {f1:.1f}")

    # Ultralytics checkpoint layout: the pickled module keeps the pruned shapes, so YOLO() and train() load it as is.
    ckpt = torch.load(A1, map_location="cpu", weights_only=False)
    ckpt.update(model=deepcopy(pruned).half(), ema=None, optimizer=None, updates=None, epoch=-1,
                date=datetime.now(timezone.utc).isoformat(), pruning={"tool": f"torch-pruning {version('torch-pruning')}", "ratio": RATIO, "round_to": ROUND_TO, "importance": "L2 magnitude", "ignored": "C2PSA, Detect"})
    pruned_pt = WORK / TAG / f"yolo11s-640-s0-{TAG}.pt"
    pruned_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, pruned_pt)
    shutil.copy2(pruned_pt, Path(r"C:\Users\srikr\uav-yolo-checkpoints") / pruned_pt.name)

    summary = {"params_before": n0, "params_after": n1, "gflops_before": f0, "gflops_after": f1, "ratio": RATIO, "round_to": ROUND_TO,
               "torch_pruning": version("torch-pruning"), "checkpoint": str(pruned_pt), "evals": {}}
    for split in ("val", "test"):
        result = studyeval.evaluate(str(pruned_pt), split=split, imgsz=IMGSZ, device=0)
        summary["evals"][split] = result
        s = result["stats"]
        print(f"[{TAG}-unrecovered {split}] AP_all {s['coco/AP_all']:.4f} small {s['coco/AP_small']:.4f} medium {s['coco/AP_medium']:.4f} large {s['coco/AP_large']:.4f}")
    (OUT / f"{TAG}-unrecovered.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
