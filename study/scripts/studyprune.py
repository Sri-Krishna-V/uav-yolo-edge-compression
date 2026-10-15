"""Uniform structured channel pruning of a trained study checkpoint, the input to a pruning-recovery run.

  python studyprune.py --source runs/baseline/yolo11s-640-s0/weights/best.pt --ratio 0.30 --out PRUNED.pt

Torch-Pruning DepGraph, L2-magnitude importance, the same ratio in every prunable layer, round_to=8 (without it
the pruned channel counts defeat the hardware kernels and buy no latency). Needs no data and no randomness, so
the same source checkpoint always gives the same pruned model: kaggle/session.py re-prunes A1 inside the session
instead of shipping the pruned file. The checkpoint records sha256 of the source and of the pruned weights.

Never pruned: C2PSA (its attention reshapes by head count) and the Detect head, whose output channels are fixed
by reg_max and the class count.

The output keeps the Ultralytics checkpoint layout with the pruned module pickled in it, so YOLO() and
studyeval load it as is. TRAIN it only through `studytrain.py --pruned`: a stock train() rebuilds the model
from its yaml and silently loads only the weights whose shapes still match.
"""

import argparse
import hashlib
import json
import types
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import studyenv  # noqa: F401  must precede ultralytics

import torch  # noqa: E402
import torch_pruning as tp  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import C2PSA, C2f, Detect  # noqa: E402
from ultralytics.utils.torch_utils import get_flops  # noqa: E402

ROUND_TO = 8


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_sha256(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def _split_forward(self, x):
    y = [self.cv0(x), self.cv1(x)]
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))


def _slice_conv(conv, rows):
    """A copy of an Ultralytics Conv keeping only the output channels in `rows` (a slice)."""
    out = deepcopy(conv)
    w = conv.conv.weight.data[rows]
    out.conv = torch.nn.Conv2d(conv.conv.in_channels, w.shape[0], conv.conv.kernel_size, conv.conv.stride, conv.conv.padding, bias=False)
    out.conv.weight.data = w.clone()
    out.bn = torch.nn.BatchNorm2d(w.shape[0], eps=conv.bn.eps, momentum=conv.bn.momentum)
    for k in ("weight", "bias", "running_mean", "running_var"):
        getattr(out.bn, k).data = getattr(conv.bn, k).data[rows].clone()
    return out


def unchunk_c2f(model):
    """Split C2f/C3k2's cv1-then-chunk into two convs (cv0, cv1); DepGraph mis-indexes the chunk otherwise.

    Torch-Pruning's YOLOv8 example does the same (C2f_v2). Numerically identical to the original.
    """
    for m in model.modules():
        if isinstance(m, C2f) and not hasattr(m, "cv0"):
            c, old = m.c, m.cv1
            m.cv0, m.cv1 = _slice_conv(old, slice(0, c)), _slice_conv(old, slice(c, 2 * c))
            m.forward = types.MethodType(_split_forward, m)
    return model.eval()  # the new BatchNorms start in train mode


def rechunk_c2f(model):
    """Fold cv0/cv1 back into one cv1, so the pruned model uses only stock Ultralytics classes and forwards.

    A pickled patched forward would tie the checkpoint to this file; the stock form loads anywhere.
    """
    for m in model.modules():
        if isinstance(m, C2f) and hasattr(m, "cv0"):
            a, b = m.cv0, m.cv1
            if a.conv.out_channels != b.conv.out_channels:
                raise RuntimeError(f"unequal C2f halves {a.conv.out_channels}/{b.conv.out_channels}: chunk(2) would mis-split")
            merged = deepcopy(a)
            merged.conv = torch.nn.Conv2d(a.conv.in_channels, 2 * a.conv.out_channels, a.conv.kernel_size, a.conv.stride, a.conv.padding, bias=False)
            merged.conv.weight.data = torch.cat([a.conv.weight.data, b.conv.weight.data])
            merged.bn = torch.nn.BatchNorm2d(2 * a.conv.out_channels, eps=a.bn.eps, momentum=a.bn.momentum)
            for k in ("weight", "bias", "running_mean", "running_var"):
                getattr(merged.bn, k).data = torch.cat([getattr(a.bn, k).data, getattr(b.bn, k).data])
            m.cv1, m.c = merged, a.conv.out_channels
            del m.cv0, m.forward
    return model.eval()


def prune(model, ratio, imgsz):
    model = unchunk_c2f(deepcopy(model).float())
    for p in model.parameters():
        p.requires_grad_(True)
    example = torch.randn(1, 3, imgsz, imgsz)
    tp.pruner.MagnitudePruner(
        model,
        example,
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=ratio,
        global_pruning=False,  # uniform: the same ratio in every prunable layer
        round_to=ROUND_TO,
        ignored_layers=[m for m in model.modules() if isinstance(m, (C2PSA, Detect))],
    ).step()
    with torch.no_grad():
        before = model(example)[0]
        after = rechunk_c2f(model)(example)[0]
    if not torch.allclose(before, after, rtol=1e-4, atol=1e-3):
        raise RuntimeError(f"re-chunking changed the pruned model's output (max diff {(before - after).abs().max():.3g})")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="a trained study checkpoint, e.g. A1's best.pt")
    p.add_argument("--ratio", type=float, required=True, help="channel fraction removed per layer, e.g. 0.30")
    p.add_argument("--imgsz", type=int, default=640, help="only shapes the trace; the pruned model is resolution-free")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    base = YOLO(a.source).model
    pruned = prune(base, a.ratio, a.imgsz)
    n0, n1 = sum(t.numel() for t in base.parameters()), sum(t.numel() for t in pruned.parameters())
    info = {
        "tool": f"torch-pruning {version('torch-pruning')}",
        "method": "DepGraph MagnitudePruner, L2 importance, uniform (global_pruning=False)",
        "ratio": a.ratio,
        "round_to": ROUND_TO,
        "ignored": ["C2PSA", "Detect"],
        "source": str(a.source),
        "source_sha256": sha256(a.source),
        "pruned_weights_sha256": weights_sha256(pruned),
        "params_before": n0,
        "params_after": n1,
        "gflops_before": get_flops(base, a.imgsz),
        "gflops_after": get_flops(pruned, a.imgsz),
    }
    ckpt = torch.load(a.source, map_location="cpu", weights_only=False)
    ckpt.update(model=deepcopy(pruned).half(), ema=None, optimizer=None, updates=None, epoch=-1,
                date=datetime.now(timezone.utc).isoformat(timespec="seconds"), pruning=info)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, a.out)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
