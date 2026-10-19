"""PROTOTYPE - throwaway. Which DepGraph pruning group blows up on YOLO26s, and why? (MEMORY-INCIDENT-2026-10-19.md)

Builds the dependency graph exactly as studyprune.prune does (unchunk_c2f, CPU example at 640), lists every
flatten index mapping DepGraph attached to a reshape, then walks get_all_groups one group at a time, logging the
root, time and private memory. A guard on the flatten mapping stops a runaway group at a few million indices
instead of 25 GB, and reports the mapping and root that ran away.

  python diagnose_groups.py --model yolo26s.pt [--ignore Attention] [--e2e off] [--cap-gb 3]
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import memcap  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--base", nargs="*", default=["C2PSA", "Detect"], help="studyprune's ignore list; drop C2PSA to test YOLO11's attention")
p.add_argument("--ignore", nargs="*", default=[], help="extra module classes to ignore, e.g. Attention")
p.add_argument("--trace-root", default=None, help="sample the first --trace-n deps added to the group rooted here")
p.add_argument("--trace-n", type=int, default=3000)
p.add_argument("--e2e", choices=["model", "off"], default="model", help="off: trace with the one-to-one branch unselected")
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--cap-gb", type=float, default=3.0)
p.add_argument("--limit", type=int, default=2_000_000, help="abort a group once one mapping emits this many indices")
p.add_argument("--out", default=None)
a = p.parse_args()
memcap.cap_this_process(a.cap_gb)

import studyenv  # noqa: E402,F401
import studyprune  # noqa: E402

import psutil  # noqa: E402
import torch  # noqa: E402
import torch_pruning as tp  # noqa: E402
from torch_pruning.dependency import index_mapping as im  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import block as B  # noqa: E402
from ultralytics.nn.modules import C2PSA, Detect  # noqa: E402

ME = psutil.Process()


def gb():
    return round(ME.memory_info().private / memcap.GB, 3)


class Runaway(Exception):
    pass


_flatten_call = im._FlattenIndexMapping.__call__


def guarded(self, idxs):
    if not self.reverse and len(idxs) * self._stride > a.limit:
        raise Runaway(f"flatten mapping stride={self._stride} asked to expand {len(idxs)} indices -> {len(idxs) * self._stride}")
    return _flatten_call(self, idxs)


im._FlattenIndexMapping.__call__ = guarded

t0 = time.time()
model = studyprune.unchunk_c2f(YOLO(a.model).model.float())
if a.e2e == "off":
    model.model[-1].end2end = False
for q in model.parameters():
    q.requires_grad_(True)
head = model.model[-1]
report = {"model": a.model, "ignore_base": a.base, "ignore_extra": a.ignore, "e2e": a.e2e, "head_end2end": bool(getattr(head, "end2end", False)),
          "has_one2one": getattr(head, "one2one_cv2", None) is not None, "torch_pruning": tp.__version__}

DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1, 3, a.imgsz, a.imgsz))
report["build_s"], report["after_build_gb"] = round(time.time() - t0, 1), gb()
im.update_index_mapping(DG)

# Every flatten mapping attached during index-mapping update: these are the only mappings that GROW index lists.
flat = []
for node in DG.module2node.values():
    for dep in node.dependencies:
        for m in dep.index_mapping:
            if isinstance(m, im._FlattenIndexMapping):
                flat.append({"from": dep.source.name, "to": dep.target.name, "stride": m._stride, "reverse": m.reverse})
report["flatten_mappings"] = flat

from ultralytics.nn.modules import head as H  # noqa: E402
ignore_cls = [getattr(B, n, None) or getattr(H, n) for n in a.base + a.ignore]
ignored = []
for m in model.modules():
    if isinstance(m, tuple(ignore_cls)):
        ignored.extend(m.modules())  # what BasePruner does with ignored_layers
names = {m: n for n, m in model.named_modules()}

groups, verdict, current = [], "all groups built", {}
_get_group = DG.get_pruning_group


def logged_group(module, fn, idxs):
    """Every root get_all_groups tries, including the ones it then rejects for touching an ignored layer."""
    current.update(root=names.get(module, "?"), t=time.time())
    g = _get_group(module, fn, idxs)
    groups.append({"root": current["root"], "deps": len(g), "max_idxs": max(len(i.idxs) for i in g),
                   "s": round(time.time() - current["t"], 2), "gb": gb()})
    return g


DG.get_pruning_group = logged_group

# Sample what the runaway group keeps adding: (target, number of indices, min, max index, distinct root indices).
trace = []
_add_dep = tp.dependency.Group.add_dep


def traced_add_dep(self, dep, idxs):
    if current.get("root") == a.trace_root:
        plain = [i.idx for i in idxs]
        trace.append([dep.target.name[:60], len(plain), min(plain), max(plain), len({i.root_idx for i in idxs})])
        if len(trace) >= a.trace_n:
            raise Runaway(f"stopped after sampling {a.trace_n} deps")
    return _add_dep(self, dep, idxs)


if a.trace_root:
    tp.dependency.Group.add_dep = traced_add_dep
try:
    yielded = sum(1 for _ in DG.get_all_groups(ignored_layers=ignored, root_module_types=(torch.nn.Conv2d, torch.nn.Linear)))
    report["groups_yielded"] = yielded
except Runaway as e:
    verdict = f"RUNAWAY in the group rooted at {current['root']}: {e}"
except MemoryError:
    verdict = f"MemoryError at the {a.cap_gb} GB cap, in the group rooted at {current['root']}"

report.update(trace_head=trace[:40], trace_tail=trace[-40:]) if trace else None
report.update(verdict=verdict, groups_built=len(groups), slowest=sorted(groups, key=lambda g: -g["s"])[:5],
              last_groups=groups[-3:], peak_gb=gb(), total_s=round(time.time() - t0, 1))
print(json.dumps(report, indent=1))
if a.out:
    Path(a.out).write_text(json.dumps({**report, "groups": groups}, indent=1), encoding="utf-8")
