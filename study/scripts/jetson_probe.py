"""Read-only report on a Jetson board for the study's Jetson column. Paste the whole output back.

  python3 study/scripts/jetson_probe.py

Standard library only, Python 3.6 or newer, so it runs on any JetPack (4.6 has 3.6, 6 has 3.10) before anything is
installed. It never calls sudo and never writes a file: a value that needs root is printed as "needs sudo" and skipped.
It imports torch only in a child process, if torch is there, to see whether CUDA works.

The verdict applies the rule the study set on 2026-10-21 and narrowed on 2026-10-22: a Jetson Orin Nano on JetPack 6
whose software installs in one day gets a small column (YOLO11s and YOLO26s; baseline as a TensorRT FP16 engine,
quantized as a TensorRT INT8 engine; 640 and 960; latency only). Anything else, JetPack 5 included, is cut and listed
as future work.

studybench.py imports the sysfs readers below (lazily, and only on a Jetson) to record the platform state of a run.
"""

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

SYS = Path("/sys")  # tests point this at a fake tree
NV_TEGRA_RELEASE = Path("/etc/nv_tegra_release")
NVPMODEL_CONF = Path("/etc/nvpmodel.conf")
NVPMODEL_STATUS = Path("/var/lib/nvpmodel/status")
EMC_RATE = Path("/sys/kernel/debug/bpmp/debug/clk/emc/rate")  # debugfs: root only
GPU_DEVFREQ_NAMES = ("gpu", "ga10b", "gv11b", "gp10b", "gm20b")  # Orin JetPack 6 / Orin JetPack 5 / Xavier / TX2 / Nano

CUT = "CUT: list the Jetson as future work"
# L4T release -> JetPack. https://developer.nvidia.com/embedded/jetpack-archive
JETPACK = {(32, 7): "4.6.x", (35, 1): "5.0.2", (35, 2): "5.1", (35, 3): "5.1.1", (35, 4): "5.1.2", (35, 5): "5.1.3",
           (35, 6): "5.1.4", (36, 2): "6.0 DP", (36, 3): "6.0", (36, 4): "6.1 or 6.2"}


def _read(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _run(argv, timeout=20, any_exit=False):
    """stdout of a command, or None when it is missing or fails (any_exit: keep stdout whatever the exit code)."""
    if not shutil.which(argv[0]) and not os.path.isfile(argv[0]):
        return None
    try:
        r = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if (r.returncode == 0 or any_exit) else None


def _ver(s):
    return tuple(int(x) for x in re.findall(r"\d+", s or "")[:3])


# ---------------------------------------------------------------- parsers (pure)


def parse_l4t(text):
    """'# R36 (release), REVISION: 4.3, GCID: ...' -> (36, 4, 3)."""
    m = re.search(r"R(\d+)\s*\(release\),\s*REVISION:\s*(\d+)(?:\.(\d+))?", text or "")
    return (int(m[1]), int(m[2]), int(m[3] or 0)) if m else None


def jetpack_for(l4t):
    if not l4t:
        return None
    if l4t[:2] == (36, 4):  # 36.4.0 is 6.1; 36.4.3 and later are 6.2.x
        return "6.1" if l4t[2] < 3 else "6.2"
    return JETPACK.get(l4t[:2], f"unknown (L4T R{l4t[0]})")


def parse_nvpmodel_conf(text):
    """/etc/nvpmodel.conf -> ([(id, name), ...], default id)."""
    modes = [(int(i), n) for i, n in re.findall(r"<\s*POWER_MODEL\s+ID=(\d+)\s+NAME=(\S+)\s*>", text or "")]
    m = re.search(r"<\s*PM_CONFIG\s+DEFAULT=(\d+)\s*>", text or "")
    return modes, int(m[1]) if m else None


def highest_mode(modes):
    """The mode the study runs in: a MAXN mode (MAXN_SUPER first), else the one with the most watts."""
    def rank(mode):
        name = mode[1].upper()
        watts = re.search(r"(\d+)W", name)
        return ("MAXN" in name, "SUPER" in name, int(watts[1]) if watts else 0)

    return max(modes, key=rank) if modes else None


# ---------------------------------------------------------------- sysfs readers (no root; None when unreadable)


def gpu_devfreq(root=None):
    base = (root or SYS) / "class" / "devfreq"
    found = sorted(d for d in base.glob("*") if any(n in d.name for n in GPU_DEVFREQ_NAMES)) if base.is_dir() else []
    return found[0] if found else None


def _int(path):
    v = _read(path)
    return int(v) if v and v.lstrip("-").isdigit() else None


def gpu_clocks(root=None):
    """GPU devfreq in MHz: cur, min, max, and the highest available frequency."""
    d = gpu_devfreq(root)
    if d is None:
        return None
    mhz = {k: _int(d / f"{k}_freq") for k in ("cur", "min", "max")}
    avail = [int(x) for x in (_read(d / "available_frequencies") or "").split() if x.isdigit()]
    out = {k: round(v / 1e6) if v else None for k, v in mhz.items()}
    out["available_max"] = round(max(avail) / 1e6) if avail else None
    out["node"] = d.name
    return out


def online_cpus(root=None):
    spec = _read((root or SYS) / "devices" / "system" / "cpu" / "online") or ""
    cpus = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        if a.isdigit():
            cpus.extend(range(int(a), int(b or a) + 1))
    return cpus


def cpu_clocks(root=None):
    """Per online CPU, kHz: scaling min / max / cur and the hardware max."""
    base = (root or SYS) / "devices" / "system" / "cpu"
    keys = {"min": "scaling_min_freq", "max": "scaling_max_freq", "cur": "scaling_cur_freq", "hw_max": "cpuinfo_max_freq"}
    return {c: {k: _int(base / f"cpu{c}" / "cpufreq" / f) for k, f in keys.items()} for c in online_cpus(root)}


def jetson_clocks_on(root=None):
    """jetson_clocks pins every minimum clock at its maximum. `jetson_clocks --show` needs sudo, so read the minimums:
    True when the GPU and every online CPU sit at their maximum, False when one does not, None when unreadable.
    EMC is not checked: its clock is in debugfs (root only)."""
    gpu, cpus = gpu_clocks(root), cpu_clocks(root)
    if not gpu or gpu["min"] is None or not cpus or any(v["min"] is None for v in cpus.values()):
        return None
    return gpu["min"] == (gpu["available_max"] or gpu["max"]) and all(v["min"] == v["hw_max"] for v in cpus.values())


def thermal(root=None):
    """{zone type: temperature in C} for every zone in /sys/devices/virtual/thermal."""
    base = (root or SYS) / "devices" / "virtual" / "thermal"
    out = {}
    for z in sorted(base.glob("thermal_zone*"), key=lambda p: int(p.name[12:]) if p.name[12:].isdigit() else 0):
        t, mc = _read(z / "type"), _int(z / "temp")
        if t and mc is not None and mc > -40000:  # some zones report -256000 when the sensor is off
            out[t] = round(mc / 1000, 1)
    return out


def oc_events(root=None):
    """Over-current throttle event counters (Orin, /sys/class/hwmon/hwmon*/oc*_event_cnt), if the kernel exposes them."""
    base = (root or SYS) / "class" / "hwmon"
    out = {}
    for f in sorted(base.glob("hwmon*/oc*_event_cnt")) if base.is_dir() else []:
        v = _int(f)
        if v is not None:
            out[f"{f.parent.name}/{f.name}"] = v
    return out


def sample(root=None):
    """The platform state studybench.py records at the start, every 50 images and the end of a run."""
    gpu = gpu_clocks(root) or {}
    return {"temp_c": thermal(root), "gpu_mhz": gpu.get("cur"), "oc_events": oc_events(root)}


# ---------------------------------------------------------------- verdict (pure)


def head_row(trt):
    """Which YOLO26 head the pinned exporter keeps for this TensorRT (engine/exporter.py L682-713)."""
    t = _ver(trt)
    if not t:
        return "unknown: TensorRT not found"
    if t[:2] < (8, 5):
        return "TensorRT older than 8.5: one-to-one head disabled at any precision, FP16 included (ultralytics #24607)"
    if t[:2] >= (10, 4):
        return "TensorRT 10.4 or newer: one-to-one head kept, INT8 included"
    if t[:2] == (10, 3):
        return "TensorRT 10.3.x on JetPack 6: one-to-one head kept in FP16, disabled for INT8 (ultralytics #23841)"
    if (8, 5) <= t[:2] < (10, 3):
        return ("TensorRT 8.5 to 10.2 (JetPack 5.1 has 8.5.2, JetPack 6.0 has 8.6.2): the exporter keeps the one-to-one "
                "head in FP16 and INT8; not in the ticket's table and never built")
    return "other"


def verdict(f):
    """The decision for the facts `f` collected by facts().

    GO       an Orin Nano on JetPack 6 with nothing in the way of a one-day install
    BLOCKED  an Orin Nano on JetPack 6, but a blocker below must be fixed first
    CUT      anything else (another board, JetPack 4 or 5): the Jetson is listed as future work
    """
    l4t = f.get("l4t") or (0, 0, 0)
    if "orin nano" not in (f.get("model") or "").lower() or l4t[0] != 36:
        return {"decision": CUT, "blockers": [], "notes": [], "route": None, "head_row": head_row(f.get("tensorrt"))}
    notes, blockers = [], []
    if not f.get("tensorrt"):
        blockers.append("TensorRT is not installed: `sudo apt install nvidia-jetpack` installs it with CUDA and cuDNN"
                        + ("" if f.get("sudo_group") else "; this account is not in the sudo group, so ask the lab"))
    if not f.get("pypi"):
        blockers.append("pip cannot reach PyPI: find the lab's proxy (set https_proxy) or a network that can")
    decision = "BLOCKED" if blockers else "GO"
    route = {(36, 4): "JetPack 6.1 / 6.2", (36, 3): "JetPack 6.0"}.get(tuple(l4t[:2]))
    mem = f.get("mem_total_mb") or 0
    if mem and mem < 5000:
        notes.append("4 GB board: the INT8 engine build at 960 may run out of memory. Build it last, headless, with swap; "
                     "if it is killed, retry with --workspace 1")
    if route == "JetPack 6.0":
        notes.append("JetPack 6.0 has TensorRT 8.6 and CUDA 12.2, not the 10.3 / 12.6 the guide was written for: "
                     "follow its JetPack 6.0 lines, or upgrade to 6.2 if the lab allows it")
    if tuple(f.get("python") or ())[:2] != (3, 10):
        notes.append("JetPack 6's Python is 3.10, and the Jetson torch wheels are cp310 only: use /usr/bin/python3")
    if not f.get("pip"):
        notes.append("no pip: `sudo apt install python3-pip python3-venv`, or without sudo "
                     "`curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - --user`")
    if not f.get("sudo_group"):
        notes.append("not in the sudo group: you cannot set nvpmodel or jetson_clocks, and cuDSS must come from its "
                     "tarball (see PLATFORMS.md). The runs then record whatever mode the board is in")
    if (f.get("disk_free_gb") or 0) < 8:
        notes.append("less than 8 GB free on the home disk: the install, dataset and 8 engines need about 6 GB")
    return {"decision": decision, "blockers": blockers, "notes": notes, "route": route, "head_row": head_row(f.get("tensorrt"))}


# ---------------------------------------------------------------- the report


def _meminfo():
    out = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        k, _, v = line.partition(":")
        if v.strip().endswith("kB"):
            out[k] = int(v.split()[0]) // 1024
    return out


def _dpkg(*patterns):
    """Installed packages matching the globs, as 'name version' lines."""
    out = _run(["dpkg-query", "-W", "-f=${db:Status-Abbrev} ${Package} ${Version}\n", *patterns], any_exit=True) or ""
    return [line[3:].strip() for line in out.splitlines() if line.startswith("ii")]


def _py(code, python=None, timeout=120):
    return _run([python or sys.executable, "-c", code], timeout=timeout)


def _cudnn_header():
    for h in [*Path("/usr/include").glob("cudnn_version*.h"), *Path("/usr/include/aarch64-linux-gnu").glob("cudnn_version*.h")]:
        v = dict(re.findall(r"#define CUDNN_(MAJOR|MINOR|PATCHLEVEL)\s+(\d+)", _read(h) or ""))
        if len(v) == 3:
            return f"{v['MAJOR']}.{v['MINOR']}.{v['PATCHLEVEL']} ({h})"
    return None


def _cuda():
    j = _read("/usr/local/cuda/version.json")
    if j:
        try:
            return json.loads(j)["cuda"]["version"]
        except (ValueError, KeyError):
            pass
    nvcc = _run(["/usr/local/cuda/bin/nvcc", "--version"])
    m = re.search(r"release ([\d.]+)", nvcc or "")
    return m[1] if m else _read("/usr/local/cuda/version.txt")


def _reach(url):
    import urllib.request

    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pip"}), timeout=15) as r:
            return r.status if hasattr(r, "status") else r.getcode()
    except Exception as e:  # noqa: BLE001  any failure is the answer
        return f"{type(e).__name__}: {e}"[:120]


def _packages():
    try:
        from importlib.metadata import PackageNotFoundError, version  # Python 3.8+
    except ImportError:
        return {}

    def v(p):
        try:
            return version(p)
        except PackageNotFoundError:
            return None

    return {p: v(p) for p in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "onnx", "onnxslim", "pyyaml", "tensorrt")}


def facts():
    """Everything the report prints, read from this board. Read-only."""
    try:
        import grp
    except ImportError:  # not Linux: the report still runs, with nothing found
        grp = None

    mem = _meminfo()
    home_free = shutil.disk_usage(str(Path.home())).free / (1 << 30)
    l4t_raw = (_read(NV_TEGRA_RELEASE) or "").splitlines()
    l4t = parse_l4t(l4t_raw[0] if l4t_raw else "")
    trt = _py("import tensorrt; print(tensorrt.__version__)", timeout=60)
    if not trt and sys.executable != "/usr/bin/python3" and os.path.exists("/usr/bin/python3"):
        trt = _py("import tensorrt; print(tensorrt.__version__)", "/usr/bin/python3", timeout=60)  # outside this venv
    conf, _ = parse_nvpmodel_conf(_read(NVPMODEL_CONF))
    groups = set()
    for g in os.getgroups() if grp else []:
        try:
            groups.add(grp.getgrgid(g).gr_name)
        except KeyError:
            pass
    reach = {name: _reach(url) for name, url in (
        ("PyPI", "https://pypi.org/simple/ultralytics/"),
        ("GitHub (Ultralytics' Jetson wheels)", "https://github.com/ultralytics/assets/releases/tag/v0.0.0"),
        ("NVIDIA downloads (cuDSS, torch)", "https://developer.download.nvidia.com/compute/redist/jp/"),
        ("jetson-ai-lab index", "https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/torch/"))}
    return {
        "model": (_read("/proc/device-tree/model") or "").rstrip("\x00") or None,
        "compatible": (_read("/proc/device-tree/compatible") or "").replace("\x00", " ").strip() or None,
        "mem_total_mb": mem.get("MemTotal"), "mem_available_mb": mem.get("MemAvailable"), "swap_total_mb": mem.get("SwapTotal"),
        "zram_devices": len(list(Path("/sys/block").glob("zram*"))),
        "disk_free_gb": round(home_free, 1),
        "root_device": next((line.split()[0] for line in (_read("/proc/mounts") or "").splitlines() if line.split()[1:2] == ["/"]), None),
        "l4t_raw": l4t_raw[0] if l4t_raw else None, "l4t": l4t, "jetpack": jetpack_for(l4t),
        "dpkg": _dpkg("nvidia-jetpack", "nvidia-l4t-core", "tensorrt", "libnvinfer[0-9]*", "python3-libnvinfer",
                      "cuda-runtime-*", "cuda-cudart-[0-9]*", "libcudnn[0-9]*", "cudss", "libcusparselt0"),
        "tensorrt": trt, "cuda": _cuda(), "cudnn": _cudnn_header(),
        "python": tuple(sys.version_info[:3]), "python_exe": sys.executable,
        "pip": _run([sys.executable, "-m", "pip", "--version"]),
        "pip_config": _run([sys.executable, "-m", "pip", "config", "list"]),
        "venv": _run([sys.executable, "-c", "import ensurepip, venv"]) is not None,
        "packages": _packages(),
        "torch_cuda": _py("import torch; print(torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"),
        "sudo_group": "sudo" in groups, "groups": sorted(groups),
        "nvpmodel_q": _run(["nvpmodel", "-q"], any_exit=True), "nvpmodel_status": _read(NVPMODEL_STATUS), "nvpmodel_modes": conf,
        "highest_mode": highest_mode(conf),
        "gpu_clocks": gpu_clocks(), "cpu_clocks": cpu_clocks(), "jetson_clocks": jetson_clocks_on(),
        "emc_rate": _read(EMC_RATE) or "needs sudo (debugfs)",
        "thermal": thermal(), "oc_events": oc_events(),
        "default_target": _run(["systemctl", "get-default"]),
        "proxy_vars_set": sorted(k for k in os.environ if k.lower() in ("http_proxy", "https_proxy", "no_proxy", "all_proxy")),
        "reach": reach, "pypi": reach["PyPI"] == 200,
    }


def report(f):
    v = verdict(f)
    cpus = f["cpu_clocks"] or {}
    cpu_line = ", ".join(f"cpu{c} min {d['min']} max {d['max']} hw {d['hw_max']}" for c, d in list(cpus.items())[:2])
    lines = [
        "=== Jetson probe (study/scripts/jetson_probe.py) ===",
        f"board         {f['model']}",
        f"compatible    {f['compatible']}",
        f"memory        {f['mem_total_mb']} MB total, {f['mem_available_mb']} MB available, swap {f['swap_total_mb']} MB, zram devices {f['zram_devices']}",
        f"disk          {f['disk_free_gb']} GB free in {Path.home()}; root on {f['root_device']}",
        f"L4T           {f['l4t_raw']}",
        f"JetPack       {f['jetpack']} (from L4T)",
        f"dpkg          {'; '.join(f['dpkg']) or 'none of the JetPack packages found'}",
        f"TensorRT      {f['tensorrt'] or 'import tensorrt failed'}",
        f"CUDA          {f['cuda']}",
        f"cuDNN         {f['cudnn']}",
        f"Python        {'.'.join(map(str, f['python']))} at {f['python_exe']}; venv module {'yes' if f['venv'] else 'NO'}",
        f"pip           {f['pip'] or 'NOT INSTALLED'}",
        f"pip config    {f['pip_config'] or '(none)'}",
        f"packages      {', '.join(f'{k} {v}' for k, v in f['packages'].items() if v) or '(none of ours)'}",
        f"torch         {f['torch_cuda'] or 'not importable'}",
        f"sudo group    {'yes' if f['sudo_group'] else 'no'} (groups: {' '.join(f['groups'])})",
        f"nvpmodel -q   {(f['nvpmodel_q'] or 'failed (may need sudo)').replace(chr(10), ' | ')}",
        f"nvpmodel      modes {f['nvpmodel_modes']}; status file {f['nvpmodel_status']!r}; highest {f['highest_mode']}",
        f"jetson_clocks {({True: 'ON', False: 'off', None: 'unknown'})[f['jetson_clocks']]} (inferred: GPU and CPU minimum clocks at their maximum)",
        f"GPU clock     {f['gpu_clocks']}",
        f"CPU clock     {cpu_line} (kHz)",
        f"EMC clock     {f['emc_rate']}",
        f"thermal       {f['thermal']}",
        f"OC events     {f['oc_events'] or 'no counters exposed'}",
        f"boot target   {f['default_target']}",
        f"proxy vars    {', '.join(f['proxy_vars_set']) or 'none'}",
        *[f"reach         {k}: {r}" for k, r in f["reach"].items()],
        "",
        f"VERDICT       {v['decision']}",
        *[f"  BLOCKER     {b}" for b in v["blockers"]],
        *[f"  note        {n}" for n in v["notes"]],
        f"install route {v['route'] or '-'} (study/PLATFORMS.md)",
        f"YOLO26 head   {v['head_row']}",
        "              (the study's engines use the one-to-many head at every precision, nms=None; this row matters only "
        "for a labelled one-to-one-head row)",
    ]
    if v["decision"] != CUT and f["highest_mode"]:
        i, name = f["highest_mode"]
        lines.append(f"run mode      sudo nvpmodel -m {i}   ({name}; it may ask to reboot), then sudo jetson_clocks")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(facts()))
