"""Check the Jetson probe's verdict and the Jetson platform-state readers on fake boards. Runs on any machine:

  python study/scripts/tests/test_jetson.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jetson_probe as jp  # noqa: E402

FAILED = []


def check(label, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        FAILED.append(label)
        if detail:
            print(f"     {detail}")


ORIN_NANO_8GB_JP62 = {
    "model": "NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super",
    "l4t": (36, 4, 3), "tensorrt": "10.3.0.30", "mem_total_mb": 7620, "python": (3, 10, 12),
    "pip": True, "pypi": True, "disk_free_gb": 40.0, "sudo_group": True,
}

v = jp.verdict(ORIN_NANO_8GB_JP62)
check("1 Orin Nano 8 GB on JetPack 6.2: GO", v["decision"] == "GO", v)
check("1 its YOLO26 head row is TensorRT 10.3.x on JetPack 6", v["head_row"].startswith("TensorRT 10.3.x on JetPack 6"), v)
check("1 no RAM warning on 8 GB", not any("4 GB" in n for n in v["notes"]), v)

ORIN_NANO_4GB_JP512 = {**ORIN_NANO_8GB_JP62, "model": "NVIDIA Jetson Orin Nano Developer Kit", "l4t": (35, 4, 1),
                       "tensorrt": "8.5.2.2", "mem_total_mb": 3500, "python": (3, 8, 10)}
v = jp.verdict(ORIN_NANO_4GB_JP512)
check("2 Orin Nano on JetPack 5.1.2: CUT", v["decision"] == "CUT: list the Jetson as future work", v)
check("2 TensorRT 8.5 keeps the one-to-one head at both precisions", v["head_row"].startswith("TensorRT 8.5 to 10.2"), v)
v = jp.verdict({**ORIN_NANO_8GB_JP62, "mem_total_mb": 3500})
check("2 Orin Nano 4 GB on JetPack 6.2: GO, with a warning about the 960 INT8 build", v["decision"] == "GO"
      and any("4 GB" in n and "960" in n for n in v["notes"]), v)
v = jp.verdict({**ORIN_NANO_4GB_JP512, "tensorrt": None, "pypi": False})
check("2 JetPack 5 stays CUT even with blockers", v["decision"].startswith("CUT"), v)

v = jp.verdict({**ORIN_NANO_8GB_JP62, "tensorrt": None, "sudo_group": False})
check("3 JetPack 6 without TensorRT and without sudo: BLOCKED", v["decision"] == "BLOCKED", v)
check("3 the blocker says what installs TensorRT", any("nvidia-jetpack" in b for b in v["blockers"]), v)
v = jp.verdict({**ORIN_NANO_8GB_JP62, "pypi": False})
check("3 PyPI out of reach: BLOCKED", v["decision"] == "BLOCKED" and any("PyPI" in b for b in v["blockers"]), v)
check("3 a clean board has no blockers", jp.verdict(ORIN_NANO_8GB_JP62)["blockers"] == [])
v = jp.verdict({**ORIN_NANO_8GB_JP62, "tensorrt": None, "tensorrt_dpkg": "libnvinfer10 10.3.0.30-1+cuda12.5"})
check("3 TensorRT libraries without the Python package: the blocker names python3-libnvinfer",
      v["decision"] == "BLOCKED" and any("python3-libnvinfer" in b for b in v["blockers"]), v)

v = jp.verdict({**ORIN_NANO_8GB_JP62, "model": "NVIDIA Jetson Nano Developer Kit", "l4t": (32, 7, 4), "tensorrt": "8.2.1.8",
                "mem_total_mb": 3956, "python": (3, 6, 9)})
check("4 original Jetson Nano on JetPack 4.6: CUT", v["decision"] == "CUT: list the Jetson as future work", v)
check("4 TensorRT 8.2 drops the one-to-one head at any precision", v["head_row"].startswith("TensorRT older than 8.5"), v)
v = jp.verdict({**ORIN_NANO_8GB_JP62, "model": "NVIDIA Jetson Orin NX Engineering Reference Developer Kit"})
check("4 an Orin NX on JetPack 6: CUT (the rule names the Orin Nano)", v["decision"].startswith("CUT"), v)

check("5 L4T 36.4.3 is JetPack 6.2 and its install route", jp.jetpack_for((36, 4, 3)) == "6.2"
      and jp.verdict(ORIN_NANO_8GB_JP62)["route"] == "JetPack 6.1 / 6.2")
check("5 L4T 36.3 is JetPack 6.0", jp.verdict({**ORIN_NANO_8GB_JP62, "l4t": (36, 3, 0), "tensorrt": "8.6.2.3"})["route"] == "JetPack 6.0")
check("5 L4T 35.4.1 is JetPack 5.1.2, with no install route", jp.jetpack_for((35, 4, 1)) == "5.1.2" and jp.verdict(ORIN_NANO_4GB_JP512)["route"] is None)
check("5 parse_l4t", jp.parse_l4t("# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64") == (36, 4, 3))

CONF = """< POWER_MODEL ID=0 NAME=15W >
CPU_ONLINE CORE_0 1
< POWER_MODEL ID=1 NAME=25W >
CPU_ONLINE CORE_0 1
< POWER_MODEL ID=2 NAME=MAXN_SUPER >
CPU_ONLINE CORE_0 1
< POWER_MODEL ID=3 NAME=7W >
< PM_CONFIG DEFAULT=2 >
"""
modes, default = jp.parse_nvpmodel_conf(CONF)
check("6 nvpmodel.conf: four modes, default 2", modes == [(0, "15W"), (1, "25W"), (2, "MAXN_SUPER"), (3, "7W")] and default == 2, (modes, default))
check("6 the highest mode is MAXN_SUPER", jp.highest_mode(modes) == (2, "MAXN_SUPER"))
check("6 without MAXN the highest is the most watts", jp.highest_mode([(0, "15W"), (1, "7W")]) == (0, "15W"))
check("6 nvpmodel -q: the current mode", jp.current_mode("NV Power Mode: MAXN_SUPER\n2") == (2, "MAXN_SUPER"))
check("6 nvpmodel -q failed: no mode", jp.current_mode(None) is None)


def fake_sysfs(gpu_min, gpu_max, cpu_min, temps=(("cpu-thermal", 51250), ("gpu-thermal", 49500))):
    root = Path(tempfile.mkdtemp())
    gpu = root / "class" / "devfreq" / "17000000.gpu"
    gpu.mkdir(parents=True)
    for k, val in (("cur_freq", gpu_max), ("min_freq", gpu_min), ("max_freq", gpu_max), ("available_frequencies", "306000000 624750000 1020000000")):
        (gpu / k).write_text(f"{val}\n")
    (root / "devices" / "system" / "cpu").mkdir(parents=True)
    (root / "devices" / "system" / "cpu" / "online").write_text("0-1\n")
    for c in (0, 1):
        d = root / "devices" / "system" / "cpu" / f"cpu{c}" / "cpufreq"
        d.mkdir(parents=True)
        for k, val in (("scaling_min_freq", cpu_min), ("scaling_max_freq", 1728000), ("cpuinfo_max_freq", 1728000), ("scaling_cur_freq", 1728000)):
            (d / k).write_text(f"{val}\n")
    for i, (t, mc) in enumerate(temps):
        z = root / "devices" / "virtual" / "thermal" / f"thermal_zone{i}"
        z.mkdir(parents=True)
        (z / "type").write_text(t + "\n")
        (z / "temp").write_text(f"{mc}\n")
    return root


check("7 jetson_clocks on: GPU and CPU minimum at their maximum", jp.jetson_clocks_on(fake_sysfs(1020000000, 1020000000, 1728000)) is True)
check("7 jetson_clocks off: GPU minimum below its maximum", jp.jetson_clocks_on(fake_sysfs(306000000, 1020000000, 1728000)) is False)
check("7 jetson_clocks off: CPU minimum below its maximum", jp.jetson_clocks_on(fake_sysfs(1020000000, 1020000000, 729600)) is False)
check("7 jetson_clocks unknown without the nodes", jp.jetson_clocks_on(Path(tempfile.mkdtemp())) is None)
s = jp.sample(fake_sysfs(1020000000, 1020000000, 1728000))
check("7 sample: GPU clock in MHz and every thermal zone in C", s["gpu_mhz"] == 1020 and s["temp_c"] == {"cpu-thermal": 51.2, "gpu-thermal": 49.5}, s)

try:
    import studybench as sb  # needs numpy and cv2: run with the study venv's python
except ImportError as e:
    print(f"skip 8 (studybench not importable here: {e})")
    sb = None
if sb:
    samples = [[0, {"temp_c": {"cpu-thermal": 50.0, "gpu-thermal": 48.0}, "gpu_mhz": 1020, "oc_events": {"hwmon1/oc1_event_cnt": 3}}],
               [50, {"temp_c": {"cpu-thermal": 61.5, "gpu-thermal": 63.0}, "gpu_mhz": 918, "oc_events": {"hwmon1/oc1_event_cnt": 5}}],
               [99, {"temp_c": {"cpu-thermal": 58.0}, "gpu_mhz": 1020, "oc_events": {"hwmon1/oc1_event_cnt": 5}}]]
    rows = [{"gpu_mhz": "1020"}, {"gpu_mhz": "918"}, {"gpu_mhz": "918"}, {"gpu_mhz": ""}]
    j = sb.jetson_summary(samples, rows, {"gpu_clocks": {"max": 1020}})
    check("8 bench: an over-current counter that rose during the run is a throttle", j["over_current_throttled"] is True
          and j["oc_events_during_run"] == {"hwmon1/oc1_event_cnt": 2}, j)
    check("8 bench: images below the pinned GPU clock are counted", j["images_gpu_below_max"] == 2, j)
    check("8 bench: hottest reading per zone, a zone missing from one sample is fine", j["temp_c_max"] == {"cpu-thermal": 61.5, "gpu-thermal": 63.0}, j)
    j = sb.jetson_summary([[0, {"temp_c": {}, "gpu_mhz": None, "oc_events": {}}]], [], None)
    check("8 bench: no counters and no clocks give unknown, not 'no throttle'", j["over_current_throttled"] is None and j["images_gpu_below_max"] is None, j)
    check("8 bench: off a Jetson there is no summary", sb.jetson_summary([], [], None) is None)


def fake_engine_modules(shapes, deserialize_ok=True):
    """Fake torch and tensorrt that move real numpy data: the engine writes 2 x the first N input values to its output.
    The TensorRT names they offer are the ones checked against the real tensorrt 10.3.0 bindings."""
    import contextlib
    import json as _json
    from types import SimpleNamespace as NS

    import numpy as np

    log, blobs, buffers = [], [], {}

    class Tensor:
        def __init__(self, shape, dtype, kind):
            self.a, self.kind = np.zeros(shape, dtype), kind
            buffers[id(self)] = self

        def data_ptr(self):
            return id(self)

        def numpy(self):
            return self.a

        def copy_(self, src, non_blocking=False):
            np.copyto(self.a, src.a)
            log.append("h2d" if self.kind == "device" else "d2h")

    class Context:
        addr = {}

        def set_tensor_address(self, name, ptr):
            self.addr[name] = ptr

        def execute_async_v3(self, stream):
            src, dst = buffers[self.addr["images"]].a, buffers[self.addr["output0"]].a
            dst.flat[:] = 2 * src.flat[:dst.size]
            log.append("exec")
            return True

    class Engine:
        num_io_tensors, device_memory_size_v2 = 2, 1234

        def get_tensor_name(self, i):
            return ["output0", "images"][i]  # output first: the runner must find the input by mode, not position

        def get_tensor_mode(self, name):
            return "IN" if name == "images" else "OUT"

        def get_tensor_shape(self, name):
            return shapes[name]

        def get_tensor_dtype(self, name):
            return "FLOAT"

        def create_execution_context(self):
            return Context()

    class Runtime:
        def __init__(self, logger):
            pass

        def deserialize_cuda_engine(self, blob):
            blobs.append(blob)
            return Engine() if deserialize_ok else None

    logger = type("Logger", (), {"WARNING": 2, "__init__": lambda self, severity: None})
    trt = NS(__version__="10.3.0", Runtime=Runtime, Logger=logger, TensorIOMode=NS(INPUT="IN", OUTPUT="OUT"), nptype=lambda d: np.float32)
    torch = NS(__version__="fake", version=NS(cuda="12.6"), from_numpy=lambda a: NS(dtype=a.dtype),
               empty=lambda shape, dtype, pin_memory=False, device=None: Tensor(shape, dtype, "device" if device else "host"),
               cuda=NS(Stream=lambda: NS(cuda_stream=7, synchronize=lambda: log.append("sync")),
                       stream=lambda s: contextlib.nullcontext(), get_device_name=lambda i: "fake GPU",
                       get_device_capability=lambda i: (8, 7)))
    return {"tensorrt": trt, "torch": torch}, log, blobs, _json


if sb:
    import numpy as np

    mods, log, blobs, _json = fake_engine_modules({"images": (1, 3, 32, 32), "output0": (1, 14, 21)})
    saved = {k: sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    try:
        eng = Path(tempfile.mkdtemp()) / "model.engine"
        header = _json.dumps({"imgsz": [32, 32], "batch": 1, "end2end": False}).encode()
        eng.write_bytes(len(header).to_bytes(4, "little") + header + b"PLAN-BYTES")
        rt = sb.TensorRT(eng, 4)
        check("9 runner: the Ultralytics header is skipped, the plan alone is deserialized", blobs == [b"PLAN-BYTES"], blobs)
        check("9 runner: input shape and header read", rt.input_shape == [1, 3, 32, 32] and rt.info["engine_header"]["imgsz"] == [32, 32], rt.info)
        x = np.random.default_rng(0).random((1, 3, 32, 32), dtype=np.float32)
        log.clear()
        y = rt(x)
        check("9 runner: the input reaches the engine and its output comes back", y.shape == (1, 14, 21)
              and np.array_equal(y.ravel(), 2 * x.ravel()[:294]))
        check("9 runner: net is H2D, execute, D2H, then a stream sync", log == ["h2d", "exec", "d2h", "sync"], log)
        rt.close()
        mods, *_ = fake_engine_modules({"images": (-1, 3, 32, 32), "output0": (1, 14, 21)})
        sys.modules.update(mods)
        try:
            sb.TensorRT(eng, 4)
            check("9 runner: a dynamic-shape engine is refused", False)
        except SystemExit as e:
            check("9 runner: a dynamic-shape engine is refused", "dynamic" in str(e), str(e))
        mods, *_ = fake_engine_modules({"images": (1, 3, 32, 32), "output0": (1, 14, 21)}, deserialize_ok=False)
        sys.modules.update(mods)
        try:
            sb.TensorRT(eng, 4)
            check("9 runner: an engine from another GPU or TensorRT says to rebuild it here", False)
        except SystemExit as e:
            check("9 runner: an engine from another GPU or TensorRT says to rebuild it here", "studyexport.py" in str(e), str(e))
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

if FAILED:
    sys.exit(f"{len(FAILED)} failed: {FAILED}")
print("all passed")
