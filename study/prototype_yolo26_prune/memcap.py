"""PROTOTYPE helper - throwaway. Hard memory and time limits for prototype stages, so a runaway stage dies alone
instead of paging the whole laptop (see MEMORY-INCIDENT-2026-10-19.md).

Two layers, because on Windows a Python MemoryError only arrives once the system commit limit is hit, i.e. after the
page file has grown and the machine has thrashed for hours:

  cap_this_process(gb)   child side: puts the process in a Windows Job Object with JOB_OBJECT_LIMIT_PROCESS_MEMORY,
                         so the kernel refuses any commit past `gb` and the allocation fails at once. Inherited by any
                         process the child starts (pnnx during NCNN export).
  run_capped(...)        parent side: launches the child, polls its private memory and the system's available RAM,
                         and kills the process tree on timeout, on the cap, or when available RAM falls below a floor.
"""

import ctypes
import ctypes.wintypes as wt
import os
import subprocess
import sys
import time

import psutil

GB = 1 << 30
_job = None  # keep the handle alive for the life of the process


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]


class _BASIC_LIMIT(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", wt.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]


class _EXTENDED_LIMIT(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BASIC_LIMIT), ("IoInfo", _IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


_JobObjectExtendedLimitInformation = 9
_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x100


def cap_this_process(gb):
    """Kernel-enforced cap on this process's committed memory. Allocations past it fail immediately."""
    global _job
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wt.HANDLE
    k32.GetCurrentProcess.restype = wt.HANDLE
    job = k32.CreateJobObjectW(None, None)
    info = _EXTENDED_LIMIT()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.ProcessMemoryLimit = int(gb * GB)
    if not k32.SetInformationJobObject(wt.HANDLE(job), _JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)):
        raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
    if not k32.AssignProcessToJobObject(wt.HANDLE(job), wt.HANDLE(k32.GetCurrentProcess())):
        raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
    _job = job


def _tree_private(proc):
    total = 0
    for p in [proc, *proc.children(recursive=True)]:
        try:
            total += p.memory_info().private
        except psutil.Error:
            pass
    return total


def _kill_tree(proc):
    for p in [*proc.children(recursive=True), proc]:
        try:
            p.kill()
        except psutil.Error:
            pass


def run_capped(argv, cap_gb, timeout_s, log_path, floor_gb=1.5, env=None):
    """Run `python -u argv...` with the child capping itself at cap_gb (via env PROTO_CAP_GB).

    Returns {"exit": code, "killed": None | reason, "peak_private_gb", "min_available_gb", "seconds"}.
    """
    env = {**os.environ, **(env or {}), "PROTO_CAP_GB": str(cap_gb), "PYTHONIOENCODING": "utf-8"}
    t0, peak, min_avail, killed = time.time(), 0, float("inf"), None
    with open(log_path, "a", encoding="utf-8") as log:
        child = subprocess.Popen([sys.executable, "-u", *argv], stdout=log, stderr=subprocess.STDOUT, env=env)
        proc = psutil.Process(child.pid)
        while child.poll() is None:
            mem, avail = _tree_private(proc), psutil.virtual_memory().available
            peak, min_avail = max(peak, mem), min(min_avail, avail)
            if time.time() - t0 > timeout_s:
                killed = f"timed out after {timeout_s / 60:.0f} min"
            elif mem > cap_gb * GB * 1.05:  # backstop: the job object should have stopped it already
                killed = f"exceeded {cap_gb} GB private memory"
            elif avail < floor_gb * GB:
                killed = f"system available RAM fell below {floor_gb} GB (child at {mem / GB:.1f} GB)"
            if killed:
                _kill_tree(proc)
                child.wait()
                break
            time.sleep(0.5)
    return {"exit": child.returncode, "killed": killed, "peak_private_gb": round(peak / GB, 2),
            "min_available_gb": round(min_avail / GB, 2), "seconds": round(time.time() - t0, 1)}


def cap_from_env():
    """Child entry: apply the cap the parent asked for, if any."""
    gb = os.environ.get("PROTO_CAP_GB")
    if gb:
        cap_this_process(float(gb))
