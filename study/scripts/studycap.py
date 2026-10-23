"""Run a heavy laptop step in a child process under a memory cap and a timeout, so it can never page the laptop (#20).

The laptop has 15 GB of RAM and often only 2-4 GB free. An uncapped job once grew to 25.8 GB and paged it for 3.5 h:
on Windows a Python MemoryError arrives only at the system commit limit, after the page file has grown, so nothing
inside the job fails fast. So on Windows the child starts suspended, joins a Job Object whose memory limit covers it
and every process it starts, and only then runs. The kernel refuses any commit past the cap and tells the parent,
which stops the whole job at once. The parent also stops it at the timeout, and when the laptop's free RAM falls below
a floor (1.5 GB; STUDY_RAM_FLOOR_GB), because a cap above what is free would still page. INVARIANTS.md section 1.4
holds the rule.

  code = studycap.run([script, *args], "scoring run X")   # the child's exit code; raises Stopped when it was stopped

On every other OS (the Pis, the Jetson, Kaggle) run() starts the same child with no cap and no timeout.
Standard library only, so it runs on every platform.
"""

import ctypes
import os
import subprocess
import sys
import time

GB = 1 << 30
WINDOWS = sys.platform == "win32"
CAP_GB, TIMEOUT_MIN = 4, 60  # INT8 export at 960 and 1280 needs 4-5 GB: raise the cap for those, with the laptop idle
FLOOR_GB = 1.5  # the step is stopped when the laptop's free RAM falls below this, whatever its own size
_INSIDE = "STUDY_CAPPED_STEP"  # set in a capped step's environment


class Stopped(SystemExit):
    """The capped step was stopped. Uncaught, it exits 1 with its message, like any other refusal in these scripts."""


if WINDOWS:
    import ctypes.wintypes as wt

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]

    class _BASIC_LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", wt.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]

    class _EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BASIC_LIMIT), ("IoInfo", _IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class _COMPLETION_PORT(ctypes.Structure):
        _fields_ = [("CompletionKey", ctypes.c_void_p), ("CompletionPort", wt.HANDLE)]

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _ntdll = ctypes.WinDLL("ntdll")
    for _name, _res, _args in [
        ("CreateJobObjectW", wt.HANDLE, [ctypes.c_void_p, wt.LPCWSTR]),
        ("SetInformationJobObject", wt.BOOL, [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]),
        ("CreateIoCompletionPort", wt.HANDLE, [wt.HANDLE, wt.HANDLE, ctypes.c_size_t, wt.DWORD]),
        ("GetQueuedCompletionStatus", wt.BOOL, [wt.HANDLE, ctypes.POINTER(wt.DWORD), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_void_p), wt.DWORD]),
        ("OpenProcess", wt.HANDLE, [wt.DWORD, wt.BOOL, wt.DWORD]),
        ("AssignProcessToJobObject", wt.BOOL, [wt.HANDLE, wt.HANDLE]),
        ("TerminateJobObject", wt.BOOL, [wt.HANDLE, wt.UINT]),
        ("CloseHandle", wt.BOOL, [wt.HANDLE]),
        ("GlobalMemoryStatusEx", wt.BOOL, [ctypes.c_void_p]),
    ]:
        getattr(_k32, _name).restype, getattr(_k32, _name).argtypes = _res, _args
    _ntdll.NtResumeProcess.restype, _ntdll.NtResumeProcess.argtypes = ctypes.c_long, [wt.HANDLE]  # an NTSTATUS, 0 on success

    _CREATE_SUSPENDED = 0x4
    _PROCESS_ALL_ACCESS = 0x1FFFFF
    _JOB_OBJECT_LIMIT_JOB_MEMORY = 0x200
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000  # the parent's death takes the job with it
    _JOB_OBJECT_MSG_JOB_MEMORY_LIMIT = 10
    _EXTENDED_LIMIT_INFORMATION, _ASSOCIATE_COMPLETION_PORT = 9, 7

    def _check(ok, what):
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error(), what)
        return ok

    def _job(cap_gb):
        """A job whose processes together cannot commit past cap_gb, and the completion port it reports the cap to."""
        job = _check(_k32.CreateJobObjectW(None, None), "CreateJobObjectW")
        info = _EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_JOB_MEMORY | _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        info.JobMemoryLimit = int(cap_gb * GB)
        _check(_k32.SetInformationJobObject(job, _EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)), "SetInformationJobObject")
        port = _check(_k32.CreateIoCompletionPort(wt.HANDLE(-1), None, 0, 1), "CreateIoCompletionPort")
        assoc = _COMPLETION_PORT(None, port)
        _check(_k32.SetInformationJobObject(job, _ASSOCIATE_COMPLETION_PORT, ctypes.byref(assoc), ctypes.sizeof(assoc)), "SetInformationJobObject")
        return job, port

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD), *[(n, ctypes.c_ulonglong) for n in (
            "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]]

    def _free_ram():
        """The laptop's available physical RAM, in bytes."""
        s = _MEMORYSTATUSEX()
        s.dwLength = ctypes.sizeof(s)
        _check(_k32.GlobalMemoryStatusEx(ctypes.byref(s)), "GlobalMemoryStatusEx")
        return s.ullAvailPhys

    def _hit_cap(port):
        """Drain the job's messages; True when one says a commit went past the cap."""
        msg, key, ov, hit = wt.DWORD(), ctypes.c_size_t(), ctypes.c_void_p(), False
        while _k32.GetQueuedCompletionStatus(port, ctypes.byref(msg), ctypes.byref(key), ctypes.byref(ov), 0):
            hit |= msg.value == _JOB_OBJECT_MSG_JOB_MEMORY_LIMIT
        return hit


def applies():
    """True when run() caps: on Windows, outside a capped step. Inside one, the outer job already holds every process."""
    return WINDOWS and not os.environ.get(_INSIDE)


def limits():
    """(cap in GB, timeout in minutes): STUDY_MEM_CAP_GB and STUDY_TIMEOUT_MIN, else 4 GB and 60 min."""
    return float(os.environ.get("STUDY_MEM_CAP_GB", CAP_GB)), float(os.environ.get("STUDY_TIMEOUT_MIN", TIMEOUT_MIN))


def add_arguments(parser):
    parser.add_argument("--mem-cap-gb", type=float, help=f"memory cap of each heavy step on Windows (default {CAP_GB} GB; STUDY_MEM_CAP_GB)")
    parser.add_argument("--timeout-min", type=float, help=f"timeout of each heavy step on Windows (default {TIMEOUT_MIN} min; STUDY_TIMEOUT_MIN)")


def use(args):
    """Apply the flags add_arguments() added. They go into the environment, so the steps' own children see them too."""
    if args.mem_cap_gb is not None:
        os.environ["STUDY_MEM_CAP_GB"] = str(args.mem_cap_gb)
    if args.timeout_min is not None:
        os.environ["STUDY_TIMEOUT_MIN"] = str(args.timeout_min)


def run(argv, step, cap_gb=None, timeout_min=None):
    """Run `python argv...`, its output going where this process's goes. Returns its exit code; raises Stopped when the
    cap stopped it. The limits default to limits()."""
    cmd = [sys.executable, *argv]
    cap_gb = limits()[0] if cap_gb is None else cap_gb
    timeout_min = limits()[1] if timeout_min is None else timeout_min
    sys.stdout.flush()  # the child writes to the same file: what this process printed first comes first, even into a pipe
    sys.stderr.flush()
    if not applies():
        return subprocess.run(cmd).returncode
    job, port = _job(cap_gb)
    try:
        child = subprocess.Popen(cmd, creationflags=_CREATE_SUSPENDED, env={**os.environ, _INSIDE: "1"})
        try:
            handle = _check(_k32.OpenProcess(_PROCESS_ALL_ACCESS, False, child.pid), "OpenProcess")
            try:
                _check(_k32.AssignProcessToJobObject(job, handle), "AssignProcessToJobObject")
                status = _ntdll.NtResumeProcess(handle)
                if status:
                    raise OSError(f"NtResumeProcess failed: NTSTATUS {status & 0xFFFFFFFF:#010x}")
            finally:
                _k32.CloseHandle(handle)
        except BaseException:  # not in the job yet, or never resumed: closing the job would not take it
            child.kill()
            child.wait()
            raise
        floor_gb = float(os.environ.get("STUDY_RAM_FLOOR_GB", FLOOR_GB))
        t0 = time.monotonic()
        while True:
            done = child.poll() is not None  # read before the messages, so a cap hit just before exiting is seen
            reason = None
            if _hit_cap(port):  # the refused allocation may have killed the child already: still the cap's doing
                reason = f"it went over its {cap_gb:g} GB memory cap"
            elif done:
                return child.returncode
            elif time.monotonic() - t0 > timeout_min * 60:
                reason = f"it ran past its {timeout_min:g} min timeout"
            elif _free_ram() < floor_gb * GB:
                reason = (f"the laptop's free RAM fell below {floor_gb:g} GB, where it starts to page (close other programs, "
                          "or lower the floor with STUDY_RAM_FLOOR_GB)")
            if reason:
                _k32.TerminateJobObject(job, 1)
                child.wait()
                raise Stopped(f"{step} was stopped: {reason}")
            time.sleep(0.25)
    finally:
        _k32.CloseHandle(port)
        _k32.CloseHandle(job)  # kills whatever the job still holds
