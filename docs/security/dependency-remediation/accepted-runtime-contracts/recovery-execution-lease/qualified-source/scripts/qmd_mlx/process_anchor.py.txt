"""Non-consuming child-exit observation for the owned process-group anchor.

Use Python's waitid(WNOWAIT) where exposed. CPython does not expose waitid on
macOS, so bind the public Darwin libc ABI (sys/signal.h and sys/wait.h). Other
platforms without this primitive are refused before a target is spawned.
"""

from __future__ import annotations

import ctypes
import errno
import os
import signal
import sys
from typing import Any, Optional


class _DarwinSigval(ctypes.Union):
    _fields_ = [("sival_int", ctypes.c_int), ("sival_ptr", ctypes.c_void_p)]


class _DarwinSiginfo(ctypes.Structure):
    _fields_ = [
        ("si_signo", ctypes.c_int), ("si_errno", ctypes.c_int),
        ("si_code", ctypes.c_int), ("si_pid", ctypes.c_int),
        ("si_uid", ctypes.c_uint), ("si_status", ctypes.c_int),
        ("si_addr", ctypes.c_void_p), ("si_value", _DarwinSigval),
        ("si_band", ctypes.c_long), ("__pad", ctypes.c_ulong * 7),
    ]


_darwin_libc = None
_darwin_waitid: Optional[Any] = None


def require_waitid_support() -> None:
    """Resolve the primitive before launch; never fall back to a PID lookup."""
    if os.name != "posix" or any(not hasattr(os, flag) for flag in (
        "P_PID", "WEXITED", "WNOHANG", "WNOWAIT",
    )):
        raise RuntimeError("Owned process supervision requires waitid(WNOWAIT)")
    if callable(getattr(os, "waitid", None)):
        return
    if sys.platform != "darwin":
        raise RuntimeError("Owned process supervision requires waitid(WNOWAIT)")
    global _darwin_libc, _darwin_waitid
    if _darwin_waitid is None:
        _darwin_libc = ctypes.CDLL(None, use_errno=True)
        try:
            function = _darwin_libc.waitid
        except AttributeError as error:
            raise RuntimeError("Darwin libc waitid(WNOWAIT) is unavailable") from error
        function.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.POINTER(_DarwinSiginfo), ctypes.c_int]
        function.restype = ctypes.c_int
        _darwin_waitid = function


def wait_child_without_reaping(pid: int) -> Optional[int]:
    """Return an exit code or None, keeping the exact child's PID reserved.

    ECHILD is intentionally not suppressed: it means the ownership anchor was
    reaped outside its handle, not that a numerical PID is safe to signal.
    """
    flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
    native_waitid: Any = getattr(os, "waitid", None)
    if callable(native_waitid):
        info = native_waitid(os.P_PID, pid, flags)
        if info is None:
            return None
        observed_pid, code, status, signo = info.si_pid, info.si_code, info.si_status, info.si_signo
        exited, killed, dumped = os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED
    else:
        require_waitid_support()
        function = _darwin_waitid
        assert function is not None
        info = _DarwinSiginfo()
        while function(os.P_PID, pid, ctypes.byref(info), flags) != 0:
            error = ctypes.get_errno()
            if error == errno.EINTR:
                continue
            raise OSError(error, os.strerror(error))
        observed_pid, code, status, signo = info.si_pid, info.si_code, info.si_status, info.si_signo
        # Darwin sys/signal.h CLD_* constants (not exported by all CPythons).
        exited, killed, dumped = 1, 2, 3
    if observed_pid == 0:
        return None
    if observed_pid != pid or signo != signal.SIGCHLD:
        raise RuntimeError("waitid returned a different child identity")
    if code == exited:
        return status
    if code in (killed, dumped):
        return -status
    raise RuntimeError(f"Unexpected non-exit waitid event: {code}")
