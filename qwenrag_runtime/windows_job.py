"""Minimal Windows Job Object wrapper used to own launcher-created processes."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from typing import Any


class WindowsJobError(RuntimeError):
    """Raised when Windows cannot create or configure the process job."""


class WindowsJob:
    """Kill all assigned processes when the final job handle is closed."""

    def __init__(self) -> None:
        self._handle: int | None = None
        if os.name != "nt":
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = (
            wintypes.HANDLE, wintypes.INT, wintypes.LPVOID, wintypes.DWORD,
        )
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise WindowsJobError("无法创建 Windows 进程作业")
        self._handle = int(handle)
        information = _JobExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            wintypes.HANDLE(self._handle), 9, ctypes.byref(information), ctypes.sizeof(information)
        )
        if not ok:
            self.close()
            raise WindowsJobError("无法配置 Windows 进程作业")

    def assign_process(self, process: Any) -> None:
        if self._handle is None:
            return
        process_handle = getattr(process, "_handle", None)
        if not process_handle:
            raise WindowsJobError("无法取得子进程 Windows 句柄")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        if not kernel32.AssignProcessToJobObject(
            wintypes.HANDLE(self._handle), wintypes.HANDLE(process_handle)
        ):
            raise WindowsJobError("无法将子进程加入 Windows 进程作业")

    def close(self) -> None:
        if self._handle is None:
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle(wintypes.HANDLE(self._handle))
        self._handle = None

    def __enter__(self) -> "WindowsJob":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


if os.name == "nt":
    ULONG_PTR = ctypes.c_size_t

    class _IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ULONG_PTR),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _JobExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]
else:
    class _JobExtendedLimitInformation:  # pragma: no cover - never instantiated off Windows
        pass
