#!/usr/bin/env python3
"""Hide console windows owned by worker descendants of one pilot launch."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from ctypes import wintypes
from pathlib import Path

import psutil


def hide_worker_windows(root: Path) -> tuple[int, bool]:
    launch = json.loads((root / "launch.json").read_text(encoding="utf-8"))
    workers: set[int] = set()
    running = False
    for row in launch["processes"]:
        try:
            process = psutil.Process(int(row["pid"]))
            if not process.is_running():
                continue
            running = True
            workers.update(child.pid for child in process.children(recursive=True)
                           if child.name().lower() in ("python.exe", "pythonw.exe"))
        except psutil.Error:
            continue
    if os.name != "nt" or not workers:
        return 0, running
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    user32.IsWindowVisible.argtypes = (wintypes.HWND,)
    user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
    hidden = 0
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def visit(hwnd, _):
        nonlocal hidden
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value in workers and user32.IsWindowVisible(hwnd):
            user32.ShowWindow(hwnd, 0)
            hidden += 1
        return True

    user32.EnumWindows(callback_type(visit), 0)
    return hidden, running


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    while True:
        hidden, running = hide_worker_windows(args.root)
        if hidden:
            print(f"hid {hidden} worker windows", flush=True)
        if not running:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
