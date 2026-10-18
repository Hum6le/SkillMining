"""Prevent Windows idle sleep only while one verified experiment is alive."""
import argparse
import ctypes
import os
from pathlib import Path
import time

import psutil


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--script", required=True)
    args = parser.parse_args()
    process = psutil.Process(args.pid)
    if not any(Path(arg).name == args.script for arg in process.cmdline()):
        raise ValueError("Target process is not the expected experiment")
    created = process.create_time()
    if os.name == "nt":
        if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001):
            raise RuntimeError("Could not request temporary Windows system wake lock")
    try:
        print(f"Temporary idle-sleep prevention active for experiment PID {args.pid}", flush=True)
        while psutil.pid_exists(args.pid):
            try:
                if psutil.Process(args.pid).create_time() != created:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(10)
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        print("Experiment ended; idle-sleep prevention released", flush=True)
