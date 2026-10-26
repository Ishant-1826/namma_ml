#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def main():
    Path("logs").mkdir(exist_ok=True)
    log_file = Path("logs") / "sfm_home2.log"
    cmd = [
        sys.executable,
        "pipeline/run_sfm.py",
        "--scene",
        "home2",
        "--multiple-models",
        "0",
        "--resume"
    ]
    with open(log_file, "w", encoding="utf-8") as out:
        p = subprocess.Popen(
            cmd,
            stdout=out,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )
    print(f"Detached SfM process started with PID: {p.pid}")
    print(f"Logging output to: {log_file.resolve()}")

if __name__ == "__main__":
    main()
