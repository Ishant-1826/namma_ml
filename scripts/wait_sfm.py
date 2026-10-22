#!/usr/bin/env python3
"""Monitor SfM progress by polling status.json every 30 seconds until done."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path


def find_status_file(project_dir: Path) -> Path:
    candidates = [
        project_dir / "logs" / "status.json",
        project_dir / "colmap" / "logs" / "status.json",
        project_dir / "status.json",
        Path("logs") / "status.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


def main() -> int:
    proj_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/home")
    status_file = find_status_file(proj_dir)
    print(f"Monitoring SfM status for '{proj_dir}' (status file: {status_file})...", flush=True)

    while True:
        cur_file = find_status_file(proj_dir)
        if cur_file.is_file():
            try:
                with open(cur_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                stage = data.get("stage", "unknown")
                progress = data.get("progress", "")
                done = bool(data.get("done", False))
                error = data.get("error")

                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{ts}] Stage: {stage:<18} | {progress}", flush=True)

                if done:
                    if error or stage == "failed":
                        print(f"[{ts}] SfM run FAILED: {error or progress}", flush=True)
                        return 1
                    else:
                        print(f"[{ts}] SfM run FINISHED successfully.", flush=True)
                        return 0
            except Exception:
                # Concurrent write or partial read
                pass
        else:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] Waiting for status.json to be created...", flush=True)

        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
