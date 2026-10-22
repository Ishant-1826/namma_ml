#!/usr/bin/env python3
"""Check the local environment for Namma Space prerequisites.

Reports: Python version, ffmpeg, COLMAP (need >=4.0 with global_mapper),
CUDA / GPU, Node.js / npm -- and what is missing.
"""

import shutil
import subprocess
import sys
import os
import re
from pathlib import Path

# On Windows, refresh PATH from registry so newly installed tools are found without restarting shell
if sys.platform == "win32":
    try:
        import winreg
        def _get_reg_path(key, subkey):
            try:
                with winreg.OpenKey(key, subkey) as k:
                    val, _ = winreg.QueryValueEx(k, "Path")
                    return val
            except Exception:
                return ""
        _m = _get_reg_path(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")
        _u = _get_reg_path(winreg.HKEY_CURRENT_USER, r"Environment")
        os.environ["PATH"] = f"{_m};{_u};{os.environ.get('PATH', '')}"
    except Exception:
        pass


def _run(cmd: list[str], timeout: int = 10) -> tuple[int, str]:
    """Run a command and return (returncode, combined stdout+stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, (result.stdout + result.stderr).strip()
    except FileNotFoundError:
        return -1, ""
    except subprocess.TimeoutExpired:
        return -2, "(timed out)"
    except Exception as e:
        return -3, str(e)


def check_python() -> bool:
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    ok = v >= (3, 10)
    status = "OK" if ok else "WARN (>=3.10 recommended)"
    print(f"  Python         : {ver_str}  [{status}]")
    return ok


def check_ffmpeg() -> bool:
    path = shutil.which("ffmpeg")
    if not path:
        print("  FFmpeg         : NOT FOUND  [X]")
        return False
    rc, out = _run(["ffmpeg", "-version"])
    first_line = out.split("\n")[0] if out else "unknown"
    print(f"  FFmpeg         : {first_line}  [OK]")
    return True


def check_colmap() -> bool:
    path = shutil.which("colmap")
    if not path:
        print("  COLMAP         : NOT FOUND  [X]")
        print("                   Need COLMAP >= 4.0 with 'global_mapper'.")
        return False

    # Get version
    rc, out = _run(["colmap", "version"])
    if rc != 0:
        # Some builds use --help or just print version on any unknown command
        rc, out = _run(["colmap", "--help"])
    version_str = "unknown"
    # Try to parse version like "COLMAP 3.8" or "4.0"
    m = re.search(r"(\d+\.\d+(\.\d+)?)", out)
    if m:
        version_str = m.group(1)

    # Check for global_mapper
    rc_gm, out_gm = _run(["colmap", "global_mapper", "--help"])
    has_gm = rc_gm == 0 or "global_mapper" in out_gm.lower()

    ok = True
    notes = []
    try:
        major = int(version_str.split(".")[0])
        if major < 4:
            notes.append(f"WARN: version {version_str} < 4.0")
            ok = False
    except ValueError:
        notes.append("could not parse version")

    if not has_gm:
        notes.append("'global_mapper' command NOT found")
        ok = False

    suffix = f"  [{'; '.join(notes)}]" if notes else "  [OK]"
    print(f"  COLMAP         : {version_str}{suffix}")
    return ok


def check_cuda_gpu() -> bool:
    found_any = False

    # nvidia-smi
    path = shutil.which("nvidia-smi")
    if path:
        rc, out = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                         "--format=csv,noheader,nounits"])
        if rc == 0 and out:
            for line in out.strip().split("\n"):
                print(f"  GPU            : {line.strip()}  [OK]")
            found_any = True
        else:
            print("  GPU            : nvidia-smi found but query failed")
    else:
        print("  nvidia-smi     : NOT FOUND")

    # nvcc (CUDA compiler)
    nvcc = shutil.which("nvcc")
    if nvcc:
        rc, out = _run(["nvcc", "--version"])
        m = re.search(r"release (\S+)", out)
        cuda_ver = m.group(1) if m else "unknown"
        print(f"  CUDA (nvcc)    : {cuda_ver}  [OK]")
        found_any = True
    else:
        print("  CUDA (nvcc)    : NOT FOUND  [X]")

    if not found_any:
        print("  !!  No GPU/CUDA detected -- splat training requires a GPU.")

    return found_any


def check_node_npm() -> bool:
    ok = True
    # Node
    node = shutil.which("node")
    if node:
        rc, out = _run(["node", "--version"])
        print(f"  Node.js        : {out}  [OK]")
    else:
        print("  Node.js        : NOT FOUND  [X]")
        ok = False

    # npm
    npm = shutil.which("npm")
    if npm:
        rc, out = _run(["npm", "--version"])
        print(f"  npm            : {out}  [OK]")
    else:
        print("  npm            : NOT FOUND  [X]")
        ok = False

    return ok


def main() -> int:
    print("=" * 60)
    print("  Namma Space -- Environment Check")
    print("=" * 60)
    print()

    results: dict[str, bool] = {}
    results["Python"] = check_python()
    results["FFmpeg"] = check_ffmpeg()
    results["COLMAP"] = check_colmap()
    results["CUDA/GPU"] = check_cuda_gpu()
    results["Node/npm"] = check_node_npm()

    print()
    missing = [k for k, v in results.items() if not v]
    if missing:
        print("!!  Missing / problematic:")
        for m in missing:
            print(f"   - {m}")
        print()
        print("Install the missing tools before running the pipeline.")
        return 1
    else:
        print("[OK]  All prerequisites found.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
