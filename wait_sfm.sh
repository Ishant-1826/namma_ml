#!/usr/bin/env bash
PROJECT_DIR="${1:-data/home}"

if [ -f "./venv/Scripts/python.exe" ]; then
    PYTHON_EXE="./venv/Scripts/python.exe"
elif [ -f "./venv/bin/python" ]; then
    PYTHON_EXE="./venv/bin/python"
else
    PYTHON_EXE="python"
fi

"$PYTHON_EXE" scripts/wait_sfm.py "$PROJECT_DIR"
