# AGENTS.md — Standing Rules for Namma Space

All AI agents (and human contributors) working in this repo **must** follow these rules.

## Pipeline Conventions

1. **Every pipeline stage is an argparse CLI** living in `pipeline/`.
   - It reads inputs from `data/<scene>/` or `outputs/<scene>/` and writes outputs there.
   - Configuration is loaded from `configs/default.yaml` (override with `--config`).
   - Each stage prints a short, human-readable summary on success.
   - On failure, exit with **non-zero** status and an **actionable** error message
     (what went wrong + what the user should do).

2. **Scene coordinate convention** (final output):
   - Metric (units = metres).
   - Right-handed, **Y-up**, floor at `y = 0`.

## Repository Hygiene

3. **Never commit** `data/`, `outputs/`, or any file larger than 5 MB.
4. After completing any task, **update `docs/STATE.md`** with:
   - What is done.
   - Exact commands to reproduce / run.
   - Known issues or TODOs.

## Environment & Dependencies

5. Always work inside a **Python virtual environment** (`venv`).
   Tell the user before installing anything system-wide.
6. **Never guess CLI flags** — run `<tool> --help` to discover the correct syntax.
7. **Ask before adding heavy dependencies** not already listed in
   `docs/ARCHITECTURE.md` (e.g., new ML frameworks, large C++ libs).

## General

8. Favour **simple, robust** choices over clever ones — deadline is 31 Oct 2026.
9. Keep docs up-to-date; stale docs are worse than no docs.
10. When making scope or priority decisions, read `docs/COMPETITION.md`. Round 1 is judged on SOP and pipeline soundness, visual quality, viewer performance, and documentation and video clarity.
