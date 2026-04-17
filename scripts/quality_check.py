"""Local quality gate for the repository.

This script is intentionally local-only. It can be used in a dev loop before
running the pipeline or committing changes.

It enforces:
- CodeSense score >= 9/10 (if configured via env)
- Ruff linting (if installed)
- Pytest
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _Cmd:
    name: str
    argv: list[str]


def _run(cmd: _Cmd) -> subprocess.CompletedProcess[str]:
    """Run a command and return the completed process."""

    proc = subprocess.run(cmd.argv, check=False, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        print(f"FAILED: {cmd.name} (exit={proc.returncode})", file=sys.stderr)
    return proc


def _codesense_cmd() -> _Cmd | None:
    """Return a configured CodeSense command or None if not configured."""

    raw = os.getenv("CODESENSE_CMD", "").strip()
    if not raw:
        return None
    return _Cmd(name="codesense", argv=raw.split())


def _min_codesense_score() -> float:
    """Return minimum acceptable CodeSense score (default 9.0)."""

    raw = os.getenv("CODESENSE_MIN_SCORE", "9").strip()
    try:
        return float(raw)
    except ValueError:
        return 9.0


def _extract_score(text: str) -> float | None:
    """Extract a score like 9.2 or 9/10 from output text."""

    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", text)
    if m:
        return float(m.group(1))
    m2 = re.search(r"\bscore\b[^0-9]*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m2:
        return float(m2.group(1))
    return None


def _run_codesense_gate(cmd: _Cmd) -> int:
    """Run CodeSense and enforce the minimum score threshold."""

    proc = _run(cmd)
    if proc.returncode != 0:
        return int(proc.returncode)
    score = _extract_score((proc.stdout or "") + "\n" + (proc.stderr or ""))
    if score is None:
        print(
            "FAILED: codesense (could not parse score). "
            "Set CODESENSE_CMD to a command that prints a score like '9.1/10'.",
            file=sys.stderr,
        )
        return 2
    minimum = _min_codesense_score()
    if score < minimum:
        print(f"FAILED: codesense score {score:.2f} < minimum {minimum:.2f}", file=sys.stderr)
        return 3
    print(f"OK: codesense score {score:.2f}/10 (min {minimum:.2f})")
    return 0


def main() -> int:
    """Run the local quality gate.

    Returns:
        Exit code (0 when all checks pass).
    """

    print(f"Python: {sys.executable}")
    print("Tip: ensure this interpreter has repo dependencies installed: `pip install -r requirements.txt`.\n")

    cmds: list[_Cmd] = []
    cs = _codesense_cmd()
    if cs is not None:
        rc = _run_codesense_gate(cs)
        if rc != 0:
            return rc
    cmds.append(_Cmd(name="ruff", argv=[sys.executable, "-m", "ruff", "check", "src", "scripts", "tests"]))
    cmds.append(_Cmd(name="pytest", argv=[sys.executable, "-m", "pytest"]))

    for cmd in cmds:
        proc = _run(cmd)
        if proc.returncode != 0:
            return int(proc.returncode)
    print("OK: quality gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

