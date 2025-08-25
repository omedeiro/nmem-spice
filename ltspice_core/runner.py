from __future__ import annotations
import os, platform, shutil, subprocess
from pathlib import Path
from typing import Tuple, Optional, List

# keep this as a LINUX path to the Windows exe (WSL can exec it)
DEFAULT_PATHS: List[str] = [
    "/mnt/c/Users/omedeiro/AppData/Local/Programs/ADI/LTspice/LTspice.exe",
]

def find_ltspice_exe() -> str:
    exe = os.environ.get("LTSPICE_EXE")
    if exe and Path(exe).exists():
        return exe
    for p in DEFAULT_PATHS:
        if Path(p).exists():
            return p
    found = shutil.which("LTspice.exe")
    if found:
        return found
    raise FileNotFoundError("LTspice.exe not found. Set LTSPICE_EXE or edit DEFAULT_PATHS.")

def _wsl_to_windows(p: str | Path) -> str:
    p = str(Path(p).resolve())
    try:
        out = subprocess.check_output(["wslpath", "-w", p], text=True).strip()
        return out.replace("\\", "/")  # forward slashes are fine
    except Exception:
        if p.startswith("/mnt/") and len(p) > 6 and p[5].isalpha() and p[6] == "/":
            drive = p[5].upper()
            rest = p[7:]
            return f"{drive}:/{rest}".replace("\\", "/")
        return f"//wsl.localhost/Ubuntu-22.04{p}".replace("\\", "/")

def run_ltspice(
    netlist: Path,
    workdir: Path,
    ascii_raw: bool = True,
    extra_args: Optional[list[str]] = None,
) -> Tuple[int, str, str]:
    """
    WSL→Windows safe:
      - exe: LINUX path (/mnt/c/...) so WSL can exec it
      - netlist arg: WINDOWS path (C:/... or //wsl.localhost/...)
      - cwd: LINUX path (your run dir)
    """
    exe = find_ltspice_exe()                       # /mnt/c/.../LTspice.exe
    nl_win = _wsl_to_windows(netlist)              # e.g. C:/.../netlist.cir or //wsl...
    args = ["-b", "-Run"]
    if ascii_raw:
        args.append("-ascii")
    if extra_args:
        args.extend(extra_args)

    cmd = [exe] + args + [nl_win]
    proc = subprocess.run(
        cmd,
        cwd=str(workdir),          # keep as Linux dir; don't convert and don't use cmd.exe
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr
