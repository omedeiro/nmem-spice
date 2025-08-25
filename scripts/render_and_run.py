# scripts/render_and_run.py
from __future__ import annotations
import argparse, json, sys
from datetime import datetime
from pathlib import Path
import importlib.util
import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined

# ---------- Path anchors (no src/ layout) ----------
HERE = Path(__file__).resolve()          # .../scripts/render_and_run.py
ROOT = HERE.parents[1]                   # repo root
PKG_CORE = ROOT / "ltspice_core"         # preferred package dir
PKG_ALT  = ROOT / "ltspice"              # fallback package dir

# Ensure repo root is importable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _import_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
import subprocess

def to_windows_path(p: Path) -> str:
    """Return an absolute Windows/UNC path for a Linux/WSL path."""
    p = p.resolve()
    # Try wslpath first (best)
    try:
        out = subprocess.check_output(["wslpath", "-w", str(p)], text=True).strip()
        # Use forward slashes – LTspice is fine with them and they avoid escaping
        return out.replace("\\", "/")
    except Exception:
        s = str(p)
        # Map /mnt/c/... to C:/...
        if s.startswith("/mnt/") and len(s) > 6 and s[5].isalpha() and s[6] == "/":
            drive = s[5].upper()
            rest = s[7:]
            return f"{drive}:/{rest}".replace("\\", "/")
        # Fallback UNC to the WSL distro (adjust distro name if different)
        return f"//wsl.localhost/Ubuntu-22.04{s}".replace("\\", "/")

# ---------- Robust local imports with fallbacks ----------
runner_mod = None
gens_mod = None
try:
    from ltspice_core.runner import run_ltspice, find_ltspice_exe  # type: ignore
    from ltspice_core.generators import WaveformGenerator          # type: ignore
except Exception:
    try:
        from ltspice.runner import run_ltspice, find_ltspice_exe    # type: ignore
        from ltspice.generators import WaveformGenerator            # type: ignore
    except Exception:
        # Last resort: import straight from files
        if (PKG_CORE / "runner.py").exists():
            runner_mod = _import_from_file("local_runner", PKG_CORE / "runner.py")
            gens_mod   = _import_from_file("local_generators", PKG_CORE / "generators.py")
        elif (PKG_ALT / "runner.py").exists():
            runner_mod = _import_from_file("local_runner", PKG_ALT / "runner.py")
            gens_mod   = _import_from_file("local_generators", PKG_ALT / "generators.py")
        else:
            raise ModuleNotFoundError(
                "Couldn't import ltspice_core/ltspice. Expected package folder at "
                f"{PKG_CORE} or {PKG_ALT} with runner.py and generators.py"
            )
        run_ltspice = runner_mod.run_ltspice
        find_ltspice_exe = runner_mod.find_ltspice_exe
        WaveformGenerator = gens_mod.WaveformGenerator

# ---------------- helpers ----------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_pwl(path: Path, t_vals, i_vals) -> None:
    lines = [f"{float(t):.12g} {float(i):.12g}\n" for t, i in zip(t_vals, i_vals)]
    path.write_text("".join(lines))

def dedupe_and_sort(t: np.ndarray, i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if t.size == 0:
        return t, i
    idx = np.argsort(t, kind="mergesort")
    t, i = t[idx], i[idx]
    last = {}
    for tt, ii in zip(t, i):
        last[float(tt)] = float(ii)
    t_out = np.array(sorted(last.keys()), dtype=float)
    i_out = np.array([last[tt] for tt in t_out], dtype=float)
    return t_out, i_out

def pad_to_tstop(t: np.ndarray, i: np.ndarray, tstop: float, pad_value: float = 0.0):
    if t.size == 0 or t[-1] < tstop:
        t = np.append(t, tstop)
        i = np.append(i, pad_value)
    return t, i

def clamp_nonfinite(t: np.ndarray, i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(t) & np.isfinite(i)
    t = np.asarray(t)[mask].astype(float)
    i = np.asarray(i)[mask].astype(float)
    i[~np.isfinite(i)] = 0.0
    t[~np.isfinite(t)] = 0.0
    return t, i

def parse_meas_from_log(log_path: Path) -> dict:
    out = {}
    if not log_path.exists():
        return out
    for raw in log_path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or ":" not in line or line.startswith("*"):
            continue
        key, rest = line.split(":", 1)
        key = key.strip().replace(" ", "_")
        tok = rest.strip().split()
        if not tok:
            continue
        try:
            val = float(tok[0])
        except ValueError:
            continue
        out[key] = val
    return out

# ---------------- main ----------------
def main():
    DEFAULT_TEMPLATE = ROOT / "netlists" / "circuits" / "memory.cir.j2"
    DEFAULT_RUNS_DIR = ROOT / "runs"
    # From run folder runs/<stamp>-<tag>/ to models under netlists/models/
    DEFAULT_MODEL_REL_FROM_RUN = "../netlists/models/hTron_behavioral.lib"

    ap = argparse.ArgumentParser(
        description="Render memory.cir.j2, generate PWL sources, and run LTspice."
    )
    ap.add_argument("--tag", default="read", help="Run tag (used in runs/<stamp>-<tag>/)")
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE,
                    help="Path to Jinja2 template (default: netlists/circuits/memory.cir.j2)")
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR,
                    help="Base runs directory (default: runs)")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL_REL_FROM_RUN,
                    help="Model path as seen from the run directory (default: ../netlists/models/hTron_behavioral.lib)")

    # circuit params (strings preserve SPICE units)
    ap.add_argument("--R1", default="1n"); ap.add_argument("--R2", default="1n")
    ap.add_argument("--R3", default="50"); ap.add_argument("--R4", default="1G")
    ap.add_argument("--tstop", default="1e-6"); ap.add_argument("--tstep", default="1e-9")
    ap.add_argument("--reltol", default="1e-6")
    ap.add_argument("--HL_w", default="250n"); ap.add_argument("--HL_L", default="2.2u"); ap.add_argument("--HL_Rh", default="150")
    ap.add_argument("--HR_w", default="350n"); ap.add_argument("--HR_L", default="4u");   ap.add_argument("--HR_Rh", default="300")

    # waveform generator overrides (optional)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--phase-offset", type=float, default=0.0)
    ap.add_argument("--cycle-time", type=float, default=None)
    ap.add_argument("--pulse-sigma", type=float, default=None)
    ap.add_argument("--hold-width-write", type=float, default=None)
    ap.add_argument("--hold-width-read", type=float, default=None)
    ap.add_argument("--hold-width-clear", type=float, default=None)
    ap.add_argument("--write-amp", type=float, default=None)
    ap.add_argument("--read-amp", type=float, default=None)
    ap.add_argument("--enab-write-amp", type=float, default=None)
    ap.add_argument("--enab-read-amp", type=float, default=None)
    ap.add_argument("--clear-amp", type=float, default=None)

    args = ap.parse_args()

    # Prepare run directory
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ensure_dir(args.runs_dir / f"{stamp}-{args.tag}")
    data_dir = ensure_dir(run_dir / "data" / "waveforms")

    # Build generator with user overrides
    gen_kwargs = {}
    for name, val in [
        ("cycle_time", args.cycle_time),
        ("pulse_sigma", args.pulse_sigma),
        ("hold_width_write", args.hold_width_write),
        ("hold_width_read", args.hold_width_read),
        ("hold_width_clear", args.hold_width_clear),
        ("write_amplitude", args.write_amp),
        ("read_amplitude", args.read_amp),
        ("enab_write_amplitude", args.enab_write_amp),
        ("enab_read_amplitude", args.enab_read_amp),
        ("clear_amplitude", args.clear_amp),
    ]:
        if val is not None:
            gen_kwargs[name] = val
    generator = WaveformGenerator(**gen_kwargs) if gen_kwargs else WaveformGenerator()

    # Generate waveforms
    t_chan, i_chan, t_enab, i_enab, ops, enab_on = generator.generate_memory_protocol_sequence(
        seed=args.seed,
        phase_offset=args.phase_offset,
    )

    # Clean and pad to tstop
    tstop_f = float(args.tstop)
    def prepare(t, i):
        t, i = np.array(t, dtype=float), np.array(i, dtype=float)
        t, i = clamp_nonfinite(t, i)
        t, i = dedupe_and_sort(t, i)
        t, i = pad_to_tstop(t, i, tstop_f, pad_value=0.0)
        return t, i

    t_chan, i_chan = prepare(t_chan, i_chan)
    t_enab, i_enab = prepare(t_enab, i_enab)

    # Write PWL files relative to run_dir
    wave_enab = data_dir / "wave_enab.txt"
    wave_chan = data_dir / "wave_chan.txt"
    write_pwl(wave_enab, t_enab, i_enab)
    write_pwl(wave_chan, t_chan, i_chan)

    # Render Jinja2 template -> netlist.cir
    tpl_path = args.template if args.template.is_absolute() else (ROOT / args.template).resolve()
    env = Environment(
        loader=FileSystemLoader(tpl_path.parent),
        undefined=StrictUndefined,
        autoescape=False, trim_blocks=True, lstrip_blocks=True,
    )
    tpl = env.get_template(tpl_path.name)

    # Model include string as it should appear in .cir
    if Path(args.model).is_absolute():
        model_path_str = str(Path(args.model)).replace("\\", "/")
    else:
        model_path_str = args.model.replace("\\", "/")  # relative to run_dir

    params = {
        "tag": args.tag,
        "R1_val": args.R1, "R2_val": args.R2, "R3_val": args.R3, "R4_val": args.R4,
        "tstop": args.tstop, "tstep": args.tstep, "reltol": args.reltol,
        "HL": {"chan_width": args.HL_w, "chan_length": args.HL_L, "heater_resistance": args.HL_Rh},
        "HR": {"chan_width": args.HR_w, "chan_length": args.HR_L, "heater_resistance": args.HR_Rh},
        "wave_enab": to_windows_path(wave_enab),
        "wave_chan": to_windows_path(wave_chan),
        # If --model is relative, resolve it against repo root; then convert to Windows
        "model_path": to_windows_path((Path(args.model) if Path(args.model).is_absolute()
                               else (Path.cwd() / args.model))),
    }

    netlist_text = tpl.render(**params)
    netlist_path = run_dir / "netlist.cir"
    netlist_path.write_text(netlist_text)

    # Run LTspice
    print("LTspice:", find_ltspice_exe())
    code, out, err = run_ltspice(netlist=netlist_path, workdir=run_dir, ascii_raw=True)

    # Collect outputs
    log_path = run_dir / "netlist.log"
    raw_path = run_dir / "netlist.raw"
    meas = parse_meas_from_log(log_path)

    artifacts = {
        "params": params,
        "waveform_meta": {
            "ops": ops,
            "enab_on": enab_on.tolist(),
            "seed": args.seed,
            "phase_offset": args.phase_offset,
            **gen_kwargs,
        },
        "meas": meas,
        "artifacts": {
            "run_dir": str(run_dir.resolve()),
            "netlist": str(netlist_path.resolve()),
            "log": str(log_path.resolve()),
            "raw": str(raw_path.resolve()),
            "wave_enab": str(wave_enab.resolve()),
            "wave_chan": str(wave_chan.resolve()),
        },
    }
    (run_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2))

    print(json.dumps({"returncode": code, "meas": meas, "run_dir": str(run_dir)}, indent=2))
    if code != 0:
        print("LTspice stderr:\n", err, file=sys.stderr)
        sys.exit(code)

if __name__ == "__main__":
    main()
