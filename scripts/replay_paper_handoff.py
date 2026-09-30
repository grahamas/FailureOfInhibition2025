"""Reproduce the dated paper-handoff checks and figure without a study archive."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/paper_handoff_replay"),
                        help="new output directory (existing paths are refused)")
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error("run without -O or PYTHONOPTIMIZE; scientific assertions are required")
    bundle = Path(__file__).resolve().parents[1] / "reproducibility/paper_handoff_20260930"
    manifest = json.loads((bundle / "inputs.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != expected:
            parser.error(f"bundled input hash mismatch: {name}")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"output already exists; choose a new directory: {output}")
    output.mkdir(parents=True)
    for name in (*manifest["files"], "inputs.json", "requirements.txt"):
        shutil.copyfile(bundle / name, output / name)
    environment = dict(os.environ, MPLCONFIGDIR=str(output / ".matplotlib"))
    for script in ("confirm_selective_contexts.py", "render_confirmed_withdrawal.py"):
        subprocess.run([sys.executable, str(output / script)], check=True, env=environment)
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(output.iterdir()) if p.is_file()}
    (output / "checksums.json").write_text(json.dumps(
        {"algorithm": "SHA-256", "files": checksums}, indent=2) + "\n")
    print(f"Report: {output / 'confirmed_withdrawal.html'}", flush=True)


if __name__ == "__main__":
    main()
