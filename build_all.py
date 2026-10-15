# -*- coding: utf-8 -*-
"""Build both independent executables when packaging is requested later."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPECS = (
    ROOT / "build" / "tool1_alcohol_vs_smoke.spec",
    ROOT / "build" / "tool2_alcohol_vs_smoke_odor.spec",
)


def main():
    for spec in SPECS:
        subprocess.run(
            [sys.executable, "-m", "PyInstaller", "--clean", str(spec)],
            cwd=str(ROOT),
            check=True,
        )


if __name__ == "__main__":
    main()
