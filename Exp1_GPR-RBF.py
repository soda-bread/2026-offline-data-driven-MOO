"""Launcher for Exp1 GPR (RBF) experiment.

- Ensures key dependencies are installed.
- Then executes `Exp1_GPR_(RBF).py` with the same CLI arguments.
"""

import runpy
import subprocess
import sys
from pathlib import Path


REQUIRED_PACKAGES = ["pymoo", "GPy", "autogluon.tabular"]


def ensure_packages_installed() -> None:
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
            print(f"{pkg} is already installed.")
        except ImportError:
            print(f"{pkg} is not installed. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            print(f"{pkg} installed.")


def main() -> None:
    ensure_packages_installed()

    target_script = Path(__file__).with_name("Exp1_GPR_(RBF).py")
    if not target_script.exists():
        raise FileNotFoundError(f"Target script not found: {target_script}")

    runpy.run_path(str(target_script), run_name="__main__")


if __name__ == "__main__":
    main()
