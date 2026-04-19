# 2026-offline-data-driven-MOO

This repository contains an offline, data-driven **multi-objective optimization (MOO)** workflow built on **NSGA-II** with multiple surrogate model options:

- **GPR (RBF)**
- **GPR (Matern)**
- **AutoGluon Quantile Regression (QR)**
- **BNN Ensemble** (ensemble neural regressors for predictive uncertainty)

---

## Notebooks

Current experiment and test notebooks in this repository:

- `Exp1 GPR (RBF).ipynb`
- `Exp2 GPR (Matern).ipynb`
- `Exp3 Autogluon_QR.ipynb`
- `Exp4 BNN.ipynb`
- `[Test] Autogluon_QR.ipynb`
- `[Test] BNN.ipynb`

All notebooks follow a similar flow:
1. Install/check dependencies.
2. Import reusable functionality from `src/`.
3. Generate data and train surrogates.
4. Evaluate uncertainty diagnostics (coverage / z-score / alpha search).
5. Run NSGA-II optimization (`run_experiment`) with standard and dual-ranking survival.

---

## Project Structure

```text
.
├── Exp1 GPR (RBF).ipynb
├── Exp2 GPR (Matern).ipynb
├── Exp3 Autogluon_QR.ipynb
├── Exp4 BNN.ipynb
├── [Test] Autogluon_QR.ipynb
├── [Test] BNN.ipynb
├── README.md
└── src
    ├── data.py            # Train/validation/test data generation
    ├── experiment.py      # NSGA-II experiment loop and metric collection
    ├── metrics.py         # HV / IGD+ setup
    ├── models.py          # GPR / AutoGluon-QR / BNN model wrappers and helpers
    ├── opt_problem.py     # pymoo Problem wrapper (GPR/QR/BNN surrogate support)
    ├── other_functions.py # Utility helpers
    ├── plotting.py        # Plotting helpers (Pareto, z-score, etc.)
    ├── survival.py        # Standard and dual-ranking survival operators
    └── uncertainty.py     # Coverage and alpha search helpers
```

---

## Requirements

Recommended Python version: **3.10+**

Main dependencies:

- numpy
- pandas
- matplotlib
- plotly
- scikit-learn
- pymoo
- GPy
- autogluon.tabular
- ipython
- jupyter


```bash
python -m pip install numpy pandas matplotlib plotly scikit-learn pymoo GPy autogluon.tabular ipython jupyter
```

> Note: The first cell in each notebook also checks/installs core packages automatically.

---

## Quick Start

### 1) Run notebooks

From the repository root, launch Jupyter and run any notebook (for example `Exp1 GPR (RBF).ipynb`) top-to-bottom.

Notebooks dynamically add the repo root to `sys.path`:

```python
from pathlib import Path
import sys
repo_root = Path.cwd().resolve()
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))
```

### 2) Reuse `src/` in scripts

Example imports:

```python
from src.opt_problem import build_problem
from src.data import generate_data
from src.experiment import run_experiment
from src.survival import Survival_standard, Survival_dual_ranking
```

---

## Surrogate and Optimization Interfaces

In `src/opt_problem.py`, `Benchmark_Problem` supports:

- `"GPR_uncertainty"`: outputs `F` and `std`
- `"QR_uncertainty"`: outputs `F` and `F_q80/F_q90/F_q95`
- `"BNN_uncertainty"`: outputs `F` and `std`

In `src/survival.py`, `Survival_dual_ranking` supports:

- **GPR/BNN mode**: hybrid ranking using `F + alpha * std`
- **QR mode**: hybrid ranking using selected quantile matrix via `alpha=0.8/0.9/0.95` (`F_q80/F_q90/F_q95`)

---

## Useful Check

Run a quick syntax check:

```bash
python -m py_compile src/models.py src/opt_problem.py src/survival.py src/uncertainty.py
```

---

## Default Notebook Parameters

Most notebooks currently default to:

- `problem_name = "dtlz1"`
- `n_var = 10`
- `n_obj = 2`
- `n_gen = 100`
- `pop_size = 100`

## Notes

- Keep `problem_name` lowercase (e.g., `dtlz1`) for consistent branch handling.
- `GPy` and `autogluon.tabular` can take longer to install depending on platform; a clean virtual environment is recommended.
