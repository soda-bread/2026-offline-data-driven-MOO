# 2026-offline-data-driven-MOO

This repository contains a multi-objective optimization workflow based on **GPR** surrogate models and **NSGA-II**, with experiment notebooks:

- `Exp_1_GPR(RBF).ipynb`
- `Exp-1 GPR(Matern).ipynb`
- `Exp-1 BNN.ipynb`
- `Exp-1 AutoGluon.ipynb`

## Current runnability status

After reviewing `src/` and the notebook:

- The source modules in `src/` form a complete pipeline (problem setup, sampling, surrogate training, uncertainty estimation, optimization, and metrics).
- The notebook is now configured for **non-Colab/local usage** by default (no `google.colab` dependency, no Drive mount requirement).
- You can run it locally in Jupyter as long as dependencies are installed.

## Project structure

```text
.
├── Exp_1_GPR(RBF).ipynb
├── Exp-1 GPR(Matern).ipynb
├── Exp-1 BNN.ipynb
├── Exp-1 AutoGluon.ipynb
├── README.md
└── src
    ├── data.py            # train/validation/test data generation
    ├── experiment.py      # optimization experiment runner
    ├── metrics.py         # HV / IGD+ metric setup
    ├── models.py          # GPR / BNN / AutoGluon-style surrogate wrappers
    ├── opt_problem.py     # surrogate problem wrapper and callback
    ├── other_functions.py # utility functions
    ├── plotting.py        # plotting helpers
    ├── survival.py        # standard and dual-ranking survival
    └── uncertainty.py     # coverage and alpha search
```

## Requirements

Recommended Python: **3.10+**

Dependencies used by this project:

- numpy
- pandas
- matplotlib
- plotly
- scikit-learn
- pymoo
- GPy
- ipython
- jupyter

Install example:

```bash
python -m pip install numpy pandas matplotlib plotly scikit-learn pymoo GPy ipython jupyter
```

## How to run

### Option A: Jupyter Notebook (local)

1. Open one notebook in Jupyter: `Exp_1_GPR(RBF).ipynb`, `Exp-1 GPR(Matern).ipynb`, `Exp-1 BNN.ipynb`, or `Exp-1 AutoGluon.ipynb`.
2. Ensure your working directory is the repository root.
3. Run all cells from top to bottom.

The notebook now appends the repository root dynamically:

```python
from pathlib import Path
import sys
repo_root = Path.cwd().resolve()
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))
```

### Option B: Reuse `src/` modules in scripts

You can import modules directly, for example:

```python
from src.opt_problem import build_problem
from src.data import generate_data
from src.models import GPR_RBF, BNN_Ensemble, AutoGluon_RF
```

## Quick checks

Run a syntax check for source files:

```bash
python -m compileall src
```

## Default notebook parameters

In `Exp_1_GPR(RBF).ipynb` and `Exp-1 GPR(Matern).ipynb` (current defaults):

- `problem_name = 'dtlz1'`
- `n_var = 10`
- `n_obj = 2`
- `n_gen = 100`
- `pop_size = 100`

For a faster smoke run, reduce the optimization scale first (for example, `n_gen=10`, `pop_size=30`).

## Notes

- Keep `problem_name` lowercase to match branch logic in `src/metrics.py`.
- `GPy` installation may vary by platform; if installation fails locally, try a clean virtual environment.

- The BNN and AutoGluon notebooks follow the same src-based pipeline (fit/predict/predict_noiseless interface).
