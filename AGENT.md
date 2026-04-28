# AGENT.md

## Purpose

This file records the expected operating workflow for running experiments in this repository and for updating code on GitHub.

## Experiment Workflow

Use this workflow when running `Exp1_GPR-RBF.py` or similar experiment scripts in this repository.

1. Work from the repository root:
   `E:\Codex\2026-offline-data-driven-MOO`
2. Prefer the rebuilt virtual environment:
   `.venv312\Scripts\python.exe`
3. If the environment is missing or broken:
   - Create a fresh environment with Python 3.12.
   - Install dependencies from `requirements.txt`.
4. Run experiments from the repository root so relative config and output paths resolve correctly.
5. Keep experiment outputs inside the repository, not in an external temp directory.
6. Keep stdout/stderr logs in the repository root when live monitoring is needed.

### Recommended run pattern

- Script:
  `Exp1_GPR-RBF.py`
- Config:
  `configs/exp1_gpr_rbf.yaml`
- Logs:
  `exp1_stdout.log`
  `exp1_stderr.log`
- Main output directory:
  `outputs/exp1_gpr_rbf/dtlz1`

### Data recording expectations

When `run_bias_variance_ea` is enabled, results should be written to:

- `outputs/exp1_gpr_rbf/dtlz1/bias_variance_summary.csv`
- `outputs/exp1_gpr_rbf/dtlz1/bias_variance_training_records.csv`
- `outputs/exp1_gpr_rbf/dtlz1/optimization_summary_seed42_bias_variance_models.csv`

### Bias/variance naming convention

- The quantity computed as:
  `np.mean((mean_pred - y_test_ref) ** 2)`
  must be named `bias_squared`.
- Do not label that quantity as `bias`.
- If true bias is needed, compute:
  `np.sqrt(bias_squared)`

## Live experiment monitoring

When a user asks to both run an experiment and see progress in chat:

1. Start the script in the repository root.
2. Redirect stdout and stderr to repo-local log files.
3. Poll `exp1_stdout.log` and `exp1_stderr.log`.
4. Report meaningful progress snapshots in chat.
5. Confirm when CSV outputs begin to appear and when they are complete.

## GitHub Update Workflow

All future code updates should follow a Pull Request workflow.

### Required flow

1. Create a new branch from `main`.
2. Make code changes on that branch.
3. Commit only the intended source/documentation files.
4. Do not include generated files unless the user explicitly requests them.
5. Push the branch to GitHub.
6. Open a Pull Request.
7. Share the PR link and a short summary with the user.

### Do not include by default

Do not include these in normal commits unless explicitly requested:

- `outputs/`
- `exp1_stdout.log`
- `exp1_stderr.log`
- `src/__pycache__/`
- other generated `.pyc` files

### GitHub fallback rule

If normal `git push` fails because of local credential/auth issues:

1. Keep the local commit.
2. Use the GitHub integration to update the remote branch or create the PR.
3. Still preserve the PR-based workflow whenever possible.

## Notes for future runs

- Run experiments from a stable environment and fixed seeds when comparing bias-variance behavior.
- For the current `Exp1_GPR-RBF.py` implementation, the test set is intentionally fixed with `test_seed=1` across runs.
- This is required for the current bias-variance comparison logic to remain meaningful.
