"""Python script version of `Exp1_GPR_(RBF).ipynb`.

This script keeps the notebook's core workflow and adds a command-line entry
for running the 30x bias-variance experiment.
"""

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
from pymoo.operators.sampling.lhs import LHS
from sklearn.metrics import mean_squared_error

from src.data import generate_data
from src.experiment import run_experiment
from src.metrics import get_metrics
from src.models import GPR_RBF, gpr_pred_mean_std
from src.opt_problem import build_problem
from src.other_functions import mean_std, print_gpr_params
from src.plotting import plot_obj_2d, plot_y_true_pred, plot_z_score
from src.survival import Survival_dual_ranking, Survival_standard
from src.uncertainty import coverage, find_alpha

warnings.filterwarnings("ignore", message=".*load_learner.*pickle.*")
np.set_printoptions(precision=4, suppress=True)


def prepare_problem(problem_name: str = "dtlz1", n_var: int = 10, n_obj: int = 2):
    problem = build_problem(problem_name=problem_name, n_var=n_var, n_obj=n_obj)

    print(f"Problem name: {problem_name}")
    print(f"Cons: {problem.n_constr}")
    print(f"Var: {n_var}")
    print(f"Obj: {n_obj}")

    hv, igd_plus, obj_min, obj_max, ref_point = get_metrics(
        problem_name=problem_name,
        problem=problem,
        n_var=n_var,
        n_obj=n_obj,
    )
    print("\nMin-Max normalization -> Min:", obj_min)
    print("Min-Max normalization -> Max:", obj_max)
    print("HV Reference points:", ref_point)

    return problem, hv, igd_plus, obj_min, obj_max


def sample_data(problem, sample_size: int, train_seed: int = 42, test_seed: int = 1):
    X_train, y_train, X_val, y_val, X_test, y_test = generate_data(
        problem=problem,
        sample_size=sample_size,
        sampling=LHS(),
        train_seed=train_seed,
        val_size=100,
        test_size=100,
        test_seed=test_seed,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def run_single_gpr(X_train, y_train, X_test, y_test, plot: bool = True):
    model_f1 = GPR_RBF()
    model_f2 = GPR_RBF()

    model_f1.fit(X_train, y_train[:, 0])
    model_f2.fit(X_train, y_train[:, 1])
    print_gpr_params(model_f1, model_f2)

    pred_mean, pred_std, mean_f1, std_f1, mean_f2, std_f2 = gpr_pred_mean_std(
        model_f1, model_f2, X_test, noiseless=True
    )
    print(f"\nGPR(RBF) MSE: {mean_squared_error(y_test, pred_mean):.2e}\n")

    if plot:
        plot_y_true_pred(y_test, pred_mean)

    return model_f1, model_f2, pred_mean, pred_std, mean_f1, std_f1, mean_f2, std_f2


def run_high_bias_or_variance(X_train, y_train, X_test, y_test, lengthscale_weight: float, plot: bool = True):
    model_f1 = GPR_RBF()
    model_f2 = GPR_RBF()
    model_f1.fit_high_bias_variance(X_train, y_train[:, 0], lengthscale_weight=lengthscale_weight)
    model_f2.fit_high_bias_variance(X_train, y_train[:, 1], lengthscale_weight=lengthscale_weight)
    print_gpr_params(model_f1, model_f2)

    pred_mean, pred_std, _, _, _, _ = gpr_pred_mean_std(model_f1, model_f2, X_test, noiseless=True)
    print(f"\nGPR(RBF) MSE: {mean_squared_error(y_test, pred_mean):.2e}")

    if plot:
        plot_y_true_pred(y_test, pred_mean)

    return pred_mean, pred_std


def run_bias_variance_experiment(problem, sample_size: int, lengthscale_weight: float, n_runs: int = 30):
    preds = []
    y_test_ref = None

    for seed in range(1, n_runs + 1):
        X_train_i, y_train_i, _, _, X_test_i, y_test_i = sample_data(
            problem=problem,
            sample_size=sample_size,
            train_seed=seed,
            test_seed=1,
        )

        model_f1_i = GPR_RBF()
        model_f2_i = GPR_RBF()
        model_f1_i.fit_high_bias_variance(
            X_train_i,
            y_train_i[:, 0],
            lengthscale_weight=lengthscale_weight,
        )
        model_f2_i.fit_high_bias_variance(
            X_train_i,
            y_train_i[:, 1],
            lengthscale_weight=lengthscale_weight,
        )

        pred_mean_i, _, _, _, _, _ = gpr_pred_mean_std(model_f1_i, model_f2_i, X_test_i, noiseless=True)
        preds.append(pred_mean_i)

        if y_test_ref is None:
            y_test_ref = y_test_i

    preds = np.array(preds)  # shape: (n_runs, n_test, n_obj)
    mean_pred = preds.mean(axis=0)
    bias = np.mean((mean_pred - y_test_ref) ** 2)
    variance = np.mean(np.var(preds, axis=0))
    return bias, variance


def run_uncertainty_eval(y_test, pred_mean, pred_std, X_val, y_val, model_f1, model_f2, mean_f1, std_f1, mean_f2, std_f2, plot: bool = True):
    for k in [1.645]:
        per_dim, overall = coverage(y_test, pred_mean, pred_std, k=k)
        print(f"k={k}: per_dim={per_dim * 100}%, overall={overall * 100:.1f}%")

    if plot:
        plot_z_score(y_test, pred_mean, pred_std)

    alpha_c90_f1 = find_alpha(X_val, y_val[:, 0], model_f1, target_coverage=0.9)
    alpha_c90_f2 = find_alpha(X_val, y_val[:, 1], model_f2, target_coverage=0.9)
    print(f"alpha_c90_f1={alpha_c90_f1:.2f}, alpha_c90_f2={alpha_c90_f2:.2f}")

    f_upper_c90 = np.stack(
        [mean_f1 + alpha_c90_f1 * std_f1, mean_f2 + alpha_c90_f2 * std_f2],
        axis=1,
    )
    print("F_upper_c90\n", f_upper_c90[0:5], "\n")

    return alpha_c90_f1, alpha_c90_f2


def run_ea_experiment(problem, problem_name, n_gen, pop_size, model_f1, model_f2, obj_min, obj_max, hv, igd_plus):
    results = run_experiment(
        problem=problem,
        problem_name=problem_name,
        n_gen=n_gen,
        pop_size=pop_size,
        model_f1=model_f1,
        model_f2=model_f2,
        obj_min=obj_min,
        obj_max=obj_max,
        hv=hv,
        igd_plus=igd_plus,
        use_surrogate="GPR_uncertainty",
        survival_function=Survival_standard(),
        use_callback=False,
        seeds=range(1, 31),
    )

    mse_list = results["mse_list"]
    igd_list = results["igd_list"]
    hv_surrogate_list = results["hv_surrogate_list"]
    hv_real_list = results["hv_real_list"]

    mean_mse, std_mse = mean_std(mse_list)
    mean_igd, std_igd = mean_std(igd_list)
    mean_hv_real, std_hv_real = mean_std(hv_real_list)
    mean_hv_surrogate, std_hv_surrogate = mean_std(hv_surrogate_list)

    print(f"MSE: Mean = {mean_mse:.2e}, Std = {std_mse:.2e}")
    print(f"IGD+: Mean = {mean_igd:.2e}, Std = {std_igd:.2e}")
    print(f"Sur HV: Mean = {mean_hv_surrogate:.2f}, Std = {std_hv_surrogate:.2f}")
    print(f"Real HV: Mean = {mean_hv_real:.2f}, Std = {std_hv_real:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run Exp1 GPR (RBF) notebook workflow as Python script.")
    parser.add_argument("--problem_name", default="dtlz1")
    parser.add_argument("--n_var", type=int, default=10)
    parser.add_argument("--n_obj", type=int, default=2)
    parser.add_argument("--n_gen", type=int, default=100)
    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--n_runs", type=int, default=30, help="Runs for bias-variance experiment.")
    parser.add_argument("--no_plot", action="store_true", help="Disable plotting calls.")
    parser.add_argument("--run_ea", action="store_true", help="Run expensive EA experiment section.")
    args = parser.parse_args()

    plot = not args.no_plot

    problem, hv, igd_plus, obj_min, obj_max = prepare_problem(
        problem_name=args.problem_name,
        n_var=args.n_var,
        n_obj=args.n_obj,
    )

    sample_size = 11 * args.n_var - 1
    X_train, y_train, X_val, y_val, X_test, y_test = sample_data(
        problem=problem,
        sample_size=sample_size,
        train_seed=42,
        test_seed=1,
    )

    print(f"Sampling X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"y_test shape: {y_test.shape}")

    if plot:
        plot_obj_2d(y_train, xlim=(-100, 700), ylim=(-100, 700))

    model_f1, model_f2, pred_mean, pred_std, mean_f1, std_f1, mean_f2, std_f2 = run_single_gpr(
        X_train, y_train, X_test, y_test, plot=plot
    )

    print("\n=== High bias model (lengthscale_weight=10) ===")
    run_high_bias_or_variance(
        X_train, y_train, X_test, y_test, lengthscale_weight=10, plot=plot
    )

    print("\n=== High variance model (lengthscale_weight=0.1) ===")
    run_high_bias_or_variance(
        X_train, y_train, X_test, y_test, lengthscale_weight=0.1, plot=plot
    )

    high_bias_bias, high_bias_variance = run_bias_variance_experiment(
        problem=problem,
        sample_size=sample_size,
        lengthscale_weight=10,
        n_runs=args.n_runs,
    )
    high_var_bias, high_var_variance = run_bias_variance_experiment(
        problem=problem,
        sample_size=sample_size,
        lengthscale_weight=0.1,
        n_runs=args.n_runs,
    )
    print(
        f"High bias model -> Bias: {high_bias_bias:.6e}, Variance: {high_bias_variance:.6e}"
    )
    print(
        f"High variance model -> Bias: {high_var_bias:.6e}, Variance: {high_var_variance:.6e}"
    )

    alpha_c90_f1, alpha_c90_f2 = run_uncertainty_eval(
        y_test=y_test,
        pred_mean=pred_mean,
        pred_std=pred_std,
        X_val=X_val,
        y_val=y_val,
        model_f1=model_f1,
        model_f2=model_f2,
        mean_f1=mean_f1,
        std_f1=std_f1,
        mean_f2=mean_f2,
        std_f2=std_f2,
        plot=plot,
    )

    if args.run_ea:
        print("\n=== Running EA experiment (this may take a while) ===")
        run_ea_experiment(
            problem=problem,
            problem_name=args.problem_name,
            n_gen=args.n_gen,
            pop_size=args.pop_size,
            model_f1=model_f1,
            model_f2=model_f2,
            obj_min=obj_min,
            obj_max=obj_max,
            hv=hv,
            igd_plus=igd_plus,
        )

        # Optional: keep variables used in notebook dual-ranking section.
        _ = Survival_dual_ranking(alpha_f1=alpha_c90_f1, alpha_f2=alpha_c90_f2)

    if plot:
        plt.show()


if __name__ == "__main__":
    main()
