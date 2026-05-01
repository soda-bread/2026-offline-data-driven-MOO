import numpy as np
import GPy
import pandas as pd
from autogluon.tabular import TabularPredictor
import torch

# Model: GPR_RBF
class GPR_RBF:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        kernel = GPy.kern.RBF(input_dim=X.shape[1],ARD=True)
        self.model = GPy.models.GPRegression(X, y, kernel, normalizer=True)
        self.model.optimize(messages=False)

    def fit_high_bias_variance(self, X, y,lengthscale_weight):
        y = y.reshape(-1, 1)
        kernel = GPy.kern.RBF(input_dim=X.shape[1], ARD=True)
        self.model = GPy.models.GPRegression(X, y, kernel, normalizer=True)
        self.model.optimize(messages=False)

        kernel = GPy.kern.RBF(
            input_dim=X.shape[1],
            ARD=True,
            variance=kernel.variance,
            lengthscale=kernel.lengthscale * lengthscale_weight)
        self.model = GPy.models.GPRegression(X, y, kernel, normalizer=True)

    def predict(self, X):
        y_mean, y_var = self.model.predict(X, include_likelihood=True)
        y_std = np.sqrt(y_var)
        return y_mean.flatten(), y_std.flatten()

    def predict_noiseless(self, X):
        y_mean, y_var = self.model.predict(X, include_likelihood=False)
        y_std = np.sqrt(y_var)
        return y_mean.flatten(), y_std.flatten()


class GPR_Matern:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        kernel = GPy.kern.Matern52(input_dim=X.shape[1], ARD=True)
        self.model = GPy.models.GPRegression(X, y, kernel, normalizer=True)
        self.model.optimize(messages=False)

    def predict(self, X):
        y_mean, y_var = self.model.predict(X, include_likelihood=True)
        y_std = np.sqrt(y_var)
        return y_mean.flatten(), y_std.flatten()

    def predict_noiseless(self, X):
        y_mean, y_var = self.model.predict(X, include_likelihood=False)
        y_std = np.sqrt(y_var)
        return y_mean.flatten(), y_std.flatten()

def gpr_pred_mean_std(model_f1, model_f2, X_test, noiseless=False, verbose=True):
    if noiseless:
        mean_f1, std_f1 = model_f1.predict_noiseless(X_test)
        mean_f2, std_f2 = model_f2.predict_noiseless(X_test)
    else:
        mean_f1, std_f1 = model_f1.predict(X_test)
        mean_f2, std_f2 = model_f2.predict(X_test)

    mean_f1 = np.asarray(mean_f1).reshape(-1)
    std_f1 = np.asarray(std_f1).reshape(-1)
    mean_f2 = np.asarray(mean_f2).reshape(-1)
    std_f2 = np.asarray(std_f2).reshape(-1)

    pred_mean = np.stack([mean_f1, mean_f2], axis=1)
    pred_std = np.stack([std_f1, std_f2], axis=1)

    if verbose:
        tag = "noiseless" if noiseless else "with_noise"

    return pred_mean, pred_std, mean_f1, std_f1, mean_f2, std_f2


def autogluon_qr_fit_predict(X_train, y_train, X_test, quantile_levels=None, random_state=42):
    if quantile_levels is None:
        quantile_levels = [0.5, 0.8, 0.9, 0.95]

    train_df = pd.DataFrame(X_train, columns=[f"x{i}" for i in range(X_train.shape[1])])
    train_df["target"] = y_train

    model = TabularPredictor(
        label="target",
        problem_type="quantile",
        quantile_levels=quantile_levels
    ).fit(
        train_data=train_df,
        verbosity=0,
        ag_args_fit={"random_state": random_state}
    )

    quantile_pred = autogluon_qr_predict(model, X_test)
    return quantile_pred, model


def autogluon_qr_predict(model, X):
    test_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    pred = model.predict(test_df)
    pred.columns = [f"y_q{q}" for q in pred.columns]
    return pred


def autogluon_qr_pred_mean_quantiles(model_f1, model_f2, X_test, verbose=True):
    pred_y1 = autogluon_qr_predict(model_f1, X_test)
    pred_y2 = autogluon_qr_predict(model_f2, X_test)

    mean_q = np.stack([pred_y1["y_q0.5"].values, pred_y2["y_q0.5"].values], axis=1)
    q80 = np.stack([pred_y1["y_q0.8"].values, pred_y2["y_q0.8"].values], axis=1)
    q90 = np.stack([pred_y1["y_q0.9"].values, pred_y2["y_q0.9"].values], axis=1)
    q95 = np.stack([pred_y1["y_q0.95"].values, pred_y2["y_q0.95"].values], axis=1)

    if verbose:
        print("[QR] y_q50\n", mean_q[:5])
        print("[QR] y_q80\n", q80[:5])
        print("[QR] y_q90\n", q90[:5])
        print("[QR] y_q95\n", q95[:5])

    return mean_q, q80, q90, q95


class BNNEnsembleRegressor:
    """
    Pyro-based Bayesian neural network regressor.
    Kept class name for backward compatibility with existing notebooks/scripts.
    """
    def __init__(
        self,
        hidden_layer_sizes=(64, 64),
        n_estimators=100,
        max_iter=5000,
        random_state=42,
        lr=1e-3,
        fixed_sigma=1e-3,
        eval_every=200,
        patience=10,
        device=None,
    ):
        if len(hidden_layer_sizes) != 2:
            raise ValueError("hidden_layer_sizes must contain exactly two hidden layer sizes, e.g. (32, 32).")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_estimators = n_estimators  # used as posterior predictive samples at inference time
        self.max_iter = max_iter
        self.random_state = random_state
        self.lr = lr
        self.fixed_sigma = fixed_sigma
        self.eval_every = eval_every
        self.patience = patience
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.guide = None

    def fit(self, X, y):
        import pyro
        import pyro.distributions as dist
        from pyro.infer import SVI, Trace_ELBO
        from pyro.infer.autoguide import AutoDiagonalNormal
        from pyro.nn import PyroModule, PyroSample
        from pyro.optim import Adam
        from torch import nn

        torch.manual_seed(self.random_state)
        pyro.set_rng_seed(self.random_state)
        pyro.clear_param_store()

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device).reshape(-1)
        h1, h2 = self.hidden_layer_sizes
        in_dim = X_t.shape[1]

        class _BNN(PyroModule):
            def __init__(self, in_dim_, h1_, h2_, fixed_sigma_):
                super().__init__()
                self.fc1 = PyroModule[nn.Linear](in_dim_, h1_)
                self.fc1.weight = PyroSample(dist.Normal(0., 0.5).expand([h1_, in_dim_]).to_event(2))
                self.fc1.bias = PyroSample(dist.Normal(0., 0.5).expand([h1_]).to_event(1))

                self.fc2 = PyroModule[nn.Linear](h1_, h2_)
                self.fc2.weight = PyroSample(dist.Normal(0., 0.5).expand([h2_, h1_]).to_event(2))
                self.fc2.bias = PyroSample(dist.Normal(0., 0.5).expand([h2_]).to_event(1))

                self.out = PyroModule[nn.Linear](h2_, 1)
                self.out.weight = PyroSample(dist.Normal(0., 0.5).expand([1, h2_]).to_event(2))
                self.out.bias = PyroSample(dist.Normal(0., 0.5).expand([1]).to_event(1))
                self.sigma = fixed_sigma_

            def forward(self, x, y_obs=None):
                x = torch.tanh(self.fc1(x))
                x = torch.tanh(self.fc2(x))
                mu = self.out(x).squeeze(-1)
                pyro.deterministic("mean", mu)
                with pyro.plate("data", x.shape[0]):
                    pyro.sample("obs", dist.Normal(mu, self.sigma), obs=y_obs)
                return mu

        self.model = _BNN(in_dim, h1, h2, self.fixed_sigma).to(self.device)
        self.guide = AutoDiagonalNormal(self.model)
        optimizer = Adam({"lr": self.lr})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())

        best_loss = float("inf")
        best_state = None
        no_improve = 0
        for step in range(1, self.max_iter + 1):
            loss = svi.step(X_t, y_t)
            if step % self.eval_every == 0:
                if loss < best_loss:
                    best_loss = loss
                    best_state = pyro.get_param_store().get_state()
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            pyro.get_param_store().set_state(best_state)

    def predict(self, X):
        if self.model is None or self.guide is None:
            raise ValueError("Model is not fitted yet.")
        import pyro
        from pyro.infer import Predictive

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        predictive = Predictive(self.model, guide=self.guide, num_samples=self.n_estimators)
        samples = predictive(X_t)
        obs_samples = samples["obs"].detach().cpu().numpy()
        return obs_samples.mean(axis=0), obs_samples.std(axis=0)


def bnn_pred_mean_std(model_f1, model_f2, X_test, verbose=True):
    mean_f1, std_f1 = model_f1.predict(X_test)
    mean_f2, std_f2 = model_f2.predict(X_test)

    mean_f1 = np.asarray(mean_f1).reshape(-1)
    std_f1 = np.asarray(std_f1).reshape(-1)
    mean_f2 = np.asarray(mean_f2).reshape(-1)
    std_f2 = np.asarray(std_f2).reshape(-1)

    pred_mean = np.stack([mean_f1, mean_f2], axis=1)
    pred_std = np.stack([std_f1, std_f2], axis=1)

    if verbose:
        print("[BNN] pred_mean\n", pred_mean[:5])
        print("[BNN] pred_std\n", pred_std[:5])
        print("[BNN] Max pred_std\n", np.max(pred_std, axis=0))

    return pred_mean, pred_std, mean_f1, std_f1, mean_f2, std_f2
