import numpy as np
import GPy
import pandas as pd
from autogluon.tabular import TabularPredictor
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule, PyroSample
from pyro.optim import Adam
from torch import nn


class BNN(PyroModule):
    def __init__(self, in_dim, hidden=32, out_dim=1, fixed_sigma=1e-3):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](in_dim, hidden)
        self.fc1.weight = PyroSample(dist.Normal(0., 0.5).expand([hidden, in_dim]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 0.5).expand([hidden]).to_event(1))
        self.fc2 = PyroModule[nn.Linear](hidden, hidden)
        self.fc2.weight = PyroSample(dist.Normal(0., 0.5).expand([hidden, hidden]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 0.5).expand([hidden]).to_event(1))
        self.out = PyroModule[nn.Linear](hidden, out_dim)
        self.out.weight = PyroSample(dist.Normal(0., 0.5).expand([out_dim, hidden]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0., 0.5).expand([out_dim]).to_event(1))
        self.sigma = fixed_sigma

    def forward(self, x, y=None):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.out(x).squeeze(-1)
        pyro.deterministic("mean", mu)
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Normal(mu, self.sigma), obs=y)
        return mu

    def train_model(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        lr=1e-3,
        max_steps=5000,
        patience=10,
        eval_every=200,
        num_val_samples=50,
        random_state=42,
        device=None,
    ):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        torch.manual_seed(random_state)
        pyro.set_rng_seed(random_state)
        pyro.clear_param_store()

        X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)
        y_train = torch.as_tensor(y_train, dtype=torch.float32, device=device).reshape(-1)
        use_val = (X_val is not None) and (y_val is not None)
        if use_val:
            X_val = torch.as_tensor(X_val, dtype=torch.float32, device=device)
            y_val = torch.as_tensor(y_val, dtype=torch.float32, device=device).reshape(-1)

        guide = AutoDiagonalNormal(self)
        svi = SVI(self, guide, Adam({"lr": lr}), loss=Trace_ELBO())
        best_score = float("inf")
        best_state = None
        wait = 0
        for step in range(1, max_steps + 1):
            loss = svi.step(X_train, y_train)
            if step % eval_every != 0:
                continue

            if use_val:
                pred = Predictive(self, guide=guide, num_samples=num_val_samples)(X_val)
                mu = pred["mean"].mean(dim=0)
                score = ((mu - y_val) ** 2).mean().item()
            else:
                score = loss

            if score < best_score:
                best_score = score
                best_state = pyro.get_param_store().get_state()
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                break

        if best_state is not None:
            pyro.get_param_store().set_state(best_state)
        self.guide = guide
        return self, guide

    def predict(self, X, guide, num_samples=100, device=None):
        device = device or next(self.parameters()).device
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        predictive = Predictive(self, guide=guide, num_samples=num_samples)
        samples = predictive(X_t)["obs"].detach().cpu().numpy()
        return samples.mean(axis=0), samples.std(axis=0)

    def predict_quantiles(self, X, guide, quantiles=(0.8, 0.9, 0.95), num_samples=100, device=None):
        device = device or next(self.parameters()).device
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        predictive = Predictive(self, guide=guide, num_samples=num_samples)
        samples = predictive(X_t)["obs"].detach().cpu().numpy()
        mean = samples.mean(axis=0)
        q_map = {q: np.percentile(samples, q * 100.0, axis=0) for q in quantiles}
        return mean, q_map


def train_bnn(X_train, y_train, X_val=None, y_val=None, hidden=32, lr=1e-3, max_steps=5000, patience=10, eval_every=200, fixed_sigma=1e-3, num_val_samples=50, random_state=42, device=None):
    model = BNN(in_dim=np.asarray(X_train).shape[1], hidden=hidden, fixed_sigma=fixed_sigma)
    return model.train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        lr=lr,
        max_steps=max_steps,
        patience=patience,
        eval_every=eval_every,
        num_val_samples=num_val_samples,
        random_state=random_state,
        device=device,
    )

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


def bnn_pred_mean_std(model_f1, model_f2, X_test, verbose=True):
    mean_f1, std_f1 = model_f1.predict(X_test, guide=model_f1.guide)
    mean_f2, std_f2 = model_f2.predict(X_test, guide=model_f2.guide)

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
