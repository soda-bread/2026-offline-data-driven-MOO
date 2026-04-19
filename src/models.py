import numpy as np
import GPy
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

# Model: GPR_RBF
class GPR_RBF:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        kernel = GPy.kern.RBF(input_dim=X.shape[1],
                              ARD=True)
        self.model = GPy.models.GPRegression(X, y, kernel, normalizer=True)
        self.model.optimize(messages=False)

    def fit_reference(self, X, y):
        y = y.reshape(-1, 1)

        kernel = GPy.kern.RBF(input_dim=X.shape[1], ARD=True)
        self.model = GPy.models.GPRegression(X, y, kernel, normalizer=True)
        self.model.optimize(messages=False)

        ls_ref = self.model.rbf.lengthscale.values.copy()
        kv_ref = float(self.model.rbf.variance.values[0])
        noise_ref = float(self.model.likelihood.variance.values[0])
        return ls_ref, kv_ref, noise_ref

    def fit_low_bias_high_variance(self, X, y, ls_ref, kv_ref, noise_ref):
        y = y.reshape(-1, 1)

        kernel = GPy.kern.RBF(
            input_dim=X.shape[1],
            ARD=True,
            variance=kv_ref,
            lengthscale=ls_ref * 0.7)

        self.model = GPy.models.GPRegression(X, y, kernel, normalizer=True)
        self.model.likelihood.variance = noise_ref * 0.3

    def fit_low_bias_low_variance(self, X, y, ls_ref, kv_ref, noise_ref):
        y = y.reshape(-1, 1)

        kernel = GPy.kern.RBF(
            input_dim=X.shape[1],
            ARD=True,
            variance=kv_ref,
            lengthscale=ls_ref * 1.3)

        self.model = GPy.models.GPRegression(X, y, kernel, normalizer=True)
        self.model.likelihood.variance = noise_ref * 3.0

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


class BNN_Ensemble:
    """Approximate BNN with an MLP ensemble to expose predictive uncertainty."""

    def __init__(self, n_estimators=5, hidden_layer_sizes=(128, 64), max_iter=600, random_state=42):
        self.n_estimators = n_estimators
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.models = []

    def fit(self, X, y):
        self.models = []
        y = np.asarray(y).reshape(-1)
        for idx in range(self.n_estimators):
            model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation="relu",
                solver="adam",
                max_iter=self.max_iter,
                random_state=self.random_state + idx,
            )
            model.fit(X, y)
            self.models.append(model)

    def predict(self, X):
        preds = np.column_stack([model.predict(X) for model in self.models])
        return preds.mean(axis=1), preds.std(axis=1)

    def predict_noiseless(self, X):
        return self.predict(X)


class AutoGluon_RF:
    """AutoGluon-style surrogate wrapper with mean/std outputs."""

    def __init__(self, n_estimators=400, random_state=42, min_samples_leaf=1):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
        )

    def fit(self, X, y):
        y = np.asarray(y).reshape(-1)
        self.model.fit(X, y)

    def predict(self, X):
        tree_preds = np.column_stack([est.predict(X) for est in self.model.estimators_])
        return tree_preds.mean(axis=1), tree_preds.std(axis=1)

    def predict_noiseless(self, X):
        return self.predict(X)


def surrogate_pred_mean_std(model_f1, model_f2, X_test, noiseless=False, verbose=True):
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
        print(f"[{tag}] pred_mean\n", pred_mean[:5])
        print(f"[{tag}] pred_std\n", pred_std[:5])
        print(f"[{tag}] Max pred_std\n", np.max(pred_std, axis=0))

    return pred_mean, pred_std, mean_f1, std_f1, mean_f2, std_f2


# Backward-compatible alias for existing notebooks.
gpr_pred_mean_std = surrogate_pred_mean_std
