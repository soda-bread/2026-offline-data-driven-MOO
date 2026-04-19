import numpy as np
import GPy

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
        print(f"[{tag}] pred_mean\n", pred_mean[:5])
        print(f"[{tag}] pred_std\n", pred_std[:5])
        print(f"[{tag}] Max pred_std\n", np.max(pred_std, axis=0))

    return pred_mean, pred_std, mean_f1, std_f1, mean_f2, std_f2
