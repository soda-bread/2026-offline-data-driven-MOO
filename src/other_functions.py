import numpy as np


def mean_std(arr):
    return np.mean(arr), np.std(arr)


def print_surrogate_params(model_f1, model_f2):
    """Print model parameters when available; works across surrogate wrappers."""
    for name, model in (("f1", model_f1), ("f2", model_f2)):
        print(f"[{name}] class: {model.__class__.__name__}")
        gpy_model = getattr(model, "model", None)
        if gpy_model is not None and hasattr(gpy_model, "kern"):
            print(f"[{name}] lengthscale:", gpy_model.kern.lengthscale.values)
            print(f"[{name}] kernel variance:", gpy_model.kern.variance.values)
            if hasattr(gpy_model, "Gaussian_noise"):
                print(f"[{name}] noise:", gpy_model.Gaussian_noise.variance.values)
        elif hasattr(model, "models"):
            print(f"[{name}] ensemble size:", len(model.models))
        else:
            print(f"[{name}] parameter details are not exposed by this model wrapper.")


# Backward-compatible alias.
print_gpr_params = print_surrogate_params
