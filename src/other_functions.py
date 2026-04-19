import numpy as np

def mean_std(arr):
    return np.mean(arr), np.std(arr)

def print_gpr_params(model_f1, model_f2):
    print("f1 lengthscale:", model_f1.model.kern.lengthscale.values)
    print("f1 kernel variance:", model_f1.model.kern.variance.values)
    print("f1 noise:", model_f1.model.Gaussian_noise.variance.values)
    print("f2 lengthscale:", model_f2.model.kern.lengthscale.values)
    print("f2 kernel variance:", model_f2.model.kern.variance.values)
    print("f2 noise:", model_f2.model.Gaussian_noise.variance.values)


