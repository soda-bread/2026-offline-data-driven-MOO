import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from sklearn.metrics import mean_squared_error

def get_metrics(problem_name, problem, n_var=None, n_obj=None):
    # Metrics: HV
    if problem_name == 'dtlz1':
        obj_min = np.array([0,0])
        obj_max = np.array([700,700])
    elif problem_name == 'dtlz2':
        obj_min = np.array([0,0])
        obj_max = np.array([2.78,2.93])
    elif problem_name == 'dtlz3':
        obj_min = np.array([0,0])
        obj_max = np.array([1605.54,1670.48])
    elif problem_name == 'dtlz4':
        obj_min = np.array([0,0])
        obj_max = np.array([2.83,2.78])
    elif problem_name == 'dtlz5':
        obj_min = np.array([0,0])
        obj_max = np.array([2.61,2.70])
    elif problem_name == 'dtlz6':
        obj_min = np.array([0,0])
        obj_max = np.array([9.78,9.78])
    elif problem_name == 'dtlz7':
        obj_min = np.array([0,0])
        obj_max = np.array([1.10,33.43])
    elif problem_name == 'omnitest':
        obj_min = np.array([-2,-2])
        obj_max = np.array([2.40,2.40])
    elif problem_name == 'bnh':
        obj_min = np.array([0,5])
        obj_max = np.array([140,50])
    elif problem_name == 'truss2d':
        obj_min = np.array([0,0])
        obj_max = np.array([0.1,1e5])
    elif problem_name == 'welded_beam':
        obj_min = np.array([0,0])
        obj_max = np.array([160,0.20])
    else:
        obj_min, obj_max = None, None

    ref_point = np.array([1.1,1.1])
    hv = HV(ref_point=ref_point)
    
    # Metrics: IGD+
    n_points = 200
    if problem_name == 'dtlz5':
        X_opt = np.full((n_points, n_var), 0.5)
        X_opt[:, 0] = np.linspace(0, 1, n_points)
        pf = problem.evaluate(X_opt)
    elif problem_name == 'dtlz6':
        X_opt = np.zeros((n_points, n_var))
        X_opt[:, 0] = np.linspace(0, 1, n_points)
        pf = problem.evaluate(X_opt)
    elif problem_name == 'dtlz7':
        X_opt = np.zeros((n_points, n_var))
        X_opt[:, :n_obj-1] = np.linspace(0, 1, n_points).reshape(-1, 1)
        pf = problem.evaluate(X_opt)
    else:
        pf = problem.pareto_front()
    
    igd_plus = IGDPlus(pf)
    
    return hv, igd_plus, obj_min, obj_max, ref_point



