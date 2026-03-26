import numpy as np
from tqdm.auto import tqdm

from optymus import Optimizer


def methods_comparison(f_obj=None, initial_point=None, **kwargs):
    METHODS = [
        "univariate",
        "powell",
        "steepest_descent",
        "conjugate_gradient",
        "bfgs",
        "newton_raphson",
        "adagrad",
        "rmsprop",
        "adam",
        "adamax",
        "yogi",
        "particle_swarm",
        "differential_evolution",
        "cmaes",
        "simulated_annealing",
    ]

    results = []
    path_dict = {}

    for method in tqdm(METHODS, desc="Comparing methods"):
        time_mean = []
        for _ in range(10):
            opt = Optimizer(f_obj=f_obj, x0=initial_point, method=method, verbose=False, **kwargs)
            time_mean.append(opt.get_results().get("time", "N/A"))

        path_dict[method] = opt.get_results()["path"]
        results.append({
            "method": opt.get_results()["method_name"],
            "x_opt": opt.get_results()["xopt"],
            "f_min": opt.get_results()["fmin"],
            "n_iter": opt.get_results()["num_iter"],
            "time": np.mean(time_mean),
        })

    results.sort(key=lambda r: r["time"])

    return {"results": results, "path_dict": path_dict}
