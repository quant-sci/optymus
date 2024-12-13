import numpy as np
import pandas as pd
from tqdm import tqdm

from optymus import Optimizer


def methods_comparison(f_obj=None, initial_point=None, **kwargs):
    METHODS = [
        "steepest_descent",
        "newton_raphson",
        "bfgs",
        "conjugate_gradient",
        "univariate",
        "powell",
        "adam",
        "adamax",
        "adagrad",
        "rmsprop",
        "yogi",
        "particle_swarm",
        "differential_evolution",
    ]

    results = pd.DataFrame(columns=["method", "x_opt", "f_min", "n_iter", "time"])
    path_dict = {}

    for method in tqdm(METHODS):
        time_mean = []
        for _ in range(10):
            opt = Optimizer(f_obj=f_obj, x0=initial_point, method=method, verbose=False, **kwargs)
            time_mean.append(opt.get_results().get("time", "N/A"))
        
        path_dict[method] = opt.get_results()["path"]
        opt_results = pd.DataFrame(
            {
                "method": [opt.get_results()["method_name"]],
                "x_opt": [opt.get_results()["xopt"]],
                "f_min": [opt.get_results()["fmin"]],
                "n_iter": [opt.get_results()["num_iter"]],
                "time": [np.mean(time_mean)],
            }
        )

        results = pd.concat([results, opt_results], axis=0, ignore_index=True)
    results.sort_values(by="time", ascending=True, inplace=True, ignore_index=True)

    return {"results": results, "path_dict": path_dict}
