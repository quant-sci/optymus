# Full path: optymus/optymus/opt/_optimize.py
# Optimizer class

import jax
jax.config.update("jax_enable_x64", True)

from optymus.utils._plots import (
    plot_optim
)

from optymus.optim._zero_order import (
    univariant,
    powell,
)
from optymus.optim._first_order import (
    gradient_descent,
    conjugate_gradients, 
    bfgs, 
    l_bfgs,
)

from optymus.optim._second_order import (
    newton_raphson
)

from optymus.optim._adaptative import (
    adagrad,
    rmsprop,
    adam,
    adamax
)

METHODS = {
    "univariant": univariant,
    "powell": powell,
    "gradient_descent": gradient_descent,
    "conjugate_gradients": conjugate_gradients, 
    "bfgs": bfgs, 
    "l_bfgs": l_bfgs,
    "newton_raphson": newton_raphson,
    "adagrad": adagrad,
    "rmsprop": rmsprop,
    "adam": adam,
    "adamax": adamax

}


class Optimizer:
    def __init__(self, f_obj, x0, method=None, tol=1e-5, max_iter=100):
        self.f_obj = f_obj
        self.x0 = x0
        self.method = method
        self.tol = tol
        self.max_iter = max_iter

        if self.method is None:
            self.opt = gradient_descent(self.f_obj, self.x0, self.tol, self.max_iter)

        elif self.method is not None:
            self.opt = METHODS[self.method](self.f_obj, self.x0, self.tol, self.max_iter)
        
        else:
            raise ValueError("Method not available")

    def check_dimension(self):
        self.dim = len(self.x0)
        return self.dim
    
    def results(self):
        return self.opt
    
    def report(self):
        print("--------------------")
        print(f"Method Name: {self.opt['method_name']}")
        print("--------------------")
        print("--------------------")
        print("Results")
        print("--------------------")
        print(f"Initial point: {self.x0}")
        print(f"Minimum: {self.opt['fmin']}")
        print(f"Number of iterations: {self.opt['num_iter']}")
        print(f"Optimal point: {self.opt['xopt']}")
        print("--------------------")
    
    def plot(self, path=True, print_opt=True):
        if self.check_dimension() == 2:
            plot_optim(f = self.f_obj, x0=self.x0, method=self.opt, path=path, print_opt=print_opt)
        else:
            print("Plot not available for this dimension. Try functions with 2 variables.")