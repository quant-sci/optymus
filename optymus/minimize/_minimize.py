# Full path: optymus/optymus/minimize/_minimize.py
# Optimizer class

from optymus.plots import plot_optim

from optymus.utils._optim_methods import (
    univariant, powell, steepest_descent, fletcher_reeves, newton_raphson, bfgs
)


class Optimizer:
    def __init__(self, f_obj, x0, grad=None, hess=None, method=None, const=None, tol=1e-5, max_iter=100):
        self.const = const
        self.f_obj = f_obj
        self.x0 = x0
        self.grad = grad
        self.hess = hess
        self.method = method
        self.tol = tol
        self.max_iter = max_iter

        if self.method is None:
            self.opt = steepest_descent(self.f_obj, self.x0, self.grad, self.tol, self.max_iter)
        elif self.method == 'powell':
            self.opt = powell(self.f_obj, self.x0, self.tol, self.max_iter)
        elif self.method == 'steepest_descent':
            self.opt = steepest_descent(self.f_obj, self.x0, self.grad, self.tol, self.max_iter)
        elif self.method == 'fletcher_reeves':
            self.opt = fletcher_reeves(self.f_obj, self.x0, self.grad, self.tol, self.max_iter)
        elif self.method == 'newton_raphson':
            self.opt = newton_raphson(self.f_obj, self.x0, self.grad, self.hess, self.tol, self.max_iter)
        elif self.method == 'bfgs':
            self.opt = bfgs(self.f_obj, self.x0, self.grad, self.tol, self.max_iter)
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