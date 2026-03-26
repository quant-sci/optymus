import numpy as np


class OptimizeResult(dict):
    """Optimization result object with attribute access and convergence info.

    Behaves as a dict for backward compatibility while also supporting
    attribute access (e.g. ``result.xopt``). Modeled after
    ``scipy.optimize.OptimizeResult``.
    """

    CONVERGED_REASONS = frozenset({
        "gradient_norm_below_tol",
        "step_size_below_tol",
        "density_change_below_tol",
        "temperature_below_min",
        "std_below_min",
        "constraint_converged",
    })

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"'OptimizeResult' object has no attribute '{name}'"
            ) from None

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None

    @property
    def converged(self):
        """True if the optimizer terminated due to a convergence criterion."""
        reason = self.get("termination_reason", "")
        return reason in self.CONVERGED_REASONS

    def __repr__(self):
        name = self.get("method_name", "OptimizeResult")
        header = f"OptimizeResult: {name}"
        separator = "-" * max(len(header), 40)

        lines = [header, separator]

        display_fields = [
            ("xopt", "Optimal Solution"),
            ("fmin", "Function Value"),
            ("num_iter", "Iterations"),
            ("termination_reason", "Termination"),
            ("time", "Time (s)"),
        ]

        max_label = max(len(label) for _, label in display_fields)

        for key, label in display_fields:
            if key in self:
                val = self[key]
                if key == "time":
                    val = f"{float(val):.4f}"
                elif key == "fmin":
                    val = f"{float(val):.6e}"
                elif key == "xopt":
                    val = np.array2string(
                        np.asarray(val), precision=4, separator=", ",
                    )
                lines.append(f"  {label:<{max_label}}   {val}")

        lines.append(f"  {'Converged':<{max_label}}   {self.converged}")
        lines.append(separator)
        return "\n".join(lines)

    def _repr_mimebundle_(self, **_):
        return {"text/plain": repr(self)}
