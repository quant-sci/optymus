import time
from collections.abc import Callable, Iterable

import jax
import jax.numpy as jnp


class ConstraintMethodError(ValueError):
    pass


class InfeasibleStartError(ValueError):
    pass


def _as_list(constraints: Iterable[Callable] | Callable | None) -> list[Callable]:
    if constraints is None:
        return []
    if isinstance(constraints, (list, tuple)):
        return list(constraints)
    return [constraints]


def _merge_constraint_values(values: list[jnp.ndarray], check_finite: bool) -> jnp.ndarray:
    merged = jnp.concatenate(values) if len(values) > 1 else values[0]
    if check_finite and not jnp.all(jnp.isfinite(merged)):
        msg = "Constraint function returned non-finite values."
        raise ConstraintMethodError(msg)
    return merged


def _stack_constraint_values(
    constraints: list[Callable],
    x,
    check_finite: bool = True,
) -> jnp.ndarray:
    if not constraints:
        return jnp.array([])
    values = [jnp.ravel(constraint(x)) for constraint in constraints]
    return _merge_constraint_values(values, check_finite=check_finite)


def combine_constraints(
    constraints: list[Callable],
    check_finite: bool = True,
) -> Callable:
    if not constraints:
        return lambda _x: jnp.array([])

    def combined(x):
        values = [jnp.ravel(constraint(x)) for constraint in constraints]
        return _merge_constraint_values(values, check_finite=check_finite)

    return combined


def _evaluate_constraints(g_fun: Callable, h_fun: Callable, x) -> tuple[jnp.ndarray, jnp.ndarray]:
    ineq_values = g_fun(x) if g_fun is not None else jnp.array([])
    eq_values = h_fun(x) if h_fun is not None else jnp.array([])
    return ineq_values, eq_values


def _constraint_violation_from_values(ineq_values: jnp.ndarray, eq_values: jnp.ndarray) -> jnp.ndarray:
    violations = []
    if ineq_values.size:
        violations.append(jnp.max(jnp.maximum(ineq_values, 0.0)))
    if eq_values.size:
        violations.append(jnp.max(jnp.abs(eq_values)))

    return jnp.max(jnp.array(violations)) if violations else jnp.array(0.0)


def _penalty_value_from_values(ineq_values: jnp.ndarray, eq_values: jnp.ndarray) -> jnp.ndarray:
    ineq_penalty = jnp.sum(jnp.maximum(ineq_values, 0.0) ** 2) if ineq_values.size else jnp.array(0.0)
    eq_penalty = jnp.sum(eq_values**2) if eq_values.size else jnp.array(0.0)

    return ineq_penalty + eq_penalty


def _barrier_value_from_values(
    ineq_values: jnp.ndarray,
    barrier_type: str,
    barrier_eps: float,
) -> jnp.ndarray:
    if not ineq_values.size:
        return jnp.array(0.0)

    safe = jnp.maximum(-ineq_values, barrier_eps)
    if barrier_type == "log":
        base = -jnp.log(safe)
    elif barrier_type == "inverse":
        base = 1.0 / safe
    else:
        msg = f"Barrier type '{barrier_type}' is not available. Use 'log' or 'inverse'."
        raise ConstraintMethodError(msg)

    infeasible = jnp.any(ineq_values >= 0.0)
    barrier_value = jnp.sum(base)
    return jnp.where(infeasible, jnp.array(jnp.inf), barrier_value)


def _prepare_constraint_functions(
    g_cons: list[Callable],
    h_cons: list[Callable],
    constraint_jit: bool,
) -> tuple[Callable, Callable, Callable, Callable]:
    check_finite_obj = not constraint_jit
    g_fun_obj = combine_constraints(g_cons, check_finite=check_finite_obj)
    h_fun_obj = combine_constraints(h_cons, check_finite=check_finite_obj)
    g_fun_checked = g_fun_obj if check_finite_obj else combine_constraints(g_cons, check_finite=True)
    h_fun_checked = h_fun_obj if check_finite_obj else combine_constraints(h_cons, check_finite=True)
    return g_fun_obj, h_fun_obj, g_fun_checked, h_fun_checked


def _maybe_warn_constraint_size(
    g_fun_checked: Callable,
    h_fun_checked: Callable,
    x0,
    warn_constraint_size: int | None,
    warnings: list[str],
) -> None:
    if warn_constraint_size is None:
        return
    g_vals, h_vals = _evaluate_constraints(g_fun_checked, h_fun_checked, x0)
    if g_vals.size > warn_constraint_size or h_vals.size > warn_constraint_size:
        warnings.append("constraint_size_warning")


def _maybe_warn_slow_iter(
    iter_time: float,
    warn_slow_iter_s: float | None,
    warnings: list[str],
) -> None:
    if warn_slow_iter_s is not None and iter_time > warn_slow_iter_s:
        if "slow_iteration_warning" not in warnings:
            warnings.append("slow_iteration_warning")


def _update_no_progress(
    no_progress_count: int,
    violation_history: list[jnp.ndarray],
    warn_no_progress_iters: int | None,
    warn_no_progress_tol: float,
    warnings: list[str],
) -> int:
    if warn_no_progress_iters is None or len(violation_history) < 2:
        return no_progress_count
    prev = violation_history[-2]
    curr = violation_history[-1]
    if jnp.abs(prev - curr) <= warn_no_progress_tol:
        no_progress_count += 1
    else:
        no_progress_count = 0
    if no_progress_count >= warn_no_progress_iters and "no_progress_warning" not in warnings:
        warnings.append("no_progress_warning")
    return no_progress_count


def _constraint_violation(g_cons: list[Callable], h_cons: list[Callable], x) -> jnp.ndarray:
    ineq_values = _stack_constraint_values(g_cons, x)
    eq_values = _stack_constraint_values(h_cons, x)
    return _constraint_violation_from_values(ineq_values, eq_values)


def _penalty_value(g_cons: list[Callable], h_cons: list[Callable], x) -> jnp.ndarray:
    ineq_values = _stack_constraint_values(g_cons, x)
    eq_values = _stack_constraint_values(h_cons, x)
    return _penalty_value_from_values(ineq_values, eq_values)


def _barrier_value(
    g_cons: list[Callable],
    x,
    barrier_type: str,
    barrier_eps: float,
) -> jnp.ndarray:
    ineq_values = _stack_constraint_values(g_cons, x)
    return _barrier_value_from_values(
        ineq_values,
        barrier_type=barrier_type,
        barrier_eps=barrier_eps,
    )


def run_penalty_method(
    f_obj: Callable,
    g_cons: list[Callable],
    h_cons: list[Callable],
    x0,
    inner_method: Callable,
    constraint_jit: bool,
    penalty_r0: float,
    penalty_factor: float,
    max_outer_iter: int,
    constraint_tol: float,
    outer_tol: float,
    warn_constraint_size: int | None,
    warn_slow_iter_s: float | None,
    warn_no_progress_iters: int | None,
    warn_no_progress_tol: float,
    inner_kwargs: dict,
) -> dict:
    g_fun_obj, h_fun_obj, g_fun_checked, h_fun_checked = _prepare_constraint_functions(
        g_cons,
        h_cons,
        constraint_jit,
    )

    def penalized_obj(x_local, r_p_local):
        g_vals, h_vals = _evaluate_constraints(g_fun_obj, h_fun_obj, x_local)
        penalty_scale = 0.5 * r_p_local
        return f_obj(x_local) + penalty_scale * _penalty_value_from_values(g_vals, h_vals)

    penalized_core = jax.jit(penalized_obj) if constraint_jit else penalized_obj

    r_p = penalty_r0
    x = x0
    outer_path = [x0]
    r_p_history = [r_p]
    violation_history = []
    warnings = []
    total_inner_iters = 0
    start_time = time.time()
    no_progress_count = 0
    last_f_val = None

    _maybe_warn_constraint_size(g_fun_checked, h_fun_checked, x0, warn_constraint_size, warnings)

    reached_max = True
    for _ in range(max_outer_iter):
        iter_start = time.time()
        penalized_eval = lambda x_local: penalized_core(x_local, r_p)
        result = inner_method(f_obj=penalized_eval, f_cons=None, x0=x, **inner_kwargs)
        iter_time = time.time() - iter_start
        _maybe_warn_slow_iter(iter_time, warn_slow_iter_s, warnings)
        total_inner_iters += result.get("num_iter", 0)
        x_new = result["xopt"]
        if not jnp.all(jnp.isfinite(x_new)):
            msg = "Objective evaluation produced non-finite values."
            raise ConstraintMethodError(msg)
        f_val = f_obj(x_new)
        if not jnp.all(jnp.isfinite(jnp.ravel(f_val))):
            msg = "Objective evaluation produced non-finite values."
            raise ConstraintMethodError(msg)
        last_f_val = f_val

        g_vals, h_vals = _evaluate_constraints(g_fun_checked, h_fun_checked, x_new)
        violation = _constraint_violation_from_values(g_vals, h_vals)
        violation_history.append(violation)
        outer_path.append(x_new)

        step_norm = jnp.linalg.norm(x_new - x)
        x = x_new
        if violation <= constraint_tol and step_norm <= outer_tol:
            reached_max = False
            break

        no_progress_count = _update_no_progress(
            no_progress_count,
            violation_history,
            warn_no_progress_iters,
            warn_no_progress_tol,
            warnings,
        )

        r_p = r_p * penalty_factor
        r_p_history.append(r_p)

    elapsed_time = time.time() - start_time
    if reached_max and violation_history:
        if violation_history[-1] > constraint_tol:
            warnings.append("max_outer_iter_reached")
    return {
        "method_name": f"{inner_method.__name__} with Penalty Method",
        "x0": x0,
        "xopt": x,
        "fmin": last_f_val if last_f_val is not None else f_obj(x),
        "num_iter": total_inner_iters,
        "outer_iters": len(outer_path) - 1,
        "path": jnp.array(outer_path),
        "constraint_violation": violation_history[-1] if violation_history else jnp.array(0.0),
        "constraint_violation_history": jnp.array(violation_history),
        "r_p_history": jnp.array(r_p_history),
        "warnings": warnings,
        "time": elapsed_time,
    }


def run_barrier_method(
    f_obj: Callable,
    g_cons: list[Callable],
    h_cons: list[Callable],
    x0,
    inner_method: Callable,
    constraint_jit: bool,
    barrier_type: str,
    barrier_r0: float,
    barrier_factor: float,
    penalty_r0: float,
    penalty_factor: float,
    max_outer_iter: int,
    constraint_tol: float,
    outer_tol: float,
    barrier_eps: float,
    warn_constraint_size: int | None,
    warn_slow_iter_s: float | None,
    warn_no_progress_iters: int | None,
    warn_no_progress_tol: float,
    inner_kwargs: dict,
) -> dict:
    g_fun_obj, h_fun_obj, g_fun_checked, h_fun_checked = _prepare_constraint_functions(
        g_cons,
        h_cons,
        constraint_jit,
    )

    ineq_values, _ = _evaluate_constraints(g_fun_checked, h_fun_checked, x0)
    if ineq_values.size and bool(jnp.any(ineq_values >= 0.0)):
        msg = "Barrier method requires a strictly feasible starting point."
        raise InfeasibleStartError(msg)

    def barrier_obj(x_local, r_p_local, rb_local):
        g_vals, h_vals = _evaluate_constraints(g_fun_obj, h_fun_obj, x_local)
        penalty_scale = 0.5 * r_p_local
        penalty_term = _penalty_value_from_values(jnp.array([]), h_vals)
        barrier_term = _barrier_value_from_values(
            g_vals,
            barrier_type=barrier_type,
            barrier_eps=barrier_eps,
        )
        return f_obj(x_local) + penalty_scale * penalty_term + rb_local * barrier_term

    barrier_core = jax.jit(barrier_obj) if constraint_jit else barrier_obj

    r_p = penalty_r0
    rb = barrier_r0
    x = x0
    outer_path = [x0]
    r_p_history = [r_p]
    rb_history = [rb]
    violation_history = []
    warnings = []
    total_inner_iters = 0
    start_time = time.time()
    no_progress_count = 0
    last_f_val = None

    _maybe_warn_constraint_size(g_fun_checked, h_fun_checked, x0, warn_constraint_size, warnings)

    reached_max = True
    for _ in range(max_outer_iter):
        iter_start = time.time()
        barrier_eval = lambda x_local: barrier_core(x_local, r_p, rb)
        result = inner_method(f_obj=barrier_eval, f_cons=None, x0=x, **inner_kwargs)
        iter_time = time.time() - iter_start
        _maybe_warn_slow_iter(iter_time, warn_slow_iter_s, warnings)
        total_inner_iters += result.get("num_iter", 0)
        x_new = result["xopt"]
        if not jnp.all(jnp.isfinite(x_new)):
            msg = "Objective evaluation produced non-finite values."
            raise ConstraintMethodError(msg)
        g_vals, h_vals = _evaluate_constraints(g_fun_checked, h_fun_checked, x_new)
        if g_vals.size and bool(jnp.any(g_vals >= 0.0)):
            direction = x_new - x
            alpha = 1.0
            for _ in range(25):
                alpha *= 0.5
                candidate = x + alpha * direction
                cand_g_vals, cand_h_vals = _evaluate_constraints(g_fun_checked, h_fun_checked, candidate)
                if cand_g_vals.size == 0 or bool(jnp.all(cand_g_vals < 0.0)):
                    x_new = candidate
                    g_vals, h_vals = cand_g_vals, cand_h_vals
                    break
            else:
                msg = "Barrier method produced an infeasible iterate."
                raise ConstraintMethodError(msg)
        f_val = f_obj(x_new)
        if not jnp.all(jnp.isfinite(jnp.ravel(f_val))):
            msg = "Objective evaluation produced non-finite values."
            raise ConstraintMethodError(msg)
        last_f_val = f_val
        violation = _constraint_violation_from_values(g_vals, h_vals)
        violation_history.append(violation)
        outer_path.append(x_new)

        step_norm = jnp.linalg.norm(x_new - x)
        x = x_new
        if violation <= constraint_tol and step_norm <= outer_tol:
            reached_max = False
            break

        no_progress_count = _update_no_progress(
            no_progress_count,
            violation_history,
            warn_no_progress_iters,
            warn_no_progress_tol,
            warnings,
        )

        rb = rb * barrier_factor
        r_p = r_p * penalty_factor
        rb_history.append(rb)
        r_p_history.append(r_p)

    elapsed_time = time.time() - start_time
    if reached_max and violation_history:
        if violation_history[-1] > constraint_tol:
            warnings.append("max_outer_iter_reached")
    return {
        "method_name": f"{inner_method.__name__} with Barrier Method",
        "x0": x0,
        "xopt": x,
        "fmin": last_f_val if last_f_val is not None else f_obj(x),
        "num_iter": total_inner_iters,
        "outer_iters": len(outer_path) - 1,
        "path": jnp.array(outer_path),
        "constraint_violation": violation_history[-1] if violation_history else jnp.array(0.0),
        "constraint_violation_history": jnp.array(violation_history),
        "r_p_history": jnp.array(r_p_history),
        "rb_history": jnp.array(rb_history),
        "barrier_type": barrier_type,
        "warnings": warnings,
        "time": elapsed_time,
    }
