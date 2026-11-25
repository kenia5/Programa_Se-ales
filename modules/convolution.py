"""
Modulo de Convolucion Continua Numerica e Interactiva con Expresion Analitica por Tramos (SymPy).
"""

import numpy as np
import scipy.integrate as integrate
import sympy as sp


def _trapz(y, x):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def continuous_convolution_at_t(x_func, h_func, t_val, tau_arr):
    """
    Calcula los componentes para un t de convolucion especifico:
    - x(tau)
    - h(t - tau)
    - prod(tau) = x(tau) * h(t - tau)
    - y(t) = integral x(tau) * h(t - tau) dtau
    """
    x_tau = x_func(tau_arr)
    h_flipped_shifted = h_func(t_val - tau_arr)
    product_tau = x_tau * h_flipped_shifted
    y_at_t = _trapz(product_tau, tau_arr)
    return x_tau, h_flipped_shifted, product_tau, y_at_t


def compute_full_numerical_convolution(x_func, h_func, t_eval_arr, tau_arr):
    """
    Calcula la curva completa de convolucion y(t) para todos los valores en t_eval_arr.
    """
    y_full = np.zeros_like(t_eval_arr, dtype=float)
    for i, t_val in enumerate(t_eval_arr):
        x_tau = x_func(tau_arr)
        h_tau = h_func(t_val - tau_arr)
        y_full[i] = _trapz(x_tau * h_tau, tau_arr)
    return y_full


def resolve_limits(a, b, c, d, t1, t2):
    t_sp = sp.Symbol("t")
    t_mid = (t1 + t2) / 2

    if (t_mid - d) < a:
        tau_low = a
    else:
        tau_low = t_sp - d

    if (t_mid - c) > b:
        tau_high = b
    else:
        tau_high = t_sp - c

    return tau_low, tau_high


def step_by_step_convolution(x_pieces, h_pieces):
    """
    Genera una lista de pasos analiticos detallados por intervalo de t
    a partir de los tramos de x(t) y h(t).
    """
    t_sp = sp.Symbol("t", real=True)
    tau_sp = sp.Symbol("tau", real=True)

    t_points = set()
    for _, (a, b) in x_pieces:
        for _, (c, d) in h_pieces:
            a_f, b_f, c_f, d_f = float(a), float(b), float(c), float(d)
            t_points.update([
                sp.Rational(str(a_f + c_f)),
                sp.Rational(str(a_f + d_f)),
                sp.Rational(str(b_f + c_f)),
                sp.Rational(str(b_f + d_f)),
            ])

    t_sorted = sorted(list(t_points))
    steps = []

    for i in range(len(t_sorted) - 1):
        t1, t2 = t_sorted[i], t_sorted[i + 1]
        t_mid = (t1 + t2) / 2

        active_terms = []
        sub_integrals = []

        for x_str, (a, b) in x_pieces:
            for h_str, (c, d) in h_pieces:
                a_r, b_r = sp.Rational(str(a)), sp.Rational(str(b))
                c_r, d_r = sp.Rational(str(c)), sp.Rational(str(d))

                low_val = max(a_r, t_mid - d_r)
                high_val = min(b_r, t_mid - c_r)

                if high_val > low_val:
                    tau_low, tau_high = resolve_limits(a_r, b_r, c_r, d_r, t1, t2)
                    x_expr = sp.sympify(x_str, locals={"t": t_sp}).subs(t_sp, tau_sp)
                    h_expr = sp.sympify(h_str, locals={"t": t_sp}).subs(t_sp, t_sp - tau_sp)
                    prod = sp.expand(x_expr * h_expr)
                    ant = sp.integrate(prod, tau_sp)
                    val = sp.expand(ant.subs(tau_sp, tau_high) - ant.subs(tau_sp, tau_low))

                    sub_integrals.append(val)
                    active_terms.append({
                        "x_tau_latex": sp.latex(x_expr),
                        "h_shifted_latex": sp.latex(h_expr),
                        "low_latex": sp.latex(tau_low),
                        "high_latex": sp.latex(tau_high),
                        "prod_latex": sp.latex(prod),
                        "ant_latex": sp.latex(ant),
                        "val_latex": sp.latex(val),
                    })

        total_expr = sp.expand(sum(sub_integrals)) if sub_integrals else sp.S.Zero
        steps.append({
            "t1_latex": sp.latex(t1),
            "t2_latex": sp.latex(t2),
            "terms": active_terms,
            "total_latex": sp.latex(total_expr),
        })

    return steps


def symbolic_piecewise_convolution(x_pieces, h_pieces):
    """
    Obtiene la expresion analitica por intervalos (por tramos) de la convolucion (x * h)(t)
    usando integracion simbolica exacta en SymPy con formato en espanol.

    x_pieces: lista de tramos de x(t) [("expr_str", (a, b)), ...]
    h_pieces: lista de tramos de h(t) [("expr_str", (c, d)), ...]
    """
    t_sp = sp.Symbol("t")
    tau_sp = sp.Symbol("tau")

    t_points = set()
    for _, (a, b) in x_pieces:
        for _, (c, d) in h_pieces:
            a_f, b_f, c_f, d_f = float(a), float(b), float(c), float(d)
            t_points.update([
                sp.Rational(str(a_f + c_f)),
                sp.Rational(str(a_f + d_f)),
                sp.Rational(str(b_f + c_f)),
                sp.Rational(str(b_f + d_f)),
            ])

    t_sorted = sorted(list(t_points))
    results = []

    for i in range(len(t_sorted) - 1):
        t1, t2 = t_sorted[i], t_sorted[i + 1]
        t_mid = (t1 + t2) / 2

        sub_integrals = []
        for x_str, (a, b) in x_pieces:
            for h_str, (c, d) in h_pieces:
                a_r, b_r = sp.Rational(str(a)), sp.Rational(str(b))
                c_r, d_r = sp.Rational(str(c)), sp.Rational(str(d))

                low_val = max(a_r, t_mid - d_r)
                high_val = min(b_r, t_mid - c_r)

                if high_val > low_val:
                    tau_low, tau_high = resolve_limits(a_r, b_r, c_r, d_r, t1, t2)
                    x_expr = sp.sympify(x_str, locals={"t": t_sp}).subs(t_sp, tau_sp)
                    h_expr = sp.sympify(h_str, locals={"t": t_sp}).subs(t_sp, t_sp - tau_sp)
                    prod = sp.expand(x_expr * h_expr)
                    ant = sp.integrate(prod, tau_sp)
                    val = sp.expand(ant.subs(tau_sp, tau_high) - ant.subs(tau_sp, tau_low))
                    sub_integrals.append(val)

        if sub_integrals:
            total_interval_expr = sp.expand(sum(sub_integrals))
        else:
            total_interval_expr = sp.S.Zero

        if total_interval_expr != 0:
            results.append((total_interval_expr, (t1, t2)))

    merged = []
    for expr, (t1, t2) in results:
        if merged and sp.simplify(merged[-1][0] - expr) == 0:
            merged[-1] = (merged[-1][0], (merged[-1][1][0], t2))
        else:
            merged.append((expr, (t1, t2)))

    # Formatear a cadena LaTeX en espanol impecable
    lines = []
    for expr, (t1, t2) in merged:
        expr_latex = sp.latex(expr)
        lines.append(f"{expr_latex}, & \\text{{para }} {sp.latex(t1)} \\le t \\le {sp.latex(t2)}")
    lines.append("0, & \\text{en otro caso}")

    spanish_latex = "\\begin{cases}\n" + " \\\\[6pt]\n".join(lines) + "\n\\end{cases}"

    # SymPy Piecewise object
    pw_args = []
    for expr, (t1, t2) in merged:
        cond = sp.And(t_sp >= t1, t_sp <= t2)
        pw_args.append((expr, cond))
    pw_args.append((0, True))
    pw_sym = sp.Piecewise(*pw_args)

    return spanish_latex, merged