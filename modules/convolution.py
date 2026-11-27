
"""
Módulo de Convolución Continua.
Incluye cálculo numérico y desarrollo simbólico paso a paso.
"""

import numpy as np
import sympy as sp


def _trapz(y, x):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def continuous_convolution_at_t(x_func, h_func, t_val, tau_arr):
    """
    Calcula x(τ), h(t-τ), su producto y la integral numérica.
    """
    x_tau = x_func(tau_arr)
    h_flipped_shifted = h_func(t_val - tau_arr)
    product_tau = x_tau * h_flipped_shifted
    y_at_t = _trapz(product_tau, tau_arr)

    return x_tau, h_flipped_shifted, product_tau, y_at_t


def compute_full_numerical_convolution(
    x_func,
    h_func,
    t_eval_arr,
    tau_arr,
):
    """
    Calcula la curva completa de convolución y(t).
    """
    y_full = np.zeros_like(t_eval_arr, dtype=float)

    x_tau = x_func(tau_arr)

    for i, t_val in enumerate(t_eval_arr):
        h_tau = h_func(t_val - tau_arr)
        y_full[i] = _trapz(x_tau * h_tau, tau_arr)

    return y_full


def _sympify_signal(expression):
    t = sp.Symbol("t", real=True)

    return sp.sympify(
        expression,
        locals={
            "t": t,
            "sin": sp.sin,
            "cos": sp.cos,
            "exp": sp.exp,
            "sqrt": sp.sqrt,
            "pi": sp.pi,
        },
    )


def symbolic_piecewise_convolution_steps(x_pieces, h_pieces):
    """
    Desarrolla la convolución por intervalos.

    Para cada intervalo de t muestra:
    - Tramos que se solapan.
    - x(τ).
    - h(t-τ).
    - Producto de ambas señales.
    - Límites de integración.
    - Antiderivada.
    - Resultado de cada integral.
    - Suma de contribuciones.
    """

    t = sp.Symbol("t", real=True)
    tau = sp.Symbol("tau", real=True)

    breakpoints = set()

    try:
        for _, (a, b) in x_pieces:
            for _, (c, d) in h_pieces:

                a = sp.Rational(str(a))
                b = sp.Rational(str(b))
                c = sp.Rational(str(c))
                d = sp.Rational(str(d))

                breakpoints.update([
                    a + c,
                    a + d,
                    b + c,
                    b + d,
                ])

        sorted_points = sorted(breakpoints)

        all_steps = []
        merged_results = []

        for index in range(len(sorted_points) - 1):

            t1 = sorted_points[index]
            t2 = sorted_points[index + 1]
            midpoint = (t1 + t2) / 2

            interval_steps = []
            interval_total = sp.Integer(0)

            for x_str, (a, b) in x_pieces:
                for h_str, (c, d) in h_pieces:

                    a = sp.Rational(str(a))
                    b = sp.Rational(str(b))
                    c = sp.Rational(str(c))
                    d = sp.Rational(str(d))

                    # Límites geométricos del solapamiento:
                    # τ pertenece a [a,b] y también a [t-d,t-c].
                    lower_candidate_1 = a
                    lower_candidate_2 = t - d

                    upper_candidate_1 = b
                    upper_candidate_2 = t - c

                    lower_mid = max(
                        lower_candidate_1,
                        lower_candidate_2.subs(t, midpoint),
                    )

                    upper_mid = min(
                        upper_candidate_1,
                        upper_candidate_2.subs(t, midpoint),
                    )

                    if upper_mid <= lower_mid:
                        continue

                    lower = (
                        lower_candidate_1
                        if lower_candidate_1 >= lower_candidate_2.subs(
                            t, midpoint
                        )
                        else lower_candidate_2
                    )

                    upper = (
                        upper_candidate_1
                        if upper_candidate_1 <= upper_candidate_2.subs(
                            t, midpoint
                        )
                        else upper_candidate_2
                    )

                    x_expr = _sympify_signal(x_str).subs(
                        sp.Symbol("t", real=True), tau
                    )

                    h_expr = _sympify_signal(h_str).subs(
                        sp.Symbol("t", real=True), t - tau
                    )

                    integrand = sp.expand(x_expr * h_expr)

                    antiderivative = sp.integrate(integrand, tau)

                    evaluated = sp.simplify(
                        antiderivative.subs(tau, upper)
                        - antiderivative.subs(tau, lower)
                    )

                    evaluated = sp.expand(evaluated)

                    interval_total += evaluated

                    interval_steps.append({
                        "x_expression": x_expr,
                        "h_expression": h_expr,
                        "integrand": integrand,
                        "lower": lower,
                        "upper": upper,
                        "antiderivative": antiderivative,
                        "result": evaluated,
                    })

            interval_total = sp.simplify(interval_total)

            if interval_steps:
                all_steps.append({
                    "interval": (t1, t2),
                    "steps": interval_steps,
                    "total": interval_total,
                })

                if interval_total != 0:
                    merged_results.append(
                        (interval_total, (t1, t2))
                    )

        # Unir intervalos consecutivos con la misma expresión.
        merged = []

        for expr, (t1, t2) in merged_results:

            if (
                merged
                and sp.simplify(merged[-1][0] - expr) == 0
                and merged[-1][1][1] == t1
            ):
                merged[-1] = (
                    merged[-1][0],
                    (merged[-1][1][0], t2),
                )
            else:
                merged.append((expr, (t1, t2)))

        return {
            "interval_steps": all_steps,
            "merged": merged,
        }

    except Exception:
        return None


def symbolic_piecewise_convolution(x_pieces, h_pieces):
    """
    Mantiene la interfaz anterior:
    devuelve LaTeX y los intervalos resultantes.
    """

    detailed = symbolic_piecewise_convolution_steps(
        x_pieces,
        h_pieces,
    )

    if detailed is None:
        raise ValueError(
            "No se pudo calcular la convolución simbólica."
        )

    t = sp.Symbol("t", real=True)
    merged = detailed["merged"]

    lines = []

    for expr, (t1, t2) in merged:
        lines.append(
            f"{sp.latex(expr)}, & "
            f"\\text{{para }} {sp.latex(t1)} "
            f"\\le t \\le {sp.latex(t2)}"
        )

    lines.append("0, & \\text{en otro caso}")

    spanish_latex = (
        "\\begin{cases}\n"
        + " \\\\[6pt]\n".join(lines)
        + "\n\\end{cases}"
    )

    return spanish_latex, merged
