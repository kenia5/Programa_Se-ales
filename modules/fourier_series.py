"""
Módulo de Series de Fourier (SEF y STF) para señales periódicas por intervalos.
Alineado con el Capítulo 3 del syllabus de Señales y Sistemas Continuos (EPN).
"""

import numpy as np
import sympy as sp


def _parse_expr(expr_str, t_sym):
    expr = sp.sympify(
        expr_str,
        locals={
            "t": t_sym,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "exp": sp.exp,
            "sqrt": sp.sqrt,
            "pi": sp.pi,
            "abs": sp.Abs,
        },
    )
    return sp.nsimplify(expr, rational=True)


def compute_fourier_series_data(pieces, t0_start, t0_end, max_k=5, n_harmonics=7):
    """
    Calcula simbólicamente los coeficientes de la Serie Exponencial (X_k)
    y Trigonométrica (X_0, X_{C_k}, X_{S_k}) para una señal periódica definida
    por tramos en el periodo [t0_start, t0_end].
    """
    t = sp.Symbol("t", real=True)
    k = sp.Symbol("k", integer=True, nonzero=True)

    a_per = sp.nsimplify(float(t0_start), rational=True)
    b_per = sp.nsimplify(float(t0_end), rational=True)
    T0 = sp.simplify(b_per - a_per)

    if T0 <= 0:
        raise ValueError("El periodo T0 debe ser mayor que cero (t_fin > t_inicio).")

    omega0 = sp.simplify(2 * sp.pi / T0)
    f0 = sp.simplify(1 / T0)

    parsed_pieces = []
    for expr_str, (p_min, p_max) in pieces:
        seg_low = max(float(p_min), float(t0_start))
        seg_high = min(float(p_max), float(t0_end))
        if seg_high > seg_low:
            expr = _parse_expr(expr_str, t)
            parsed_pieces.append(
                (expr, sp.nsimplify(seg_low, rational=True), sp.nsimplify(seg_high, rational=True))
            )

    # Valor promedio / componente DC: X_0
    x0_integral_sum = sp.Integer(0)
    x0_steps = []
    for idx, (expr, seg_a, seg_b) in enumerate(parsed_pieces, start=1):
        val_seg = sp.simplify(sp.integrate(expr, (t, seg_a, seg_b)))
        x0_integral_sum += val_seg
        x0_steps.append({
            "index": idx,
            "expr": expr,
            "a": seg_a,
            "b": seg_b,
            "integral_val": val_seg,
        })
    X0 = sp.simplify(x0_integral_sum / T0)

    # Fórmula general simbólica de X_k para k != 0
    xk_general_terms = []
    for expr, seg_a, seg_b in parsed_pieces:
        term_int = sp.integrate(expr * sp.exp(-sp.I * k * omega0 * t), (t, seg_a, seg_b))
        xk_general_terms.append(sp.simplify(term_int / T0))
    Xk_general = sp.simplify(sum(xk_general_terms)) if xk_general_terms else sp.Integer(0)

    # Coeficientes puntuales para k en [-max_k, max_k]
    harmonics_dict = {}
    k_range = list(range(-int(max_k), int(max_k) + 1))

    for k_val in k_range:
        if k_val == 0:
            c_val = sp.simplify(X0)
        else:
            val_acc = sp.Integer(0)
            for expr, seg_a, seg_b in parsed_pieces:
                integrand_k = expr * sp.exp(-sp.I * k_val * omega0 * t)
                val_acc += sp.integrate(integrand_k, (t, seg_a, seg_b))
            c_val = sp.simplify(val_acc / T0)

        re_part = sp.simplify(sp.re(c_val))
        im_part = sp.simplify(sp.im(c_val))
        mag_val = sp.simplify(sp.Abs(c_val))
        phase_val = sp.simplify(sp.arg(c_val))

        harmonics_dict[k_val] = {
            "k": k_val,
            "Xk": c_val,
            "mag": mag_val,
            "phase": phase_val,
            "mag_num": float(sp.N(mag_val)),
            "phase_num": float(sp.N(phase_val)),
            "XC": sp.simplify(2 * re_part) if k_val > 0 else sp.Integer(0),
            "XS": sp.simplify(-2 * im_part) if k_val > 0 else sp.Integer(0),
        }

    # Potencia promedio por periodo y suma de Parseval hasta max_k
    power_integral_sum = sp.Integer(0)
    for expr, seg_a, seg_b in parsed_pieces:
        power_integral_sum += sp.integrate(sp.Abs(expr) ** 2, (t, seg_a, seg_b))
    Px_exact = sp.simplify(power_integral_sum / T0)
    Px_num = float(sp.N(Px_exact))

    parseval_partial = sp.simplify(
        sum(harmonics_dict[kv]["mag"] ** 2 for kv in k_range)
    )
    parseval_partial_num = float(sp.N(parseval_partial))

    # Síntesis numérica con N armónicos (k = -N ... N)
    n_harm = max(0, int(n_harmonics))
    t_sym_local = sp.Symbol("t", real=True)
    approx_expr = sp.simplify(X0)
    for kv in range(1, n_harm + 1):
        xc_k = harmonics_dict.get(kv, {}).get("XC", sp.Integer(0))
        xs_k = harmonics_dict.get(kv, {}).get("XS", sp.Integer(0))
        approx_expr += sp.simplify(xc_k * sp.cos(kv * omega0 * t_sym_local) + xs_k * sp.sin(kv * omega0 * t_sym_local))

    approx_func = sp.lambdify(t_sym_local, sp.re(approx_expr), modules=["numpy", "math"])

    return {
        "a": a_per,
        "b": b_per,
        "T0": T0,
        "T0_num": float(sp.N(T0)),
        "f0": f0,
        "omega0": omega0,
        "omega0_num": float(sp.N(omega0)),
        "X0": X0,
        "x0_steps": x0_steps,
        "Xk_general": Xk_general,
        "harmonics": harmonics_dict,
        "k_range": k_range,
        "Px_exact": Px_exact,
        "Px_num": Px_num,
        "parseval_partial": parseval_partial,
        "parseval_partial_num": parseval_partial_num,
        "approx_expr": approx_expr,
        "approx_func": approx_func,
    }


def evaluate_periodic_piecewise(pieces, t0_start, t0_end, t_arr):
    """
    Extiende periódicamente la señal definida en [t0_start, t0_end]
    sobre todo el arreglo de tiempo t_arr.
    """
    t_arr = np.asarray(t_arr, dtype=float)
    T0 = float(t0_end) - float(t0_start)
    if T0 <= 0:
        return np.zeros_like(t_arr, dtype=float)

    t_wrapped = ((t_arr - float(t0_start)) % T0) + float(t0_start)
    y = np.zeros_like(t_wrapped, dtype=float)
    t_sp = sp.Symbol("t", real=True)

    for expr_str, (p_min, p_max) in pieces:
        mask = (t_wrapped >= float(p_min) - 1e-12) & (t_wrapped <= float(p_max) + 1e-12)
        if np.any(mask):
            try:
                expr = _parse_expr(expr_str, t_sp)
                f_num = sp.lambdify(t_sp, expr, modules=["numpy", "math"])
                vals = f_num(t_wrapped[mask])
                y[mask] = float(vals) if np.isscalar(vals) else np.asarray(vals, dtype=float)
            except Exception:
                y[mask] = 0.0

    return y