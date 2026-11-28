"""
Módulo de Análisis de Señales de Potencia y Energía.
Alineado con el Capítulo 3 (Conceptos Básicos) de Señales y Sistemas Continuos.
Incluye cálculo numérico, clasificación y desarrollo simbólico paso a paso por intervalos.
"""

import numpy as np
import sympy as sp


def _trapz(y, x):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def compute_energy_power_metrics(signal_func, t_max=15.0, num_points=2500):
    """
    Calcula numéricamente la evolución de la energía acumulada E(T)
    y la potencia promedio P(T) en ventanas simétricas [-T, T].
    """
    t_max = max(float(t_max), 2.0)
    T_axis = np.linspace(0.5, t_max, 120)
    E_T_curve = np.zeros_like(T_axis, dtype=float)
    P_T_curve = np.zeros_like(T_axis, dtype=float)

    for i, T in enumerate(T_axis):
        t_arr = np.linspace(-T, T, num_points)
        y_val = np.asarray(signal_func(t_arr), dtype=complex)

        E_curr = _trapz(np.abs(y_val) ** 2, t_arr)
        P_curr = E_curr / (2.0 * T)

        E_T_curve[i] = E_curr
        P_T_curve[i] = P_curr

    t_full = np.linspace(-t_max, t_max, num_points)
    y_full = np.asarray(signal_func(t_full), dtype=complex)

    Ex_total = float(E_T_curve[-1])
    Px_avg = float(P_T_curve[-1])

    mean_val = float(np.real(_trapz(y_full, t_full) / (2.0 * t_max)))
    rms_val = float(np.sqrt(max(Px_avg, 0.0)))

    return {
        "Ex": Ex_total,
        "Px": Px_avg,
        "mean": mean_val,
        "rms": rms_val,
        "T_axis": T_axis,
        "E_T_curve": E_T_curve,
        "P_T_curve": P_T_curve,
    }


def classify_energy_power_signal(
    Ex_val,
    Px_val,
    T_axis,
    E_T_curve,
    P_T_curve,
    is_periodic=False,
):
    """
    Clasifica la señal según el comportamiento de E(T) y P(T).
    """
    idx_start = max(1, int(0.75 * len(T_axis)))
    dT = float(T_axis[-1] - T_axis[idx_start])
    slope_E = float((E_T_curve[-1] - E_T_curve[idx_start]) / max(dT, 1e-6))
    slope_P = float((P_T_curve[-1] - P_T_curve[idx_start]) / max(dT, 1e-6))

    energy_saturates = (slope_E < 1e-2) and (Px_val < 1e-2)

    if Ex_val == 0.0 and Px_val == 0.0:
        category = "Señal Nula (Sin Energía ni Potencia útil)"
        ex_latex = r"E_x = \int_{-\infty}^{\infty} |x(t)|^2\,dt = 0"
        px_latex = r"P_x = \lim_{T\to\infty} \frac{1}{T}\int_{-T/2}^{T/2} |x(t)|^2\,dt = 0"
        rms_latex = r"X_{\mathrm{RMS}} = \sqrt{P_x} = 0"
        justification = (
            "La señal es idénticamente nula (o cero en todo el intervalo "
            "evaluado), por lo que su energía total y su potencia promedio "
            "son iguales a cero."
        )
    elif energy_saturates and not is_periodic:
        category = "Señal de Energía"
        ex_latex = (
            r"E_x = \int_{-\infty}^{\infty} |x(t)|^2\,dt \approx "
            + f"{Ex_val:.4f}"
            + r" < \infty"
        )
        px_latex = r"P_x = \lim_{T\to\infty} \frac{1}{T}\int_{-T/2}^{T/2} |x(t)|^2\,dt = 0"
        rms_latex = r"X_{\mathrm{RMS}} = \sqrt{P_x} = 0"
        justification = (
            f"La energía acumulada converge a un valor finito "
            f"(E_x ≈ {Ex_val:.4f}), mientras que la potencia promedio "
            "tiende a 0 al extender el tiempo de observación. "
            "Es el caso típico de pulsos de duración finita o señales "
            "exponenciales decrecientes."
        )
    elif abs(slope_P) < 0.03 and Px_val >= 1e-4:
        category = "Señal de Potencia"
        rms_num = float(np.sqrt(Px_val))
        ex_latex = r"E_x = \int_{-\infty}^{\infty} |x(t)|^2\,dt \to \infty"
        px_latex = r"P_x = \langle |x(t)|^2 \rangle \approx " + f"{Px_val:.4f}" + r" < \infty"
        rms_latex = r"X_{\mathrm{RMS}} = \sqrt{P_x} \approx " + f"{rms_num:.4f}"
        justification = (
            "La energía total crece sin límite al aumentar T, "
            "pero la potencia promedio se estabiliza en un valor "
            f"finito (P_x ≈ {Px_val:.4f}), como ocurre en señales "
            "periódicas o escalones."
        )
    else:
        category = "Ni de Energía ni de Potencia"
        ex_latex = r"E_x = \int_{-\infty}^{\infty} |x(t)|^2\,dt \to \infty"
        px_latex = r"P_x = \lim_{T\to\infty} \frac{1}{T}\int_{-T/2}^{T/2} |x(t)|^2\,dt \to \infty"
        rms_latex = r"X_{\mathrm{RMS}} \to \infty"
        justification = (
            "Tanto la energía acumulada como la potencia promedio "
            "crecen sin estabilizarse al aumentar la ventana de tiempo "
            "(por ejemplo, en señales crecientes como rampas o "
            "exponenciales crecientes)."
        )

    return {
        "category": category,
        "ex_latex": ex_latex,
        "px_latex": px_latex,
        "rms_latex": rms_latex,
        "justification": justification,
    }


def symbolic_energy_power_steps(pieces, window_min, window_max):
    """
    Calcula simbólicamente paso a paso la energía en cada tramo
    recortado a la ventana [window_min, window_max].
    """
    t = sp.Symbol("t", real=True)
    total_energy = sp.Integer(0)
    steps = []

    try:
        w_min = float(window_min)
        w_max = float(window_max)
        if w_max <= w_min:
            return None, None

        for expr_str, (t_min, t_max) in pieces:
            lower = max(float(t_min), w_min)
            upper = min(float(t_max), w_max)

            if upper <= lower:
                continue

            expr = sp.sympify(
                expr_str,
                locals={
                    "t": t,
                    "sin": sp.sin,
                    "cos": sp.cos,
                    "exp": sp.exp,
                    "sqrt": sp.sqrt,
                    "pi": sp.pi,
                },
            )

            a = sp.nsimplify(lower, rational=True)
            b = sp.nsimplify(upper, rational=True)

            integrand = sp.simplify(sp.Abs(expr) ** 2)
            antiderivative = sp.integrate(integrand, t)
            val_upper = sp.simplify(antiderivative.subs(t, b))
            val_lower = sp.simplify(antiderivative.subs(t, a))
            result = sp.simplify(val_upper - val_lower)

            steps.append({
                "expression": expr,
                "integrand": integrand,
                "lower": a,
                "upper": b,
                "antiderivative": antiderivative,
                "val_upper": val_upper,
                "val_lower": val_lower,
                "result": result,
            })

            total_energy += result

        return steps, sp.simplify(total_energy)

    except Exception:
        return None, None


def symbolic_energy_power_development(pieces, window_min, window_max):
    """
    Desarrolla energía, valor medio, potencia promedio y valor RMS
    en la ventana [window_min, window_max], tramo por tramo.
    """
    t = sp.Symbol("t", real=True)
    w_min_sym = sp.nsimplify(float(window_min), rational=True)
    w_max_sym = sp.nsimplify(float(window_max), rational=True)
    duration = sp.simplify(w_max_sym - w_min_sym)

    if duration <= 0:
        return None

    energy_steps, total_energy = symbolic_energy_power_steps(pieces, window_min, window_max)
    if energy_steps is None or total_energy is None:
        return None

    mean_integral_sum = sp.Integer(0)
    mean_steps = []
    for index, item in enumerate(energy_steps, start=1):
        a = item["lower"]
        b = item["upper"]
        expr = item["expression"]
        mean_anti = sp.integrate(expr, t)
        mean_val_seg = sp.simplify(mean_anti.subs(t, b) - mean_anti.subs(t, a))
        mean_integral_sum += mean_val_seg
        mean_steps.append({
            "index": index,
            "expression": expr,
            "lower": a,
            "upper": b,
            "antiderivative": mean_anti,
            "integral_val": mean_val_seg,
            "mean_contribution": sp.simplify(mean_val_seg / duration),
        })

    average_mean = sp.simplify(mean_integral_sum / duration)
    average_power = sp.simplify(total_energy / duration)
    rms_value = sp.simplify(sp.sqrt(average_power))

    power_steps = []
    for index, item in enumerate(energy_steps, start=1):
        power_steps.append({
            "index": index,
            "lower": item["lower"],
            "upper": item["upper"],
            "energy_result": item["result"],
            "duration": duration,
            "power_contribution": sp.simplify(item["result"] / duration),
        })

    return {
        "energy_steps": energy_steps,
        "total_energy": total_energy,
        "mean_steps": mean_steps,
        "mean_integral_sum": sp.simplify(mean_integral_sum),
        "average_mean": average_mean,
        "duration": duration,
        "average_power": average_power,
        "rms": rms_value,
        "power_steps": power_steps,
        "window_min": w_min_sym,
        "window_max": w_max_sym,
    }


def symbolic_energy_power_from_pieces(pieces):
    """
    Calcula simbólicamente la energía total de una señal por tramos.
    """
    t = sp.Symbol("t", real=True)
    total_E = sp.Integer(0)

    try:
        for expr_str, (t_min, t_max) in pieces:
            expr = sp.sympify(expr_str, locals={"t": t})
            a = sp.Rational(str(t_min))
            b = sp.Rational(str(t_max))
            if b > a:
                total_E += sp.integrate(sp.Abs(expr) ** 2, (t, a, b))
        return sp.simplify(total_E)
    except Exception:
        return None
