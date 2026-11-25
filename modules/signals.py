"""
Modulo de Senales Basicas y Funciones de Evaluacion.
Soporta evaluacion numerica (NumPy) y simbolica (SymPy), incluyendo impulsos δ(t) y dobletes δ'(t).
"""

import numpy as np
import sympy as sp

t_sym = sp.Symbol("t", real=True)
tau_sym = sp.Symbol("tau", real=True)


class BasicSignal:
    """Clase base y creador de senales continuas basico."""

    @staticmethod
    def step(t_val, t0=0.0, amplitude=1.0):
        """Escalon unitario u(t - t0) * amplitude"""
        if isinstance(t_val, np.ndarray):
            return amplitude * np.where(t_val >= t0, 1.0, 0.0)
        return amplitude * 1.0 if t_val >= t0 else 0.0

    @staticmethod
    def ramp(t_val, t0=0.0, slope=1.0):
        """Rampa r(t - t0) = (t - t0)*u(t - t0) * slope"""
        if isinstance(t_val, np.ndarray):
            return slope * np.maximum(0.0, t_val - t0)
        return slope * max(0.0, t_val - t0)

    @staticmethod
    def rect(t_val, center=0.0, width=1.0, amplitude=1.0):
        """Pulso rectangular rect((t - center)/width) * amplitude"""
        half_w = width / 2.0
        if isinstance(t_val, np.ndarray):
            return amplitude * np.where(np.abs(t_val - center) <= half_w, 1.0, 0.0)
        return amplitude * 1.0 if abs(t_val - center) <= half_w else 0.0

    @staticmethod
    def tri(t_val, center=0.0, width=1.0, amplitude=1.0):
        """Pulso triangular tri((t - center)/width) * amplitude"""
        half_w = width / 2.0
        if isinstance(t_val, np.ndarray):
            val = amplitude * (1.0 - np.abs(t_val - center) / half_w)
            return np.where(np.abs(t_val - center) <= half_w, val, 0.0)
        if abs(t_val - center) <= half_w:
            return amplitude * (1.0 - abs(t_val - center) / half_w)
        return 0.0

    @staticmethod
    def impulse(t_val, t0=0.0, amplitude=1.0, eps=0.05):
        """Aproximacion numerica de impulso delta(t - t0) para visualizacion."""
        if isinstance(t_val, np.ndarray):
            return amplitude * np.where(np.abs(t_val - t0) < eps, 1.0 / (2 * eps), 0.0)
        return amplitude * (1.0 / (2 * eps)) if abs(t_val - t0) < eps else 0.0

    @staticmethod
    def doublet(t_val, t0=0.0, amplitude=1.0, eps=0.08):
        """Aproximacion numerica del doblete unitario delta'(t - t0) (derivada del impulso)."""
        if isinstance(t_val, np.ndarray):
            y = np.zeros_like(t_val, dtype=float)
            mask_pos = (t_val >= t0 - eps) & (t_val < t0)
            mask_neg = (t_val >= t0) & (t_val <= t0 + eps)
            y[mask_pos] = amplitude / (eps ** 2)
            y[mask_neg] = -amplitude / (eps ** 2)
            return y
        if t0 - eps <= t_val < t0:
            return amplitude / (eps ** 2)
        if t0 <= t_val <= t0 + eps:
            return -amplitude / (eps ** 2)
        return 0.0

    @staticmethod
    def exponential(t_val, t0=0.0, alpha=-1.0, amplitude=1.0):
        """Exponencial A * exp(alpha * (t - t0)) * u(t - t0)"""
        if isinstance(t_val, np.ndarray):
            u = np.where(t_val >= t0, 1.0, 0.0)
            return amplitude * np.exp(alpha * (t_val - t0)) * u
        return amplitude * np.exp(alpha * (t_val - t0)) if t_val >= t0 else 0.0

    @staticmethod
    def sine(t_val, freq=1.0, phase=0.0, amplitude=1.0):
        """Senoidal A * sin(2*pi*freq*t + phase)"""
        return amplitude * np.sin(2 * np.pi * freq * t_val + phase)


def parse_piecewise_signal(pieces, t_arr):
    """
    Evalua una senal definida por tramos expresados como pares (expr_str, (t_min, t_max)).
    """
    y = np.zeros_like(t_arr, dtype=float)
    t_sp = sp.Symbol("t", real=True)

    for expr_str, (t_min, t_max) in pieces:
        mask = (t_arr >= t_min) & (t_arr <= t_max)
        if np.any(mask):
            try:
                expr = sp.sympify(expr_str)
                f_num = sp.lambdify(t_sp, expr, modules=["numpy", "math"])
                vals = f_num(t_arr[mask])
                if np.isscalar(vals):
                    y[mask] = float(vals)
                else:
                    y[mask] = vals
            except Exception:
                y[mask] = 0.0
    return y


def compute_piecewise_derivative(pieces):
    """
    Calcula analiticamente la derivada de una senal por tramos x'(t) = dx/dt,
    identificando derivadas continuas en tramos e impulsos delta(t - t0) en saltos de discontinuidad.
    Retorna (deriv_pieces, impulses_list, latex_str)
    """
    t_sp = sp.Symbol("t", real=True)
    deriv_pieces = []
    impulses = []

    for expr_str, (t_min, t_max) in pieces:
        try:
            expr = sp.sympify(expr_str)
            syms = [s for s in expr.free_symbols if s.name == "t"]
            t_var = syms[0] if syms else t_sp
            dexpr = sp.diff(expr, t_var)
            deriv_pieces.append((str(dexpr), (t_min, t_max)))
        except Exception:
            deriv_pieces.append(("0", (t_min, t_max)))

    points = set()
    for _, (t_min, t_max) in pieces:
        points.add(t_min)
        points.add(t_max)
    sorted_points = sorted(list(points))

    for tp in sorted_points:
        val_left = 0.0
        for expr_str, (t_min, t_max) in pieces:
            if abs(t_max - tp) < 1e-6:
                try:
                    expr = sp.sympify(expr_str)
                    syms = [s for s in expr.free_symbols if s.name == "t"]
                    t_var = syms[0] if syms else t_sp
                    val_left = float(expr.subs(t_var, tp))
                except Exception:
                    pass

        val_right = 0.0
        for expr_str, (t_min, t_max) in pieces:
            if abs(t_min - tp) < 1e-6:
                try:
                    expr = sp.sympify(expr_str)
                    syms = [s for s in expr.free_symbols if s.name == "t"]
                    t_var = syms[0] if syms else t_sp
                    val_right = float(expr.subs(t_var, tp))
                except Exception:
                    pass

        jump = val_right - val_left
        if abs(jump) > 1e-5:
            impulses.append((tp, jump))

    latex_parts = []
    pw_args = []
    for dexpr_str, (t_min, t_max) in deriv_pieces:
        try:
            expr = sp.sympify(dexpr_str)
            cond = sp.And(t_sp > t_min, t_sp < t_max)
            pw_args.append((expr, cond))
        except Exception:
            pass
    pw_args.append((0, True))
    pw_sym = sp.Piecewise(*pw_args)
    pw_latex = sp.latex(pw_sym)
    pw_latex = pw_latex.replace("\\text{for}\\:", "\\text{para }").replace("\\text{otherwise}", "\\text{en otro caso}")
    latex_parts.append(pw_latex)

    for tp, amp in impulses:
        sign = "+" if amp > 0 else "-"
        abs_amp = abs(amp)
        amp_str = f"{abs_amp:.2g}" if abs_amp != 1 else ""
        t_shift = f" - {tp:.2g}" if tp > 0 else (f" + {abs(tp):.2g}" if tp < 0 else "")
        latex_parts.append(f"{sign} {amp_str} \\delta(t{t_shift})")

    full_latex = " \\quad ".join(latex_parts)
    return deriv_pieces, impulses, full_latex


def step_by_step_derivative(pieces):
    """
    Genera el desglose paso a paso para derivar una señal por tramos:
    1) Derivada en cada subintervalo abierto.
    2) Evaluación de saltos Δx = x(t_p^+) - x(t_p^-) en puntos frontera.
    3) Construcción de impulsos Δx · δ(t - t_p).
    """
    t_sp = sp.Symbol("t", real=True)
    segment_steps = []

    for expr_str, (t_min, t_max) in pieces:
        expr = sp.sympify(expr_str)
        syms = [s for s in expr.free_symbols if s.name == "t"]
        t_var = syms[0] if syms else t_sp
        dexpr = sp.diff(expr, t_var)
        segment_steps.append({
            "t_min_latex": sp.latex(sp.Rational(str(t_min))),
            "t_max_latex": sp.latex(sp.Rational(str(t_max))),
            "expr_latex": sp.latex(expr),
            "dexpr_latex": sp.latex(dexpr),
        })

    points = sorted({pt for _, (a, b) in pieces for pt in (a, b)})
    jump_steps = []

    for tp in points:
        val_left = 0.0
        for expr_str, (t_min, t_max) in pieces:
            if abs(t_max - tp) < 1e-6:
                expr = sp.sympify(expr_str)
                syms = [s for s in expr.free_symbols if s.name == "t"]
                t_var = syms[0] if syms else t_sp
                val_left = float(expr.subs(t_var, tp))

        val_right = 0.0
        for expr_str, (t_min, t_max) in pieces:
            if abs(t_min - tp) < 1e-6:
                expr = sp.sympify(expr_str)
                syms = [s for s in expr.free_symbols if s.name == "t"]
                t_var = syms[0] if syms else t_sp
                val_right = float(expr.subs(t_var, tp))

        jump = val_right - val_left
        jump_steps.append({
            "tp_latex": sp.latex(sp.Rational(str(tp))),
            "left_val": f"{val_left:.4g}",
            "right_val": f"{val_right:.4g}",
            "jump": jump,
            "jump_latex": sp.latex(sp.Rational(str(round(jump, 6)))),
        })

    return segment_steps, jump_steps


def piecewise_to_sympy(pieces):
    """Convierte una lista de tramos en una expresion sp.Piecewise de SymPy."""
    t_sp = sp.Symbol("t", real=True)
    args = []
    for expr_str, (t_min, t_max) in pieces:
        try:
            expr = sp.sympify(expr_str)
            cond = sp.And(t_sp >= t_min, t_sp <= t_max)
            args.append((expr, cond))
        except Exception:
            pass
    args.append((0, True))
    return sp.Piecewise(*args)
