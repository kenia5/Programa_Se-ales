"""
Modulo de Senales Basicas y Funciones de Evaluacion.
Soporta evaluacion numerica (NumPy) y simbolica (SymPy), incluyendo impulsos δ(t) y dobletes δ'(t).
"""

import numpy as np
import sympy as sp

# Variable simbolica global t
t_sym = sp.Symbol('t', real=True)
tau_sym = sp.Symbol('tau', real=True)


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
        elif t0 <= t_val <= t0 + eps:
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


def get_piecewise_latex_definition(
    signal_type,
    amp=1.0,
    t0=0.0,
    width=2.0,
    alpha=-1.0,
    freq=0.5,
    phase=0.0,
    pieces=None,
):
    r"""
    Genera la expresión matemática explícita por tramos en notación LaTeX
    priorizando intervalos claros (ej. \begin{cases} ... \end{cases}).
    """
    if "Triángulo x1(t)" in signal_type:
        return (
            r"x_1(t) = \begin{cases} "
            r"t+2, & \text{para } -2 \le t \le 0 \\[4pt] "
            r"-t+2, & \text{para } 0 \le t \le 2 \\[4pt] "
            r"0, & \text{en otro caso} \end{cases}"
        )
    elif "Rectangular" in signal_type:
        t_left = t0 - width / 2.0
        t_right = t0 + width / 2.0
        amp_str = f"{amp:g}"
        return (
            f"x(t) = \\begin{{cases}} {amp_str}, & \\text{{para }} {t_left:g} \\le t \\le {t_right:g} \\\\[4pt] "
            f"0, & \\text{{en otro caso}} \\end{{cases}}"
        )
    elif "Triangular" in signal_type:
        t_left = t0 - width / 2.0
        t_mid = t0
        t_right = t0 + width / 2.0
        h_w = width / 2.0
        m = amp / h_w
        e1_str = f"{m:g}(t - {t_left:g})" if t_left != 0 else (f"{m:g}t" if m != 1 else "t")
        e2_str = f"{-m:g}(t - {t_right:g})"
        return (
            f"x(t) = \\begin{{cases}} {e1_str}, & \\text{{para }} {t_left:g} \\le t \\le {t_mid:g} \\\\[4pt] "
            f"{e2_str}, & \\text{{para }} {t_mid:g} < t \\le {t_right:g} \\\\[4pt] "
            f"0, & \\text{{en otro caso}} \\end{{cases}}"
        )
    elif "Escalón" in signal_type:
        amp_str = f"{amp:g}"
        return (
            f"x(t) = \\begin{{cases}} {amp_str}, & \\text{{para }} t \\ge {t0:g} \\\\[4pt] "
            f"0, & \\text{{para }} t < {t0:g} \\end{{cases}}"
        )
    elif "Rampa" in signal_type:
        amp_str = f"{amp:g}"
        ramp_expr = f"{amp_str}(t - {t0:g})" if t0 != 0 else (f"{amp_str}t" if amp != 1 else "t")
        return (
            f"x(t) = \\begin{{cases}} {ramp_expr}, & \\text{{para }} t \\ge {t0:g} \\\\[4pt] "
            f"0, & \\text{{para }} t < {t0:g} \\end{{cases}}"
        )
    elif "Exponencial" in signal_type:
        amp_str = f"{amp:g}" if amp != 1 else ""
        t_shift = f"(t - {t0:g})" if t0 != 0 else "t"
        exp_str = f"{amp_str} e^{{{alpha:g} {t_shift}}}"
        return (
            f"x(t) = \\begin{{cases}} {exp_str}, & \\text{{para }} t \\ge {t0:g} \\\\[4pt] "
            f"0, & \\text{{para }} t < {t0:g} \\end{{cases}}"
        )
    elif "Senoidal" in signal_type:
        amp_str = f"{amp:g}"
        return (
            f"x(t) = {amp_str} \\sin(2\\pi \\cdot {freq:g} t + {phase:g}) "
            f"\\quad (\\text{{Señal continua periódica para todo }} t \\in \\mathbb{{R}})"
        )
    elif "Impulso" in signal_type:
        amp_str = f"{amp:g}" if amp != 1 else ""
        return (
            f"x(t) = {amp_str} \\delta(t - {t0:g}) "
            f"\\quad \\left(\\text{{Impulso unitario: }} \\int_{{-\\infty}}^{{\\infty}} \\delta(t - {t0:g}) dt = 1\\right)"
        )
    elif "Doblete" in signal_type:
        amp_str = f"{amp:g}" if amp != 1 else ""
        return (
            f"x(t) = {amp_str} \\delta'(t - {t0:g}) "
            f"\\quad \\left(\\text{{Doblete unitario: }} \\frac{{d\\delta(t - {t0:g})}}{{dt}}\\right)"
        )
    else:
        if pieces:
            t_sp = sp.Symbol('t', real=True)
            args = []
            for expr_str, (t_min, t_max) in pieces:
                try:
                    expr = sp.sympify(expr_str)
                    cond = sp.And(t_sp >= t_min, t_sp <= t_max)
                    args.append((expr, cond))
                except Exception:
                    pass
            args.append((0, True))
            pw = sp.Piecewise(*args)
            pw_latex = (
                sp.latex(pw)
                .replace(r"\text{for}\:", r"\text{para }")
                .replace(r"\text{otherwise}", r"\text{en otro caso}")
            )
            return f"x(t) = {pw_latex}"
        return r"x(t) = \text{Señal personalizada definida por tramos}"


def parse_piecewise_signal(pieces, t_arr):
    """
    Evalua una senal definida por tramos expresados como pares (expr_str, (t_min, t_max)).
    Ejemplo de pieces: [("t + 1", (-1, 0)), ("1", (0, 2)), ("-t + 3", (2, 3))]
    """
    y = np.zeros_like(t_arr, dtype=float)
    t_sp = sp.Symbol('t', real=True)

    for expr_str, (t_min, t_max) in pieces:
        mask = (t_arr >= t_min) & (t_arr <= t_max)
        if np.any(mask):
            try:
                expr = sp.sympify(expr_str)
                f_num = sp.lambdify(t_sp, expr, modules=['numpy', 'math'])
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
    t_sp = sp.Symbol('t', real=True)
    deriv_pieces = []
    impulses = []

    for expr_str, (t_min, t_max) in pieces:
        try:
            expr = sp.sympify(expr_str)
            syms = [s for s in expr.free_symbols if s.name == 't']
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
                    syms = [s for s in expr.free_symbols if s.name == 't']
                    t_var = syms[0] if syms else t_sp
                    val_left = float(expr.subs(t_var, tp))
                except Exception:
                    pass
        val_right = 0.0
        for expr_str, (t_min, t_max) in pieces:
            if abs(t_min - tp) < 1e-6:
                try:
                    expr = sp.sympify(expr_str)
                    syms = [s for s in expr.free_symbols if s.name == 't']
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

    pw_latex = (
        pw_latex.replace("\\text{for}\\:", "\\text{para }")
        .replace("\\text{otherwise}", "\\text{en otro caso}")
    )
    latex_parts.append(pw_latex)

    for tp, amp in impulses:
        sign = "+" if amp > 0 else "-"
        abs_amp = abs(amp)
        amp_str = f"{abs_amp:.2g}" if abs_amp != 1 else ""
        t_shift = f" - {tp:.2g}" if tp > 0 else (f" + {abs(tp):.2g}" if tp < 0 else "")
        latex_parts.append(f"{sign} {amp_str} \\delta(t{t_shift})")

    full_latex = " \\quad ".join(latex_parts)
    return deriv_pieces, impulses, full_latex


def piecewise_to_sympy(pieces):
    """Convierte una lista de tramos en una expresion sp.Piecewise de SymPy."""
    t_sp = sp.Symbol('t', real=True)
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


def _num_latex(v):
    """LaTeX limpio de un número: entero, fracción o decimal."""
    try:
        return sp.latex(sp.nsimplify(v, rational=True))
    except Exception:
        return str(v)


def step_by_step_derivative(pieces):
    """
    Paso a paso de la derivada de una señal por tramos.
    Retorna (seg_steps, jump_steps).
    """
    parsed = []
    seg_steps = []
    for expr_str, (t_min, t_max) in pieces:
        try:
            expr = sp.nsimplify(sp.sympify(expr_str, locals={'t': t_sym}), rational=True)
        except Exception:
            continue
        a = sp.nsimplify(t_min, rational=True)
        b = sp.nsimplify(t_max, rational=True)
        parsed.append((expr, a, b))
        dexpr = sp.simplify(sp.diff(expr, t_sym))
        seg_steps.append({
            't_min_latex': _num_latex(a),
            't_max_latex': _num_latex(b),
            'expr_latex': sp.latex(expr),
            'dexpr_latex': sp.latex(dexpr),
        })

    puntos = sorted({p for _, a, b in parsed for p in (a, b)}, key=float)
    jump_steps = []
    for tp in puntos:
        left = sp.Integer(0)
        right = sp.Integer(0)
        for expr, a, b in parsed:
            if b == tp:
                left = sp.nsimplify(expr.subs(t_sym, tp))
            if a == tp:
                right = sp.nsimplify(expr.subs(t_sym, tp))
        jump = sp.simplify(right - left)
        jump_steps.append({
            'tp_latex': _num_latex(tp),
            'left_val': sp.latex(left),
            'right_val': sp.latex(right),
            'jump_latex': sp.latex(jump),
            'jump': float(jump),
        })
    return seg_steps, jump_steps