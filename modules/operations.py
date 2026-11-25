"""
Modulo de Operaciones, Manipulacion y Propiedades de Senales.
"""

import numpy as np
import scipy.integrate as integrate
import sympy as sp
from .signals import BasicSignal


def _trapz(y, x):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def evaluate_transformed_signal(signal_func, t_arr, scale_a=1.0, shift_b=0.0, amp_A=1.0, offset_C=0.0):
    """
    Calcula la senal transformada y(t) = A * x(a*t - b) + C.
    Donde:
      a: Escalamiento temporal (a > 1 compresion, 0 < a < 1 expansion, a < 0 inversion)
      b: Desplazamiento temporal (b > 0 retardo a la derecha)
      A: Modificacion de amplitud
      C: Desplazamiento vertical (offset)
    """
    t_prime = scale_a * t_arr - shift_b
    return amp_A * signal_func(t_prime) + offset_C


def compute_signal_area(y_arr, t_arr):
    """Calcula el area bajo la senal usando integracion numerica trapezoidal."""
    return _trapz(y_arr, t_arr)


def compute_signal_energy(y_arr, t_arr):
    """Calcula la energia de la senal E_x = integral |x(t)|^2 dt."""
    return _trapz(np.abs(y_arr) ** 2, t_arr)


def decompose_even_odd(signal_eval_func, t_arr):
    """
    Descompone una senal x(t) en sus componentes par e impar:
    x_e(t) = [x(t) + x(-t)] / 2
    x_o(t) = [x(t) - x(-t)] / 2
    Retorna (y_x, y_even, y_odd, symmetry_type)
    """
    y_pos = signal_eval_func(t_arr)
    y_neg = signal_eval_func(-t_arr)

    y_even = 0.5 * (y_pos + y_neg)
    y_odd = 0.5 * (y_pos - y_neg)

    diff_even = np.max(np.abs(y_pos - y_neg))
    diff_odd = np.max(np.abs(y_pos + y_neg))
    max_val = np.max(np.abs(y_pos))

    tol = 1e-4 if max_val == 0 else 1e-3 * max_val

    if diff_even < tol:
        symmetry = "Par (Simétrica respecto al eje Y)"
    elif diff_odd < tol:
        symmetry = "Impar (Antisimétrica respecto al origen)"
    else:
        symmetry = "Sin simetría (Asimétrica)"

    return y_pos, y_even, y_odd, symmetry


def step_by_step_transformation(scale_a=1.0, shift_b=0.0, amp_A=1.0, offset_C=0.0, area_x=0.0, energy_x=0.0):
    """
    Genera los pasos analiticos para y(t) = A * x(a*t - b) + C y sus efectos
    en el eje temporal, amplitud, area y energia.
    """
    steps = []

    steps.append({
        "title": "Paso 1: Identificación de la transformación",
        "latex": rf"y(t) = {amp_A:g} \cdot x\left({scale_a:g}t - ({shift_b:g})\right) + ({offset_C:g})",
        "note": "Se evalúa la señal original x en el argumento modificado u(t) = a·t - b y luego se escala/desplaza verticalmente.",
    })

    if abs(scale_a) > 1:
        time_desc = f"Compresión temporal por factor {abs(scale_a):g}"
    elif 0 < abs(scale_a) < 1:
        time_desc = f"Expansión temporal por factor {1.0 / abs(scale_a):g} (argumento {scale_a:g}t)"
    else:
        time_desc = "Factor temporal nulo"

    if scale_a < 0:
        time_desc += " + inversión temporal (reflexión respecto al eje vertical)"

    shift_dir = "derecha" if shift_b > 0 else ("izquierda" if shift_b < 0 else "sin desplazamiento")
    steps.append({
        "title": "Paso 2: Manipulación en el eje temporal",
        "latex": rf"u(t) = {scale_a:g}t - ({shift_b:g})",
        "note": f"{time_desc}; desplazamiento efectivo hacia la {shift_dir} (b = {shift_b:g}).",
    })

    steps.append({
        "title": "Paso 3: Escalamiento y desplazamiento vertical",
        "latex": rf"y(t) = {amp_A:g}\,x(u(t)) + ({offset_C:g})",
        "note": f"Amplitud escalada por A = {amp_A:g} y offset vertical C = {offset_C:g}.",
    })

    if abs(scale_a) > 1e-12:
        area_scaled = (abs(amp_A) / abs(scale_a)) * area_x
        energy_scaled = ((amp_A ** 2) / abs(scale_a)) * energy_x
        steps.append({
            "title": "Paso 4: Efecto en área y energía (sin offset C)",
            "latex": (
                rf"E_{{y,0}} = \frac{{A^2}}{{|a|}} E_x = \frac{{{amp_A:g}^2}}{{|{scale_a:g}|}} ({energy_x:.4f}) = {energy_scaled:.4f}"
            ),
            "note": f"Área base x(t): {area_x:.4f} | Energía base x(t): {energy_x:.4f} | Energía escalada de referencia: {energy_scaled:.4f} (área con |A|/|a|: {area_scaled:.4f}).",
        })

    return steps


def step_by_step_even_odd(sym_type, area_x=0.0, energy_x=0.0):
    """Genera los pasos analiticos de descomposición en parte par e impar."""
    return [
        {
            "title": "Paso 1: Definición de reflexión temporal",
            "latex": r"x_{\text{ref}}(t) = x(-t)",
            "note": "Se refleja la señal x(t) alrededor del eje vertical t = 0.",
        },
        {
            "title": "Paso 2: Cálculo de la componente par",
            "latex": r"x_e(t) = \frac{x(t) + x(-t)}{2}",
            "note": "Contiene la simetría par tal que x_e(-t) = x_e(t).",
        },
        {
            "title": "Paso 3: Cálculo de la componente impar",
            "latex": r"x_o(t) = \frac{x(t) - x(-t)}{2}",
            "note": "Contiene la antisimetría impar tal que x_o(-t) = -x_o(t).",
        },
        {
            "title": "Paso 4: Reconstrucción y clasificación",
            "latex": r"x(t) = x_e(t) + x_o(t)",
            "note": f"Clasificación detectada: {sym_type} (Área x(t): {area_x:.4f}, Energía x(t): {energy_x:.4f}).",
        },
    ]
