"""
Modulo de Operaciones, Manipulacion y Propiedades de Senales.
"""

import numpy as np
import scipy.integrate as integrate
import sympy as sp
from .signals import BasicSignal

def _trapz(y, x):
    if hasattr(np, 'trapezoid'):
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
    return _trapz(np.abs(y_arr)**2, t_arr)

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
    
    # Determinar tipo de simetria con tolerancia numerica
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
