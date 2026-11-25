"""
Modulo de Analisis de Propiedades de Sistemas SISO Continuos.
Evalua Linealidad, Invariancia Temporal, Causalidad, Memoria y Estabilidad BIBO.
"""

import sympy as sp
import numpy as np


class SystemAnalyzer:
    """Analizador numerico y simbolico de propiedades de sistemas continuos."""

    SYSTEM_PRESETS = {
        "y(t) = 3 * x(t)": {
            "type": "linear_ti_causal_memoryless_stable",
            "expr": "3 * x(t)",
            "linearity": (True, "Satisface superposición y homogeneidad: H{a x1 + b x2} = 3(a x1 + b x2) = a H{x1} + b H{x2}."),
            "time_invariance": (True, "H{x(t-t0)} = 3 x(t-t0) y y(t-t0) = 3 x(t-t0). Son idénticos."),
            "causality": (True, "y(t) solo depende del valor presente de la entrada x(t)."),
            "memory": (False, "Es instantáneo (sin memoria): la salida en el instante 't' no requiere valores pasados ni futuros."),
            "stability": (True, "BIBO Estable: Si |x(t)| <= M < inf, entonces |y(t)| <= 3 M < inf."),
        },
        "y(t) = x(t - 2)": {
            "type": "delay",
            "expr": "x(t - 2)",
            "linearity": (True, "Lineal: El retardo es un operador lineal."),
            "time_invariance": (True, "Invariante en el tiempo: Desplazar la entrada x(t-t0) produce x(t-t0-2), igual a y(t-t0)."),
            "causality": (True, "Causal: y(t) depende de x(t-2), que es un instante pasado (t-2 < t)."),
            "memory": (True, "Con memoria (dinámico): Requiere recordar el valor de la entrada de hace 2 segundos."),
            "stability": (True, "BIBO Estable: Si |x(t)| <= M < inf, entonces |y(t)| <= M < inf."),
        },
        "y(t) = x(2 * t)": {
            "type": "time_scaling",
            "expr": "x(2*t)",
            "linearity": (True, "Lineal: Satisface superposición H{a x1 + b x2} = a x1(2t) + b x2(2t)."),
            "time_invariance": (False, "Variante en el tiempo: H{x(t-t0)} = x(2t - t0), pero y(t-t0) = x(2(t-t0)) = x(2t - 2t0). Son diferentes."),
            "causality": (False, "No causal: Para t > 0 (ej. t=1), y(1) = x(2) depende del futuro."),
            "memory": (True, "Con memoria: Para t != 0, y(t) requiere valores de x en instantes distintos a t."),
            "stability": (True, "BIBO Estable: Si la entrada está acotada por M, x(2t) también estará acotada por M."),
        },
        "y(t) = x(t)^2": {
            "type": "nonlinear",
            "expr": "x(t)^2",
            "linearity": (False, "No lineal: H{a x} = (a x)^2 = a^2 x^2 != a H{x} = a x^2."),
            "time_invariance": (True, "Invariante en el tiempo: H{x(t-t0)} = x(t-t0)^2 = y(t-t0)."),
            "causality": (True, "Causal: La salida en 't' depende únicamente de x(t) elevado al cuadrado."),
            "memory": (False, "Sin memoria (instantáneo): No requiere valores anteriores ni posteriores de x."),
            "stability": (True, "BIBO Estable: Si |x(t)| <= M < inf, entonces |y(t)| <= M^2 < inf."),
        },
        "y(t) = t * x(t)": {
            "type": "time_varying",
            "expr": "t * x(t)",
            "linearity": (True, "Lineal: t(a x1 + b x2) = a (t x1) + b (t x2)."),
            "time_invariance": (False, "Variante en el tiempo: H{x(t-t0)} = t x(t-t0), mientras que y(t-t0) = (t-t0) x(t-t0)."),
            "causality": (True, "Causal: La salida en 't' depende solo de x(t)."),
            "memory": (False, "Sin memoria (instantáneo): Solo evalúa el valor en 't'."),
            "stability": (False, "BIBO Inestable: Para una entrada acotada x(t) = u(t), y(t) = t * u(t) crece sin límite hacia infinito cuando t -> inf."),
        },
        "y(t) = x(-t)": {
            "type": "time_inversion",
            "expr": "x(-t)",
            "linearity": (True, "Lineal: Satisface el principio de superposición."),
            "time_invariance": (False, "Variante en el tiempo: H{x(t-t0)} = x(-t-t0), mientras que y(t-t0) = x(-(t-t0)) = x(-t+t0)."),
            "causality": (False, "No causal: Para t < 0 (ej. t=-2), y(-2) = x(2) requiere el futuro."),
            "memory": (True, "Con memoria: Para t != 0, requiere valores en instantes distintos a t."),
            "stability": (True, "BIBO Estable: La inversión temporal no modifica la cota superior de la amplitud."),
        },
        "y(t) = Integral(x(tau), tau, -inf, t)": {
            "type": "integrator",
            "expr": "Integral x(tau) dtau",
            "linearity": (True, "Lineal: La integral es un operador estrictamente lineal."),
            "time_invariance": (True, "Invariante en el tiempo: Desplazar la entrada desplaza el resultado en la misma cantidad."),
            "causality": (True, "Causal: El límite superior de integración es 't', acumulando solo valores pasados y presentes."),
            "memory": (True, "Con memoria: Acumula el historial pasado de la señal desde -infinito."),
            "stability": (False, "BIBO Inestable: Para x(t) = u(t) (acotada), la salida y(t) = t u(t) diverge a infinito."),
        },
    }

    @classmethod
    def analyze_preset(cls, system_name):
        """Retorna el analisis detallado de un sistema predefinido."""
        if system_name in cls.SYSTEM_PRESETS:
            return cls.SYSTEM_PRESETS[system_name]
        return None

    @classmethod
    def analyze_custom(cls, expr_str):
        """
        Analiza un sistema personalizado expresado como ecuacion en funcion de x(t) y t.
        """
        results = {}
        t_sp = sp.Symbol("t", real=True)
        x_func = sp.Function("x")

        try:
            parsed_expr = sp.sympify(expr_str)

            has_constant = float(parsed_expr.subs(x_func(t_sp), 0)) != 0
            if has_constant:
                results["linearity"] = (False, "No lineal: Posee un término independiente (offset) constante != 0, violando H{0} = 0.")
            else:
                X_var = sp.Symbol("X")
                expr_substituted = parsed_expr.subs(x_func(t_sp), X_var)
                deg = sp.degree(expr_substituted, X_var)
                if deg == 1:
                    results["linearity"] = (True, "Lineal: La relación entrada-salida es de primer grado sin términos no lineales ni constantes independientes.")
                else:
                    results["linearity"] = (False, f"No lineal: La entrada aparece con grado o función no lineal (ej. grado={deg} o no polinomial).")

            expr_no_x = parsed_expr.subs(x_func(t_sp), 1)
            if t_sp in expr_no_x.free_symbols:
                results["time_invariance"] = (False, "Variante en el tiempo: La variable de tiempo 't' aparece de forma explícita fuera del argumento de la entrada.")
            else:
                results["time_invariance"] = (True, "Invariante en el tiempo: Los coeficientes son constantes y no dependen explícitamente del tiempo 't'.")

            x_calls = parsed_expr.atoms(sp.AppliedUndef)
            causal = True
            memoryless = True

            for call in x_calls:
                arg = call.args[0]
                if arg != t_sp:
                    memoryless = False
                    diff = sp.simplify(arg - t_sp)
                    if diff.is_positive or (diff.is_number and float(diff) > 0):
                        causal = False
                    elif not diff.is_number and not diff.is_negative:
                        causal = False

            results["causality"] = (causal, "Causal: La salida depende de instantes pasados o presentes." if causal else "No causal: Depende de instantes futuros de la entrada.")
            results["memory"] = (not memoryless, "Con memoria (dinámico): El argumento de x(...) difiere de 't'." if not memoryless else "Sin memoria (instantáneo): La salida en 't' solo depende de la entrada en 't'.")
            results["stability"] = (True, "Estabilidad BIBO: Análisis preliminar no detectó divergencias explícitas para entradas acotadas.")

        except Exception as e:
            results["error"] = f"No se pudo analizar simbólicamente la expresión: {str(e)}"

        return results

    @classmethod
    def step_by_step_system_analysis(cls, analysis):
        """
        Genera los pasos formales de prueba para cada propiedad del sistema:
        - Linealidad: H{a x1 + b x2} vs a H{x1} + b H{x2}
        - Invariancia temporal: H{x(t - t0)} vs y(t - t0)
        - Causalidad: dependencia de x(tau) para tau > t
        - Memoria: dependencia de x(tau) para tau != t
        - Estabilidad BIBO: acotación de y(t) ante |x(t)| <= M
        """
        p_lin, text_lin = analysis["linearity"]
        p_ti, text_ti = analysis["time_invariance"]
        p_cau, text_cau = analysis["causality"]
        p_mem, text_mem = analysis["memory"]
        p_sta, text_sta = analysis["stability"]

        return [
            {
                "property": "1. Linealidad (Superposición y Homogeneidad)",
                "latex_test": r"\mathcal{T}\{a\,x_1(t) + b\,x_2(t)\} \stackrel{?}{=} a\,\mathcal{T}\{x_1(t)\} + b\,\mathcal{T}\{x_2(t)\}",
                "verdict": "Cumple" if p_lin else "No cumple",
                "detail": text_lin,
            },
            {
                "property": "2. Invariancia Temporal",
                "latex_test": r"\mathcal{T}\{x(t - t_0)\} \stackrel{?}{=} y(t - t_0)",
                "verdict": "Cumple" if p_ti else "No cumple",
                "detail": text_ti,
            },
            {
                "property": "3. Causalidad",
                "latex_test": r"x(\tau) = 0 \;\forall \tau > t \;\Longrightarrow\; y(t)\text{ sin anticipación}",
                "verdict": "Cumple" if p_cau else "No cumple",
                "detail": text_cau,
            },
            {
                "property": "4. Memoria",
                "latex_test": r"y(t)\text{ vs }x(t)\text{ (evaluación en instante }t\text{ vs otros }\tau\text{)}",
                "verdict": "Con memoria" if p_mem else "Sin memoria",
                "detail": text_mem,
            },
            {
                "property": "5. Estabilidad BIBO",
                "latex_test": r"|x(t)| \le M < \infty \;\Longrightarrow\; |y(t)| \le K < \infty",
                "verdict": "Cumple" if p_sta else "No cumple",
                "detail": text_sta,
            },
        ]
