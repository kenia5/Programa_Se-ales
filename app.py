"""
Calculadora de Senales y Sistemas Continuos
Aplicacion Interactiva con Streamlit y Matplotlib.
Autores: Basado en el syllabus de Senales y Sistemas Continuos (EPN)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from modules.signals import (
    BasicSignal,
    parse_piecewise_signal,
    piecewise_to_sympy,
    compute_piecewise_derivative,
    step_by_step_derivative,
)
from modules.operations import (
    evaluate_transformed_signal,
    compute_signal_area,
    compute_signal_energy,
    decompose_even_odd,
    step_by_step_transformation,
    step_by_step_even_odd,
)
from modules.systems import SystemAnalyzer
from modules.convolution import (
    continuous_convolution_at_t,
    compute_full_numerical_convolution,
    symbolic_piecewise_convolution,
    step_by_step_convolution,
)

# Configuración de página Streamlit
st.set_page_config(
    page_title="Calculadora de Señales y Sistemas Continuos",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilo CSS personalizado
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.0rem;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        text-align: center;
        margin-bottom: 1.2rem;
        opacity: 0.85;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2563EB;
        margin-bottom: 0.8rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-header">⚡ Calculadora de Señales y Sistemas Continuos</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Herramienta interactiva para resolución de ejercicios universitarios de Señales y Sistemas</div>',
    unsafe_allow_html=True,
)


def render_math(latex_str, label=""):
    """
    Renderiza expresiones LaTeX usando st.latex para evitar conflictos de tema,
    desbordamiento o recuadros blancos en pantallas pequeñas.
    """
    if label:
        st.markdown(f"**{label}**")
    cleaned = str(latex_str).strip()
    if cleaned.startswith("$$") and cleaned.endswith("$$"):
        cleaned = cleaned[2:-2].strip()
    elif cleaned.startswith("$") and cleaned.endswith("$"):
        cleaned = cleaned[1:-1].strip()
    cleaned = cleaned.replace(r"\displaystyle", "").strip()
    st.latex(cleaned)


# Pestañas principales enfocadas en el temario esencial
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Señales Básicas, Impulsos y Dobletes",
    "2️⃣ Operaciones, Manipulación y Simetría",
    "3️⃣ Respuesta Impulsiva y Convolución Continua",
    "4️⃣ Análisis de Sistemas (Opcional)",
])

# Configuración comun de eje de tiempo
t_domain = np.linspace(-10, 10, 2000)

# -----------------------------------------------------------------------------
# TAB 1: CONCEPTOS, SEÑALES BÁSICAS E IMPULSOS / DOBLETES
# -----------------------------------------------------------------------------
with tab1:
    st.header("1. Conceptos, Señales Básicas e Impulsos / Dobletes")
    st.write("Genera y visualiza las señales continuas fundamentales, funciones singulares impulsivas y su diferenciación.")

    col1, col2 = st.columns([1, 2])

    with col1:
        signal_type = st.selectbox(
            "Selecciona el tipo de señal básica:",
            [
                "Triángulo x1(t) [-2, 2] (Ejercicio de Examen)",
                "Impulso Unitario δ(t - t0)",
                "Doblete Unitario δ'(t - t0) (Derivada del Impulso)",
                "Escalón Unitario u(t - t0)",
                "Rampa Unitario r(t - t0)",
                "Pulso Rectangular rect((t - t0)/W)",
                "Pulso Triangular tri((t - t0)/W)",
                "Exponencial A*exp(α(t - t0))*u(t - t0)",
                "Senoidal A*sin(2πf t + φ)",
                "Definida por Tramos (Personalizada)",
            ],
        )

        amp = st.number_input("Amplitud (A):", value=1.0, step=0.5, key="t1_amp")
        t0 = st.number_input("Desplazamiento / Centro (t0):", value=0.0, step=0.5, key="t1_t0")

        if "Rectangular" in signal_type or "Triangular" in signal_type:
            width = st.number_input("Ancho (W):", value=2.0, min_value=0.1, step=0.5, key="t1_w")
        elif "Exponencial" in signal_type:
            alpha = st.number_input("Factor α (exponencial):", value=-1.0, step=0.5, key="t1_alpha")
        elif "Senoidal" in signal_type:
            freq = st.number_input("Frecuencia (f en Hz):", value=0.5, step=0.1, key="t1_freq")
            phase = st.number_input("Fase φ (rad):", value=0.0, step=0.1, key="t1_phase")
        elif "Personalizada" in signal_type:
            st.markdown("**Definición por tramos:**")
            pieces_text = st.text_area(
                "Ingresa tramos (formato: expr, t_min, t_max por línea):",
                value="t + 2, -2, 0\n-t + 2, 0, 2",
                key="t1_pieces",
            )

    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        pieces = []

        if "Triángulo x1(t)" in signal_type:
            pieces = [("t + 2", (-2, 0)), ("-t + 2", (0, 2))]
            y_sig = parse_piecewise_signal(pieces, t_domain)
            latex_expr = r"x_1(t) = \begin{cases} t+2, & \text{para } -2 \le t \le 0 \\[4pt] -t+2, & \text{para } 0 \le t \le 2 \\[4pt] 0, & \text{en otro caso} \end{cases}"
            desc_text = "**Señales por Tramos:** Representación matemática de la señal continua definida por intervalos lineales."
        elif "Doblete" in signal_type:
            y_sig = BasicSignal.doublet(t_domain, t0=t0, amplitude=amp)
            latex_expr = f"x(t) = {amp} \\cdot \\delta'(t - {t0})"
            desc_text = "**Doblete Unitario δ'(t):** Es la primera derivada del impulso unitario dδ(t)/dt. Representa un par de impulsos infinitesimalmente cercanos de polaridad opuesta."
        elif "Impulso" in signal_type:
            y_sig = BasicSignal.impulse(t_domain, t0=t0, amplitude=amp)
            latex_expr = f"x(t) = {amp} \\cdot \\delta(t - {t0})"
            desc_text = "**Impulso Unitario δ(t):** Función singular de área unitaria concentrada en un instante de tiempo t0. Es la derivada del escalón u(t)."
        elif "Escalón" in signal_type:
            y_sig = BasicSignal.step(t_domain, t0=t0, amplitude=amp)
            pieces = [(f"{amp}", (t0, 5.0))]
            latex_expr = f"x(t) = {amp} \\cdot u(t - {t0})"
            desc_text = "**Escalón Unitario u(t):** Vale 1 para t >= t0 y 0 para t < t0. Sirve para modelar encendidos de sistemas."
        elif "Rampa" in signal_type:
            y_sig = BasicSignal.ramp(t_domain, t0=t0, slope=amp)
            latex_expr = f"x(t) = {amp} \\cdot r(t - {t0})"
            desc_text = "**Rampa Unitario r(t):** Es la integral del escalón u(t). Crece linealmente a partir de t0."
        elif "Rectangular" in signal_type:
            y_sig = BasicSignal.rect(t_domain, center=t0, width=width, amplitude=amp)
            pieces = [(f"{amp}", (t0 - width / 2.0, t0 + width / 2.0))]
            latex_expr = f"x(t) = {amp} \\cdot \\text{{rect}}\\left(\\frac{{t - {t0}}}{{{width}}}\\right)"
            desc_text = "**Pulso Rectangular:** Ventana de ancho W centrada en t0."
        elif "Triangular" in signal_type:
            y_sig = BasicSignal.tri(t_domain, center=t0, width=width, amplitude=amp)
            half_w = width / 2.0
            pieces = [(f"{amp} * (1 - (t - {t0})/{half_w})", (t0 - half_w, t0)), (f"{amp} * (1 + (t - {t0})/{half_w})", (t0, t0 + half_w))]
            latex_expr = f"x(t) = {amp} \\cdot \\text{{tri}}\\left(\\frac{{t - {t0}}}{{{width}}}\\right)"
            desc_text = "**Pulso Triangular:** Pulso simétrico de base W y altura A."
        elif "Exponencial" in signal_type:
            y_sig = BasicSignal.exponential(t_domain, t0=t0, alpha=alpha, amplitude=amp)
            latex_expr = f"x(t) = {amp} e^{{{alpha}(t - {t0})}} u(t - {t0})"
            desc_text = "**Exponencial Causal:** Describe la respuesta natural en circuitos RC/RL y amortiguación de sistemas."
        elif "Senoidal" in signal_type:
            y_sig = BasicSignal.sine(t_domain, freq=freq, phase=phase, amplitude=amp)
            latex_expr = f"x(t) = {amp} \\sin(2\\pi \\cdot {freq} t + {phase})"
            desc_text = "**Senoidal:** Señal periódica fundamental en análisis armónico de Fourier."
        else:
            pieces = []
            for line in pieces_text.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 3:
                        pieces.append((parts[0], (float(parts[1]), float(parts[2]))))
            y_sig = parse_piecewise_signal(pieces, t_domain)
            latex_expr = r"x(t) \text{ definida por tramos personalizada}"
            desc_text = "**Señal por Tramos Personalizada:** Definida directamente por el usuario por intervalos."

        ax.plot(t_domain, y_sig, color="#2563EB", linewidth=2, label="x(t)")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlim(-6, 6)
        ax.set_title("Gráfica de la Señal Continua x(t)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Tiempo t (s)")
        ax.set_ylabel("Amplitud x(t)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        st.pyplot(fig)

        render_math(latex_expr, "Definición Matemática de la Señal:")

        with st.expander("📐 Procedimiento Analítico y Propiedades Básicas de la Señal"):
            st.write(desc_text)
            area_val = compute_signal_area(y_sig, t_domain)
            energy_val = compute_signal_energy(y_sig, t_domain)
            st.markdown(f"- **Área total calculada:** `{area_val:.4f}`")
            st.markdown(f"- **Energía total de la señal $E_x$:** `{energy_val:.4f}`")

        if pieces:
            with st.expander("⚡ Derivada de la Señal dx/dt (Análisis de Impulsos en Saltos y Dobletes)"):
                try:
                    _, impulses, deriv_latex = compute_piecewise_derivative(pieces)
                    render_math(deriv_latex, "Primera Derivada x'(t) = dx/dt:")

                    seg_steps, jump_steps = step_by_step_derivative(pieces)
                    st.markdown("**Paso 1 — Derivada en cada intervalo:**")
                    for s_item in seg_steps:
                        st.latex(
                            rf"\text{{Para }} {s_item['t_min_latex']} < t < {s_item['t_max_latex']}: "
                            rf"\quad \frac{{d}}{{dt}}\left({s_item['expr_latex']}\right) = {s_item['dexpr_latex']}"
                        )

                    st.markdown("**Paso 2 — Evaluación de saltos en puntos frontera:**")
                    for j_item in jump_steps:
                        st.latex(
                            rf"\text{{En }} t = {j_item['tp_latex']}: \quad "
                            rf"\Delta x = x({j_item['tp_latex']}^+) - x({j_item['tp_latex']}^-) "
                            rf"= {j_item['right_val']} - ({j_item['left_val']}) = {j_item['jump_latex']}"
                        )

                    if impulses:
                        st.markdown("**Paso 3 — Impulsos $\\delta(t)$ generados por discontinuidades de salto:**")
                        for tp, amp_imp in impulses:
                            sign_str = "+" if amp_imp > 0 else "-"
                            st.info(
                                f"📍 En t = {tp:.2f}: Salto de amplitud Δx = {amp_imp:+.2f} "
                                f"=> Impulso {sign_str}{abs(amp_imp):.2f} δ(t - {tp:.2f})"
                            )
                    else:
                        st.success("La señal es continua en sus bordes; no genera impulsos en su primera derivada.")
                except Exception as e:
                    st.info(f"Ingresa una señal definida por tramos válida para obtener su derivada simbólica exacta ({e}).")


# -----------------------------------------------------------------------------
# TAB 2: OPERACIONES, MANIPULACIÓN Y PROPIEDADES DE SEÑALES
# -----------------------------------------------------------------------------
with tab2:
    st.header("2. Operaciones, Manipulación y Simetría de Señales")
    st.write("Aplica transformaciones de la variable independiente (inversión, escalamiento, desplazamiento) y analiza simetría Par/Impar.")

    col_op1, col_op2 = st.columns([1, 2])

    with col_op1:
        st.subheader("Configuración de la Señal Base")
        sig_base_choice = st.selectbox(
            "Selecciona la señal base:",
            [
                "Triángulo x1(t) [-2, 2] (Ejercicio)",
                "Pulso Triangular tri(t/2)",
                "Pulso Rectangular rect(t/2)",
                "Exponencial exp(-t)u(t)",
                "Señal Impar Ejemplo t [-1, 1]",
                "Por Tramos (Personalizada)",
            ],
            key="t2_sig_choice",
        )

        if sig_base_choice == "Por Tramos (Personalizada)":
            pieces_text_t2 = st.text_area(
                "Ingresa tramos para x(t):",
                value="t + 2, -2, 0\n-t + 2, 0, 2",
                key="t2_pieces",
            )
            pieces_t2 = []
            for line in pieces_text_t2.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 3:
                        pieces_t2.append((parts[0], (float(parts[1]), float(parts[2]))))
            x_base_func = lambda t_arr: parse_piecewise_signal(pieces_t2, t_arr)
        elif sig_base_choice == "Triángulo x1(t) [-2, 2] (Ejercicio)":
            pieces_t2 = [("t + 2", (-2, 0)), ("-t + 2", (0, 2))]
            x_base_func = lambda t_arr: parse_piecewise_signal(pieces_t2, t_arr)
        elif sig_base_choice == "Señal Impar Ejemplo t [-1, 1]":
            pieces_t2 = [("t", (-1, 1))]
            x_base_func = lambda t_arr: parse_piecewise_signal(pieces_t2, t_arr)
        elif sig_base_choice == "Pulso Triangular tri(t/2)":
            x_base_func = lambda t_arr: BasicSignal.tri(t_arr, center=0.0, width=2.0)
        elif sig_base_choice == "Pulso Rectangular rect(t/2)":
            x_base_func = lambda t_arr: BasicSignal.rect(t_arr, center=0.0, width=2.0)
        else:
            x_base_func = lambda t_arr: BasicSignal.exponential(t_arr, t0=0.0, alpha=-1.0)

        st.subheader("Transformación: y(t) = A · x(a·t - b) + C")
        op_A = st.number_input("Modificación de Amplitud (A):", value=1.0, step=0.5, key="t2_A")
        op_a = st.number_input("Escalamiento / Inversión Temporal (a):", value=1.0, step=0.5, key="t2_a")
        op_b = st.number_input("Desplazamiento Temporal (b):", value=0.0, step=0.5, key="t2_b")
        op_C = st.number_input("Desplazamiento Vertical (C):", value=0.0, step=0.5, key="t2_C")

    with col_op2:
        y_orig = x_base_func(t_domain)
        y_trans = evaluate_transformed_signal(x_base_func, t_domain, amp_A=op_A, scale_a=op_a, shift_b=op_b, offset_C=op_C)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(t_domain, y_orig, color="#9CA3AF", linewidth=1.5, linestyle="--", label="Original x(t)")
        ax2.plot(t_domain, y_trans, color="#2563EB", linewidth=2.2, label=f"Transformada y(t) = {op_A}·x({op_a}t - {op_b}) + {op_C}")
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_xlim(-6, 6)
        ax2.set_title("Comparación entre Señal Original y Transformada", fontsize=11, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend()
        st.pyplot(fig2)

        area_x = compute_signal_area(y_orig, t_domain)
        energy_x = compute_signal_energy(y_orig, t_domain)
        _, y_even, y_odd, sym_type = decompose_even_odd(x_base_func, t_domain)

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Área bajo la señal", f"{area_x:.4f}")
        col_m2.metric("Energía de la señal E_x", f"{energy_x:.4f}")
        col_m3.metric("Simetría", sym_type)

        with st.expander("📐 Ver Procedimiento Paso a Paso de la Transformación"):
            t_steps = step_by_step_transformation(
                scale_a=op_a,
                shift_b=op_b,
                amp_A=op_A,
                offset_C=op_C,
                area_x=area_x,
                energy_x=energy_x,
            )
            for ts in t_steps:
                st.markdown(f"**{ts['title']}**")
                st.latex(ts["latex"])
                st.caption(ts["note"])

        with st.expander("⚖️ Ver Descomposición en Componente Par xe(t) e Impar xo(t) Paso a Paso"):
            eo_steps = step_by_step_even_odd(sym_type, area_x=area_x, energy_x=energy_x)
            for eos in eo_steps:
                st.markdown(f"**{eos['title']}**")
                st.latex(eos["latex"])
                st.caption(eos["note"])

            fig_eo, ax_eo = plt.subplots(figsize=(8, 3))
            ax_eo.plot(t_domain, y_even, color="#059669", linewidth=2, label="Componente Par xe(t)")
            ax_eo.plot(t_domain, y_odd, color="#D97706", linewidth=2, label="Componente Impar xo(t)")
            ax_eo.axhline(0, color="black", linewidth=0.8)
            ax_eo.axvline(0, color="black", linewidth=0.8)
            ax_eo.set_xlim(-6, 6)
            ax_eo.grid(True, linestyle=":", alpha=0.6)
            ax_eo.legend()
            st.pyplot(fig_eo)


# -----------------------------------------------------------------------------
# TAB 3: RESPUESTA IMPULSIVA Y CONVOLUCIÓN CONTINUA
# -----------------------------------------------------------------------------
with tab3:
    st.header("3. Respuesta Impulsiva y Convolución Continua")
    st.write("Calculadora interactiva paso a paso de la integral de convolución y(t) = x(t) * h(t).")

    col_c1, col_c2 = st.columns([1, 2])

    with col_c1:
        st.markdown("**Configuración de las Señales x(t) y h(t):**")

        sig_x_type = st.selectbox(
            "Señal de Entrada x(t):",
            ["Triángulo x1(t) [-2, 2] (Ejercicio)", "Pulso Rectangular [0, 2]", "Triángulo [0, 2]", "Exponencial e^(-t) u(t)", "Definida por Tramos (Personalizada)"],
            key="t4_x_type",
        )

        if sig_x_type == "Definida por Tramos (Personalizada)":
            x_pieces_text = st.text_area(
                "Ingresa tramos para x(t) (expr, t_min, t_max):",
                value="t + 2, -2, 0\n-t + 2, 0, 2",
                key="t4_x_text",
            )
            x_pieces = []
            for line in x_pieces_text.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 3:
                        x_pieces.append((parts[0], (float(parts[1]), float(parts[2]))))
            x_func = lambda t_arr: parse_piecewise_signal(x_pieces, t_arr)
        elif sig_x_type == "Triángulo x1(t) [-2, 2] (Ejercicio)":
            x_pieces = [("t + 2", (-2, 0)), ("-t + 2", (0, 2))]
            x_func = lambda t_arr: parse_piecewise_signal(x_pieces, t_arr)
        elif sig_x_type == "Pulso Rectangular [0, 2]":
            x_pieces = [("1", (0, 2))]
            x_func = lambda t_arr: BasicSignal.rect(t_arr, center=1.0, width=2.0)
        elif sig_x_type == "Triángulo [0, 2]":
            x_pieces = [("t", (0, 1)), ("2 - t", (1, 2))]
            x_func = lambda t_arr: BasicSignal.tri(t_arr, center=1.0, width=2.0)
        else:
            x_pieces = [("exp(-t)", (0, 5))]
            x_func = lambda t_arr: BasicSignal.exponential(t_arr, t0=0.0, alpha=-1.0)

        sig_h_type = st.selectbox(
            "Respuesta Impulsiva h(t):",
            ["Pulso Rectangular h(t) [3, 5] (Ejercicio)", "Pulso Rectangular [0, 1]", "Escalón u(t)", "Triángulo [0, 1]", "Definida por Tramos (Personalizada)"],
            key="t4_h_type",
        )

        if sig_h_type == 