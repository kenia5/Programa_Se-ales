"""
Calculadora de Señales y Sistemas Continuos
Aplicación Interactiva con Streamlit, Matplotlib, NumPy, SciPy y SymPy.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from modules.signals import (
    BasicSignal,
    parse_piecewise_signal,
    compute_piecewise_derivative,
    get_piecewise_latex_definition,
    step_by_step_derivative,
)
from modules.operations import (
    evaluate_transformed_signal,
    compute_signal_area,
    compute_signal_energy,
    decompose_even_odd,
)
from modules.energy_power import (
    compute_energy_power_metrics,
    classify_energy_power_signal,
    symbolic_energy_power_development,
    symbolic_energy_power_from_pieces,
)
from modules.systems import SystemAnalyzer
from modules.convolution import (
    continuous_convolution_at_t,
    compute_full_numerical_convolution,
    symbolic_piecewise_convolution,
)

st.set_page_config(
    page_title="Calculadora de Señales y Sistemas Continuos",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: clamp(1.6rem, 3.5vw, 2.3rem);
        color: #1E3A8A;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: clamp(0.95rem, 2vw, 1.15rem);
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .math-box {
        font-size: clamp(0.95rem, 2.5vw, 1.15rem);
        padding: 12px;
        background-color: #F8FAFC;
        border-radius: 8px;
        margin-bottom: 12px;
        text-align: center;
        overflow-x: auto;
        max-width: 100%;
        border: 1px solid #E2E8F0;
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


def render_math_box(latex_str, label=""):
    """Renderiza expresiones LaTeX usando st.latex para evitar cuadros en blanco."""
    if label:
        st.markdown(f"**{label}**")
    clean_latex = (latex_str or "").strip()
    if not clean_latex:
        st.info("No hay expresión matemática disponible para esta selección.")
        return
    st.latex(clean_latex)


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ Señales Básicas, Impulsos y Dobletes",
    "2️⃣ Señales de Potencia y Energía",
    "3️⃣ Operaciones y Simetría",
    "4️⃣ Convolución Continua",
    "5️⃣ Sistemas (Opcional)",
])

t_domain = np.linspace(-10, 10, 2000)

# -----------------------------------------------------------------------------
# TAB 1: SEÑALES BÁSICAS, IMPULSOS Y DOBLETES
# -----------------------------------------------------------------------------
with tab1:
    st.header("1. Conceptos, Señales Básicas e Impulsos / Dobletes")
    st.write(
        "Genera y visualiza señales continuas fundamentales con su definición explícita por intervalos y diferenciación."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        signal_type = st.selectbox(
            "Selecciona el tipo de señal básica:",
            [
                "Triángulo x1(t) por intervalos [-2, 2]",
                "Impulso Unitario δ(t - t0)",
                "Doblete Unitario δ'(t - t0) (Derivada del Impulso)",
                "Escalón Unitario u(t - t0)",
                "Rampa Unitaria r(t - t0)",
                "Pulso Rectangular por intervalos",
                "Pulso Triangular por intervalos",
                "Exponencial A*exp(α(t - t0)) por intervalos",
                "Senoidal A*sin(2πf t + φ)",
                "Definida por Tramos (Personalizada)",
            ],
        )

        amp = st.number_input("Amplitud (A):", value=1.0, step=0.5, key="t1_amp")
        t0 = st.number_input("Desplazamiento / Centro (t0):", value=0.0, step=0.5, key="t1_t0")

        width = 2.0
        alpha = -1.0
        freq = 0.5
        phase = 0.0
        pieces = []

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
            for line in pieces_text.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 3:
                        pieces.append((parts[0], (float(parts[1]), float(parts[2]))))

    with col2:
        fig, ax = plt.subplots(figsize=(8, 3.8))

        if "Triángulo x1(t)" in signal_type:
            pieces = [("t + 2", (-2, 0)), ("-t + 2", (0, 2))]
            y_sig = parse_piecewise_signal(pieces, t_domain)
            desc_text = "**Señal por Tramos:** Definida por intervalos lineales en [-2, 0] y [0, 2]."
        elif "Doblete" in signal_type:
            y_sig = BasicSignal.doublet(t_domain, t0=t0, amplitude=amp)
            desc_text = "**Doblete Unitario δ'(t):** Primera derivada del impulso unitario."
        elif "Impulso" in signal_type:
            y_sig = BasicSignal.impulse(t_domain, t0=t0, amplitude=amp)
            desc_text = "**Impulso Unitario δ(t):** Función singular concentrada en t0."
        elif "Escalón" in signal_type:
            y_sig = BasicSignal.step(t_domain, t0=t0, amplitude=amp)
            pieces = [(f"{amp:g}", (t0, 10.0))]
            desc_text = "**Escalón Unitario por Intervalos:** Vale A para t ≥ t0 y 0 para t < t0."
        elif "Rampa" in signal_type:
            y_sig = BasicSignal.ramp(t_domain, t0=t0, slope=amp)
            ramp_expr = f"{amp:g}*(t - {t0:g})" if t0 != 0 else f"{amp:g}*t"
            pieces = [(ramp_expr, (t0, 10.0))]
            desc_text = "**Rampa por Intervalos:** Crece linealmente a partir de t0."
        elif "Rectangular" in signal_type:
            y_sig = BasicSignal.rect(t_domain, center=t0, width=width, amplitude=amp)
            pieces = [(f"{amp:g}", (t0 - width / 2.0, t0 + width / 2.0))]
            desc_text = "**Pulso por Intervalos:** Vale A dentro del intervalo [t0 - W/2, t0 + W/2] y 0 fuera."
        elif "Triangular" in signal_type:
            y_sig = BasicSignal.tri(t_domain, center=t0, width=width, amplitude=amp)
            hw = width / 2.0
            m = amp / hw
            pieces = [
                (f"{m:g}*(t - {t0 - hw:g})", (t0 - hw, t0)),
                (f"{-m:g}*(t - {t0 + hw:g})", (t0, t0 + hw)),
            ]
            desc_text = "**Pulso Triangular por Intervalos:** Definido por dos tramos lineales simétricos."
        elif "Exponencial" in signal_type:
            y_sig = BasicSignal.exponential(t_domain, t0=t0, alpha=alpha, amplitude=amp)
            t_shift_expr = f"(t - {t0:g})" if t0 != 0 else "t"
            pieces = [(f"{amp:g}*exp({alpha:g}*{t_shift_expr})", (t0, 10.0))]
            desc_text = "**Exponencial por Intervalos:** Definida para t ≥ t0 y nula para t < t0."
        elif "Senoidal" in signal_type:
            y_sig = BasicSignal.sine(t_domain, freq=freq, phase=phase, amplitude=amp)
            desc_text = "**Senoidal:** Señal periódica continua."
        else:
            y_sig = parse_piecewise_signal(pieces, t_domain)
            desc_text = "**Señal por Tramos Personalizada:** Definida explícitamente por intervalos."

        ax.plot(t_domain, y_sig, color="#2563EB", linewidth=2, label="x(t)")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlim(-6, 6)
        ax.set_title("Gráfica de la Señal Continua x(t)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Tiempo t (s)")
        ax.set_ylabel("Amplitud x(t)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")
        st.pyplot(fig)

        latex_piecewise_expr = get_piecewise_latex_definition(
            signal_type,
            amp=amp,
            t0=t0,
            width=width,
            alpha=alpha,
            freq=freq,
            phase=phase,
            pieces=pieces,
        )
        render_math_box(latex_piecewise_expr, "Definición Matemática Explícita por Intervalos:")

        with st.expander("📐 Procedimiento Analítico y Propiedades Básicas"):
            st.write(desc_text)
            area_val = compute_signal_area(y_sig, t_domain)
            energy_val = compute_signal_energy(y_sig, t_domain)
            st.write(f"- **Área total calculada:** `{area_val:.4f}`")
            st.write(f"- **Energía total acumulada:** `{energy_val:.4f}`")

        if pieces:
            with st.expander("⚡ Derivada de la Señal dx/dt (Paso a Paso e Impulsos en Saltos)"):
                try:
                    _, impulses, deriv_latex = compute_piecewise_derivative(pieces)
                    seg_steps, jump_steps = step_by_step_derivative(pieces)

                    st.markdown("**1. Derivada por tramos:**")
                    for idx_s, s_item in enumerate(seg_steps, start=1):
                        st.latex(
                            f"\\text{{Tramo {idx_s} }} ({s_item['t_min_latex']} \\le t \\le {s_item['t_max_latex']}): "
                            f"\\quad \\frac{{d}}{{dt}}\\left({s_item['expr_latex']}\\right) = {s_item['dexpr_latex']}"
                        )

                    st.markdown("**2. Saltos en fronteras ($\\Delta x = x(t_p^+) - x(t_p^-)$):**")
                    for j_item in jump_steps:
                        st.latex(
                            f"t_p = {j_item['tp_latex']}: \\quad \\Delta x = {j_item['right_val']} - ({j_item['left_val']}) = {j_item['jump_latex']}"
                        )

                    st.markdown("**3. Expresión completa de la derivada:**")
                    st.latex(r"x'(t) = " + deriv_latex)

                    if impulses:
                        for tp, amp_imp in impulses:
                            sign_str = "+" if amp_imp > 0 else "-"
                            st.info(
                                f"📍 En t = {tp:.2f}: Salto Δx = {amp_imp:+.2f} "
                                f"=> Impulso {sign_str}{abs(amp_imp):.2f} * δ(t - {tp:.2f})"
                            )
                except Exception:
                    st.info("No se pudo calcular el paso a paso de la derivada.")


# -----------------------------------------------------------------------------
# TAB 2: SEÑALES DE POTENCIA Y ENERGÍA (PASO A PASO POR INTERVALOS)
# -----------------------------------------------------------------------------
with tab2:
    st.header("2. Conceptos Básicos: Señales de Potencia y Energía")
    st.write(
        "Cálculo numérico, clasificación y **desarrollo paso a paso por intervalos** de "
        "Energía ($E_x$), Valor Medio ($\\langle x(t) \\rangle$), Potencia ($P_x$) y Valor RMS ($X_{\\mathrm{RMS}}$)."
    )

    col_ep1, col_ep2 = st.columns([1, 2])

    with col_ep1:
        st.subheader("Selección o Definición por Intervalos")
        ep_sig_choice = st.selectbox(
            "Selecciona una señal para analizar:",
            [
                "Triángulo x1(t) por intervalos [-2, 2]",
                "Pulso por intervalos [-1, 1]",
                "Exponencial decreciente por intervalos t >= 0",
                "Senoidal periódica A*sin(2π f t)",
                "Escalón por intervalos t >= 0",
                "Rampa por intervalos t >= 0",
                "Definida por Tramos (Personalizada)",
            ],
            key="ep_choice",
        )

        ep_tmax = st.slider(
            "Límite máximo de ventana T ([-T, T]):",
            min_value=2.0,
            max_value=30.0,
            value=15.0,
            step=1.0,
            key="ep_tmax",
        )

        st.markdown("**Ventana para desarrollo paso a paso $[T_a, T_b]$:**")
        c_wa, c_wb = st.columns(2)
        win_a = c_wa.number_input("Inicio T_a:", value=-2.0, step=0.5, key="ep_win_a")
        win_b = c_wb.number_input("Fin T_b:", value=2.0, step=0.5, key="ep_win_b")

        ep_pieces = []
        is_periodic_choice = False

        if ep_sig_choice == "Definida por Tramos (Personalizada)":
            ep_text = st.text_area(
                "Ingresa tramos para x(t) (expr, t_min, t_max):",
                value="t + 2, -2, 0\n-t + 2, 0, 2",
                key="ep_text",
            )
            for line in ep_text.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 3:
                        ep_pieces.append((parts[0], (float(parts[1]), float(parts[2]))))
            ep_func = lambda t_arr: parse_piecewise_signal(ep_pieces, t_arr)
        elif "Triángulo x1(t)" in ep_sig_choice:
            ep_pieces = [("t + 2", (-2, 0)), ("-t + 2", (0, 2))]
            ep_func = lambda t_arr: parse_piecewise_signal(ep_pieces, t_arr)
        elif "Pulso por intervalos" in ep_sig_choice:
            ep_pieces = [("1", (-1, 1))]
            ep_func = lambda t_arr: parse_piecewise_signal(ep_pieces, t_arr)
        elif "Exponencial decreciente" in ep_sig_choice:
            ep_pieces = [("exp(-t)", (0, 30))]
            ep_func = lambda t_arr: BasicSignal.exponential(t_arr, t0=0.0, alpha=-1.0, amplitude=1.0)
        elif "Senoidal periódica" in ep_sig_choice:
            is_periodic_choice = True
            ep_pieces = [("sin(2*pi*0.5*t)", (-1, 1))]
            ep_func = lambda t_arr: BasicSignal.sine(t_arr, freq=0.5, phase=0.0, amplitude=1.0)
        elif "Escalón por intervalos" in ep_sig_choice:
            is_periodic_choice = True
            ep_pieces = [("1", (0, 2))]
            ep_func = lambda t_arr: BasicSignal.step(t_arr, t0=0.0, amplitude=1.0)
        else:
            ep_pieces = [("t", (0, 2))]
            ep_func = lambda t_arr: BasicSignal.ramp(t_arr, t0=0.0, slope=1.0)

    with col_ep2:
        metrics = compute_energy_power_metrics(ep_func, t_max=ep_tmax)
        classification = classify_energy_power_signal(
            metrics["Ex"],
            metrics["Px"],
            metrics["T_axis"],
            metrics["E_T_curve"],
            metrics["P_T_curve"],
            is_periodic=is_periodic_choice,
        )

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Energía E_x(T_max)", f"{metrics['Ex']:.4f}")
        col_m2.metric("Potencia P_x(T_max)", f"{metrics['Px']:.4f}")
        col_m3.metric("Valor Medio <x(t)>", f"{metrics['mean']:.4f}")
        col_m4.metric("Valor RMS", f"{metrics['rms']:.4f}")

        st.subheader(f"Clasificación: {classification['category']}")
        st.info(classification["justification"])

        st.markdown("**Energía total ($E_x$):**")
        st.latex(classification["ex_latex"])

        st.markdown("**Potencia promedio ($P_x$):**")
        st.latex(classification["px_latex"])

        st.markdown("**Valor eficaz / RMS ($X_{\\mathrm{RMS}}$):**")
        st.latex(classification["rms_latex"])

        if ep_pieces:
            sym_E = symbolic_energy_power_from_pieces(ep_pieces)
            if sym_E is not None and sym_E.is_finite:
                st.markdown("**Energía exacta por intervalos (SymPy):**")
                st.latex(r"E_x = " + sp.latex(sym_E))

        with st.expander("📝 Paso a Paso Detallado: Energía, Valor Medio, Potencia y Valor RMS por Intervalos", expanded=True):
            dev = symbolic_energy_power_development(ep_pieces, win_a, win_b)
            if dev is None:
                st.warning("Revisa que el intervalo $[T_a, T_b]$ cumpla $T_b > T_a$ y tenga tramos válidos.")
            else:
                st.markdown(
                    f"**Paso 1 — Energía por intervalos en $[{sp.latex(dev['window_min'])}, {sp.latex(dev['window_max'])}]$:**"
                )
                for idx_e, step in enumerate(dev["energy_steps"], start=1):
                    st.latex(
                        f"\\text{{Tramo {idx_e} }} ({sp.latex(step['lower'])} \\le t \\le {sp.latex(step['upper'])}): "
                        f"\\quad E_{{{idx_e}}} = \\int_{{{sp.latex(step['lower'])}}}^{{{sp.latex(step['upper'])}}} "
                        f"\\left|{sp.latex(step['expression'])}\\right|^2 dt "
                        f"= \\left[{sp.latex(step['antiderivative'])}\\right]_{{{sp.latex(step['lower'])}}}^{{{sp.latex(step['upper'])}}} "
                        f"= {sp.latex(step['result'])}"
                    )
                st.latex(r"E_{\text{ventana}} = " + sp.latex(dev["total_energy"]))

                st.markdown("**Paso 2 — Valor medio en la ventana:**")
                st.latex(
                    r"\Delta T = " + sp.latex(dev["window_max"])
                    + r" - (" + sp.latex(dev["window_min"]) + r") = "
                    + sp.latex(dev["duration"])
                )
                for m_step in dev["mean_steps"]:
                    st.latex(
                        f"\\text{{Aporte tramo {m_step['index']} }} ({sp.latex(m_step['lower'])} \\le t \\le {sp.latex(m_step['upper'])}): "
                        f"\\quad \\int_{{{sp.latex(m_step['lower'])}}}^{{{sp.latex(m_step['upper'])}}} "
                        f"({sp.latex(m_step['expression'])})\\,dt = {sp.latex(m_step['integral_val'])}"
                    )
                st.latex(
                    r"\langle x(t) \rangle_{\text{ventana}} = \frac{1}{\Delta T}\sum \int x(t)\,dt = "
                    r"\frac{" + sp.latex(dev["mean_integral_sum"]) + r"}{" + sp.latex(dev["duration"]) + r"} = "
                    + sp.latex(dev["average_mean"])
                )

                st.markdown("**Paso 3 — Potencia promedio por intervalos:**")
                for p_step in dev["power_steps"]:
                    st.latex(
                        f"P_{{{p_step['index']}}} = \\frac{{E_{{{p_step['index']}}}}}{{\\Delta T}} = "
                        f"\\frac{{{sp.latex(p_step['energy_result'])}}}{{{sp.latex(p_step['duration'])}}} = "
                        f"{sp.latex(p_step['power_contribution'])}"
                    )
                st.latex(
                    r"P_{\text{ventana}} = \frac{E_{\text{ventana}}}{\Delta T} = "
                    r"\frac{" + sp.latex(dev["total_energy"]) + r"}{" + sp.latex(dev["duration"]) + r"} = "
                    + sp.latex(dev["average_power"])
                )

                st.markdown("**Paso 4 — Valor eficaz / RMS:**")
                st.latex(
                    r"X_{\mathrm{RMS}} = \sqrt{P_{\text{ventana}}} = \sqrt{"
                    + sp.latex(dev["average_power"]) + r"} = " + sp.latex(dev["rms"])
                )

        fig_ep, (ax_e, ax_p) = plt.subplots(1, 2, figsize=(9, 3.5))

        ax_e.plot(metrics["T_axis"], metrics["E_T_curve"], color="#2563EB", linewidth=2)
        ax_e.set_title("Energía Acumulada E(T) vs Ventana T", fontsize=10, fontweight="bold")
        ax_e.set_xlabel("Tamaño de ventana T (s)")
        ax_e.set_ylabel("E(T)")
        ax_e.grid(True, linestyle=":", alpha=0.6)

        ax_p.plot(metrics["T_axis"], metrics["P_T_curve"], color="#D97706", linewidth=2)
        ax_p.set_title("Potencia Promedio P(T) vs Ventana T", fontsize=10, fontweight="bold")
        ax_p.set_xlabel("Tamaño de ventana T (s)")
        ax_p.set_ylabel("P(T)")
        ax_p.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        st.pyplot(fig_ep)


# -----------------------------------------------------------------------------
# TAB 3: OPERACIONES, MANIPULACIÓN Y SIMETRÍA
# -----------------------------------------------------------------------------
with tab3:
    st.header("3. Operaciones, Manipulación y Propiedades de Simetría")
    st.write("Realiza transformaciones combinadas y calcula las componentes Par e Impar.")

    col_op1, col_op2 = st.columns([1, 2])

    with col_op1:
        st.subheader("Configuración de la Señal Base")
        sig_base_choice = st.selectbox(
            "Selecciona la señal base:",
            [
                "Triángulo x1(t) por intervalos [-2, 2]",
                "Pulso triangular por intervalos [-1, 1]",
                "Pulso por intervalos [-1, 1]",
                "Exponencial por intervalos t >= 0",
                "Por Tramos (Personalizada)",
            ],
            key="t3_sig_choice",
        )

        if sig_base_choice == "Por Tramos (Personalizada)":
            pieces_text_t3 = st.text_area(
                "Ingresa tramos para x(t):",
                value="t + 2, -2, 0\n-t + 2, 0, 2",
                key="t3_pieces",
            )
            pieces_t3 = []
            for line in pieces_text_t3.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 3:
                        pieces_t3.append((parts[0], (float(parts[1]), float(parts[2]))))
            x_base_func = lambda t_arr: parse_piecewise_signal(pieces_t3, t_arr)
        elif "Triángulo x1(t)" in sig_base_choice:
            pieces_t3 = [("t + 2", (-2, 0)), ("-t + 2", (0, 2))]
            x_base_func = lambda t_arr: parse_piecewise_signal(pieces_t3, t_arr)
        elif "Pulso triangular" in sig_base_choice:
            x_base_func = lambda t_arr: BasicSignal.tri(t_arr, center=0.0, width=2.0)
        elif "Pulso por intervalos" in sig_base_choice:
            x_base_func = lambda t_arr: BasicSignal.rect(t_arr, center=0.0, width=2.0)
        else:
            x_base_func = lambda t_arr: BasicSignal.exponential(t_arr, t0=0.0, alpha=-1.0)

        st.subheader("Transformación: y(t) = A · x(a·t - b) + C")
        op_A = st.number_input("Modificación de Amplitud (A):", value=1.0, step=0.5, key="t3_A")
        op_a = st.number_input("Escalamiento / Inversión Temporal (a):", value=1.0, step=0.5, key="t3_a")
        op_b = st.number_input("Desplazamiento Temporal (b):", value=0.0, step=0.5, key="t3_b")
        op_C = st.number_input("Desplazamiento Vertical (C):", value=0.0, step=0.5, key="t3_C")

    with col_op2:
        y_orig = x_base_func(t_domain)
        y_trans = evaluate_transformed_signal(
            x_base_func, t_domain, amp_A=op_A, scale_a=op_a, shift_b=op_b, offset_C=op_C
        )

        fig2, ax2 = plt.subplots(figsize=(8, 3.8))
        ax2.plot(t_domain, y_orig, color="#9CA3AF", linewidth=1.5, linestyle="--", label="Original x(t)")
        ax2.plot(
            t_domain,
            y_trans,
            color="#2563EB",
            linewidth=2.2,
            label=f"y(t) = {op_A}·x({op_a}t - {op_b}) + {op_C}",
        )
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

        with st.expander("Ver Descomposición en Componente Par xe(t) e Impar xo(t)"):
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
# TAB 4: RESPUESTA IMPULSIVA Y CONVOLUCIÓN CONTINUA
# -----------------------------------------------------------------------------
with tab4:
    st.header("4. Respuesta Impulsiva y Convolución Continua")
    st.write("Calculadora interactiva paso a paso de la integral de convolución y(t) = x(t) * h(t).")

    col_c1, col_c2 = st.columns([1, 2])

    with col_c1:
        st.markdown("**Configuración de las Señales x(t) y h(t) por intervalos:**")

        sig_x_type = st.selectbox(
            "Señal de Entrada x(t):",
            [
                "Triángulo x1(t) por intervalos [-2, 2]",
                "Pulso por intervalos [0, 2]",
                "Triángulo por intervalos [0, 2]",
                "Exponencial por intervalos t >= 0",
                "Definida por Tramos (Personalizada)",
            ],
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
        elif "Triángulo x1(t)" in sig_x_type:
            x_pieces = [("t + 2", (-2, 0)), ("-t + 2", (0, 2))]
            x_func = lambda t_arr: parse_piecewise_signal(x_pieces, t_arr)
        elif "Pulso por intervalos" in sig_x_type:
            x_pieces = [("1", (0, 2))]
            x_func = lambda t_arr: BasicSignal.rect(t_arr, center=1.0, width=2.0)
        elif "Triángulo por intervalos" in sig_x_type:
            x_pieces = [("t", (0, 1)), ("2 - t", (1, 2))]
            x_func = lambda t_arr: BasicSignal.tri(t_arr, center=1.0, width=2.0)
        else:
            x_pieces = [("exp(-t)", (0, 10))]
            x_func = lambda t_arr: BasicSignal.exponential(t_arr, t0=0.0, alpha=-1.0)

        sig_h_type = st.selectbox(
            "Respuesta Impulsiva h(t):",
            [
                "Pulso por intervalos [3, 5]",
                "Pulso por intervalos [0, 1]",
                "Escalón por intervalos t >= 0",
                "Triángulo por intervalos [0, 1]",
                "Definida por Tramos (Personalizada)",
            ],
            key="t4_h_type",
        )

        if sig_h_type == "Definida por Tramos (Personalizada)":
            h_pieces_text = st.text_area(
                "Ingresa tramos para h(t) (expr, t_min, t_max):",
                value="1, 3, 5",
                key="t4_h_text",
            )
            h_pieces = []
            for line in h_pieces_text.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 3:
                        h_pieces.append((parts[0], (float(parts[1]), float(parts[2]))))
            h_func = lambda t_arr: parse_piecewise_signal(h_pieces, t_arr)
        elif "[3, 5]" in sig_h_type:
            h_pieces = [("1", (3, 5))]
            h_func = lambda t_arr: parse_piecewise_signal(h_pieces, t_arr)
        elif "[0, 1]" in sig_h_type and "Pulso" in sig_h_type:
            h_pieces = [("1", (0, 1))]
            h_func = lambda t_arr: BasicSignal.rect(t_arr, center=0.5, width=1.0)
        elif "Escalón" in sig_h_type:
            h_pieces = [("1", (0, 10))]
            h_func = lambda t_arr: BasicSignal.step(t_arr)
        else:
            h_pieces = [("2*t", (0, 0.5)), ("2 - 2*t", (0.5, 1.0))]
            h_func = lambda t_arr: BasicSignal.tri(t_arr, center=0.5, width=1.0)

        t_shift = st.slider(
            "Desplazamiento t (tiempo actual de convolución):",
            min_value=-4.0,
            max_value=9.0,
            value=3.5,
            step=0.1,
        )

    with col_c2:
        tau_axis = np.linspace(-6, 9, 1500)
        t_eval_axis = np.linspace(-4, 9, 350)

        x_tau, h_shifted, prod_tau, y_curr = continuous_convolution_at_t(x_func, h_func, t_shift, tau_axis)
        y_full_curve = compute_full_numerical_convolution(x_func, h_func, t_eval_axis, tau_axis)

        fig_conv, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
        fig_conv.subplots_adjust(hspace=0.35)

        axes[0].plot(tau_axis, x_tau, color="#2563EB", linewidth=2, label="x(τ)")
        axes[0].plot(tau_axis, h_func(tau_axis), color="#D97706", linewidth=2, linestyle="--", label="h(τ)")
        axes[0].set_title("1. Señales Originales en la variable de integración τ", fontsize=10, fontweight="bold")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, linestyle=":", alpha=0.6)

        axes[1].plot(tau_axis, x_tau, color="#2563EB", linewidth=2, label="x(τ)")
        axes[1].plot(tau_axis, h_shifted, color="#DC2626", linewidth=2, label=f"h({t_shift:.1f} - τ)")
        axes[1].set_title(f"2. Inversión y Desplazamiento: h(t - τ) para t = {t_shift:.2f}", fontsize=10, fontweight="bold")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, linestyle=":", alpha=0.6)

        axes[2].plot(tau_axis, prod_tau, color="#059669", linewidth=2, label="Producto p(τ) = x(τ)·h(t-τ)")
        axes[2].fill_between(tau_axis, 0, prod_tau, color="#34D399", alpha=0.5, label=f"Área = {y_curr:.4f}")
        axes[2].set_title("3. Producto de Solapamiento p(τ) (Área sombreada)", fontsize=10, fontweight="bold")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, linestyle=":", alpha=0.6)

        axes[3].plot(t_eval_axis, y_full_curve, color="#7C3AED", linewidth=2.2, label="y(t) = (x * h)(t)")
        axes[3].plot(t_shift, y_curr, "o", color="#DC2626", markersize=8, label=f"y({t_shift:.2f}) = {y_curr:.4f}")
        axes[3].axvline(t_shift, color="#DC2626", linestyle=":", alpha=0.8)
        axes[3].set_title("4. Resultado de la Integral de Convolución y(t)", fontsize=10, fontweight="bold")
        axes[3].set_xlabel("Tiempo (t / τ)")
        axes[3].legend(loc="upper right")
        axes[3].grid(True, linestyle=":", alpha=0.6)

        st.pyplot(fig_conv)

        st.subheader("Expresión Analítica por Intervalos (SymPy)")
        try:
            spanish_latex, merged_pieces = symbolic_piecewise_convolution(x_pieces, h_pieces)
            st.latex(r"y(t) = " + spanish_latex)

            with st.expander("📋 Ver Tramos Resultantes"):
                for expr_item, (t1_item, t2_item) in merged_pieces:
                    st.code(f"y(t) = {expr_item}   para   {t1_item} <= t <= {t2_item}", language="text")
        except Exception:
            st.info("No se pudo obtener la expresión analítica exacta para las funciones seleccionadas.")


# -----------------------------------------------------------------------------
# TAB 5: ANÁLISIS DE SISTEMAS (OPCIONAL)
# -----------------------------------------------------------------------------
with tab5:
    st.header("5. Sistemas: Conceptos y Propiedades (Opcional)")
    st.write(
        "Analiza las 5 propiedades fundamentales de sistemas SISO continuos: "
        "Linealidad, Invariancia Temporal, Causalidad, Memoria y Estabilidad BIBO."
    )

    sys_choice = st.selectbox(
        "Selecciona un sistema de estudio o ingresa uno personalizado:",
        [
            "y(t) = 3 * x(t)",
            "y(t) = x(t - 2)",
            "y(t) = x(2 * t)",
            "y(t) = x(t)^2",
            "y(t) = t * x(t)",
            "y(t) = x(-t)",
            "y(t) = Integral(x(tau), tau, -inf, t)",
            "Sistema Personalizado (Ecuación)",
        ],
    )

    if sys_choice == "Sistema Personalizado (Ecuación)":
        custom_expr = st.text_input("Ingresa la ecuación del sistema en función de x(t) y t:", value="2*x(t) + 1")
        analysis = SystemAnalyzer.analyze_custom(custom_expr)
    else:
        analysis = SystemAnalyzer.analyze_preset(sys_choice)

    if analysis and "error" not in analysis:
        st.subheader("Dictamen y Justificación Teórica de Propiedades")

        p_lin, text_lin = analysis["linearity"]
        p_ti, text_ti = analysis["time_invariance"]
        p_cau, text_cau = analysis["causality"]
        p_mem, text_mem = analysis["memory"]
        p_sta, text_sta = analysis["stability"]

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown(f"**1. Linealidad:** {'✅ LINEAL' if p_lin else '❌ NO LINEAL'}")
            st.info(text_lin)

            st.markdown(f"**2. Invariancia Temporal:** {'✅ INVARIANTE' if p_ti else '❌ VARIANTE'}")
            st.info(text_ti)

            st.markdown(f"**3. Causalidad:** {'✅ CAUSAL' if p_cau else '❌ NO CAUSAL'}")
            st.info(text_cau)

        with col_s2:
            st.markdown(f"**4. Memoria:** {'🧠 CON MEMORIA (Dinámico)' if p_mem else '⚡ SIN MEMORIA (Instantáneo)'}")
            st.info(text_mem)

            st.markdown(f"**5. Estabilidad BIBO:** {'✅ ESTABLE' if p_sta else '❌ INESTABLE'}")
            st.info(text_sta)

# -----------------------------------------------------------------------------
# TAB 6: SERIES DE FOURIER (SEF Y STF)
# -----------------------------------------------------------------------------
with tab6:
    st.header("6. Series de Fourier (Exponencial y Trigonométrica)")
    st.write(
        "Calcula los coeficientes de la **SEF** ($X_k$) y la **STF** ($X_0, X_{C_k}, X_{S_k}$), "
        "la aproximación con $N$ armónicos, los espectros de magnitud y fase, y la potencia por Parseval."
    )

    col_fs1, col_fs2 = st.columns([1, 2])

    with col_fs1:
        fs_preset = st.selectbox(
            "Señal periódica de ejemplo o personalizada:",
            [
                "Tren de pulsos rectangulares en [-1, 1] (T0 = 2)",
                "Onda triangular periódica en [-1, 1] (T0 = 2)",
                "Onda diente de sierra en [0, 2] (T0 = 2)",
                "Personalizada por tramos en un periodo",
            ],
            key="fs_preset",
        )

        if fs_preset == "Tren de pulsos rectangulares en [-1, 1] (T0 = 2)":
            fs_t0_start, fs_t0_end = -1.0, 1.0
            fs_pieces_default = "1, -0.5, 0.5"
        elif fs_preset == "Onda triangular periódica en [-1, 1] (T0 = 2)":
            fs_t0_start, fs_t0_end = -1.0, 1.0
            fs_pieces_default = "1 + t, -1, 0\n1 - t, 0, 1"
        elif fs_preset == "Onda diente de sierra en [0, 2] (T0 = 2)":
            fs_t0_start, fs_t0_end = 0.0, 2.0
            fs_pieces_default = "t, 0, 2"
        else:
            fs_t0_start, fs_t0_end = -1.0, 1.0
            fs_pieces_default = "1, -0.5, 0.5"

        c_p1, c_p2 = st.columns(2)
        per_start = c_p1.number_input("Inicio periodo t_a:", value=fs_t0_start, step=0.5, key="fs_per_a")
        per_end = c_p2.number_input("Fin periodo t_b:", value=fs_t0_end, step=0.5, key="fs_per_b")

        fs_text = st.text_area(
            "Tramos dentro del periodo (expr, t_min, t_max):",
            value=fs_pieces_default,
            key="fs_pieces_text",
        )
        fs_pieces = []
        for line in fs_text.strip().split("\n"):
            if line.strip():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 3:
                    fs_pieces.append((parts[0], (float(parts[1]), float(parts[2]))))

        max_k_harm = st.slider("Rango de armónicos |k| para espectros:", min_value=1, max_value=10, value=5, key="fs_max_k")
        n_synth = st.slider("Número de armónicos N para reconstruir x_N(t):", min_value=1, max_value=25, value=7, key="fs_n_synth")

    with col_fs2:
        try:
            fs_data = compute_fourier_series_data(
                fs_pieces,
                t0_start=per_start,
                t0_end=per_end,
                max_k=max_k_harm,
                n_harmonics=n_synth,
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Periodo T0", f"{fs_data['T0_num']:.4f}")
            m2.metric("Frecuencia ω0", f"{fs_data['omega0_num']:.4f} rad/s")
            m3.metric("Potencia P_x (tiempo)", f"{fs_data['Px_num']:.4f}")
            m4.metric(f"Parseval (|k|≤{max_k_harm})", f"{fs_data['parseval_partial_num']:.4f}")

            st.markdown("**Parámetros fundamentales y coeficiente DC ($X_0$):**")
            st.latex(
                r"T_0 = " + sp.latex(fs_data["T0"])
                + r", \quad f_0 = " + sp.latex(fs_data["f0"])
                + r", \quad \omega_0 = " + sp.latex(fs_data["omega0"])
                + r", \quad X_0 = " + sp.latex(fs_data["X0"])
            )

            st.markdown("**Fórmula general de la SEF ($X_k$ para $k \neq 0$):**")
            st.latex(
                r"X_k = \frac{1}{T_0}\int_{" + sp.latex(fs_data["a"]) + r"}^{" + sp.latex(fs_data["b"])
                + r"} \tilde{x}(t)e^{-jk\omega_0 t}\,dt = " + sp.latex(fs_data["Xk_general"])
            )

            t_fs_axis = np.linspace(per_start - 1.5 * fs_data["T0_num"], per_end + 1.5 * fs_data["T0_num"], 1800)
            y_periodic_true = evaluate_periodic_piecewise(fs_pieces, per_start, per_end, t_fs_axis)
            y_periodic_synth = np.asarray(fs_data["approx_func"](t_fs_axis), dtype=float)

            fig_fs, ax_fs = plt.subplots(figsize=(8.5, 3.6))
            ax_fs.plot(t_fs_axis, y_periodic_true, color="#2563EB", linewidth=2, label="Señal periódica x̃(t)")
            ax_fs.plot(t_fs_axis, y_periodic_synth, color="#DC2626", linewidth=1.8, linestyle="--", label=f"Aproximación N={n_synth}")
            ax_fs.axhline(0, color="black", linewidth=0.8)
            ax_fs.set_title("Señal Periódica y Síntesis con N Armónicos", fontsize=10.5, fontweight="bold")
            ax_fs.set_xlabel("Tiempo t (s)")
            ax_fs.grid(True, linestyle=":", alpha=0.6)
            ax_fs.legend(loc="upper right")
            st.pyplot(fig_fs)

            k_vals_arr = np.array(fs_data["k_range"], dtype=float)
            mag_vals_arr = np.array([fs_data["harmonics"][int(kv)]["mag_num"] for kv in k_vals_arr])
            phase_vals_arr = np.array([fs_data["harmonics"][int(kv)]["phase_num"] for kv in k_vals_arr])

            fig_sp, (ax_m, ax_ph) = plt.subplots(1, 2, figsize=(9, 3.2))
            ax_m.stem(k_vals_arr, mag_vals_arr, basefmt=" ")
            ax_m.set_title("Espectro de Magnitud |X_k|", fontsize=10, fontweight="bold")
            ax_m.set_xlabel("Armónico k")
            ax_m.grid(True, linestyle=":", alpha=0.6)

            ax_ph.stem(k_vals_arr, phase_vals_arr, basefmt=" ")
            ax_ph.set_title("Espectro de Fase ∠X_k (rad)", fontsize=10, fontweight="bold")
            ax_ph.set_xlabel("Armónico k")
            ax_ph.grid(True, linestyle=":", alpha=0.6)

            plt.tight_layout()
            st.pyplot(fig_sp)

            with st.expander("📋 Coeficientes SEF (X_k) y STF (X_{C_k}, X_{S_k}) por armónico"):
                for kv in range(1, max_k_harm + 1):
                    item_k = fs_data["harmonics"][kv]
                    st.latex(
                        f"k={kv}: \\quad X_{{{kv}}} = {sp.latex(item_k['Xk'])}, "
                        f"\\quad X_{{C_{kv}}} = {sp.latex(item_k['XC'])}, "
                        f"\\quad X_{{S_{kv}}} = {sp.latex(item_k['XS'])}"
                    )

        except Exception as exc:
            st.error(f"No se pudo calcular la Serie de Fourier: {exc}")