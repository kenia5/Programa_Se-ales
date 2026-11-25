# ⚡ Calculadora de Señales y Sistemas Continuos

Una aplicación interactiva moderna desarrollada en Python con **Streamlit**, **Matplotlib**, **NumPy**, **SciPy** y **SymPy**, diseñada específicamente para resolver ejercicios universitarios del curso **Señales y Sistemas Continuos** (Escuela Politécnica Nacional / EPN).

---

## 📌 Características Principales

La aplicación está dividida en 4 módulos interactivos basados en el temario oficial:

### 1. Conceptos y Señales Básicas
- Creación y graficación interactiva de señales elementales:
  - **Escalón Unitario** $u(t - t_0)$
  - **Impulso Unitario** $\delta(t - t_0)$
  - **Rampa Unitario** $r(t - t_0)$
  - **Pulso Rectangular** $\text{rect}((t-t_0)/W)$
  - **Pulso Triangular** $\text{tri}((t-t_0)/W)$
  - **Exponencial Causal** $A e^{\alpha (t-t_0)} u(t-t_0)$
  - **Senoidal** $A \sin(2\pi f t + \phi)$
  - **Señales Definidas por Tramos** (funciones a trozos personalizadas ingresadas por el usuario).
- Visualización gráfica de alta calidad y renderizado de la expresión analítica en LaTeX.

### 2. Operaciones, Manipulación y Propiedades de Señales
- Aplicación de transformaciones en tiempo y amplitud:
  - **Inversión temporal** $x(-t)$
  - **Desplazamiento temporal** $x(t - t_0)$ (retardos y adelantos)
  - **Escalamiento temporal** $x(a t)$ (compresión si $a > 1$, expansión si $0 < a < 1$)
  - **Modificación de amplitud y desplazamiento vertical** $y(t) = A x(a t - b) + C$
- Cálculos cuantitativos automáticos:
  - **Área bajo la señal** $\int_{-\infty}^{\infty} x(t) dt$
  - **Energía de la señal** $E_x = \int_{-\infty}^{\infty} |x(t)|^2 dt$
  - **Identificación de Simetría** (Par, Impar, Asimétrica)
  - **Descomposición en Componente Par** $x_e(t) = \frac{x(t) + x(-t)}{2}$ e **Impar** $x_o(t) = \frac{x(t) - x(-t)}{2}$.

### 3. Sistemas: Conceptos y Propiedades
- Evaluación automática y justificación teórica paso a paso de las 5 propiedades fundamentales de sistemas SISO continuos:
  1. **Linealidad** (Principio de Superposición y Homogeneidad)
  2. **Invariancia Temporal**
  3. **Causalidad**
  4. **Memoria** (Instantáneo vs. Dinámico)
  5. **Estabilidad BIBO** (Bounded-Input Bounded-Output)
- Incluye sistemas clásicos de examen ($y(t) = 3x(t)$, $y(t) = x(t-2)$, $y(t) = x(2t)$, $y(t) = x^2(t)$, $y(t) = t x(t)$, $y(t) = x(-t)$, integradores) y soporte para ecuaciones personalizadas.

### 4. Respuesta Impulsiva y Convolución Continua
- Herramienta interactiva para la integral de convolución $y(t) = (x * h)(t) = \int_{-\infty}^{\infty} x(\tau) h(t - \tau) d\tau$.
- **Panel gráfico dinámico en 4 etapas:**
  1. Señales originales $x(\tau)$ y $h(\tau)$ en la variable dummy de integración.
  2. Reflexión y desplazamiento $h(t - \tau)$ en tiempo real al mover el control deslizante (slider) de $t$.
  3. Producto punto a punto $p(\tau) = x(\tau) \cdot h(t - \tau)$ con el área de solapamiento sombreada.
  4. Curva resultado completa de $y(t)$ con marcador resaltado en la posición actual de $t$.
- **Solución numérica:** Valor numérico de $y(t)$ para el $t$ seleccionado.
- **Solución analítica por tramos:** Expresión simbólica exacta en SymPy para cada intervalo de tiempo.

---

## 🛠️ Estructura del Código Modular

```text
calculadora_senales/
├── app.py                     # Interfaz gráfica principal Streamlit
├── requirements.txt           # Lista de dependencias
├── README.md                  # Manual e instrucciones
└── modules/
    ├── signals.py             # Generación y evaluación de señales continuas
    ├── operations.py          # Transformaciones, área, energía y simetría
    ├── systems.py             # Analizador de propiedades de sistemas
    └── convolution.py         # Motor de convolución numérica y simbólica
```

---

## 🚀 Instalación y Ejecución

1. **Clonar o extraer los archivos del proyecto:**
   ```bash
   cd calculadora_senales
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar la aplicación:**
   ```bash
   streamlit run app.py
   ```

4. Abre la dirección web local indicada en tu navegador (por defecto `http://localhost:8501`).

---

## 📚 Referencia Académica
Groundeado en los contenidos del capítulo 2 de *Señales y Sistemas Continuos* (PhD. Patricio J. Cruz / Escuela Politécnica Nacional).
