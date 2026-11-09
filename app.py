"""app.py
Capa de presentación (Streamlit) del Perceptrón Monocapa – compuerta AND.
Ejecutar con:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from pdf_generator import generate_pdf
from perceptron import Perceptron

st.set_page_config(page_title="Perceptrón AND", page_icon="🧠", layout="wide")


# ---------------------------------------------------------------- utilidades de vista
def n(v: float) -> str:
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def p(v: float) -> str:
    """Número para LaTeX, entre paréntesis si es negativo."""
    return rf"\left({n(v)}\right)" if v < 0 else n(v)


def network_svg(w1, w2, b, x1=None, x2=None, z=None, y=None) -> str:
    """Esquema del perceptrón como SVG (se actualiza con pesos, entradas, z e y)."""
    ink, blue, orange, green = "#1f2937", "#2563eb", "#d97706", "#059669"
    fx = lambda v: "x" if v is None else n(v)
    zt = "z" if z is None else f"z = {n(z)}"
    yt = "y" if y is None else f"y = {y}"
    yfill = "#ffffff" if y is None else ("#bbf7d0" if y == 1 else "#fecaca")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 320" width="760" height="320" font-family="Arial, sans-serif">
<rect width="760" height="320" rx="12" fill="#ffffff" stroke="#d1d5db"/>
<line x1="110" y1="110" x2="355" y2="170" stroke="{blue}" stroke-width="2"/>
<line x1="110" y1="240" x2="355" y2="190" stroke="{blue}" stroke-width="2"/>
<line x1="300" y1="45" x2="375" y2="140" stroke="{orange}" stroke-width="2"/>
<line x1="430" y1="180" x2="520" y2="180" stroke="{ink}" stroke-width="2"/>
<line x1="610" y1="180" x2="665" y2="180" stroke="{ink}" stroke-width="2"/>
<circle cx="80" cy="110" r="30" fill="#eff6ff" stroke="{blue}" stroke-width="2"/>
<text x="80" y="106" text-anchor="middle" font-size="15" fill="{ink}">x₁</text>
<text x="80" y="124" text-anchor="middle" font-size="13" fill="{blue}">{fx(x1)}</text>
<circle cx="80" cy="240" r="30" fill="#eff6ff" stroke="{blue}" stroke-width="2"/>
<text x="80" y="236" text-anchor="middle" font-size="15" fill="{ink}">x₂</text>
<text x="80" y="254" text-anchor="middle" font-size="13" fill="{blue}">{fx(x2)}</text>
<circle cx="275" cy="45" r="28" fill="#fffbeb" stroke="{orange}" stroke-width="2"/>
<text x="275" y="41" text-anchor="middle" font-size="15" fill="{ink}">b</text>
<text x="275" y="58" text-anchor="middle" font-size="13" fill="{orange}">{n(b)}</text>
<text x="215" y="125" text-anchor="middle" font-size="14" fill="{blue}" font-weight="bold">w₁ = {n(w1)}</text>
<text x="215" y="232" text-anchor="middle" font-size="14" fill="{blue}" font-weight="bold">w₂ = {n(w2)}</text>
<circle cx="395" cy="180" r="38" fill="#f9fafb" stroke="{ink}" stroke-width="2"/>
<text x="395" y="188" text-anchor="middle" font-size="30" fill="{ink}">Σ</text>
<text x="395" y="240" text-anchor="middle" font-size="13" fill="{ink}">{zt}</text>
<rect x="520" y="145" width="90" height="70" rx="8" fill="#f9fafb" stroke="{ink}" stroke-width="2"/>
<polyline points="533,195 565,195 565,165 597,165" fill="none" stroke="{green}" stroke-width="3"/>
<text x="565" y="232" text-anchor="middle" font-size="12" fill="{ink}">Escalón f(z)</text>
<circle cx="695" cy="180" r="30" fill="{yfill}" stroke="{ink}" stroke-width="2"/>
<text x="695" y="185" text-anchor="middle" font-size="15" fill="{ink}">{yt}</text>
<text x="380" y="300" text-anchor="middle" font-size="12" fill="#6b7280">z = x₁·w₁ + x₂·w₂ + b  →  y = f(z) = 1 si z ≥ 0, 0 si z &lt; 0</text>
</svg>"""


# ---------------------------------------------------------------- barra lateral (REQ 2)
st.title("🧠 Perceptrón Monocapa – Compuerta AND")
st.caption("Simulación pedagógica del aprendizaje paso a paso (regla delta, activación escalón).")

with st.sidebar:
    st.header("Hiperparámetros iniciales")
    w1_0 = st.number_input("Peso inicial w₁", value=0.1, step=0.1, format="%.3f")
    w2_0 = st.number_input("Peso inicial w₂", value=0.3, step=0.1, format="%.3f")
    b_0 = st.number_input("Bias inicial b", value=0.2, step=0.1, format="%.3f")
    eta = st.number_input("Tasa de aprendizaje η", value=0.1, min_value=0.001, step=0.01, format="%.3f")
    max_epochs = st.slider("Límite máximo de épocas", 1, 100, 4)
    train_clicked = st.button("▶ Entrenar perceptrón", type="primary", width="stretch")

if train_clicked:
    st.session_state["model"] = Perceptron(w1_0, w2_0, b_0, eta).train(max_epochs)

model: Perceptron | None = st.session_state.get("model")

# ---------------------------------------------------------------- REQ 1: esquema
st.subheader("1. Esquema de la red")
if model is None:
    st.image(network_svg(w1_0, w2_0, b_0))
    st.info("Configura los valores en la barra lateral y pulsa **Entrenar perceptrón**.")
    st.stop()

if model.converged:
    st.success(f"Convergió en {len(model.epochs)} época(s): la última no tuvo errores.")
else:
    st.warning(f"No convergió en {len(model.epochs)} época(s). Aumenta el límite de épocas.")
st.image(network_svg(model.w1, model.w2, model.b))
st.caption("Pesos y bias mostrados: valores finales tras el entrenamiento.")

# ---------------------------------------------------------------- REQ 4: tabla
st.subheader("2. Tabla resumen por épocas")
df = pd.DataFrame([
    {"Época": e.epoch, "w₁": e.w1, "w₂": e.w2, "b": e.b, "Errores": e.errors}
    for e in model.epochs
])
st.dataframe(df, hide_index=True, width="stretch")

# ---------------------------------------------------------------- REQ 5: PDF
st.download_button("📄 Descargar informe PDF", data=generate_pdf(model),
                   file_name="informe_perceptron_AND.pdf", mime="application/pdf")

# ---------------------------------------------------------------- REQ 3: procedimiento
st.subheader("3. Procedimiento matemático paso a paso")
st.latex(r"z=(x_1\cdot w_1)+(x_2\cdot w_2)+b \quad y=f(z)=\begin{cases}1 & z\geq 0\\0 & z<0\end{cases}"
         r"\quad e=t-y")
st.latex(r"w^{(n)}=w^{(prev)}+\eta\cdot e\cdot x \qquad b^{(n)}=b^{(prev)}+\eta\cdot e")

for ep in model.epochs:
    with st.expander(f"Época {ep.epoch} — errores: {ep.errors}", expanded=True):
        for s in model.steps_of_epoch(ep.epoch):
            st.markdown(f"**Muestra {s.sample}** — entrada $({n(s.x1)},\\,{n(s.x2)})$, objetivo $t={s.t}$ · "
                        f"$w_1={n(s.w1_prev)},\\ w_2={n(s.w2_prev)},\\ b={n(s.b_prev)}$")
            st.latex(rf"z=({n(s.x1)}\cdot {p(s.w1_prev)})+({n(s.x2)}\cdot {p(s.w2_prev)})+{p(s.b_prev)}={n(s.z)}")
            st.latex(rf"y=f({n(s.z)})={s.y}\qquad e={s.t}-{s.y}={s.e}")
            if s.e == 0:
                st.markdown("_e = 0 → no hay actualización._")
            else:
                st.latex(rf"w_1^{{(n)}}={p(s.w1_prev)}+{n(model.eta)}\cdot {p(s.e)}\cdot {n(s.x1)}={n(s.w1_new)}")
                st.latex(rf"w_2^{{(n)}}={p(s.w2_prev)}+{n(model.eta)}\cdot {p(s.e)}\cdot {n(s.x2)}={n(s.w2_new)}")
                st.latex(rf"b^{{(n)}}={p(s.b_prev)}+{n(model.eta)}\cdot {p(s.e)}={n(s.b_new)}")
            st.divider()
        st.markdown(f"**Fin de época {ep.epoch}:** $w_1={n(ep.w1)},\\ w_2={n(ep.w2)},\\ b={n(ep.b)}$, "
                    f"errores = {ep.errors}")

# ---------------------------------------------------------------- REQ 6: inferencia
st.subheader("4. Módulo de inferencia / prueba")
c1, c2 = st.columns([1, 2])
with c1:
    xi1 = st.number_input("x₁", value=1.0, step=1.0, format="%.2f", key="inf_x1")
    xi2 = st.number_input("x₂", value=1.0, step=1.0, format="%.2f", key="inf_x2")
    z, y = model.predict(xi1, xi2)
    st.latex(rf"z=({n(xi1)}\cdot {p(model.w1)})+({n(xi2)}\cdot {p(model.w2)})+{p(model.b)}={n(z)}")
    st.latex(rf"y=f({n(z)})={y}")
    if xi1 in (0, 1) and xi2 in (0, 1):
        expected = int(xi1 == 1 and xi2 == 1)
        (st.success if y == expected else st.error)(f"Salida esperada de AND: {expected}")
with c2:
    st.image(network_svg(model.w1, model.w2, model.b, xi1, xi2, z, y))
