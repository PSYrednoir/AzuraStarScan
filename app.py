import streamlit as st
import joblib
import numpy as np

# Configuración de página
st.set_page_config(
    page_title="Azura StarScan",
    page_icon="🌟",
    layout="centered"
)

# ============ FONDO Y ESTILO ============
st.markdown("""
<style>
/* Fondo espacial oscuro */
.stApp {
    background: linear-gradient(135deg, #0a0a2e 0%, #1a1a4e 40%, #0d0d3b 100%);
    color: white;
}

/* Estrellas animadas en el fondo */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image: 
        radial-gradient(1px 1px at 10% 20%, white, transparent),
        radial-gradient(1px 1px at 30% 50%, white, transparent),
        radial-gradient(1px 1px at 50% 10%, white, transparent),
        radial-gradient(1px 1px at 70% 80%, white, transparent),
        radial-gradient(1px 1px at 90% 30%, white, transparent),
        radial-gradient(1px 1px at 20% 70%, white, transparent),
        radial-gradient(1px 1px at 60% 40%, white, transparent),
        radial-gradient(1px 1px at 80% 60%, white, transparent),
        radial-gradient(2px 2px at 15% 85%, #a0c4ff, transparent),
        radial-gradient(2px 2px at 45% 25%, #ffd6ff, transparent),
        radial-gradient(2px 2px at 75% 55%, #caffbf, transparent);
    pointer-events: none;
    z-index: 0;
}

/* Títulos */
h1, h2, h3 {
    color: #c9a8ff !important;
    text-shadow: 0 0 20px #7b2fff;
}

/* Tarjetas de inputs */
.stNumberInput > div {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 10px;
    border: 1px solid rgba(160, 100, 255, 0.3);
}

/* Botón principal */
.stButton > button {
    background: linear-gradient(90deg, #7b2fff, #00b4d8);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 12px 40px;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 0 20px rgba(123, 47, 255, 0.5);
}
.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 35px rgba(123, 47, 255, 0.8);
}

/* Métrica */
.stMetric {
    background: rgba(123, 47, 255, 0.15);
    border-radius: 12px;
    padding: 10px;
    border: 1px solid rgba(123, 47, 255, 0.4);
}

/* Links */
a {
    color: #00b4d8 !important;
}

/* Separador */
hr {
    border-color: rgba(123, 47, 255, 0.3) !important;
}

/* Labels de inputs */
label {
    color: #c9a8ff !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ============ CARGAR MODELOS ============
rf = joblib.load("modelos/modelo_rf.pkl")
le = joblib.load("modelos/label_encoder.pkl")

# ============ HEADER ============
st.markdown("""
<div style='text-align: center; padding: 20px 0;'>
    <h1 style='font-size: 3em;'>🌟 Azura StarScan</h1>
    <p style='color: #a0c4ff; font-size: 1.1em;'>
        Clasificador de Exoplanetas — Datos del Telescopio Kepler · NASA
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

col_info1, col_info2 = st.columns(2)
with col_info1:
    st.markdown("👩‍🚀 **Allison Carolina Negreiros Castillo**")
    st.markdown("🎓 NRC: 6817")
with col_info2:
    st.markdown("[📓 Ver cuaderno en Google Colab](TU_LINK_COLAB_AQUI)")

st.markdown("---")

# ============ DESCRIPCIÓN ============
st.markdown("""
<div style='background: rgba(123,47,255,0.1); border-radius: 15px; 
            padding: 20px; border-left: 4px solid #7b2fff; margin-bottom: 20px;'>
    <h4 style='color: #c9a8ff;'>🔭 ¿Qué hace Azura StarScan?</h4>
    <p style='color: #ddd;'>
    Analiza las características físicas y orbitales de objetos detectados por el 
    telescopio Kepler para predecir si son <strong style='color:#caffbf'>exoplanetas confirmados</strong> 
    o <strong style='color:#ff9999'>falsos positivos</strong>, usando un modelo 
    de Random Forest entrenado con datos reales de la NASA.
    </p>
</div>
""", unsafe_allow_html=True)

# ============ INPUTS ============
st.markdown("### 🪐 Características del objeto")
st.markdown("""
<p style='color: #a0c4ff; font-size: 0.9em;'>
💡 No sabes qué valores ingresar? Usa los valores por defecto como ejemplo 
de un exoplaneta típico.
</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    periodo = st.number_input("⏱️ Período orbital (días)",
                               min_value=0.1, max_value=1000.0, value=10.0)
    st.caption("Tiempo que tarda el planeta en dar una vuelta completa a su estrella. La Tierra tiene 365 días.")

    radio = st.number_input("📏 Radio del planeta (R⊕)",
                             min_value=0.1, max_value=100.0, value=2.0)
    st.caption("Tamaño del planeta comparado con la Tierra. 1.0 = igual que la Tierra, 11.0 = igual que Júpiter.")

    temperatura = st.number_input("🌡️ Temperatura de equilibrio (K)",
                                   min_value=100, max_value=5000, value=800)
    st.caption("Temperatura estimada de la superficie en Kelvin. La Tierra tiene ~255 K. Más de 1000 K es muy caliente.")

    insolacion = st.number_input("☀️ Flujo de insolación",
                                  min_value=0.0, max_value=10000.0, value=100.0)
    st.caption("Cantidad de energía que recibe el planeta de su estrella comparado con la Tierra. 1.0 = igual que la Tierra.")

with col2:
    snr = st.number_input("📡 Señal-ruido del tránsito",
                           min_value=0.0, max_value=1000.0, value=20.0)
    st.caption("Qué tan clara fue la señal del planeta al pasar frente a su estrella. Valores más altos = señal más confiable.")

    temp_estelar = st.number_input("⭐ Temperatura estelar (K)",
                                    min_value=3000, max_value=10000, value=5500)
    st.caption("Temperatura de la estrella anfitriona. El Sol tiene ~5778 K. Estrellas más frías son naranjas o rojas.")

    radio_estelar = st.number_input("🌞 Radio estelar (R☉)",
                                     min_value=0.1, max_value=10.0, value=1.0)
    st.caption("Tamaño de la estrella comparado con el Sol. 1.0 = igual que el Sol.")
# ============ PREDICCIÓN ============
if st.button("🚀 Analizar objeto estelar"):
    entrada = np.array([[periodo, radio, temperatura,
                         insolacion, snr, temp_estelar, radio_estelar]])
    prediccion = rf.predict(entrada)
    probabilidad = rf.predict_proba(entrada)
    resultado = le.inverse_transform(prediccion)[0]
    confianza = max(probabilidad[0]) * 100

    if resultado == "CONFIRMED":
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(0,200,100,0.2), rgba(0,100,50,0.2));
                    border-radius: 15px; padding: 25px; text-align: center;
                    border: 2px solid #00c864; margin: 10px 0;'>
            <h2 style='color: #caffbf;'>✅ EXOPLANETA CONFIRMADO</h2>
            <p style='color: #aaa;'>Este objeto presenta características consistentes con un planeta real.</p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(200,0,0,0.2), rgba(100,0,0,0.2));
                    border-radius: 15px; padding: 25px; text-align: center;
                    border: 2px solid #ff4444; margin: 10px 0;'>
            <h2 style='color: #ffaaaa;'>❌ FALSO POSITIVO</h2>
            <p style='color: #aaa;'>Este objeto no corresponde a un exoplaneta real.</p>
        </div>
        """, unsafe_allow_html=True)

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("🎯 Confianza", f"{confianza:.1f}%")
    with col_m2:
        st.metric("🌡️ Temp. ingresada", f"{temperatura} K")
    with col_m3:
        st.metric("📏 Radio ingresado", f"{radio} R⊕")

# ============ FOOTER ============
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #555; font-size: 0.85em;'>
    Azura StarScan © 2026 · Datos: NASA Kepler Objects of Interest
</div>
""", unsafe_allow_html=True)
