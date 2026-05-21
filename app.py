import streamlit as st
import joblib
import numpy as np

# Configuración de página
st.set_page_config(
    page_title="Azura StarScan",
    page_icon="🌟",
    layout="centered"
)

# Cargar modelos
rf = joblib.load("modelos/modelo_rf.pkl")
le = joblib.load("modelos/label_encoder.pkl")

# Header
st.title("🌟 Azura StarScan")
st.markdown("### Clasificador de Exoplanetas — NASA Kepler")
st.markdown("---")
st.markdown("**Allison Carolina Negreiros Castillo** | NRC: 6817")
st.markdown("[📓 Ver cuaderno en Google Colab](TU_LINK_COLAB_AQUI)")
st.markdown("---")

st.markdown("""
Este modelo predice si un objeto de interés planetario (KOI)
detectado por el telescopio Kepler es un **exoplaneta confirmado**
o un **falso positivo**, basándose en sus características astronómicas.
""")

st.markdown("### 🔭 Ingresa las características del objeto")

col1, col2 = st.columns(2)

with col1:
    periodo = st.number_input("Período orbital (días)", 
                               min_value=0.1, max_value=1000.0, value=10.0)
    radio = st.number_input("Radio del planeta (R⊕)", 
                             min_value=0.1, max_value=100.0, value=2.0)
    temperatura = st.number_input("Temperatura de equilibrio (K)", 
                                   min_value=100, max_value=5000, value=800)

with col2:
    insolacion = st.number_input("Flujo de insolación", 
                                  min_value=0.0, max_value=10000.0, value=100.0)
    snr = st.number_input("Señal-ruido del tránsito", 
                           min_value=0.0, max_value=1000.0, value=20.0)
    temp_estelar = st.number_input("Temperatura estelar (K)", 
                                    min_value=3000, max_value=10000, value=5500)
    radio_estelar = st.number_input("Radio estelar (R☉)", 
                                     min_value=0.1, max_value=10.0, value=1.0)

st.markdown("---")

if st.button("🚀 Analizar objeto"):
    entrada = np.array([[periodo, radio, temperatura,
                         insolacion, snr, temp_estelar, radio_estelar]])
    prediccion = rf.predict(entrada)
    probabilidad = rf.predict_proba(entrada)
    resultado = le.inverse_transform(prediccion)[0]

    if resultado == "CONFIRMED":
        st.success(f"✅ **EXOPLANETA CONFIRMADO**")
        st.balloons()
    else:
        st.error(f"❌ **FALSO POSITIVO**")

    confianza = max(probabilidad[0]) * 100
    st.metric("Confianza del modelo", f"{confianza:.1f}%")

st.markdown("---")
st.caption("Datos: NASA Kepler Objects of Interest | Azura StarScan © 2026")