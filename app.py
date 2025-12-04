import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN VISUAL Y TEMA DENTAL
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DentalCare AI Analytics",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS PROFESIONAL (DISEÑO CLÍNICO)
st.markdown("""
    <style>
    /* Fondo General - Un gris muy suave para descansar la vista */
    .stApp {
        background-color: #F0F4F8;
    }
    
    /* Barra Lateral */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E6E9EF;
    }
    
    /* Títulos Principales en Azul Dental */
    h1, h2, h3 {
        color: #0056b3;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Tarjetas de Métricas (KPIs) */
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #D1D9E6;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-color: #007BFF;
    }
    
    /* Botones */
    div.stButton > button {
        background-color: #007BFF;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #0056b3;
    }
    
    /* Ajuste de gráficos */
    .plotly-graph-div {
        background-color: white;
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CARGA DE DATOS INTELIGENTE
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("Dataset_Pacientes_Tendencia_Fuerte.xlsx")
        df = df.drop_duplicates()
        
        # Normalización
        if df['sexo'].dtype == 'O':
            df["sexo"] = df["sexo"].astype(str).str.strip().str.upper()
            df["sexo"] = df["sexo"].map({"M": 1, "F": 0})
            
        # Etiquetas legibles para gráficos
        df['sexo_txt'] = df['sexo'].map({1: 'Masculino', 0: 'Femenino'})
        df['vuelve_txt'] = df['vuelve'].map({1: 'Fidelizado', 0: 'Perdido'})
        df['caries_txt'] = df['tiene_caries_previas'].map({1: 'Sí', 0: 'No'})
        
        return df
    except FileNotFoundError:
        return None

df = load_data()

# -----------------------------------------------------------------------------
# 3. NAVEGACIÓN LATERAL (MENU TIPO APP)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966334.png", width=80)
    st.markdown("## DentalCare Manager")
    st.markdown("Sistema de Inteligencia Clínica")
    st.markdown("---")
    
    # Menú de navegación
    opcion = st.radio(
        "Navegación:",
        ["🏠 Inicio / Dashboard", "🔍 Análisis de Datos", "🤖 Predicción IA", "📂 Base de Datos"],
        index=0
    )
    
    st.markdown("---")
    st.info("💡 **Tip:** Usa la sección de IA para evaluar pacientes nuevos antes de que salgan de la clínica.")

# -----------------------------------------------------------------------------
# 4. LÓGICA DE LAS PÁGINAS
# -----------------------------------------------------------------------------

if df is None:
    st.error("⚠️ ERROR CRÍTICO: No se encuentra el archivo Excel. Asegúrate de cargarlo en la carpeta del proyecto.")
    st.stop()

# --- ENTRENAMIENTO DEL MODELO (BACKEND) ---
features = ['edad', 'sexo', 'dolor_reportado', 'tiene_caries_previas', 'frecuencia_visitas_anual']
X = df[features]
y = df['vuelve']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)

# =============================================================================
# PÁGINA 1: DASHBOARD (RESUMEN EJECUTIVO)
# =============================================================================
if opcion == "🏠 Inicio / Dashboard":
    st.title("📊 Resumen Ejecutivo de la Clínica")
    st.markdown("Estado actual de la fidelización de pacientes.")
    
    # 1. TARJETAS SUPERIORES (KPIs)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Pacientes Activos", len(df), delta="Base Total")
    with k2:
        tasa = df['vuelve'].mean()
        st.metric("Tasa de Retención", f"{tasa:.1%}", delta="Objetivo: >60%", delta_color="normal" if tasa > 0.6 else "inverse")
    with k3:
        dolor_prom = df['dolor_reportado'].mean()
        st.metric("Nivel Dolor Promedio", f"{dolor_prom:.1f}/10", delta="- Menor es mejor", delta_color="inverse")
    with k4:
        st.metric("Precisión del Modelo IA", f"{acc:.1%}", delta="Confiable")

    st.markdown("---")

    # 2. GRÁFICOS PRINCIPALES
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("👥 Distribución de Pacientes")
        fig_pie = px.pie(df, names='sexo_txt', title='Género de Pacientes', 
                         color_discrete_sequence=['#007BFF', '#00C6FF'], hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        st.subheader("🎂 Retención por Edad")
        fig_hist = px.histogram(df, x="edad", color="vuelve_txt", 
                                title="¿Qué grupo de edad fidelizamos más?",
                                color_discrete_map={'Fidelizado':'#28a745', 'Perdido':'#dc3545'},
                                barmode="group")
        st.plotly_chart(fig_hist, use_container_width=True)

# =============================================================================
# PÁGINA 2: ANÁLISIS (CIENCIA DE DATOS)
# =============================================================================
elif opcion == "🔍 Análisis de Datos":
    st.title("🔬 Análisis Profundo de Comportamiento")
    
    tab1, tab2 = st.tabs(["Correlaciones", "Factores de Riesgo"])
    
    with tab1:
        st.markdown("#### ¿Qué variables están conectadas?")
        st.write("Este mapa de calor nos muestra qué factores influyen más en el retorno del paciente.")
        numeric_df = df.select_dtypes(include=['number']).drop(columns=['id_paciente'], errors='ignore')
        corr = numeric_df.corr()
        fig_hm = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", aspect="auto")
        st.plotly_chart(fig_hm, use_container_width=True)
        
    with tab2:
        st.markdown("#### Dolor vs. Fidelización")
        fig_box = px.box(df, x="vuelve_txt", y="dolor_reportado", color="vuelve_txt",
                         color_discrete_map={'Fidelizado':'#28a745', 'Perdido':'#dc3545'},
                         title="Impacto del dolor en la decisión de volver")
        st.plotly_chart(fig_box, use_container_width=True)

# =============================================================================
# PÁGINA 3: PREDICCIÓN (IA) - EL PLATO FUERTE
# =============================================================================
elif opcion == "🤖 Predicción IA":
    st.title("🤖 Simulador de Probabilidad de Retorno")
    
    c_izq, c_der = st.columns([1, 2])
    
    with c_izq:
        st.markdown("### 📋 Datos del Paciente")
        with st.form("form_prediccion"):
            input_edad = st.slider("Edad", 18, 90, 30)
            input_sexo = st.selectbox("Sexo", options=[1, 0], format_func=lambda x: "Masculino" if x==1 else "Femenino")
            input_dolor = st.slider("Nivel de Dolor (1-10)", 1, 10, 5)
            input_caries = st.selectbox("Caries Previas", options=[1, 0], format_func=lambda x: "Sí" if x==1 else "No")
            input_visitas = st.number_input("Visitas Anuales Previas", 0, 20, 2)
            
            submit_val = st.form_submit_button("CALCULAR AHORA")
            
    with c_der:
        st.markdown("### 🎯 Resultado del Análisis")
        if submit_val:
            # Crear dataframe input
            dato_nuevo = pd.DataFrame([[input_edad, input_sexo, input_dolor, input_caries, input_visitas]], columns=features)
            
            # Predicción
            pred = model.predict(dato_nuevo)[0]
            prob = model.predict_proba(dato_nuevo)[0][1]
            
            # Mostrar resultado visualmente atractivo
            if pred == 1:
                st.success("✅ **PRONÓSTICO FAVORABLE**")
                st.metric("Probabilidad de Retorno", f"{prob:.1%}", delta="Alta")
                st.progress(prob)
                st.markdown("""
                    **Acción Recomendada:**
                    * Programar cita de seguimiento estándar.
                    * Enviar recordatorio por WhatsApp en 6 meses.
                """)
                st.balloons()
            else:
                st.error("⚠️ **RIESGO DE FUGA DETECTADO**")
                st.metric("Probabilidad de Retorno", f"{prob:.1%}", delta="- Baja", delta_color="inverse")
                st.progress(prob)
                st.markdown("""
                    **Acción Recomendada URGENTE:**
                    * 🛑 Ofrecer descuento del 10% en próxima visita.
                    * 📞 Realizar llamada de seguimiento de calidad post-tratamiento.
                """)
        else:
            st.info("👈 Ingresa los datos a la izquierda y presiona 'Calcular' para ver la magia de la IA.")
            st.image("https://cdn-icons-png.flaticon.com/512/3209/3209079.png", width=150)

# =============================================================================
# PÁGINA 4: BASE DE DATOS
# =============================================================================
elif opcion == "📂 Base de Datos":
    st.title("📂 Registro Completo de Pacientes")
    st.dataframe(df, use_container_width=True)
    
    # Botón de descarga
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar datos en CSV",
        data=csv,
        file_name='pacientes_dental_data.csv',
        mime='text/csv',
    )