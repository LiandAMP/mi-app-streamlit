import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN VISUAL Y TEMA DENTAL
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DentalCare AI Analytics",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS PROFESIONAL
st.markdown("""
    <style>
    .stApp { background-color: #F0F4F8; }
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E6E9EF;
    }
    h1, h2, h3 { color: #0056b3; }
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #D1D9E6;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-color: #007BFF;
    }
    .plotly-graph-div {
        background-color: white;
        border-radius: 12px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CARGA DE DATOS
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("Dataset_Pacientes_Tendencia_Fuerte.xlsx")
        df = df.drop_duplicates()

        if df['sexo'].dtype == 'O':
            df["sexo"] = df["sexo"].astype(str).str.strip().str.upper()
            df["sexo"] = df["sexo"].map({"M": 1, "F": 0})

        df['sexo_txt'] = df['sexo'].map({1: 'Masculino', 0: 'Femenino'})
        df['vuelve_txt'] = df['vuelve'].map({1: 'Fidelizado', 0: 'Perdido'})
        df['caries_txt'] = df['tiene_caries_previas'].map({1: 'Sí', 0: 'No'})

        return df
    except FileNotFoundError:
        return None

df = load_data()

# -----------------------------------------------------------------------------
# 3. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966334.png", width=80)
    st.markdown("## DentalCare Manager")
    st.markdown("Sistema de Inteligencia Clínica")
    st.markdown("---")

    opcion = st.radio(
        "Navegación:",
        ["🏠 Inicio / Dashboard", "🔍 Análisis de Datos", "🤖 Predicción IA", "📂 Base de Datos"],
        index=0
    )

    st.markdown("---")
    st.info("💡 Tip: Usa la IA para evaluar pacientes nuevos.")

# -----------------------------------------------------------------------------
# 4. MODELO BASE
# -----------------------------------------------------------------------------
if df is None:
    st.error("⚠️ ERROR: No se encuentra el archivo Excel.")
    st.stop()

features = ['edad', 'sexo', 'dolor_reportado', 'tiene_caries_previas', 'frecuencia_visitas_anual']
X = df[features]
y = df['vuelve']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)
acc_rf = modelo_rf.score(X_test, y_test)

# =============================================================================
# PÁGINA 1: DASHBOARD
# =============================================================================
if opcion == "🏠 Inicio / Dashboard":
    st.title("📊 Resumen Ejecutivo de la Clínica")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pacientes Activos", len(df))
    k2.metric("Tasa de Retención", f"{df['vuelve'].mean():.1%}")
    k3.metric("Dolor Promedio", f"{df['dolor_reportado'].mean():.1f}/10")
    k4.metric("Precisión Modelo IA", f"{acc_rf:.1%}")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(df, names="sexo_txt", title="Distribución por Género")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.histogram(df, x="edad", color="vuelve_txt",
                            title="Retención por Edad")
        st.plotly_chart(fig2, use_container_width=True)

# =============================================================================
# PÁGINA 2: ANÁLISIS Y COMPARACIÓN DE MODELOS
# =============================================================================
elif opcion == "🔍 Análisis de Datos":
    st.title("🔬 Análisis Clínico y Comparación de Modelos")

    tab1, tab2, tab3 = st.tabs(["📈 Correlaciones", "⚠ Factores de Riesgo", "🤖 Comparación de Modelos"])

    # ---------------- TAB 1 ----------------
    with tab1:
        corr = df.select_dtypes(include=['number']).corr()
        st.subheader("Mapa de Correlaciones Clínicas")
        st.plotly_chart(px.imshow(corr, text_auto=True), use_container_width=True)

    # ---------------- TAB 2 ----------------
    with tab2:
        st.subheader("Dolor vs Decisión de Retorno")
        fig_box = px.box(df, x="vuelve_txt", y="dolor_reportado")
        st.plotly_chart(fig_box, use_container_width=True)

    # ---------------- TAB 3: COMPARACIÓN DE ALGORITMOS -----------
    with tab3:
        st.subheader("🤖 Comparación de Algoritmos Predictivos")

        modelos = {
            "Regresión Logística": LogisticRegression(max_iter=200),
            "Árbol de Decisión": DecisionTreeClassifier(max_depth=5),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }

        resultados = []
        for nombre, modelo in modelos.items():
            modelo.fit(X_train, y_train)
            pred = modelo.predict(X_test)
            acc = accuracy_score(y_test, pred)
            resultados.append({"Algoritmo": nombre, "Precisión (%)": round(acc * 100, 2)})

        df_alg = pd.DataFrame(resultados)
        st.dataframe(df_alg)

        fig_alg = px.bar(df_alg, x="Algoritmo", y="Precisión (%)",
                         text="Precisión (%)",
                         title="Comparación de Algoritmos")
        st.plotly_chart(fig_alg, use_container_width=True)

# =============================================================================
# PÁGINA 3: PREDICCIÓN IA CON SELECTOR DE MODELO
# =============================================================================
elif opcion == "🤖 Predicción IA":
    st.title("🤖 Simulador de Probabilidad de Retorno")

    st.sidebar.markdown("### 🔧 Selecciona el método de predicción")
    modelo_sel = st.sidebar.selectbox(
        "Modelo a utilizar:",
        ["Regresión Logística", "Árbol de Decisión", "Random Forest"]
    )

    modelos_pred = {
        "Regresión Logística": LogisticRegression(max_iter=200),
        "Árbol de Decisión": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    modelo_elegido = modelos_pred[modelo_sel]
    modelo_elegido.fit(X_train, y_train)

    c1, c2 = st.columns([1, 2])

    with c1:
        with st.form("form_pred"):
            edad = st.slider("Edad", 18, 90, 30)
            sexo = st.selectbox("Sexo", [1, 0], format_func=lambda x: "Masculino" if x==1 else "Femenino")
            dolor = st.slider("Dolor (1-10)", 1, 10, 5)
            caries = st.selectbox("Caries Previas", [1, 0], format_func=lambda x: "Sí" if x==1 else "No")
            visitas = st.number_input("Visitas Anuales", 0, 20, 2)

            submit = st.form_submit_button("CALCULAR")

    with c2:
        if submit:
            dato = pd.DataFrame([[edad, sexo, dolor, caries, visitas]], columns=features)
            pred = modelo_elegido.predict(dato)[0]
            prob = modelo_elegido.predict_proba(dato)[0][1]

            st.info(f"🔍 Modelo Seleccionado: **{modelo_sel}**")

            if pred == 1:
                st.success("Paciente con alta probabilidad de retorno")
            else:
                st.error("Riesgo de fuga detectado")

            st.metric("Probabilidad", f"{prob:.1%}")

# =============================================================================
# PÁGINA 4: BASE DE DATOS
# =============================================================================
elif opcion == "📂 Base de Datos":
    st.title("📂 Registro de Pacientes")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar CSV", data=csv, file_name="pacientes.csv")
