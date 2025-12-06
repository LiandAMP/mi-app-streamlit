import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN VISUAL Y TEMA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DentalCare AI Analytics",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background-color: #F0F4F8; }
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E6E9EF;
    }
    h1, h2, h3 { color: #0056b3; }
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

        if df["sexo"].dtype == "O":
            df["sexo"] = df["sexo"].str.upper().map({"M": 1, "F": 0})

        df["sexo_txt"] = df["sexo"].map({1: "Masculino", 0: "Femenino"})
        df["vuelve_txt"] = df["vuelve"].map({1: "Fidelizado", 0: "Perdido"})
        return df
    except:
        return None

df = load_data()

if df is None:
    st.error("⚠️ ERROR: No se encuentra el archivo Excel.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966334.png", width=80)
    st.subheader("DentalCare Manager")
    opcion = st.radio(
        "Menú",
        ["🏠 Dashboard", "🔍 Análisis de Datos", "🤖 Predicción IA", "📊 Comparación Completa", "📂 Base de Datos"]
    )

# -----------------------------------------------------------------------------
# 4. VARIABLES BASE
# -----------------------------------------------------------------------------
features_full = ['edad', 'sexo', 'dolor_reportado', 'tiene_caries_previas', 'frecuencia_visitas_anual']
X_full = df[features_full]
y = df["vuelve"]

X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)

# Modelo base (RF)
modelo_base = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_base.fit(X_train, y_train)
acc_base = modelo_base.score(X_test, y_test)

# =============================================================================
# 🏠 DASHBOARD
# =============================================================================
if opcion == "🏠 Dashboard":
    st.title("📊 Resumen Ejecutivo de la Clínica")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pacientes Registrados", len(df))
    c2.metric("Tasa de Retención", f"{df['vuelve'].mean():.1%}")
    c3.metric("Dolor Promedio", f"{df['dolor_reportado'].mean():.1f}/10")
    c4.metric("Precisión IA", f"{acc_base:.1%}")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(df, names="sexo_txt", title="Distribución por Género")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.histogram(df, x="edad", color="vuelve_txt", title="Retención por Edad")
        st.plotly_chart(fig2, use_container_width=True)

# =============================================================================
# 🔍 ANÁLISIS
# =============================================================================
elif opcion == "🔍 Análisis de Datos":
    st.title("🔍 Análisis Clínico")

    tab1, tab2 = st.tabs(["📈 Correlaciones", "⚠ Factores de Riesgo"])

    with tab1:
        st.subheader("Mapa de Correlaciones")
        corr = df.select_dtypes(include=["number"]).corr()
        st.plotly_chart(px.imshow(corr, text_auto=True), use_container_width=True)

    with tab2:
        st.subheader("Dolor vs Retorno del Paciente")
        fig = px.box(df, x="vuelve_txt", y="dolor_reportado")
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 📊 COMPARACIÓN COMPLETA (Variables y Algoritmos)
# =============================================================================
elif opcion == "📊 Comparación Completa":
    st.title("📊 Comparación Completa de Modelos y Variables")

    st.header("1️⃣ Comparación entre Variables (Modelo A, B, C)")

    modelos_variables = {
        "Modelo A (Básico)": ['edad', 'sexo'],
        "Modelo B (Clínico)": ['dolor_reportado', 'tiene_caries_previas'],
        "Modelo C (Completo)": features_full
    }

    resultados_var = []

    for nombre, vars_usadas in modelos_variables.items():
        X = df[vars_usadas]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)
        acc = accuracy_score(y_test, modelo.predict(X_test))

        resultados_var.append({"Modelo": nombre, "Variables": ", ".join(vars_usadas), "Precisión (%)": round(acc*100, 2)})

    df_var = pd.DataFrame(resultados_var)
    st.dataframe(df_var)

    st.plotly_chart(px.bar(df_var, x="Modelo", y="Precisión (%)", text="Precisión (%)",
                           title="Comparación entre Variables"), use_container_width=True)

    st.markdown("---")
    st.header("2️⃣ Comparación entre Algoritmos")

    modelos_alg = {
        "Regresión Logística": LogisticRegression(max_iter=200),
        "Árbol de Decisión": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    resultados_alg = []

    for nombre, modelo in modelos_alg.items():
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, pred)
        resultados_alg.append({"Algoritmo": nombre, "Precisión (%)": round(acc*100, 2)})

    df_alg = pd.DataFrame(resultados_alg)
    st.dataframe(df_alg)

    st.plotly_chart(px.bar(df_alg, x="Algoritmo", y="Precisión (%)", text="Precisión (%)",
                           title="Comparación entre Algoritmos"), use_container_width=True)

# =============================================================================
# 🤖 PREDICCIÓN CON SELECTOR DE MODELO
# =============================================================================
elif opcion == "🤖 Predicción IA":
    st.title("🤖 Predicción Inteligente del Retorno del Paciente")

    st.sidebar.subheader("Elige el algoritmo:")
    modelo_sel = st.sidebar.selectbox(
        "Modelo:",
        ["Regresión Logística", "Árbol de Decisión", "Random Forest"]
    )

    modelos_pred = {
        "Regresión Logística": LogisticRegression(max_iter=300),
        "Árbol de Decisión": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    modelo_elegido = modelos_pred[modelo_sel]
    modelo_elegido.fit(X_train, y_train)

    st.subheader("Ingresa los datos del paciente:")

    edad = st.slider("Edad", 18, 90, 30)
    sexo = st.selectbox("Sexo", [1, 0], format_func=lambda x: "Masculino" if x==1 else "Femenino")
    dolor = st.slider("Dolor (1–10)", 1, 10, 5)
    caries = st.slider("Caries Previas", 0, 10, 0)
    visitas = st.number_input("Visitas Anuales", 0, 20, 2)

    if st.button("🔮 Predecir"):
        dato = pd.DataFrame([[edad, sexo, dolor, caries, visitas]], columns=features_full)
        pred = modelo_elegido.predict(dato)[0]
        prob = modelo_elegido.predict_proba(dato)[0][1]

        st.info(f"🔧 Modelo seleccionado: **{modelo_sel}**")

        if pred == 1:
            st.success("✔ Alta probabilidad de retorno")
        else:
            st.error("⚠ Riesgo de no retorno")

        st.metric("Probabilidad", f"{prob:.1%}")

# =============================================================================
# 📂 BASE DE DATOS
# =============================================================================
elif opcion == "📂 Base de Datos":
    st.title("📂 Base de Datos Completa")
    st.dataframe(df, use_container_width=True)

    st.download_button("📥 Descargar CSV", df.to_csv(index=False), "pacientes.csv")

