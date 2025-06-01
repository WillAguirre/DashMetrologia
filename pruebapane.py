import streamlit as st
import pandas as pd
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm


CARPETA_EXCEL = r"Datos"
def listar_archivos():
    archivos = [f for f in os.listdir(CARPETA_EXCEL) if f.endswith('.xlsx') or f.endswith('.xls')]
    return archivos
@st.cache_data
def cargar_excel(nombre_archivo):
    path = os.path.join(CARPETA_EXCEL, nombre_archivo)
    df = pd.read_excel(path)
    return df
def selector_archivo():
    archivos = listar_archivos()
    if len(archivos) == 0:
        st.error(f"No se encontraron archivos Excel en la carpeta '{CARPETA_EXCEL}'.")
        return None
    archivo = st.selectbox("Selecciona el libro de Excel", archivos)
    return archivo
def pagina_analisis_exploratorio():
    st.title("📊 Análisis Exploratorio de Datos")
    archivo = selector_archivo()
    if archivo is None:
        return
    df = cargar_excel(archivo)
    if "Resultado real" not in df.columns:
        st.error("La columna 'Resultado real' no está en el archivo seleccionado.")
        return
    serie = df["Resultado real"].dropna()
    st.subheader("📋 Estadísticas Descriptivas")
    st.write(serie.describe())
    st.subheader("📋 Grafico de frecuencia")
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(20, 12))  # Aumentamos el tamaño aquí
    sns.histplot(serie, kde=True, color="#e74c3c", edgecolor='black', linewidth=0.6, ax=ax)
    media = np.mean(serie)
    mediana = np.median(serie)
    ax.axvline(media, color='black', linestyle='--', linewidth=2, label=f'Media: {media:.2f}')
    ax.axvline(mediana, color='gray', linestyle=':', linewidth=2, label=f'Mediana: {mediana:.2f}')
    ax.set_xlabel("Valor", fontsize=24, labelpad=15, color='black')
    ax.set_ylabel("Frecuencia", fontsize=24, labelpad=15, color='black')
    ax.tick_params(axis='both', labelsize=18, colors='black')
    ax.legend(fontsize=20, frameon=True, facecolor='white', edgecolor='gray')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
def pagina_analisis_normalidad():
    st.title(" 📈 Análisis de Normalidad")
    archivo = selector_archivo()
    if archivo is None:
        return
    df = cargar_excel(archivo)
    if "Resultado real" not in df.columns:
        st.error("La columna 'Resultado real' no está en el archivo seleccionado.")
        return
    serie = df["Resultado real"].dropna()
    stat, p = stats.shapiro(serie)
    st.write(f"Estadístico de Shapiro-Wilk: {stat:.4f}")
    st.write(f"p-valor: {p:.4f}")
    if p > 0.05:
        st.success(" ✅ Los datos parecen seguir una distribución normal (p > 0.05)")
    else:
        st.warning(" 🚨 Los datos NO parecen seguir una distribución normal (p ≤ 0.05)")
    st.subheader("📊 Grafica Q-Q")
    # Gráfico Q-Q plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    sm.qqplot(serie, line='45', ax=ax, color='red')
    ax.set_title('Q-Q Plot de Resultado real', fontsize=16)
    ax.set_xlabel('Cuantiles Teóricos')
    ax.set_ylabel('Cuantiles de la Serie')
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig, use_container_width=True)
def pagina_analisis_varianza():
    st.title("📐 Análisis de Varianza")
    archivo = selector_archivo()
    if archivo is None:
        return
    df = cargar_excel(archivo)
    if "Resultado real" not in df.columns:
        st.error("La columna 'Resultado real' no está en el archivo seleccionado.")
        return
    serie = df["Resultado real"].dropna()
    st.subheader("📊 Estadísticas de dispersión")
    varianza = np.var(serie, ddof=1)
    desviacion_std = np.std(serie, ddof=1)
    q1 = np.percentile(serie, 25)
    q3 = np.percentile(serie, 75)
    iqr = q3 - q1
    rango_total = serie.max() - serie.min()
    st.markdown(f"""
    <div style='font-size: 20px;'>
        <ul>
            <li><strong>Varianza</strong>: {varianza:.4f}</li>
            <li><strong>Desviación estándar</strong>: {desviacion_std:.4f}</li>
            <li><strong>Rango intercuartílico (IQR)</strong>: {iqr:.4f}</li>
            <li><strong>Rango total (max - min)</strong>: {rango_total:.4f}</li>
        </ul>
    </div>
""", unsafe_allow_html=True)
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    outliers = serie[(serie < limite_inferior) | (serie > limite_superior)]
    normales = serie[(serie >= limite_inferior) & (serie <= limite_superior)]
    st.subheader("🔍 Gráfica de dispersión con detección de anomalias")
    df_dispersion = pd.DataFrame({
        "Índice": serie.index,
        "Valor": serie,
        "Tipo": ["Outlier" if (v < limite_inferior or v > limite_superior) else "Normal" for v in serie]
    })
    fig = px.scatter(
        df_dispersion,
        x="Índice",
        y="Valor",
        color="Tipo",
        color_discrete_map={"Normal": "#e74c3c", "Outlier": "#2c3e50"},
        height=700,
        width=1100
    )
    fig.update_layout(
        showlegend=True,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title=None,
        yaxis_title=None,
        font=dict(size=18, color="black"),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
def pagina_proximos_calculos():
    st.title(" ⚙️ Próximos cálculos")
    st.info("Esta sección estará disponible próximamente.")

def main():
    st.sidebar.title(" 📋 Navegación")
    pagina = st.sidebar.radio("Ir a", 
        ["Análisis Exploratorio", "Análisis de Normalidad", "Análisis de Varianza", "Próximos cálculos"])
    st.caption("Creado por: Wilson Aguirre")

    if pagina == "Análisis Exploratorio":
        pagina_analisis_exploratorio()
    elif pagina == "Análisis de Normalidad":
        pagina_analisis_normalidad()
    elif pagina == "Próximos cálculos":
        pagina_proximos_calculos()
    elif pagina == "Análisis de Varianza":
        pagina_analisis_varianza()
    

if __name__ == "__main__":
    main()
