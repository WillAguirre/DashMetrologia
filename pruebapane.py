import streamlit as st
import pandas as pd
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Carpeta donde tienes los archivos Excel
CARPETA_EXCEL = r"/home/wilson/Documentos/Proyectos/IsamelRoldan/Datos"

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
    st.title(" 📊 Análisis Exploratorio de Datos")
    
    archivo = selector_archivo()
    if archivo is None:
        return
    
    df = cargar_excel(archivo)

    if "Resultado real" not in df.columns:
        st.error("La columna 'Resultado real' no está en el archivo seleccionado.")
        return
    
    serie = df["Resultado real"].dropna()

    st.write(serie.describe())

    fig, ax = plt.subplots()
    sns.histplot(serie, kde=True, ax=ax)
    st.pyplot(fig)

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

def pagina_proximos_calculos():
    st.title(" ⚙️ Próximos cálculos")
    st.info("Esta sección estará disponible próximamente.")

def main():
    st.sidebar.title(" 📋 Navegación")
    pagina = st.sidebar.radio("Ir a", 
                              ["Análisis Exploratorio", "Análisis de Normalidad", "Próximos cálculos"])
    st.caption("Creado por: Wilson Aguirre")

    if pagina == "Análisis Exploratorio":
        pagina_analisis_exploratorio()
    elif pagina == "Análisis de Normalidad":
        pagina_analisis_normalidad()
    elif pagina == "Próximos cálculos":
        pagina_proximos_calculos()
    

if __name__ == "__main__":
    main()