
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import plotly.express as px
import io
from sklearn.linear_model import LinearRegression
from streamlit_option_menu import option_menu

# Definicion de una clase para obtener los datos de cada investigacion y que sea de facil acceso
class CitationAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.start_line = 4
        self.df = self.load_data()
        self.autor = self.get_author()
        self.min_year = self.df['Publication Year'].min()

    def load_data(self):
        df1 = pd.read_csv(self.file_path, delimiter='\t')
        df = pd.read_csv(self.file_path, skiprows=self.start_line - 1, delimiter=',')
        df['Publication Year'] = pd.to_numeric(df['Publication Year'], errors='coerce')
        return df

    def get_author(self):
        with open(self.file_path, 'r') as file:
            first_line = file.readline().strip()
        autor = first_line.replace(" (Author)", "").strip()
        return autor

    def plot_graphs(self):
        publicaciones_por_año = self.df.groupby('Publication Year')['Title'].count()
        citas_por_año = self.df.groupby('Publication Year')['Total Citations'].sum()
        citas_acumuladas = citas_por_año.cumsum()

        fig, ax1 = plt.subplots(figsize=(12, 8))

        ax1.bar(publicaciones_por_año.index, publicaciones_por_año.values, color='b', alpha=0.7, label='Publicaciones')
        ax1.set_xlabel('Año de Publicación', fontsize=14)
        ax1.set_ylabel('Cantidad de Publicaciones', fontsize=14, color='b')
        ax1.tick_params('y', labelcolor='b')

        ax1.set_xticks(np.arange(int(publicaciones_por_año.index.min()), int(publicaciones_por_año.index.max()) + 1, 1))
        ax1.set_xticklabels(np.arange(int(publicaciones_por_año.index.min()), int(publicaciones_por_año.index.max()) + 1, 1), rotation=45)


        ax2 = ax1.twinx()
        ax2.plot(citas_acumuladas.index, citas_acumuladas.values, color='r', marker='o', linestyle='-', label='Citas Acumuladas')
        ax2.set_ylabel('Citas Totales Acumuladas', fontsize=14, color='r')
        ax2.tick_params('y', labelcolor='r')

        ax1.set_ylim(0,publicaciones_por_año.max()+1)
        ax2.set_ylim(0,citas_acumuladas.max()+1)

        plt.title('Publicaciones y Citas Acumuladas por Año', fontsize=16)
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
        plt.xticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        return fig

    def calculate_metrics(self):
        total_publicaciones = len(self.df)
        total_articulos_citados = self.df['Total Citations'].count()
        total_citas = self.df['Total Citations'].sum()
        promedio_citas_total = total_citas / total_publicaciones if total_publicaciones > 0 else 0
        promedio_citas_citados = total_citas / total_articulos_citados if total_articulos_citados > 0 else 0
        citas_ordenadas = self.df['Total Citations'].sort_values(ascending=False)
        indice_h = sum(citas_ordenadas >= range(1, len(citas_ordenadas) + 1))

        metrics_text = f"Nombre: {self.autor}\n" \
                       f"Total de publicaciones: {total_publicaciones}\n" \
                       f"Número total de artículos citados: {total_articulos_citados}\n" \
                       f"Número total de citas: {total_citas}\n" \
                       f"Número promedio de citas del total de artículos: {promedio_citas_total:.2f}\n" \
                       f"Número promedio de citas de artículos citados: {promedio_citas_citados:.2f}\n" \
                       f"Índice H: {indice_h}"

        return metrics_text

# Función para leer los nombres de los investigadores desde GitHub
def leer_nombres(file_name):
    url = f"https://raw.githubusercontent.com/GiovanniFranciscoP/IA/refs/heads/main/Investigadores/{file_name}"
    df_nombres = pd.read_csv(url, encoding = 'latin-1')
    investigadores = df_nombres.iloc[:, 0].tolist()
    return investigadores

def leer_archivo_investigador(file_name):
    url = f"https://raw.githubusercontent.com/GiovanniFranciscoP/IA/refs/heads/main/Investigadores/{file_name}"  # Cambia esto por tu URL
    response = requests.get(url)

    if response.status_code == 200:
        with open(file_name, 'wb') as file:
            file.write(response.content)
        return file_name
    else:
        raise FileNotFoundError(f"Archivo no encontrado en: {url}")

def leer_patentes_desde_git(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Verifica que no hubo errores en la solicitud
        file_data = io.BytesIO(response.content)  # Maneja el archivo como un archivo en memoria
        df_patentes = pd.read_excel(file_data)
        return df_patentes
    except requests.exceptions.HTTPError as e:
        st.error(f"Error al cargar el archivo desde GitHub: {e}")
        return None

# Streamlit app
st.title("Análisis de Publicaciones")

# Sidebar with collapsible menu and icons
with st.sidebar:
    menu_option = option_menu(
        "Menú",
        ["Inicio", "Buscar", "Todos los Investigadores", "Análisis de Patentes"],
        icons=["house", "search", "people", "gear"],
        menu_icon="cast",
        default_index=0
    )

if menu_option == "Inicio":
    st.header("Bienvenido")
    st.write("Esta es una aplicación para analizar las publicaciones de investigadores.")

elif menu_option == "Buscar":
    nombres_investigadores = leer_nombres('Investigadores.csv')
    researcher_name = st.selectbox("Seleccione el investigador:", nombres_investigadores)

    if researcher_name:

        file_name = f"{researcher_name}.txt"
        try:
            local_file = leer_archivo_investigador(file_name)
            analysis = CitationAnalysis(local_file)

            st.header(f"Información de {analysis.autor}")

            #Mostrar la tabla de publicaciones por año y citas
            st.subheader("Publicaciones")
            df_filtered = analysis.df[['Title', 'Publication Year', 'Total Citations']]
            df_filtered['Publication Year'] = df_filtered['Publication Year'].astype(str)
            df_filtered = df_filtered.sort_values('Publication Year')
            st.dataframe(df_filtered)

            # Mostrar las métricas
            metrics_text = analysis.calculate_metrics()
            st.text(metrics_text)

            # Mostrar la gráfica
            fig = analysis.plot_graphs()
            st.pyplot(fig)

        except FileNotFoundError:
            st.error(f"Archivo no encontrado: {file_name}")

elif menu_option == "Todos los Investigadores":
    nombres_investigadores = leer_nombres('Investigadores.csv')
    archivos_no_encontrados = []
    data_publicaciones = {}
    data_citas = {}

    for nombre in nombres_investigadores:
        file_name = f"{nombre}.txt"
        try:
            local_file = leer_archivo_investigador(file_name)
            analysis = CitationAnalysis(file_name)
            df_filtered = analysis.df[(analysis.df['Publication Year'] >= 2000) & (analysis.df['Publication Year'] <= 2023)]
            publicaciones_por_año = df_filtered.groupby('Publication Year')['Title'].count()
            citas_por_año = df_filtered.groupby('Publication Year')['Total Citations'].sum()
            data_publicaciones[nombre] = publicaciones_por_año
            data_citas[nombre] = citas_por_año
        except FileNotFoundError:
            print(f"File not found: {file_name}")
            archivos_no_encontrados.append(file_name)

    # Convertir los diccionarios en formato largo
    df_publicaciones_long = pd.DataFrame(data_publicaciones).stack().reset_index()
    df_publicaciones_long.columns = ['Año', 'Investigador', 'Publicaciones']
    df_citas_long = pd.DataFrame(data_citas).stack().reset_index()
    df_citas_long.columns = ['Año', 'Investigador', 'Citas']

    # Gráfica de barras de publicaciones por año usando Plotly
    st.subheader("Relación entre Publicaciones por Año y Citas por Investigador")
    
    fig_publicaciones = px.bar(df_publicaciones_long, 
                                x='Año', 
                                y='Publicaciones', 
                                color='Investigador',
                                title='Relación entre Publicaciones por Año',
                                labels={'Año': 'Año', 'Publicaciones': 'Número de Publicaciones'})
    
    fig_publicaciones.update_layout(barmode='group')  # Agrupar las barras
    st.plotly_chart(fig_publicaciones)

    # Gráfica de barras de citas por año usando Plotly
    st.subheader("Relación entre Citas por Año y Publicaciones por Investigador")

    fig_citas = px.bar(df_citas_long, 
                        x='Año', 
                        y='Citas', 
                        color='Investigador',
                        title='Relación entre Citas por Año',
                        labels={'Año': 'Año', 'Citas': 'Número de Citas'})
    
    fig_citas.update_layout(barmode='group')  # Agrupar las barras
    st.plotly_chart(fig_citas)

elif menu_option == "Análisis de Patentes":
    st.header("Análisis de Patentes")

    # Cargar el archivo de patentes desde GitHub
    url_patentes = "https://raw.githubusercontent.com/GiovanniFranciscoP/IA/refs/heads/main/Investigadores/Patentes.xlsx"
    df_patentes = leer_patentes_desde_git(url_patentes)

    if df_patentes is not None:
        # Convertir la columna de fechas a formato datetime
        df_patentes['Filing Date'] = pd.to_datetime(df_patentes['Filing Date'], format='%Y-%m-%d')

        # Selección del inventor y conversión a mayúsculas
        inventor_seleccionado = st.selectbox("Selecciona un inventor", df_patentes['Inventor'].unique())
        inventor_seleccionado = inventor_seleccionado  # Convertir a mayúsculas

        # Filtrar patentes del inventor seleccionado
        patentes_inventor = df_patentes[df_patentes['Inventor'] == inventor_seleccionado]

        patente_seleccionada = st.selectbox("Selecciona una patente", patentes_inventor['Patent'].unique())

        # Obtener la fecha de la patente seleccionada y extraer el año
        fecha_patente = patentes_inventor[patentes_inventor['Patent'] == patente_seleccionada]['Filing Date'].values[0]
        año_patente = pd.to_datetime(fecha_patente).year

        # Inicializar el botón "Mostrar antes y después" como verdadero
        mostrar_solo_antes = st.button("Antes de la patente")
        mostrar_solo_despues = st.button("Después de la patente")

        # Cargar archivo del investigador seleccionado
        file_name = f"{inventor_seleccionado}.txt"
        try:
            local_file = leer_archivo_investigador(file_name)
            analysis = CitationAnalysis(local_file)

            # Inicializar publicaciones_filtradas como un DataFrame vacío
            publicaciones_filtradas = pd.DataFrame()

            # Definir el filtro de años
            if mostrar_solo_antes:
                publicaciones_filtradas = analysis.df[
                    (analysis.df['Publication Year'] < año_patente) &
                    (analysis.df['Publication Year'].between(2000, 2023))
                ]
            elif mostrar_solo_despues:
                publicaciones_filtradas = analysis.df[
                    (analysis.df['Publication Year'] >= año_patente) &
                    (analysis.df['Publication Year'].between(2000, 2023))
                ]
            else:
                # Mostrar antes y después por defecto
                publicaciones_filtradas = analysis.df[
                    (analysis.df['Publication Year'].between(2000, 2023))
                ]

            # Agrupar por año y sumar publicaciones y citas solo si hay publicaciones filtradas
            if not publicaciones_filtradas.empty:
                resumen_anual = publicaciones_filtradas.groupby('Publication Year').agg(
                    Publicaciones=('Title', 'count'),
                    Citas=('Total Citations', 'sum')
                ).reset_index()

                # Crear la gráfica interactiva con plotly.express
                fig = px.line(resumen_anual, x='Publication Year', y=['Publicaciones', 'Citas'],
                              labels={'value': 'Número', 'Publication Year': 'Año'},
                              title="Análisis de Publicaciones y Citas por Año")

                # Añadir una línea vertical en el año de la patente seleccionada
                fig.add_vline(x=año_patente, line_color='green', line_dash='dash', 
                              annotation_text=f'Año de Patente: {año_patente}', 
                              annotation_position="top right")

                for column in ['Publicaciones', 'Citas']:
                    x = resumen_anual['Publication Year'].values.reshape(-1, 1)  # Reshape para sklearn
                    y = resumen_anual[column].values

                    # Crear un modelo de regresión lineal
                    model = LinearRegression()
                    model.fit(x, y)  # Ajustar el modelo

                    # Predecir los valores
                    y_pred = model.predict(x)

                    # Agregar la línea de regresión a la gráfica
                    fig.add_scatter(x=resumen_anual['Publication Year'], y=y_pred, mode='lines', name=f'Tendencia {column}', 
                                     line=dict(dash='dash'))
                # Agregar un punto en cada publicación y cita
                fig.add_scatter(x=resumen_anual['Publication Year'], y=resumen_anual['Publicaciones'], mode='markers', 
                                 name='Publicaciones', marker=dict(size=8, color='blue'))

                fig.add_scatter(x=resumen_anual['Publication Year'], y=resumen_anual['Citas'], mode='markers', 
                                 name='Citas', marker=dict(size=8, color='orange'))

                # Mostrar la gráfica en Streamlit
                st.plotly_chart(fig)

                # Contar publicaciones antes y después de la patente
                publicaciones_antes = len(publicaciones_filtradas[publicaciones_filtradas['Publication Year'] < año_patente])
                publicaciones_despues = len(publicaciones_filtradas[publicaciones_filtradas['Publication Year'] >= año_patente])

                # Mostrar los resultados
                st.write(f"Publicaciones antes de la patente: {publicaciones_antes}")
                st.write(f"Publicaciones después de la patente: {publicaciones_despues}")

            else:
                st.warning("No hay publicaciones filtradas para mostrar.")

        except FileNotFoundError:
            st.error(f"Archivo no encontrado: {file_name}")

