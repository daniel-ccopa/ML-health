import streamlit as st
st.set_page_config(layout="wide", page_icon=":hospital:")
# Importamos librerías
import pandas as pd
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
import base64  # Importamos base64 para codificar la imagen de fondo
plt.style.use('fivethirtyeight')
plt.style.use('dark_background')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.metrics import precision_recall_fscore_support as score, mean_squared_error
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

# Función para añadir imagen de fondo
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{encoded_string.decode()});
        background-size: cover;
    }}
    </style>
    """,
        unsafe_allow_html=True
    )

# Llama a la función con la ruta de tu imagen
add_bg_from_local('fondohealth.png')  # Asegúrate de que la imagen esté en la ruta correcta

#--------------------------------------------------------------------------------------------------------------------------------------------------------
start_time = time.time()  # Tiempo de inicio del programa

# Estilos CSS personalizados
st.markdown(
    """
    <style>
    /* Estilos para el título */
    .titulo {
        text-align: center;
        font-size: 50px;
        color: #FFFFFF;
        text-decoration: underline;
    }
    /* Estilos para los subtítulos */
    .subtitulo {
        color: #FFFFFF;
    }
    /* Estilos para el texto */
    .texto {
        color: #FFFFFF;
    }
    /* Estilos para las métricas */
    .metricas {
        font-size: 18px;
        color: #FFFFFF;
    }
    /* Estilos para los inputs */
    .stNumberInput>div>input {
        background-color: #f0f0f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Títulos
tit1, tit2 = st.columns((4, 1))
tit1.markdown("<h1 class='titulo'>HealthNet-AI: Machine Learning en Salud</h1>", unsafe_allow_html=True)
tit2.image("HealthNet.png")
st.sidebar.title("Conjunto de Datos y Clasificador")

dataset_name = st.sidebar.selectbox("Seleccionar Conjunto de Datos: ", ('Ataque Cardíaco', "Cáncer de Mama"))
classifier_name = st.sidebar.selectbox("Seleccionar Clasificador: ", ("Regresión Logística", "KNN", "SVM", "Árboles de Decisión",
                                                                     "Bosque Aleatorio", "Aumento de Gradiente", "XGBoost"))

LE = LabelEncoder()
def get_dataset(dataset_name):
    if dataset_name == "Ataque Cardíaco":
        data = pd.read_csv("https://raw.githubusercontent.com/daniel-ccopa/ML-health/main/Data/heart.csv")
        st.header("Predicción de Ataque Cardíaco")
        return data

    else:
        data = pd.read_csv("https://raw.githubusercontent.com/daniel-ccopa/ML-health/main/Data/BreastCancer.csv")
        
        data["diagnosis"] = LE.fit_transform(data["diagnosis"])
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data["diagnosis"] = pd.to_numeric(data["diagnosis"], errors="coerce")
        st.header("Predicción de Cáncer de Mama")
        return data

data = get_dataset(dataset_name)

def selected_dataset(dataset_name):
    if dataset_name == "Ataque Cardíaco":
        X = data.drop(["output"], axis=1)
        Y = data.output
        return X, Y

    elif dataset_name == "Cáncer de Mama":
        X = data.drop(["id", "diagnosis"], axis=1)
        Y = data.diagnosis
        return X, Y

X, Y = selected_dataset(dataset_name)

# Graficar variable de salida
def plot_op(dataset_name):
    col1, col2 = st.columns((1, 5))
    plt.figure(figsize=(12, 3))
    plt.title("Clases en 'Y'")
    if dataset_name == "Ataque Cardíaco":
        col1.write(Y)
        sns.countplot(Y, palette='gist_heat')
        # Capturamos la figura actual
        fig = plt.gcf()
        col2.pyplot(fig)  # Pasamos la figura a st.pyplot()

    elif dataset_name == "Cáncer de Mama":
        col1.write(Y)
        sns.countplot(Y, palette='gist_heat')
        fig = plt.gcf()
        col2.pyplot(fig)

st.write(data)
st.write("Forma del conjunto de datos: ", data.shape)
st.write("Número de clases: ", Y.nunique())
plot_op(dataset_name)

def add_parameter_ui(clf_name):
    params = {}
    st.sidebar.write("Seleccionar valores: ")

    if clf_name == "Regresión Logística":
        R = st.sidebar.slider("Regularización", 0.1, 10.0, step=0.1)
        MI = st.sidebar.slider("Máx iteraciones", 50, 400, step=50)
        params["R"] = R
        params["MI"] = MI

    elif clf_name == "KNN":
        K = st.sidebar.slider("n_neighbors", 1, 20)
        params["K"] = K

    elif clf_name == "SVM":
        C = st.sidebar.slider("Regularización", 0.01, 10.0, step=0.01)
        kernel = st.sidebar.selectbox("Kernel", ("linear", "poly", "rbf", "sigmoid", "precomputed"))
        params["C"] = C
        params["kernel"] = kernel

    elif clf_name == "Árboles de Decisión":
        M = st.sidebar.slider("Profundidad Máxima", 2, 20)
        C = st.sidebar.selectbox("Criterio", ("gini", "entropy"))
        SS = st.sidebar.slider("Mín muestras para dividir", 2, 10)
        params["M"] = M
        params["C"] = C
        params["SS"] = SS

    elif clf_name == "Bosque Aleatorio":
        N = st.sidebar.slider("n_estimators", 50, 500, step=50, value=100)
        M = st.sidebar.slider("Profundidad Máxima", 2, 20)
        C = st.sidebar.selectbox("Criterio", ("gini", "entropy"))
        params["N"] = N
        params["M"] = M
        params["C"] = C

    elif clf_name == "Aumento de Gradiente":
        N = st.sidebar.slider("n_estimators", 50, 500, step=50, value=100)
        LR = st.sidebar.slider("Tasa de Aprendizaje", 0.01, 0.5)
        L = st.sidebar.selectbox("Pérdida", ('log_loss', 'exponential'))
        M = st.sidebar.slider("Profundidad Máxima", 2, 20)
        params["N"] = N
        params["LR"] = LR
        params["L"] = L
        params["M"] = M

    elif clf_name == "XGBoost":
        N = st.sidebar.slider("n_estimators", 50, 500, step=50, value=50)
        LR = st.sidebar.slider("Tasa de Aprendizaje", 0.01, 0.5, value=0.1)
        O = st.sidebar.selectbox("Objetivo", ('binary:logistic', 'reg:logistic', 'reg:squarederror', "reg:gamma"))
        M = st.sidebar.slider("Profundidad Máxima", 1, 20, value=6)
        G = st.sidebar.slider("Gamma", 0, 10, value=5)
        L = st.sidebar.slider("reg_lambda", 1.0, 5.0, step=0.1)
        A = st.sidebar.slider("reg_alpha", 0.0, 5.0, step=0.1)
        CS = st.sidebar.slider("colsample_bytree", 0.5, 1.0, step=0.1)
        params["N"] = N
        params["LR"] = LR
        params["O"] = O
        params["M"] = M
        params["G"] = G
        params["L"] = L
        params["A"] = A
        params["CS"] = CS

    RS = st.sidebar.slider("Estado Aleatorio", 0, 100)
    params["RS"] = RS
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    global clf
    if clf_name == "Regresión Logística":
        clf = LogisticRegression(C=params["R"], max_iter=params["MI"])

    elif clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])

    elif clf_name == "SVM":
        clf = SVC(kernel=params["kernel"], C=params["C"])

    elif clf_name == "Árboles de Decisión":
        clf = DecisionTreeClassifier(max_depth=params["M"], criterion=params["C"], min_samples_split=params["SS"])

    elif clf_name == "Bosque Aleatorio":
        clf = RandomForestClassifier(n_estimators=params["N"], max_depth=params["M"], criterion=params["C"])

    elif clf_name == "Aumento de Gradiente":
        clf = GradientBoostingClassifier(n_estimators=params["N"], learning_rate=params["LR"], loss=params["L"], max_depth=params["M"])

    elif clf_name == "XGBoost":
        clf = XGBClassifier(booster="gbtree", n_estimators=params["N"], max_depth=params["M"], learning_rate=params["LR"],
                            objective=params["O"], gamma=params["G"], reg_alpha=params["A"], reg_lambda=params["L"], colsample_bytree=params["CS"])

    return clf

clf = get_classifier(classifier_name, params)

# Construir Modelo
def model():
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=65)

    # Escalado estándar / Normalización de datos
    Std_scaler = StandardScaler()
    X_train = Std_scaler.fit_transform(X_train)
    X_test = Std_scaler.transform(X_test)

    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)

    return Y_pred, Y_test

Y_pred, Y_test = model()

# Graficar Salida
def compute(Y_pred, Y_test):
    # Graficar PCA
    pca = PCA(2)
    X_projected = pca.fit_transform(X)
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]
    plt.figure(figsize=(16, 8))
    plt.scatter(x1, x2, c=Y, alpha=0.8, cmap="viridis")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.colorbar()
    fig_pca = plt.gcf()
    st.pyplot(fig_pca)

    c1, c2 = st.columns((4, 3))
    # Gráfico de salida
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(Y_pred)), Y_pred, color="yellow", lw=5, label="Predicciones")
    plt.scatter(range(len(Y_test)), Y_test, color="red", label="Actual")
    plt.title("Valores de Predicción vs Valores Reales")
    plt.legend()
    plt.grid(True)
    fig_pred = plt.gcf()
    c1.pyplot(fig_pred)

    # Matriz de Confusión
    cm = confusion_matrix(Y_test, Y_pred)
    class_label = ["Bajo Riesgo", "Alto Riesgo"]
    df_cm = pd.DataFrame(cm, index=class_label, columns=class_label)
    plt.figure(figsize=(12, 7.5))
    sns.heatmap(df_cm, annot=True, cmap='Pastel1', linewidths=2, fmt='d')
    plt.title("Matriz de Confusión", fontsize=15)
    plt.xlabel("Predicho")
    plt.ylabel("Verdadero")
    fig_cm = plt.gcf()
    c2.pyplot(fig_cm)

    # Calcular Métricas
    acc = accuracy_score(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    precision, recall, fscore, train_support = score(Y_test, Y_pred, pos_label=1, average='binary')
    st.subheader("Métricas del modelo: ")
    st.markdown(f"""
    <div class='metricas'>
    <ul>
    <li><strong>Precisión:</strong> {round(precision, 3)}</li>
    <li><strong>Recall:</strong> {round(recall, 3)}</li>
    <li><strong>F1-Score:</strong> {round(fscore, 3)}</li>
    <li><strong>Exactitud:</strong> {round((acc * 100), 3)} %</li>
    <li><strong>Error Cuadrático Medio:</strong> {round((mse), 3)}</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.header(f"1) Modelo para la Predicción de {dataset_name}")
st.subheader(f"Clasificador Usado: {classifier_name}")
compute(Y_pred, Y_test)

# Tiempo de Ejecución
end_time = time.time()
st.info(f"Tiempo total de ejecución: {round((end_time - start_time),4)} segundos")

# Obtener valores del usuario
def user_inputs_ui(dataset_name, data):
    user_val = {}
    
    if dataset_name == "Cáncer de Mama":
        X = data.drop(["id", "diagnosis"], axis=1)
        cols = st.columns(3)
        
        # Lista de variables con descripciones
        variables = [
            ("radius_mean", "Media de las distancias desde el centro a los puntos del perímetro."),
            ("texture_mean", "Desviación estándar de los valores de escala de grises."),
            ("perimeter_mean", "Media del perímetro de las células."),
            ("area_mean", "Media del área de las células."),
            ("smoothness_mean", "Media de la suavidad local de las variaciones en la longitud del radio."),
            ("compactness_mean", "Media de la compactación (perímetro² / área - 1.0)."),
            ("concavity_mean", "Media de la concavidad (grados de hendiduras en la superficie)."),
            ("concave points_mean", "Media del número de puntos cóncavos en el contorno."),
            ("symmetry_mean", "Media de la simetría."),
            ("fractal_dimension_mean", "Media de la aproximación de la costa de la dimensión fractal ('aproximación de la costa')."),
            ("radius_se", "Error estándar del radio."),
            ("texture_se", "Error estándar de la textura."),
            ("perimeter_se", "Error estándar del perímetro."),
            ("area_se", "Error estándar del área."),
            ("smoothness_se", "Error estándar de la suavidad."),
            ("compactness_se", "Error estándar de la compacidad."),
            ("concavity_se", "Error estándar de la concavidad."),
            ("concave points_se", "Error estándar de los puntos cóncavos."),
            ("symmetry_se", "Error estándar de la simetría."),
            ("fractal_dimension_se", "Error estándar de la dimensión fractal."),
            ("radius_worst", "Peor valor del radio."),
            ("texture_worst", "Peor valor de la textura."),
            ("perimeter_worst", "Peor valor del perímetro."),
            ("area_worst", "Peor valor del área."),
            ("smoothness_worst", "Peor valor de la suavidad."),
            ("compactness_worst", "Peor valor de la compacidad."),
            ("concavity_worst", "Peor valor de la concavidad."),
            ("concave points_worst", "Peor valor de los puntos cóncavos."),
            ("symmetry_worst", "Peor valor de la simetría."),
            ("fractal_dimension_worst", "Peor valor de la dimensión fractal."),
            # ... (mantener la lista de variables y descripciones)
        ]
        
        for idx, (name, description) in enumerate(variables):
            col = cols[idx % 3]
            col.markdown(f"**{name}**: {description}")
            col_data = X[name]
            std = col_data.std()
            mean = col_data.mean()
            min_data = col_data.min()
            max_data = col_data.max()
            if pd.api.types.is_integer_dtype(col_data):
                min_val = int(np.floor(min_data - std))
                max_val = int(np.ceil(max_data + std))
                default_value = int(np.round(mean))
                # Asegurar que default_value esté entre min_val y max_val
                default_value = max(min_val, min(default_value, max_val))
                col_input = col.number_input(name, min_value=min_val, max_value=max_val, value=default_value)
            else:
                min_val = min_data - std
                max_val = max_data + std
                default_value = mean
                # Asegurar que default_value esté entre min_val y max_val
                default_value = max(min_val, min(default_value, max_val))
                col_input = col.number_input(name, min_value=min_val, max_value=max_val, value=default_value)
            user_val[name] = col_input

    elif dataset_name == "Ataque Cardíaco":
        X = data.drop(["output"], axis=1)
        cols = st.columns(3)
        variables = [
            ("age", "Edad de la persona en años."),
            ("sex", "Sexo de la persona (0 = Mujer, 1 = Hombre)."),
            ("cp", "Tipo de dolor en el pecho (0 = Angina típica, 1 = Angina atípica, 2 = Dolor no anginoso, 3 = Asintomático)."),
            ("trtbps", "Presión arterial en reposo (en mm Hg)."),
            ("chol", "Nivel de colesterol (mg/dl)."),
            ("fbs", "Azúcar en sangre en ayunas > 120 mg/dl (1 = Verdadero, 0 = Falso)."),
            ("restecg", "Resultados del electrocardiograma en reposo (0 = Normal, 1 = Anormalidad de ST-T, 2 = Hipertrofia ventricular izquierda)."),
            ("thalachh", "Frecuencia cardíaca máxima alcanzada."),
            ("exng", "Ejercicio inducido por angina (1 = Sí, 0 = No)."),
            ("oldpeak", "Depresión de ST inducida por el ejercicio en relación con el reposo."),
            ("slp", "Pendiente del segmento de ST al máximo esfuerzo (0 = Descendente, 1 = Plano, 2 = Ascendente)."),
            ("caa", "Número de vasos principales (0-3) coloreados por fluoroscopia."),
            ("thall", "Defecto talámico (1 = Normal, 2 = Defecto fijo, 3 = Defecto reversible)."),
        ]
        for idx, (name, description) in enumerate(variables):
            col = cols[idx % 3]
            col.markdown(f"**{name}**: {description}")
            if name in ["sex", "fbs", "exng"]:
                # Variables binarias 0 o 1
                min_val = 0
                max_val = 1
                default_value = int(X[name].mode()[0])
                col_input = col.number_input(name, min_value=min_val, max_value=max_val, value=default_value, step=1)
            elif name in ["cp", "restecg", "slp"]:
                # Variables categóricas con rangos conocidos
                if name == "cp":
                    min_val = 0
                    max_val = 3
                elif name == "restecg":
                    min_val = 0
                    max_val = 2
                elif name == "slp":
                    min_val = 0
                    max_val = 2
                default_value = int(X[name].mode()[0])
                col_input = col.number_input(name, min_value=min_val, max_value=max_val, value=default_value, step=1)
            elif name == "thall":
                min_val = 1
                max_val = 3
                default_value = int(X[name].mode()[0])
                col_input = col.number_input(name, min_value=min_val, max_value=max_val, value=default_value, step=1)
            elif name == "caa":
                min_val = 0
                max_val = 4
                default_value = int(X[name].mode()[0])
                col_input = col.number_input(name, min_value=min_val, max_value=max_val, value=default_value, step=1)
            else:
                col_data = X[name]
                std = col_data.std()
                mean = col_data.mean()
                min_data = col_data.min()
                max_data = col_data.max()
                if pd.api.types.is_integer_dtype(col_data):
                    min_val = int(np.floor(min_data - std))
                    max_val = int(np.ceil(max_data + std))
                    default_value = int(np.round(mean))
                    default_value = max(min_val, min(default_value, max_val))
                    col_input = col.number_input(name, min_value=min_val, max_value=max_val, value=default_value)
                else:
                    min_val = min_data - std
                    max_val = max_data + std
                    default_value = mean
                    default_value = max(min_val, min(default_value, max_val))
                    col_input = col.number_input(name, min_value=min_val, max_value=max_val, value=default_value)
            user_val[name] = col_input

    return user_val

# Valores del Usuario
st.markdown("<hr>", unsafe_allow_html=True)
st.header("2) Valores del Usuario")
with st.expander("Ver más"):
    st.markdown("""
    En esta sección, puedes usar tus propios valores para predecir la variable objetivo. 
    Ingresa los valores requeridos a continuación y obtendrás tu estado basado en los valores. <br>
    <p style='color: red;'> 1 - Alto Riesgo </p> <p style='color: green;'> 0 - Bajo Riesgo </p>
    """, unsafe_allow_html=True)

user_val = user_inputs_ui(dataset_name, data)

# Añadir el botón "Predecir"
if st.button("Predecir"):
    def user_predict():
        global U_pred
        if dataset_name == "Cáncer de Mama":
            X = data.drop(["id", "diagnosis"], axis=1)
            U_pred = clf.predict([[user_val[col] for col in X.columns]])
    
        elif dataset_name == "Ataque Cardíaco":
            X = data.drop(["output"], axis=1)
            U_pred = clf.predict([[user_val[col] for col in X.columns]])
    
        st.subheader("Tu Estado: ")
        if U_pred == 0:
            st.write(U_pred[0], " - No estás en alto riesgo :)")
        else:
            st.write(U_pred[0], " - Estás en alto riesgo :(")
    user_predict()  # Predecir el estado del usuario.

#-------------------------------------------------------------------------FIN------------------------------------------------------------------------#
