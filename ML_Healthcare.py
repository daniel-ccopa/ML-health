import streamlit as st
st.set_page_config(layout="wide", page_icon=":hospital:", page_title="Aprendizaje Automático en Salud")

# Importamos librerías
import pandas as pd
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, precision_recall_fscore_support as score
from sklearn.decomposition import PCA

# CSS personalizado para darle estilo
st.markdown("""
    <style>
        body {
            background-image: url("https://via.placeholder.com/1920x1080.png"); /* Cambia esta URL por una imagen de fondo de tu preferencia */
            background-size: cover;
        }
        .main {
            background-color: rgba(255, 255, 255, 0.9); /* Fondo semi-transparente */
            padding: 20px;
            border-radius: 10px;
        }
        h1 {
            color: #FF4B4B;
            font-size: 3em;
            text-align: center;
            margin-bottom: 20px;
        }
        h2, h3 {
            color: #333333;
            font-size: 1.8em;
            text-align: center;
        }
        .stSidebar {
            background-color: #f7f7f7;
            color: #333;
        }
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
            font-size: 1.2em;
        }
        .step-box {
            background-color: #fff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .description {
            font-size: 1.1em;
            margin-bottom: 15px;
            color: #555;
        }
        .result-box {
            background-color: #F5F5F5;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

# Títulos principales
st.markdown("<h1>Aprendizaje Automático en Salud</h1>", unsafe_allow_html=True)

# Secciones de la barra lateral
st.sidebar.title("Conjunto de Datos y Clasificador")
dataset_name = st.sidebar.selectbox("Seleccionar Conjunto de Datos:", ('Ataque Cardíaco', 'Cáncer de Mama'))
classifier_name = st.sidebar.selectbox("Seleccionar Clasificador:", 
                                       ["Regresión Logística", "KNN", "SVM", "Árboles de Decisión", 
                                        "Bosque Aleatorio", "Aumento de Gradiente", "XGBoost"])

# Cargar dataset
LE = LabelEncoder()
def get_dataset(dataset_name):
    if dataset_name == "Ataque Cardíaco":
        data = pd.read_csv("https://raw.githubusercontent.com/advikmaniar/ML-Healthcare-Web-App/main/Data/heart.csv")
        st.header("Predicción de Ataque Cardíaco")
        return data
    else:
        data = pd.read_csv("https://raw.githubusercontent.com/advikmaniar/ML-Healthcare-Web-App/main/Data/BreastCancer.csv")
        data["diagnosis"] = LE.fit_transform(data["diagnosis"])
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data["diagnosis"] = pd.to_numeric(data["diagnosis"], errors="coerce")
        st.header("Predicción de Cáncer de Mama")
        return data

data = get_dataset(dataset_name)

def selected_dataset(dataset_name):
    if dataset_name == "Ataque Cardíaco":
        X = data.drop(["output"], axis=1)
        Y = data["output"]
        return X, Y
    elif dataset_name == "Cáncer de Mama":
        X = data.drop(["id", "diagnosis"], axis=1)
        Y = data["diagnosis"]
        return X, Y

X, Y = selected_dataset(dataset_name)

# Graficar la variable de salida
def plot_op(dataset_name):
    col1, col2 = st.columns((1, 5))
    plt.figure(figsize=(12, 3))
    plt.title("Clases en 'Y'")
    if dataset_name == "Ataque Cardíaco":
        col1.write(Y)
        sns.countplot(Y, palette='gist_heat')
        fig = plt.gcf()
        col2.pyplot(fig)
    elif dataset_name == "Cáncer de Mama":
        col1.write(Y)
        sns.countplot(Y, palette='gist_heat')
        fig = plt.gcf()
        col2.pyplot(fig)

st.write("Datos Cargados:")
st.write(data)
st.write(f"Forma del conjunto de datos: {data.shape}")
st.write(f"Número de clases: {Y.nunique()}")
plot_op(dataset_name)

# Seleccionar parámetros
def add_parameter_ui(clf_name):
    params = {}
    if clf_name == "Regresión Logística":
        params["R"] = st.sidebar.slider("Regularización", 0.1, 10.0)
        params["MI"] = st.sidebar.slider("Máx iteraciones", 50, 400, step=50)
    elif clf_name == "KNN":
        params["K"] = st.sidebar.slider("n_neighbors", 1, 20)
    elif clf_name == "SVM":
        params["C"] = st.sidebar.slider("Regularización", 0.01, 10.0)
        params["kernel"] = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
    elif clf_name == "Árboles de Decisión":
        params["M"] = st.sidebar.slider("Profundidad Máxima", 2, 20)
        params["C"] = st.sidebar.selectbox("Criterio", ["gini", "entropy"])
    elif clf_name == "Bosque Aleatorio":
        params["N"] = st.sidebar.slider("n_estimators", 50, 500, step=50)
        params["M"] = st.sidebar.slider("Profundidad Máxima", 2, 20)
    elif clf_name == "Aumento de Gradiente":
        params["N"] = st.sidebar.slider("n_estimators", 50, 500, step=50)
        params["LR"] = st.sidebar.slider("Tasa de Aprendizaje", 0.01, 0.5)
    elif clf_name == "XGBoost":
        params["N"] = st.sidebar.slider("n_estimators", 50, 500, step=50)
        params["LR"] = st.sidebar.slider("Tasa de Aprendizaje", 0.01, 0.5)
    return params

params = add_parameter_ui(classifier_name)

# Obtener el clasificador basado en los parámetros
def get_classifier(clf_name, params):
    if clf_name == "Regresión Logística":
        clf = LogisticRegression(C=params["R"], max_iter=params["MI"])
    elif clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"], kernel=params["kernel"])
    elif clf_name == "Árboles de Decisión":
        clf = DecisionTreeClassifier(max_depth=params["M"], criterion=params["C"])
    elif clf_name == "Bosque Aleatorio":
        clf = RandomForestClassifier(n_estimators=params["N"], max_depth=params["M"])
    elif clf_name == "Aumento de Gradiente":
        clf = GradientBoostingClassifier(n_estimators=params["N"], learning_rate=params["LR"])
    elif clf_name == "XGBoost":
        clf = XGBClassifier(n_estimators=params["N"], learning_rate=params["LR"])
    return clf

clf = get_classifier(classifier_name, params)

# Entrenar y mostrar el modelo
def model():
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    return Y_pred, Y_test

Y_pred, Y_test = model()

# Mostrar gráficos y resultados
def compute(Y_pred, Y_test):
    c1, c2 = st.columns((4, 3))
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(Y_pred)), Y_pred, color="yellow", lw=5, label="Predicciones")
    plt.scatter(range(len(Y_test)), Y_test, color="red", label="Actual")
    plt.legend()
    fig_pred = plt.gcf()
    c1.pyplot(fig_pred)
    
    # Matriz de Confusión
    cm = confusion_matrix(Y_test, Y_pred)
    df_cm = pd.DataFrame(cm, index=["Alto Riesgo", "Bajo Riesgo"], columns=["Predicho Alto", "Predicho Bajo"])
    sns.heatmap(df_cm, annot=True, fmt='d')
    fig_cm = plt.gcf()
    c2.pyplot(fig_cm)

    # Métricas del modelo
    precision, recall, fscore, _ = score(Y_test, Y_pred, pos_label=1, average='binary')
    st.subheader("Métricas del Modelo:")
    st.text(f'Precisión: {round(precision, 3)}\nRecall: {round(recall, 3)}\nF1-Score: {round(fscore, 3)}')

compute(Y_pred, Y_test)

# Mostrar valores de usuario en tres columnas
def user_inputs_ui():
    col1, col2, col3 = st.columns(3)
    user_val = {}
    
    with col1:
        st.markdown("### Descripción de las Variables")
        age = st.number_input("Edad", 18, 100)
        user_val["age"] = age
        cp = st.number_input("Dolor en Pecho (cp)", 0, 4)
        user_val["cp"] = cp
        restecg = st.number_input("Electrocardiograma en reposo (restecg)", 0, 2)
        user_val["restecg"] = restecg

    with col2:
        sex = st.number_input("Sexo (0 = Mujer, 1 = Hombre)", 0, 1)
        user_val["sex"] = sex
        trtbps = st.number_input("Presión arterial en reposo (trtbps)", 80, 200)
        user_val["trtbps"] = trtbps
        thalachh = st.number_input("Frecuencia cardíaca máxima (thalachh)", 60, 200)
        user_val["thalachh"] = thalachh

    with col3:
        chol = st.number_input("Colesterol (chol)", 100, 600)
        user_val["chol"] = chol
        fbs = st.number_input("Azúcar en sangre (fbs)", 0, 1)
        user_val["fbs"] = fbs
        exng = st.number_input("Angina inducida por ejercicio (exng)", 0, 1)
        user_val["exng"] = exng

    return user_val

st.header("Valores del Usuario")
user_val = user_inputs_ui()

# Fin del código
st.markdown("<h2>¡Gracias por usar la aplicación!</h2>", unsafe_allow_html=True)
st.markdown("<h3>¡Espero que hayas disfrutado de la experiencia!</h3>", unsafe_allow_html=True)
