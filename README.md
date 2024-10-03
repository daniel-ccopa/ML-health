# ML-Healthcare-Web-App

Esta es una aplicación web interactiva de Machine Learning "ML en Salud", desarrollada usando Python y StreamLit. Utiliza algoritmos de Machine Learning para construir modelos precisos y potentes que predicen el riesgo (Alto / Bajo) de que el usuario sufra un ataque cardíaco o cáncer de mama, en función de atributos específicos del usuario como edad, sexo, frecuencia cardíaca, nivel de azúcar en sangre, entre otros.

## **Ver la Aplicación Aquí:**

[![StreamLit App](https://static.streamlit.io/badges/streamlit_badge_white.svg)](https://share.streamlit.io/advikmaniar/ml-healthcare-web-app/main/ML_Healthcare.py)

---

Esta aplicación tiene dos secciones principales:

## 1) - Construcción de Modelos

En esta sección, se construyen 7 modelos diferentes utilizando diversos algoritmos de Machine Learning. Estos son:

1. Regresión Logística
1. KNN
1. SVM
1. Árboles de Decisión
1. Random Forest
1. Gradient Boosting
1. XGBoost

Los modelos se entrenan usando los datos disponibles en https://archive.ics.uci.edu/ml/index.php, particularmente los datasets de [Predicción de Ataque Cardíaco](https://github.com/daniel-ccopa/ML-health/blob/main/Data/heart.csv) y [Cáncer de Mama (Wisconsin)](https://github.com/daniel-ccopa/ML-health/blob/main/Data/BreastCancer.csv).

Se ha creado un panel lateral interactivo utilizando la llamada `st.sidebar` de Streamlit, que permite al usuario realizar las siguientes acciones:
1. Elegir el conjunto de datos - `Ataque Cardíaco / Cáncer de Mama`
2. Elegir el algoritmo - `Regresión Logística, KNN, SVM, Árboles de Decisión, Random Forest, Gradient Boosting, XGBoost.`
3. Modificar los parámetros importantes para cada modelo - `Tasa de Aprendizaje, Random State, Coeficiente de Regularización, Gamma, Kernel, n_estimators`, etc.

Después de entrenar usando los parámetros seleccionados por el usuario, el modelo ajustado se construye y está listo para ser probado con nuestros datos de prueba. Se muestran el gráfico de clasificación y la matriz de confusión para el modelo seleccionado, junto con las métricas del modelo: `Exactitud, Precisión, Recall, F1-Score, Error Cuadrático Medio, Tiempo de Ejecución`. El usuario puede observar cambios en tiempo real en los gráficos y métricas a medida que modifica los parámetros del modelo.

> **Esta es una excelente forma de comprender los diferentes algoritmos de ML y cómo se ven influenciados al ajustar los hiperparámetros.**

![imagen](https://raw.githubusercontent.com/daniel-ccopa/ML-health/refs/heads/main/Results/Section%201%20-%20Model.PNG)

Los 7 modelos (ajustados óptimamente) tuvieron el siguiente rendimiento:
`Criterio: Exactitud`
Modelo | Exactitud (Ataque Cardíaco / Cáncer de Mama)
------------ | -------------
Regresión Logística | **91.803% / 100.0%**
KNN | **86.89% / 96.49%**
SVM | **93.44% / 100.0%**
Árboles de Decisión | **52.56% / 60.53%**
Random Forest | **90.164% / 98.24%**
Gradient Boosting | **88.53% / 96.49%**
XGBoost | **95.08% / 94.737%**

## 2) - Predicción del Usuario

En esta sección, el usuario puede utilizar cualquier modelo desarrollado anteriormente para predecir su riesgo (Alto / Bajo) usando sus propios valores. (Ya sea para Ataque Cardíaco o Cáncer de Mama)

![imagen](https://raw.githubusercontent.com/daniel-ccopa/ML-health/refs/heads/main/Results/Section%202%20-%20User%20(1).PNG)

![imagen](https://raw.githubusercontent.com/daniel-ccopa/ML-health/refs/heads/main/Results/Section%202%20-%20User%20(2).PNG)

Puedes ver el video final [aquí](Results/Video.mp4).

---

# ¡Gracias!

---