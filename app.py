
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import janitor

# Personalización de colores en Streamlit
st.set_page_config(page_title="¿Sobrevivirías al Titanic?", page_icon="🛳️", layout="centered")

# Cargar datos
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Limpieza con Janitor
df = df.clean_names()  # Limpia los nombres de las columnas

# Usando SimpleImputer para manejar los valores faltantes de manera más efectiva
imputer = SimpleImputer(strategy="most_frequent")  # Imputación de moda para 'Embarked'
df["age"] = imputer.fit_transform(df[["age"]])
df["embarked"] = imputer.fit_transform(df[["embarked"]])

# Filtrar columnas innecesarias
df.drop(columns=["cabin", "ticket", "name", "passengerid"], inplace=True)

# Codificar variables
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["embarked"] = df["embarked"].map({"s": 0, "c": 1, "q": 2})

# Definir variables
X = df.drop("survived", axis=1)
y = df["survived"]

# Modelo
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X, y)

# Streamlit App
st.title("¿Sobrevivirías al Titanic? 🛳️")
st.write("Ingresa tus datos para predecir si sobrevivirías")

# Entradas del usuario
pclass = st.selectbox("Clase del pasajero (1 = Primera, 2 = Segunda, 3 = Tercera)", [1, 2, 3], key="pclass")
sex = st.selectbox("Sexo", ["Hombre", "Mujer"], key="sex")
age = st.slider("Edad", 0, 80, 30, key="age")
sibsp = st.number_input("Hermanos/cónyuge a bordo", 0, 10, 0, key="sibsp")
parch = st.number_input("Padres/hijos a bordo", 0, 10, 0, key="parch")
fare = st.slider("Tarifa del billete (£)", 0.0, 600.0, 50.0, key="fare")
embarked = st.selectbox("Puerto de embarque", ["Southampton", "Cherbourg", "Queenstown"], key="embarked")

# Procesar inputs
sexo_cod = 0 if sex == "Hombre" else 1
embarked_cod = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}[embarked]

# Crear dataframe del pasajero
pasajero = pd.DataFrame([[pclass, sexo_cod, age, sibsp, parch, fare, embarked_cod]],
                        columns=["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"])

# Predicción
pred = modelo.predict(pasajero)[0]
st.subheader("Resultado:")
if pred == 1:
    st.markdown('<p style="color:green; font-size: 24px;">🎉 ¡Sobrevivirías!</p>', unsafe_allow_html=True)
else:
    st.markdown('<p style="color:red; font-size: 24px;">😢 No sobrevivirías...</p>', unsafe_allow_html=True)
