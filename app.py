
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import janitor

# Personalización de la página
st.set_page_config(page_title="¿Sobrevivirías al Titanic?", page_icon="🛳️", layout="centered")
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
        }
        .main {
            background-color: #f5f5f5;
        }
        h1, h2, h3, h4 {
            color: #0097b2;
        }
        .stButton>button {
            background-color: #0097b2;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #f9a620;
            color: black;
        }
        /* Slider personalizado */
        input[type=range]::-webkit-slider-thumb {
            background: #0097b2;
        }
        input[type=range]::-webkit-slider-runnable-track {
            background: #f9a620;
            height: 8px;
            border-radius: 8px;
        }
        input[type=range]::-moz-range-thumb {
            background: #0097b2;
        }
        input[type=range]::-moz-range-track {
            background: #f9a620;
            height: 8px;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Cargar datos
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Limpieza de nombres de columnas
df = df.clean_names()

# Imputación de datos faltantes
imputer = SimpleImputer(strategy="most_frequent")
df["age"] = imputer.fit_transform(df[["age"]]).ravel()
df["embarked"] = imputer.fit_transform(df[["embarked"]]).ravel()

# Eliminar columnas innecesarias
df.drop(columns=["cabin", "ticket", "name", "passengerid"], inplace=True)

# Codificación de variables categóricas
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["embarked"] = df["embarked"].map({"S": 0, "C": 1, "Q": 2})

# Definir variables predictoras y objetivo
X = df.drop("survived", axis=1)
y = df["survived"]

# Modelo de árbol de decisión
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X, y)

# Interfaz de usuario
st.title("¿Sobrevivirías al Titanic? 🛳️")
st.write("Ingresa tus datos para predecir si habrías sobrevivido.")

# Entradas del usuario
pclass = st.selectbox("Clase del pasajero (1 = Primera, 2 = Segunda, 3 = Tercera)", [1, 2, 3])
sex = st.selectbox("Sexo", ["Hombre", "Mujer"])
age = st.slider("Edad", 0, 80, 30)
sibsp = st.number_input("Cónyuge y/o Hermanos a bordo", 0, 10, 0)
parch = st.number_input("Padres y/o hijos a bordo", 0, 10, 0)
fare = st.slider("Tarifa del billete (£)", 0.0, 600.0, 50.0)
embarked = st.selectbox("Puerto de embarque", ["Southampton", "Cherbourg", "Queenstown"])

# Preprocesamiento de entrada del usuario
sexo_cod = 0 if sex == "Hombre" else 1
embarked_cod = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}[embarked]

# Crear dataframe del pasajero
pasajero = pd.DataFrame([[pclass, sexo_cod, age, sibsp, parch, fare, embarked_cod]],
                        columns=["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"])

# Predicción
pred = modelo.predict(pasajero)[0]

# Mostrar resultado
st.subheader("Resultado:")
if pred == 1:
    st.markdown('<p style="color:green; font-size: 24px;">🎉 ¡Sobrevivirías!</p>', unsafe_allow_html=True)
else:
    st.markdown('<p style="color:red; font-size: 24px;">😢 No sobrevivirías...</p>', unsafe_allow_html=True)
