
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Cargar datos
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Limpieza
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df.drop(columns=["Cabin", "Ticket", "Name", "PassengerId"], inplace=True)

# Codificar variables
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Definir variables
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Modelo
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X, y)

# Streamlit App
st.title("¿Sobrevivirías al Titanic? 🛳️")
st.write("Ingresa tus datos para predecir si sobrevivirías")

# Entradas del usuario
pclass = st.selectbox("Clase del pasajero (1 = Primera, 2 = Segunda, 3 = Tercera)", [1, 2, 3])
sex = st.selectbox("Sexo", ["Hombre", "Mujer"])
age = st.slider("Edad", 0, 80, 30)
sibsp = st.number_input("Hermanos/cónyuge a bordo", 0, 10, 0)
parch = st.number_input("Padres/hijos a bordo", 0, 10, 0)
fare = st.slider("Tarifa del billete (£)", 0.0, 600.0, 50.0)
embarked = st.selectbox("Puerto de embarque", ["Southampton", "Cherbourg", "Queenstown"])

# Procesar inputs
sexo_cod = 0 if sex == "Hombre" else 1
embarked_cod = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}[embarked]

# Crear dataframe del pasajero
pasajero = pd.DataFrame([[pclass, sexo_cod, age, sibsp, parch, fare, embarked_cod]],
                        columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])

# Predicción
pred = modelo.predict(pasajero)[0]
st.subheader("Resultado:")
st.write("🎉 ¡Sobrevivirías!" if pred == 1 else "😢 No sobrevivirías...")
