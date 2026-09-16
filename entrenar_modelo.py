"""
Entrena un clasificador de flores Iris (Setosa, Versicolor, Virginica) a
partir de las 4 medidas clasicas de la flor: largo y ancho del sepalo, largo
y ancho del petalo. Usa una red neuronal pequena hecha con Keras.

Guarda dos cosas en modelo_web/:
  - modelo_iris.h5     : el modelo entrenado (arquitectura + pesos), en el
                          formato de Keras. Es el modelo que carga app.py
                          para predecir y para corregir.
  - configuracion.json : la media y desviacion usadas para normalizar los
                          datos (hace falta aplicar la misma normalizacion
                          antes de cada prediccion), mas la lista de
                          especies en el orden que usa el modelo.
"""

import json
import os

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

NOMBRES_ESPECIES = ["Setosa", "Versicolor", "Virginica"]

datos = load_iris()
X = datos.data.astype("float32")
y = datos.target

# Normalizamos cada medida (media 0, desviación 1) para que la red entrene
# mejor. Guardamos la media y la desviación para aplicar esta misma
# normalización más tarde, en el navegador.
media = X.mean(axis=0)
desviacion = X.std(axis=0)
X_normalizado = (X - media) / desviacion

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X_normalizado, y, test_size=0.2, random_state=42, stratify=y
)

modelo = keras.Sequential(
    [
        layers.Input(shape=(4,)),
        layers.Dense(8, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ]
)

modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
modelo.fit(X_entrenamiento, y_entrenamiento, epochs=300, verbose=0)

perdida, precision = modelo.evaluate(X_prueba, y_prueba, verbose=0)
print(f"Precisión en datos de prueba: {precision:.2%}")

os.makedirs("modelo_web", exist_ok=True)
modelo.save("modelo_web/modelo_iris.h5")

with open("modelo_web/configuracion.json", "w", encoding="utf-8") as archivo:
    json.dump(
        {
            "especies": NOMBRES_ESPECIES,
            "media": media.tolist(),
            "desviacion": desviacion.tolist(),
        },
        archivo,
    )

print("Listo: modelo_web/modelo_iris.h5 y modelo_web/configuracion.json")
