"""
Entrena un clasificador de flores Iris (Setosa, Versicolor, Virginica) a
partir de las 4 medidas clasicas de la flor: largo y ancho del sepalo, largo
y ancho del petalo. Usa una red neuronal pequena hecha con Keras.

Guarda dos cosas:
  - modelo_iris.h5      : el modelo entrenado, en el formato original de Keras.
  - modelo_web/pesos_modelo.json : los pesos de esa misma red, mas la media y
                          desviacion usadas para normalizar los datos, para
                          poder reconstruir la red y usarla en el navegador
                          con TensorFlow.js (estaticos/js/script.js).
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

modelo.save("modelo_iris.h5")

# Exportamos los pesos a JSON para reconstruir esta misma red en el
# navegador con TensorFlow.js (el orden coincide con model.get_weights():
# [pesos_capa1, sesgo_capa1, pesos_capa2, sesgo_capa2, ...]).
pesos = [capa.tolist() for capa in modelo.get_weights()]

os.makedirs("modelo_web", exist_ok=True)
with open("modelo_web/pesos_modelo.json", "w", encoding="utf-8") as archivo:
    json.dump(
        {
            "especies": NOMBRES_ESPECIES,
            "media": media.tolist(),
            "desviacion": desviacion.tolist(),
            "pesos": pesos,
        },
        archivo,
    )

print("Listo: modelo_iris.h5 y modelo_web/pesos_modelo.json")
