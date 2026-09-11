"""
Clasificador de flores Iris — servidor Flask
---------------------------------------------
Sirve la página (que usa TensorFlow.js para predecir en el navegador, sin
pasar por el servidor) y expone un único endpoint:

  - /api/corregir : recibe las 4 medidas de una flor y la especie correcta,
                     ajusta el modelo de Keras con esos datos (unos pocos
                     pasos de entrenamiento) y guarda los pesos actualizados
                     en modelo_web/pesos_modelo.json. Así, la próxima vez que
                     cualquier visitante cargue la página (no solo quien
                     corrigió), el modelo ya viene corregido.

Importante sobre el plan gratuito de Render: el disco no es permanente
entre reinicios del servicio. Mientras el servicio siga corriendo, las
correcciones quedan guardadas y las ve cualquier visitante; si Render
reinicia el servicio (por inactividad, o al desplegar un cambio nuevo),
se vuelve a la última versión que subiste a GitHub.
"""

import json
import os
import threading

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from tensorflow import keras
from tensorflow.keras import layers

app = Flask(__name__, static_folder="estaticos", static_url_path="/estaticos")

RUTA_PESOS = "modelo_web/pesos_modelo.json"
RUTA_H5 = "modelo_iris.h5"
ESPECIES = ["Setosa", "Versicolor", "Virginica"]

# Protege al modelo de que dos correcciones lleguen al mismo tiempo desde
# distintos visitantes.
candado_modelo = threading.Lock()


def construir_modelo():
    modelo = keras.Sequential(
        [
            layers.Input(shape=(4,)),
            layers.Dense(8, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(3, activation="softmax"),
        ]
    )
    modelo.compile(optimizer=keras.optimizers.Adam(0.05), loss="sparse_categorical_crossentropy")
    return modelo


with open(RUTA_PESOS, encoding="utf-8") as archivo:
    estado_inicial = json.load(archivo)

# Se guarda una copia de los pesos originales (los que vinieron de
# entrenar_modelo.py) para poder restablecerlos si hace falta.
pesos_originales = estado_inicial["pesos"]
media = np.array(estado_inicial["media"], dtype="float32")
desviacion = np.array(estado_inicial["desviacion"], dtype="float32")

modelo = construir_modelo()
modelo.set_weights([np.array(capa, dtype="float32") for capa in estado_inicial["pesos"]])


def guardar_pesos_actuales():
    pesos = [capa.tolist() for capa in modelo.get_weights()]
    with open(RUTA_PESOS, "w", encoding="utf-8") as archivo:
        json.dump(
            {"especies": ESPECIES, "media": media.tolist(), "desviacion": desviacion.tolist(), "pesos": pesos},
            archivo,
        )
    modelo.save(RUTA_H5)


@app.route("/")
@app.route("/index.html")
def index():
    return send_from_directory(".", "index.html")


@app.route("/datos.html")
def pagina_datos():
    return send_from_directory(".", "datos.html")


@app.route("/modelo_web/<path:nombre_archivo>")
def modelo_web(nombre_archivo):
    respuesta = send_from_directory("modelo_web", nombre_archivo)
    # Sin caché: si el modelo se corrigió, el navegador tiene que traer la
    # versión nueva sí o sí, no una copia vieja guardada localmente.
    respuesta.headers["Cache-Control"] = "no-store"
    return respuesta


@app.route("/api/corregir", methods=["POST"])
def corregir():
    datos = request.get_json(force=True, silent=True) or {}
    valores = datos.get("valores")
    especie_correcta = datos.get("especie")

    if not isinstance(valores, list) or len(valores) != 4 or especie_correcta not in ESPECIES:
        return jsonify({"error": "Faltan datos o la especie no es válida."}), 400

    indice = ESPECIES.index(especie_correcta)
    entrada = (np.array(valores, dtype="float32") - media) / desviacion
    entrada = entrada.reshape(1, 4)
    etiqueta = np.array([indice])

    with candado_modelo:
        modelo.fit(entrada, etiqueta, epochs=15, verbose=0)
        guardar_pesos_actuales()

    return jsonify({"ok": True})


@app.route("/api/restablecer", methods=["POST"])
def restablecer():
    with candado_modelo:
        modelo.set_weights([np.array(capa, dtype="float32") for capa in pesos_originales])
        guardar_pesos_actuales()

    return jsonify({"ok": True})


if __name__ == "__main__":
    # Render (y la mayoría de plataformas en la nube) asignan el puerto por
    # la variable de entorno PORT y hay que escuchar en 0.0.0.0, no en
    # 127.0.0.1. En tu computadora, sin esa variable, sigue usando el 5000.
    puerto = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=puerto, debug=False)
